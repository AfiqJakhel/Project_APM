"""
train.py
========
Expanding Window Cross-Validation untuk Prediksi Harga Cabai (Merah & Rawit)
Model XGBoost multi-horizon H+1, H+3, H+7

Perubahan v6 (Unified):
  ✓ Menggabungkan training untuk Cabai Merah dan Cabai Rawit dalam satu script.
  ✓ Output folder dipisah: expanding_window_merah/ dan expanding_window_rawit/.
  ✓ Target: DELTA (perubahan harga), dengan inverse transform ke harga absolut.

Cara pakai:
  python machine_learning/train.py

Dependencies:
  pip install xgboost scikit-learn pandas numpy matplotlib joblib shap
"""

import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
import sys
import io
import time

# Fix unicode error on windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data" / "processed"
RANDOM_STATE = 42

# Konfigurasi Expanding Window
MIN_TRAIN_DAYS = 365
STEP_DAYS      = 30
TEST_DAYS      = 30

LOG_LINES: list = []

def log(msg: str) -> None:
    print(msg)
    LOG_LINES.append(msg)


# =============================================================================
# KELAS KONFIGURASI KOMODITAS
# =============================================================================
class KomoditasConfig:
    def __init__(self, nama, folder_output, target_utama, target_map, non_features, ref_inverse):
        self.nama = nama
        self.folder_output = folder_output
        self.model_dir = BASE_DIR / "machine_learning" / "output" / folder_output
        self.plots_dir = self.model_dir / "plots"
        self.metrics_dir = self.model_dir / "metrics"
        self.target_utama = target_utama
        self.target_map = target_map
        self.non_features = non_features
        self.ref_inverse = ref_inverse

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)


CONFIGS = {
    "merah": KomoditasConfig(
        nama="Cabai Merah",
        folder_output="expanding_window_merah",
        target_utama="harga_cabai_merah",
        target_map={
            "h1": "target_h1",
            "h3": "target_h3",
            "h7": "target_h7",
        },
        non_features=[
            "tanggal", "harga_cabai_rawit", "harga_cabai_merah",
            "target_h1", "target_h3", "target_h7",
            "target_rawit_h1", "target_rawit_h3", "target_rawit_h7",
            "arah_target_h1", "arah_target_h3", "arah_target_h7",
            "harga_hari_ini", "harga_rawit_hari_ini",
        ],
        ref_inverse="harga_hari_ini",
    ),
    "rawit": KomoditasConfig(
        nama="Cabai Rawit",
        folder_output="expanding_window_rawit",
        target_utama="harga_cabai_rawit",
        target_map={
            "rawit_h1": "target_rawit_h1",
            "rawit_h3": "target_rawit_h3",
            "rawit_h7": "target_rawit_h7",
        },
        non_features=[
            "tanggal", "harga_cabai_rawit", "harga_cabai_merah",
            "target_h1", "target_h3", "target_h7",
            "target_rawit_h1", "target_rawit_h3", "target_rawit_h7",
            "arah_target_h1", "arah_target_h3", "arah_target_h7",
            "harga_hari_ini", "harga_rawit_hari_ini",
        ],
        ref_inverse="harga_rawit_hari_ini",
    )
}


# =============================================================================
# BAGIAN 1 — LOAD DATASET & SPLIT WINDOWS
# =============================================================================
def load_dataset() -> pd.DataFrame:
    log("\n[1] Memuat dataset_preprocessed.csv ...")
    path = DATA_DIR / "dataset_preprocessed.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} tidak ditemukan. Jalankan preprocessing.py dulu.")

    df = pd.read_csv(path, parse_dates=["tanggal"])
    df = df.sort_values("tanggal").reset_index(drop=True)
    log(f"    -> Shape   : {df.shape}")
    log(f"    -> Rentang : {df['tanggal'].min().date()} s/d {df['tanggal'].max().date()}")
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    log("\n[1B] Menambahkan fitur kalender dan hari raya (menggunakan library)...")
    df = df.copy()
    
    from hijridate import Gregorian, Hijri
    import holidays
    
    # 1. Fitur Kalender Dasar
    df['day_of_week'] = df['tanggal'].dt.dayofweek
    df['day_of_month'] = df['tanggal'].dt.day
    df['week_of_year'] = df['tanggal'].dt.isocalendar().week.astype(int)
    df['month'] = df['tanggal'].dt.month
    
    # 5. is_weekend (0/1)
    # Cukup menggunakan pandas karena akhir pekan di Indonesia standar (Sabtu-Minggu)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # (Opsional) Menggunakan library 'holidays' khusus Indonesia
    id_holidays = holidays.ID()
    df['is_libur_nasional_lib'] = df['tanggal'].apply(lambda x: 1 if x in id_holidays else 0)
    
    # 6. is_ramadan (0/1) menggunakan hijri-converter
    def check_ramadan(date):
        try:
            h = Gregorian(date.year, date.month, date.day).to_hijri()
            return 1 if h.month == 9 else 0
        except OverflowError:
            return 0
            
    df['is_ramadan'] = df['tanggal'].apply(check_ramadan)
        
    # 7. days_to_lebaran — SUDAH DIPINDAHKAN ke preprocessing.py
    # Tidak perlu dihitung ulang di sini karena sudah ada di dataset_preprocessed.csv
    # (pemindahan ini memastikan fitur tersedia saat inference di production)
    # Baris lama di-skip agar tidak menimpa kolom yang lebih akurat dari preprocessing.
    if 'days_to_lebaran' not in df.columns:
        # Fallback jika belum ada di dataset (backward compat)
        lebaran_cache = {}
        min_year = df['tanggal'].min().year
        max_year = df['tanggal'].max().year
        for y in range(min_year, max_year + 2):
            for hy in [y - 580, y - 579, y - 578]:
                lebaran_g = Hijri(hy, 10, 1).to_gregorian()
                if lebaran_g.year == y:
                    lebaran_cache[y] = pd.Timestamp(year=lebaran_g.year, month=lebaran_g.month, day=lebaran_g.day)
                    break
        def get_days_to_lebaran(date):
            lebaran_this_year = lebaran_cache.get(date.year)
            if lebaran_this_year:
                return (lebaran_this_year - date).days
            return 0
        df['days_to_lebaran'] = df['tanggal'].apply(get_days_to_lebaran)
    
    # 8. is_natal_newyear (minggu ke-3 des - minggu ke-1 jan, ~15 Des s/d 7 Jan)
    df['is_natal_newyear'] = (
        ((df['tanggal'].dt.month == 12) & (df['tanggal'].dt.day >= 15)) |
        ((df['tanggal'].dt.month == 1) & (df['tanggal'].dt.day <= 7))
    ).astype(int)
    
    # 9. is_awal_bulan (1-5)
    df['is_awal_bulan'] = (df['tanggal'].dt.day <= 5).astype(int)
    
    # 10. is_akhir_bulan (25-31)
    df['is_akhir_bulan'] = (df['tanggal'].dt.day >= 25).astype(int)
    
    # Fitur kalender mungkin sudah ada di df dengan nama 'bulan', 'is_natal_tahunbaru'
    # biarkan saja atau fitur ini akan menimpa/menambah. XGBoost akan menyeleksi.
    log(f"    -> Fitur kalender berhasil ditambahkan. Total fitur saat ini: {df.shape[1] - 2}")
    return df

def buang_fitur_sampah(df: pd.DataFrame) -> pd.DataFrame:
    # FITUR YANG DIBUANG: hanya yang secara statistik terbukti redundan atau bising.
    # Fitur cuaca (curah_hujan, lag_hujan, roll_hujan) TIDAK lagi dibuang —
    # biarkan XGBoost menilai kontribusinya sendiri melalui tree splitting.
    # Fitur baru roll_hujan_60 dan hujan_ekstrem_3hari tetap dipertahankan.
    sampah = [
        # 1. Cuaca harian mentah yang memang lemah secara individual
        #    (suhu, kelembaban, penyinaran, angin) tetap dibuang karena benar-benar
        #    tidak berpola. Namun curah_hujan dan turunannya KINI DIPERTAHANKAN.
        "suhu_rata", "kelembaban", "lama_penyinaran", "kec_angin",
        # 2. Kalender Redundan
        "bulan", "kuartal", "is_akhir_bulan", "is_natal_newyear", "is_libur_nasional",
        # 3. Arah yang Kalah Informatif dari pct_change
        "arah_lag1", "arah_lag2", "arah_lag3"
    ]
    to_drop = [c for c in sampah if c in df.columns]
    if to_drop:
        log(f"\n[1C] Membuang {len(to_drop)} fitur bising (Feature Selection XGBoost)...")
        df = df.drop(columns=to_drop)
        log(f"    -> Fitur curah hujan & turunannya DIPERTAHANKAN untuk dievaluasi XGBoost")
        log(f"    -> Fitur tersisa untuk XGBoost: {df.shape[1] - 2} fitur")
    return df


def buat_expanding_windows(df: pd.DataFrame) -> list[dict]:
    log(f"\n[2] Membuat expanding windows ...")
    total = len(df)
    windows = []
    window_id = 1
    train_end = MIN_TRAIN_DAYS

    while True:
        test_end = train_end + TEST_DAYS
        if test_end > total:
            break
        windows.append({
            "window"               : window_id,
            "train_start"          : 0,
            "train_end"            : train_end,
            "test_start"           : train_end,
            "test_end"             : test_end,
            "tanggal_test_mulai"   : df["tanggal"].iloc[train_end],
            "tanggal_test_selesai" : df["tanggal"].iloc[test_end - 1],
        })
        train_end += STEP_DAYS
        window_id += 1

    log(f"    -> Total windows : {len(windows)}")
    return windows


# =============================================================================
# BAGIAN 2 — METRIK & ML
# =============================================================================
def hitung_da(y_true: np.ndarray, y_pred: np.ndarray, ref_val: np.ndarray) -> float:
    # Arah pergerakan dievaluasi antara harga target masa depan terhadap harga saat prediksi dibuat (ref_val)
    diff_true = y_true - ref_val
    diff_pred = y_pred - ref_val
    mask = diff_true != 0
    if np.sum(mask) == 0: return float("nan")
    return float(np.mean(np.sign(diff_true[mask]) == np.sign(diff_pred[mask])) * 100)

def hitung_semua_metrik(y_true: np.ndarray, y_pred: np.ndarray, ref_val: np.ndarray) -> dict:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    smape = float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)
    
    mask_mape = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask_mape] - y_pred[mask_mape]) / y_true[mask_mape])) * 100)
    
    return {
        "MAE"  : float(mean_absolute_error(y_true, y_pred)),
        "RMSE" : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE" : mape,
        "sMAPE": smape,
        "R2"   : float(r2_score(y_true, y_pred)),
        "DA"   : hitung_da(y_true, y_pred, ref_val),
    }

def buat_sample_weights(n: int, decay: float = 0.001) -> np.ndarray:
    """
    Data terbaru mendapat bobot mendekati 1.0,
    data terlama mendapat bobot mendekati exp(-decay * n).
    decay=0.001 → data 1000 hari lalu bobotnya ~0.37
    """
    indices = np.arange(n)
    weights = np.exp(decay * (indices - n + 1))
    return weights / weights.mean()

def pisahkan_fitur(df_split: pd.DataFrame, target_col: str, non_features: list):
    drop_cols = [c for c in non_features if c in df_split.columns]
    return df_split.drop(columns=drop_cols), df_split[target_col]

def train_xgb_window(X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame, params: dict | None = None) -> tuple:
    if params is None: params = {}
    split_val = max(int(len(X_train) * 0.9), len(X_train) - 60)
    X_tr, X_val = X_train.iloc[:split_val], X_train.iloc[split_val:]
    y_tr, y_val = y_train.iloc[:split_val], y_train.iloc[split_val:]

    decay = params.get("decay", 0.001)
    w_tr = buat_sample_weights(len(X_tr), decay)
    xgb_params = {k: v for k, v in params.items() if k != "decay"}

    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_STATE, verbosity=0, eval_metric="rmse", early_stopping_rounds=50,
        n_estimators=xgb_params.get("n_estimators", 150), max_depth=xgb_params.get("max_depth", 2),
        learning_rate=xgb_params.get("learning_rate", 0.05), subsample=xgb_params.get("subsample", 0.6),
        colsample_bytree=xgb_params.get("colsample_bytree", 0.6), min_child_weight=xgb_params.get("min_child_weight", 5),
        gamma=xgb_params.get("gamma", 0.5), reg_alpha=xgb_params.get("reg_alpha", 1.0), reg_lambda=xgb_params.get("reg_lambda", 10.0),
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
    return model.predict(X_test), model

def ridge_baseline_window(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    model = Ridge(alpha=1.0)
    model.fit(X_tr_sc, y_train)
    return model.predict(X_te_sc)


# =============================================================================
# BAGIAN 3 — PIPELINE PER KOMODITAS
# =============================================================================
def proses_komoditas(df: pd.DataFrame, windows: list[dict], cfg: KomoditasConfig):
    # [FIX-2]
    global LOG_LINES
    LOG_LINES = []

    log(f"\n{'='*70}")
    log(f"  MEMULAI PROSES: {cfg.nama.upper()}")
    log(f"{'='*70}")

    all_results = {}
    
    for label, target_col in cfg.target_map.items():
        if target_col not in df.columns or df[target_col].notna().sum() < MIN_TRAIN_DAYS:
            log(f"  ! Skip {label}: Kolom target tidak cukup data.")
            continue
            
        log(f"\n[PROSES] {label.upper()} ({target_col})")
        
        # ── 1. Hyperparameter Tuning ──
        best_params = tuning_final(df, target_col, label, cfg)
        
        # ── 2. Expanding Window Eval ──
        df_results = expanding_window_eval(df, windows, target_col, label, cfg, best_params)
        all_results[label] = df_results
        
        # ── 3. Plot Expanding Window ──
        plot_expanding_window(df_results, label, cfg)
        
        # ── 4. Train Model Final (Seluruh Data) ──
        model, features = latih_model_final(df, target_col, label, cfg, best_params)
        
        # ── 5. SHAP Analysis ──
        # [FIX-5]
        df_valid_shap = df.dropna(subset=[target_col])
        X_shap, _ = pisahkan_fitur(df_valid_shap.iloc[-500:], target_col, cfg.non_features)
        shap_final(model, X_shap, label, cfg)

    # Laporan Akhir Komoditas
    laporan_ringkasan(all_results, cfg)


# ── Modul-modul internal pipeline ──

def tuning_final(df: pd.DataFrame, target_col: str, label: str, cfg: KomoditasConfig) -> dict:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    import json
    from datetime import datetime

    cache_path = cfg.metrics_dir / f"best_params_{label}.json"
    
    # FIX 3: Ambil dictionary 'params' saja saat membaca format JSON yang baru
    # [FIX-6] Aktifkan kembali cache dengan flag CLI "--no-cache"
    USE_CACHE = "--no-cache" not in sys.argv
    if USE_CACHE and cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
            log(f"    -> Cache ditemukan, lewati tuning ({label})")
            return data.get("params", data)
        
    log(f"    -> Tuning Hyperparameter (Optuna + Time-Decay Weights)...")
    df_valid = df.dropna(subset=[target_col]).copy()
    X, y = pisahkan_fitur(df_valid, target_col, cfg.non_features)
    
    # FIX 1 & 2: Ambil gap dari label dan pertahankan arsitektur cfg
    # [FIX-3]
    import re
    match = re.search(r'h(\d+)$', label)
    gap = int(match.group(1)) if match else 1
    tscv = TimeSeriesSplit(n_splits=10, gap=gap)
    
    def objective(trial):
        param = {
            "n_estimators"     : trial.suggest_int("n_estimators", 100, 300),
            "max_depth"        : trial.suggest_int("max_depth", 2, 4),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "min_child_weight" : trial.suggest_int("min_child_weight", 5, 15),
            "gamma"            : trial.suggest_float("gamma", 0.0, 0.01),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 0.0, 0.1),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 0.1, 5.0),
        }
        decay = trial.suggest_float("decay", 0.0005, 0.005)

        mses = []
        for step, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            
            w_tr = buat_sample_weights(len(train_idx), decay)
            
            model = XGBRegressor(
                objective="reg:squarederror", tree_method="hist", random_state=RANDOM_STATE, verbosity=0,
                **param
            )
            model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
            preds = model.predict(X_va)
            mse = mean_squared_error(y_va, preds)
            mses.append(mse)
            
            trial.report(mse, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        return np.mean(mses)
        
    # Mute optuna logger
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=MedianPruner())
    study.optimize(objective, n_trials=200, timeout=300)
    
    best = study.best_params
    best_mse = study.best_value
    
    log(f"    -> Best MSE (CV): {best_mse:,.6f}")
    
    # FIX 4: Simpan dengan metadata berstruktur
    output = {
        "params": best,
        "best_cv_mse": round(best_mse, 6),
        "label": label,
        "horizon_days": gap,
        "n_splits": 10,
        "n_iter": 200,
        "tuning_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(cache_path, "w") as f:
        json.dump(output, f, indent=4)
        
    # Kembalikan hanya params untuk training
    return best


def expanding_window_eval(df: pd.DataFrame, windows: list[dict], target_col: str, label: str, cfg: KomoditasConfig, params: dict) -> pd.DataFrame:
    log(f"    -> Expanding Window Evaluation")
    records = []
    global_y_true = []
    global_y_pred = []
    
    for w in windows:
        df_tr = df.iloc[w["train_start"]:w["train_end"]].dropna(subset=[target_col])
        df_te = df.iloc[w["test_start"]:w["test_end"]].dropna(subset=[target_col])
        if len(df_tr) < 100 or len(df_te) < 5: continue

        X_train, y_train = pisahkan_fitur(df_tr, target_col, cfg.non_features)
        X_test,  y_test  = pisahkan_fitur(df_te, target_col, cfg.non_features)

        # Inverse transform: log-return -> harga absolut
        # prediksi_rp = harga_hari_ini * exp(log_return)
        ref_test = df_te[cfg.ref_inverse].values
        y_true_abs = ref_test * np.exp(y_test.values)

        # XGBoost
        try:
            y_pred_log_xgb, _ = train_xgb_window(X_train, y_train, X_test, params)
            y_pred_abs_xgb = ref_test * np.exp(y_pred_log_xgb)
            # Kliping: prediksi tidak boleh < Rp 1000 (sanity check)
            y_pred_abs_xgb = np.clip(y_pred_abs_xgb, 1000, None)
            m_xgb = hitung_semua_metrik(y_true_abs, y_pred_abs_xgb, ref_test)
            global_y_true.extend(y_true_abs)
            global_y_pred.extend(y_pred_abs_xgb)
        except Exception: continue

        # Ridge
        try:
            y_pred_log_ridge = ridge_baseline_window(X_train, y_train, X_test)
            y_pred_abs_ridge = np.clip(ref_test * np.exp(y_pred_log_ridge), 1000, None)
            m_ridge = hitung_semua_metrik(y_true_abs, y_pred_abs_ridge, ref_test)
        except Exception as e:
            # [FIX-4]
            log(f"    ! Ridge error di window {w['window']}: {e}")
            m_ridge = {"MAE": 0, "RMSE": 0, "DA": 0}

        # Naive (prediksi log_return=0 -> harga = harga hari ini)
        m_naive = hitung_semua_metrik(y_true_abs, ref_test, ref_test)

        records.append({
            "window"              : w["window"],
            "tanggal_test_mulai"  : w["tanggal_test_mulai"].date(),
            "xgb_MAE": m_xgb["MAE"], "xgb_RMSE": m_xgb["RMSE"], "xgb_MAPE": m_xgb["MAPE"], "xgb_sMAPE": m_xgb["sMAPE"], "xgb_DA": m_xgb["DA"], "xgb_R2": m_xgb["R2"],
            "ridge_MAE": m_ridge["MAE"], "naive_MAE": m_naive["MAE"], "naive_DA": m_naive["DA"],
        })

    df_res = pd.DataFrame(records)
    if len(df_res) > 0:
        from sklearn.metrics import r2_score, mean_squared_error
        xgb_m, n_m = df_res['xgb_MAE'].mean(), df_res['naive_MAE'].mean()
        
        # Kalkulasi R2 Global dari seluruh OOS predictions (bukan rata-rata dari sub-window)
        global_r2 = r2_score(global_y_true, global_y_pred)
        global_rmse = mean_squared_error(global_y_true, global_y_pred) ** 0.5
        
        log(f"       MAE: XGB Rp{xgb_m:,.0f} | Naive Rp{n_m:,.0f} | DA: {df_res['xgb_DA'].mean():.1f}%")
        log(f"       R² : {global_r2:.4f} | RMSE: Rp{global_rmse:,.0f} | (Global Out-of-Sample)")
        
        # Simpan nilai global ke dataframe agar CSV laporan membaca nilai yang benar
        df_res['xgb_R2'] = global_r2
        df_res['xgb_RMSE'] = global_rmse
        
    return df_res


def plot_expanding_window(df_results: pd.DataFrame, label: str, cfg: KomoditasConfig):
    if len(df_results) == 0: return
    tanggal = pd.to_datetime(df_results["tanggal_test_mulai"])
    plt.figure(figsize=(12, 5))
    plt.plot(tanggal, df_results["xgb_MAE"], "o-", label="XGBoost")
    plt.plot(tanggal, df_results["naive_MAE"], "^:", label="Naive (Harga Tetap)")
    plt.title(f"MAE per Window — {cfg.nama} ({label.upper()})")
    plt.ylabel("MAE (Rp)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(cfg.plots_dir / f"expanding_window_{label}.png", bbox_inches="tight")
    plt.close()


def latih_model_final(df: pd.DataFrame, target_col: str, label: str, cfg: KomoditasConfig, params: dict):
    df_valid = df.dropna(subset=[target_col]).copy()
    X, y = pisahkan_fitur(df_valid, target_col, cfg.non_features)
    
    # Train full data (eval set on last 10% to prevent overfitting)
    split_val = int(len(X) * 0.9)
    X_tr, y_tr = X.iloc[:split_val], y.iloc[:split_val]
    
    decay = params.get("decay", 0.001)
    w_tr = buat_sample_weights(len(X_tr), decay)
    xgb_params = {k: v for k, v in params.items() if k != "decay"}
    
    # [FIX-1]
    model = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=RANDOM_STATE, n_estimators=xgb_params.get("n_estimators", 150), max_depth=xgb_params.get("max_depth", 2), learning_rate=xgb_params.get("learning_rate", 0.05), subsample=xgb_params.get("subsample", 0.6), colsample_bytree=xgb_params.get("colsample_bytree", 0.6), min_child_weight=xgb_params.get("min_child_weight", 5), gamma=xgb_params.get("gamma", 0.5), reg_alpha=xgb_params.get("reg_alpha", 1.0), reg_lambda=xgb_params.get("reg_lambda", 10.0))
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X.iloc[split_val:], y.iloc[split_val:])], verbose=False)

    # Save Model & Features
    ts = datetime.now().strftime("%Y%m%d")
    model_path = cfg.model_dir / f"model_final_{label}_{ts}.pkl"
    joblib.dump(model, model_path)
    
    feat_path = cfg.model_dir / f"feature_cols_{label}.json"
    with open(feat_path, "w") as f: json.dump(list(X.columns), f, indent=2)
    
    log(f"    -> Model Final: {model_path.name} | Fitur: {len(X.columns)}")
    return model, list(X.columns)


def shap_final(model, X_sample: pd.DataFrame, label: str, cfg: KomoditasConfig):
    if not SHAP_AVAILABLE: return
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="dot", max_display=15, show=False)
        plt.title(f"SHAP Summary — {cfg.nama} ({label.upper()})")
        plt.savefig(cfg.plots_dir / f"shap_summary_{label}.png", bbox_inches="tight")
        plt.close()
        
        # Cek dominasi lag_1 dan print top 10
        mean_abs = np.abs(shap_values).mean(axis=0)
        df_shap = pd.DataFrame({"fitur": X_sample.columns, "shap_mean": mean_abs}).sort_values("shap_mean", ascending=False)
        
        log(f"    -> SHAP Top 10 Fitur:")
        for i, row in df_shap.head(10).iterrows():
            log(f"         {row['fitur']:<20}: {row['shap_mean']:.4f}")
            
        if df_shap.iloc[0]["fitur"] == "lag_1":
            ratio = df_shap.iloc[0]["shap_mean"] / df_shap.iloc[1]["shap_mean"]
            log(f"    -> SHAP: lag_1 masih dominan ({ratio:.1f}x dari fitur #2)")
        else:
            log(f"    -> SHAP: lag_1 bukan #1 (Detrending berhasil. Fitur top: {df_shap.iloc[0]['fitur']})")
    except Exception as e:
        log(f"    ! SHAP error: {e}")


def laporan_ringkasan(all_results: dict, cfg: KomoditasConfig):
    log(f"\n  RINGKASAN AKHIR: {cfg.nama.upper()}")
    log(f"  {'Model':<10} {'XGB-MAE':>10} {'Naive-MAE':>10} {'Beat%':>6}")
    log(f"  {'─'*10} {'─'*10} {'─'*10} {'─'*6}")
    
    rows = []
    for label, df_res in all_results.items():
        if len(df_res) == 0: continue
        xgb_m = df_res['xgb_MAE'].mean()
        na_m = df_res['naive_MAE'].mean()
        beat = (df_res['xgb_MAE'] < df_res['naive_MAE']).mean() * 100
        log(f"  {label:<10} {xgb_m:>10,.0f} {na_m:>10,.0f} {beat:>5.1f}%")
        rows.append({
            "horizon": label, "xgb_MAE_mean": round(xgb_m, 2), "naive_MAE_mean": round(na_m, 2),
            "xgb_RMSE_mean": round(df_res["xgb_RMSE"].mean(), 2), "xgb_MAPE_mean": round(df_res["xgb_MAPE"].mean(), 4),
            "xgb_sMAPE_mean": round(df_res["xgb_sMAPE"].mean(), 4), "xgb_R2": round(df_res["xgb_R2"].mean(), 4),
            "xgb_DA_mean": round(df_res["xgb_DA"].mean(), 2), "beat_naive_%": round(beat, 1)
        })
        
    if rows:
        pd.DataFrame(rows).to_csv(cfg.metrics_dir / f"ringkasan_{cfg.folder_output}.csv", index=False)
        
    with open(cfg.model_dir / f"log_{cfg.folder_output}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    log("=" * 70)
    log("  TRAINING MODEL CABAI MERAH & RAWIT (UNIFIED v6 DELTA)")
    log("  Kota Padang, Sumatera Barat")
    log("=" * 70)

    start_time = time.time()
    df = load_dataset()
    df = add_calendar_features(df)
    df = buang_fitur_sampah(df)
    windows = buat_expanding_windows(df)

    # 1. Train Merah
    proses_komoditas(df, windows, CONFIGS["merah"])
    
    # 2. Train Rawit
    proses_komoditas(df, windows, CONFIGS["rawit"])

    log(f"\n[SELESAI] Semua model berhasil dilatih dalam {time.time()-start_time:.1f} detik.")

if __name__ == "__main__":
    main()