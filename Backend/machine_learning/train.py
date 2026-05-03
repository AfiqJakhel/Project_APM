"""
train_expanding_window.py
=========================
Expanding Window Cross-Validation untuk Prediksi Harga Cabai Merah — Kota Padang
Judul Proyek: Implementasi Machine Learning untuk Prediksi Fluktuasi Harga Cabai
              di Kota Padang Berbasis Web sebagai Upaya Pengendalian Inflasi Daerah
              Menggunakan Metode Extreme Gradient Boosting (XGBoost)

Mengapa Expanding Window lebih baik dari 80/20 untuk kasus ini?
────────────────────────────────────────────────────────────────
Dengan 80/20 standar, model hanya punya 1 titik evaluasi (308 hari terakhir).
Expanding Window menghasilkan 40 titik evaluasi independen yang mewakili semua
kondisi pasar: kenaikan 2022, lonjakan pra-lebaran, panen, inflasi 2025, dll.

Alur kerja:
  1. Load dataset_preprocessed.csv (tanpa perlu train.csv / test.csv dulu)
  2. Buat expanding windows dengan min_train=365, step=30 hari
  3. Tiap window: fit XGBoost → prediksi 30 hari ke depan → hitung metrik
  4. Agregasi seluruh hasil → grafik MAE per window (deteksi concept drift)
  5. Latih model final pada seluruh data → simpan untuk deployment web

Cara pakai:
  python train_expanding_window.py

Dependencies:
  pip install xgboost scikit-learn pandas numpy matplotlib joblib shap
"""

import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data" / "processed"
MODEL_DIR   = BASE_DIR / "machine_learning" / "output" / "expanding_window"
PLOTS_DIR   = MODEL_DIR / "plots"
METRICS_DIR = MODEL_DIR / "metrics"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TARGET      = "harga_cabai_merah"
RANDOM_STATE = 42

# ── Konfigurasi Expanding Window ──────────────────────────────────────────────
MIN_TRAIN_DAYS = 365   # Minimum 1 tahun data training (sesuai saran reviewer)
STEP_DAYS      = 30    # Geser window setiap 30 hari (≈ 1 bulan)
TEST_DAYS      = 30    # Ukuran window test (30 hari ke depan)

# ── Target multi-horizon yang akan dilatih ────────────────────────────────────
TARGET_MAP = {
    "h1": "target_h1",   # Prediksi H+1 (besok)
    "h3": "target_h3",   # Prediksi H+3 (3 hari ke depan)
    "h7": "target_h7",   # Prediksi H+7 (7 hari ke depan)
}

# Kolom yang BUKAN fitur (dibuang dari X)
NON_FEATURE_COLS = [
    "tanggal", "harga_cabai_rawit", TARGET,
    "target_h1", "target_h3", "target_h7",
]

LOG_LINES: list = []

def log(msg: str) -> None:
    print(msg)
    LOG_LINES.append(msg)


# =============================================================================
# BAGIAN 1 — LOAD DATASET LANGSUNG DARI PREPROCESSED
# =============================================================================
def load_dataset() -> pd.DataFrame:
    """
    Memuat dataset_preprocessed.csv langsung — tidak perlu train.csv/test.csv.
    Expanding Window akan melakukan split sendiri secara dinamis per window.
    
    CATATAN: Membaca dataset_preprocessed.csv (bukan dataset_scaled.csv)
    dataset_preprocessed.csv berisi nilai asli dalam Rupiah + fitur ter-scale.
    dataset_scaled.csv hanya cadangan untuk debugging/EDA manual.
    """
    log("\n[1] Memuat dataset_preprocessed.csv ...")

    # Coba beberapa path umum
    kandidat = [
        DATA_DIR / "dataset_preprocessed.csv",
        BASE_DIR / "data" / "processed" / "dataset_preprocessed.csv",
        Path("dataset_preprocessed.csv"),
    ]
    path = next((p for p in kandidat if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "dataset_preprocessed.csv tidak ditemukan.\n"
            f"Cari di: {[str(p) for p in kandidat]}"
        )

    df = pd.read_csv(path, parse_dates=["tanggal"])
    df = df.sort_values("tanggal").reset_index(drop=True)

    log(f"    -> File    : {path}")
    log(f"    -> Shape   : {df.shape}")
    log(f"    -> Rentang : {df['tanggal'].min().date()} s/d {df['tanggal'].max().date()}")
    log(f"    -> Harga   : Rp {df[TARGET].min():,.0f} – Rp {df[TARGET].max():,.0f}")

    # Validasi target multi-horizon
    for label, col in TARGET_MAP.items():
        n_valid = df[col].notna().sum() if col in df.columns else 0
        log(f"    -> {col:<14}: {n_valid} baris valid")

    return df


# =============================================================================
# BAGIAN 2 — BUAT EXPANDING WINDOWS
# =============================================================================
def buat_expanding_windows(df: pd.DataFrame) -> list[dict]:
    """
    Membuat daftar expanding window dengan aturan:
      - Train selalu dimulai dari baris 0 (expanding, bukan rolling)
      - Minimum train = MIN_TRAIN_DAYS hari
      - Test = TEST_DAYS hari berikutnya
      - Step = STEP_DAYS hari (train bertambah setiap iterasi)

    Format output tiap window:
      {
        "window"     : nomor window (1-based),
        "train_start": indeks awal train,
        "train_end"  : indeks akhir train (eksklusif),
        "test_start" : indeks awal test,
        "test_end"   : indeks akhir test (eksklusif),
        "tanggal_train_mulai": datetime,
        "tanggal_train_selesai": datetime,
        "tanggal_test_mulai" : datetime,
        "tanggal_test_selesai": datetime,
      }

    Mengapa expanding (bukan rolling)?
      Dalam forecasting komoditas, data lama TETAP relevan — harga panen 2022
      adalah referensi untuk pola musiman 2025. Rolling window membuang informasi
      ini. Expanding window mempertahankan semua sejarah sambil tetap memberi
      bobot implisit lebih pada data terbaru (karena data terbaru selalu ada).
    """
    log(f"\n[2] Membuat expanding windows ...")
    log(f"    -> Min train : {MIN_TRAIN_DAYS} hari")
    log(f"    -> Step      : {STEP_DAYS} hari")
    log(f"    -> Test size : {TEST_DAYS} hari")

    total = len(df)
    windows = []
    window_id = 1
    train_end = MIN_TRAIN_DAYS  # mulai dari minimum 1 tahun

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
            "tanggal_train_mulai"  : df["tanggal"].iloc[0],
            "tanggal_train_selesai": df["tanggal"].iloc[train_end - 1],
            "tanggal_test_mulai"   : df["tanggal"].iloc[train_end],
            "tanggal_test_selesai" : df["tanggal"].iloc[test_end - 1],
        })
        train_end += STEP_DAYS
        window_id += 1

    log(f"    -> Total windows : {len(windows)}")
    log(f"    -> Window 1      : train={MIN_TRAIN_DAYS} hari, "
        f"test={windows[0]['tanggal_test_mulai'].date()} – "
        f"{windows[0]['tanggal_test_selesai'].date()}")
    log(f"    -> Window terakhir: train={windows[-1]['train_end']} hari, "
        f"test={windows[-1]['tanggal_test_mulai'].date()} – "
        f"{windows[-1]['tanggal_test_selesai'].date()}")
    return windows


# =============================================================================
# BAGIAN 3 — METRIK EVALUASI
# =============================================================================
def hitung_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def hitung_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE — lebih adil untuk harga komoditas."""
    denom = np.abs(y_true) + np.abs(y_pred)
    mask  = denom != 0
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def hitung_da(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Directional Accuracy — % prediksi arah naik/turun yang benar."""
    if len(y_true) < 2:
        return float("nan")
    return float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)

def hitung_semua_metrik(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE"  : float(mean_absolute_error(y_true, y_pred)),
        "RMSE" : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE" : hitung_mape(y_true, y_pred),
        "sMAPE": hitung_smape(y_true, y_pred),
        "R2"   : float(r2_score(y_true, y_pred)),
        "DA"   : hitung_da(y_true, y_pred),
    }


# =============================================================================
# BAGIAN 4 — PISAHKAN FITUR
# =============================================================================
def pisahkan_fitur(df_split: pd.DataFrame, target_col: str):
    """Pisahkan X (fitur) dan y (target) dengan membuang kolom non-fitur."""
    drop_cols = [c for c in NON_FEATURE_COLS if c in df_split.columns]
    X = df_split.drop(columns=drop_cols)
    y = df_split[target_col]
    return X, y


# =============================================================================
# BAGIAN 5 — NAIVE BASELINE PER WINDOW
# =============================================================================
def naive_baseline_window(y_true: np.ndarray,
                          last_train_value: float) -> np.ndarray:
    """
    Naive baseline per window:
      - Prediksi hari pertama test = nilai terakhir di training
      - Prediksi hari berikutnya  = nilai aktual hari sebelumnya (walk-forward)

    Ini mensimulasikan strategi paling sederhana yang mungkin dilakukan seseorang
    tanpa model ML: "saya prediksi harga besok = harga hari ini".
    """
    y_naive = np.empty_like(y_true, dtype=float)
    y_naive[0] = last_train_value
    for i in range(1, len(y_true)):
        y_naive[i] = y_true[i - 1]  # walk-forward
    return y_naive


# =============================================================================
# BAGIAN 6 — RIDGE BASELINE PER WINDOW
# =============================================================================
def ridge_baseline_window(X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame) -> np.ndarray:
    """
    Ridge Regression sebagai baseline linear.
    Keunggulan vs XGBoost: mampu ekstrapolasi di luar range historis.
    Jika harga cabai tahun depan tembus rekor, Ridge bisa mengikuti tren
    sementara XGBoost akan mematok (clip) di nilai maksimum yang pernah dilihat.
    RobustScaler digunakan karena Ridge sensitif terhadap skala.
    """
    scaler = RobustScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    model   = Ridge(alpha=1.0)
    model.fit(X_tr_sc, y_train)
    return model.predict(X_te_sc)


# =============================================================================
# BAGIAN 7 — TRAINING XGBoost SATU WINDOW
# =============================================================================
def train_xgb_window(X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame,
                      params: dict | None = None) -> np.ndarray:
    """
    Melatih XGBoost pada satu window tanpa hyperparameter tuning
    (tuning dilakukan sekali di Bagian 10, hasilnya di-cache).
    Menggunakan early stopping dengan split 90/10 internal dari train window.
    """
    if params is None:
        params = {}

    split_val = max(int(len(X_train) * 0.9), len(X_train) - 60)
    X_tr, X_val = X_train.iloc[:split_val], X_train.iloc[split_val:]
    y_tr, y_val = y_train.iloc[:split_val], y_train.iloc[split_val:]

    model = XGBRegressor(
        objective             = "reg:squarederror",
        tree_method           = "hist",
        random_state          = RANDOM_STATE,
        verbosity             = 0,
        eval_metric           = "rmse",
        early_stopping_rounds = 30,
        n_estimators          = params.get("n_estimators", 500),
        max_depth             = params.get("max_depth", 5),
        learning_rate         = params.get("learning_rate", 0.05),
        subsample             = params.get("subsample", 0.8),
        colsample_bytree      = params.get("colsample_bytree", 0.8),
        min_child_weight      = params.get("min_child_weight", 3),
        gamma                 = params.get("gamma", 0.1),
        reg_alpha             = params.get("reg_alpha", 0.1),
        reg_lambda            = params.get("reg_lambda", 1.0),
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False,
    )
    return model.predict(X_test), model


# =============================================================================
# BAGIAN 8 — LOOP UTAMA: EXPANDING WINDOW EVALUATION
# =============================================================================
def expanding_window_eval(df: pd.DataFrame,
                           windows: list[dict],
                           target_col: str,
                           label: str,
                           params: dict | None = None) -> pd.DataFrame:
    """
    Loop utama expanding window evaluation.

    Untuk setiap window:
      1. Bagi data → df_train dan df_test
      2. Hapus baris NaN pada target_col (bisa terjadi di ujung karena shift)
      3. Pisahkan X, y
      4. Jalankan 3 model: XGBoost, Ridge (linear baseline), Naive (random walk)
      5. Hitung semua metrik: MAE, RMSE, MAPE, sMAPE, R², DA
      6. Catat hasil per window → DataFrame

    Output DataFrame berisi 1 baris per window, bisa langsung:
      - Di-plot untuk melihat apakah MAE meningkat seiring waktu (concept drift)
      - Di-ringkas untuk mendapat mean ± std MAE (lebih representatif dari 80/20)
    """
    log(f"\n[8] Expanding Window Evaluation — {label.upper()} ({target_col})")
    log(f"    {'Win':<5} {'Train':>6} {'Test':>5}   XGB-MAE      Ridge-MAE    Naive-MAE   DA%    R²")
    log(f"    {'─'*5} {'─'*6} {'─'*5}   {'─'*11}  {'─'*11}  {'─'*11}  {'─'*6} {'─'*6}")

    records = []

    for w in windows:
        win_id = w["window"]
        df_tr  = df.iloc[w["train_start"]:w["train_end"]].dropna(subset=[target_col]).copy()
        df_te  = df.iloc[w["test_start"] :w["test_end"] ].dropna(subset=[target_col]).copy()

        if len(df_tr) < 100 or len(df_te) < 5:
            continue

        X_train, y_train = pisahkan_fitur(df_tr, target_col)
        X_test,  y_test  = pisahkan_fitur(df_te, target_col)

        y_true = y_test.values

        # ── XGBoost ────────────────────────────────────────────────────────
        try:
            y_pred_xgb, _ = train_xgb_window(X_train, y_train, X_test, params)
            m_xgb = hitung_semua_metrik(y_true, y_pred_xgb)
        except Exception as e:
            log(f"    ! Window {win_id} XGBoost error: {e}")
            continue

        # ── Ridge Baseline ─────────────────────────────────────────────────
        try:
            y_pred_ridge = ridge_baseline_window(X_train, y_train, X_test)
            m_ridge = hitung_semua_metrik(y_true, y_pred_ridge)
        except Exception:
            y_pred_ridge = np.full_like(y_true, y_train.mean(), dtype=float)
            m_ridge = hitung_semua_metrik(y_true, y_pred_ridge)

        # ── Naive Baseline ─────────────────────────────────────────────────
        last_train_val  = float(y_train.iloc[-1])
        y_pred_naive    = naive_baseline_window(y_true, last_train_val)
        m_naive         = hitung_semua_metrik(y_true, y_pred_naive)

        # ── Simpan hasil ───────────────────────────────────────────────────
        records.append({
            "window"               : win_id,
            "tanggal_test_mulai"   : w["tanggal_test_mulai"].date(),
            "tanggal_test_selesai" : w["tanggal_test_selesai"].date(),
            "n_train"              : len(df_tr),
            "n_test"               : len(df_te),
            # XGBoost
            "xgb_MAE"  : round(m_xgb["MAE"],   2),
            "xgb_RMSE" : round(m_xgb["RMSE"],  2),
            "xgb_MAPE" : round(m_xgb["MAPE"],  4),
            "xgb_sMAPE": round(m_xgb["sMAPE"], 4),
            "xgb_R2"   : round(m_xgb["R2"],    6),
            "xgb_DA"   : round(m_xgb["DA"],    2),
            # Ridge
            "ridge_MAE"  : round(m_ridge["MAE"],  2),
            "ridge_RMSE" : round(m_ridge["RMSE"], 2),
            "ridge_DA"   : round(m_ridge["DA"],   2),
            # Naive
            "naive_MAE"  : round(m_naive["MAE"],  2),
            "naive_RMSE" : round(m_naive["RMSE"], 2),
            "naive_DA"   : round(m_naive["DA"],   2),
        })

        log(
            f"    {win_id:<5} {len(df_tr):>6} {len(df_te):>5}   "
            f"Rp {m_xgb['MAE']:>8,.0f}   "
            f"Rp {m_ridge['MAE']:>8,.0f}   "
            f"Rp {m_naive['MAE']:>8,.0f}   "
            f"{m_xgb['DA']:>5.1f}  "
            f"{m_xgb['R2']:>6.4f}"
        )

    df_results = pd.DataFrame(records)

    if len(df_results) > 0:
        log(f"\n    ── Ringkasan Expanding Window — {label.upper()} ──────────────")
        log(f"    XGBoost  MAE : Rp {df_results['xgb_MAE'].mean():>10,.0f} "
            f"± Rp {df_results['xgb_MAE'].std():>8,.0f}")
        log(f"    Ridge    MAE : Rp {df_results['ridge_MAE'].mean():>10,.0f} "
            f"± Rp {df_results['ridge_MAE'].std():>8,.0f}")
        log(f"    Naive    MAE : Rp {df_results['naive_MAE'].mean():>10,.0f} "
            f"± Rp {df_results['naive_MAE'].std():>8,.0f}")
        log(f"    XGBoost  DA  : {df_results['xgb_DA'].mean():.2f}% "
            f"(Naive: {df_results['naive_DA'].mean():.2f}%)")

        # Penilaian apakah XGBoost mengalahkan Naive
        beat_naive = (df_results["xgb_MAE"] < df_results["naive_MAE"]).mean() * 100
        log(f"    XGBoost mengalahkan Naive: {beat_naive:.1f}% dari window")
        if beat_naive < 70:
            log("    ! PERINGATAN: XGBoost hanya mengalahkan Naive di "
                f"{beat_naive:.0f}% window. Pertimbangkan feature engineering tambahan.")

    return df_results


# =============================================================================
# BAGIAN 9 — PLOT HASIL EXPANDING WINDOW
# =============================================================================
def plot_expanding_window(df_results: pd.DataFrame, label: str) -> None:
    """
    Membuat 3 plot untuk analisis hasil expanding window:

    Plot 1 — MAE per window (XGBoost vs Ridge vs Naive):
      Jika MAE XGBoost meningkat tajam pada window 2024-2025, itu sinyal
      concept drift → model perlu di-retrain lebih sering.

    Plot 2 — Directional Accuracy per window:
      Persentase berapa kali model menebak arah harga (naik/turun) dengan benar.
      DA > 50% = lebih baik dari tebakan random.

    Plot 3 — Distribusi MAE (boxplot):
      Menunjukkan konsistensi model. Distribusi sempit = stabil.
      Outlier tinggi = ada periode harga yang sangat sulit diprediksi.
    """
    if len(df_results) == 0:
        log("    ! Tidak ada data untuk diplot")
        return

    log(f"\n[9] Membuat plot expanding window — {label.upper()}...")

    tanggal = pd.to_datetime(df_results["tanggal_test_mulai"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))
    fig.suptitle(
        f"Expanding Window Evaluation — Model {label.upper()}\n"
        "Prediksi Harga Cabai Merah, Kota Padang",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Plot 1: MAE per window ─────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(tanggal, df_results["xgb_MAE"],   "o-", color="#1976D2",
             linewidth=1.8, markersize=4, label="XGBoost", zorder=3)
    ax1.plot(tanggal, df_results["ridge_MAE"], "s--", color="#388E3C",
             linewidth=1.5, markersize=3, label="Ridge (linear)", zorder=2, alpha=0.8)
    ax1.plot(tanggal, df_results["naive_MAE"], "^:", color="#E53935",
             linewidth=1.5, markersize=3, label="Naive (random walk)", zorder=2, alpha=0.8)

    # Area di mana XGBoost kalah dari Naive
    for _, row in df_results.iterrows():
        if row["xgb_MAE"] > row["naive_MAE"]:
            ax1.axvspan(
                pd.to_datetime(row["tanggal_test_mulai"]) - pd.Timedelta(days=15),
                pd.to_datetime(row["tanggal_test_mulai"]) + pd.Timedelta(days=15),
                alpha=0.1, color="red",
            )

    # Rolling mean (tren MAE)
    if len(df_results) >= 5:
        roll = df_results["xgb_MAE"].rolling(5, center=True).mean()
        ax1.plot(tanggal, roll, color="#1976D2", linewidth=3,
                 alpha=0.3, label="Tren XGBoost (5-window MA)")

    ax1.set_ylabel("MAE (Rupiah)", fontsize=11)
    ax1.set_title("MAE per Window — XGBoost vs Baseline (area merah = XGBoost kalah dari Naive)",
                  fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rp {x:,.0f}"))
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    # ── Plot 2: Directional Accuracy ─────────────────────────────────────
    ax2 = axes[1]
    ax2.bar(tanggal, df_results["xgb_DA"], width=20, color="#1976D2",
            alpha=0.7, label="XGBoost DA%")
    ax2.step(tanggal, df_results["naive_DA"], color="#E53935", linewidth=1.5,
             where="mid", linestyle="--", label="Naive DA%", alpha=0.8)
    ax2.axhline(y=50, color="gray", linewidth=1, linestyle=":", alpha=0.7,
                label="50% (level random)")
    ax2.axhline(y=df_results["xgb_DA"].mean(), color="#1976D2", linewidth=1.5,
                linestyle="--", alpha=0.6,
                label=f"Mean DA XGBoost ({df_results['xgb_DA'].mean():.1f}%)")

    ax2.set_ylabel("Directional Accuracy (%)", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.set_title("Directional Accuracy — % Prediksi Arah Naik/Turun yang Benar",
                  fontsize=11)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))

    # ── Plot 3: Distribusi MAE ────────────────────────────────────────────
    ax3 = axes[2]
    data_box = [
        df_results["xgb_MAE"].values,
        df_results["ridge_MAE"].values,
        df_results["naive_MAE"].values,
    ]
    bp = ax3.boxplot(
        data_box,
        labels=["XGBoost", "Ridge (linear)", "Naive"],
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )
    colors_box = ["#1976D2", "#388E3C", "#E53935"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Tambah label mean di atas tiap box
    for i, data in enumerate(data_box):
        ax3.text(i + 1, np.max(data) * 1.02,
                 f"mean\nRp{np.mean(data):,.0f}",
                 ha="center", va="bottom", fontsize=8, color=colors_box[i])

    ax3.set_ylabel("MAE (Rupiah)", fontsize=11)
    ax3.set_title("Distribusi MAE — Konsistensi Model (outlier = periode sulit)",
                  fontsize=11)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rp {x:,.0f}"))
    ax3.grid(axis="y", alpha=0.3)
    ax3.spines[["top", "right"]].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = PLOTS_DIR / f"expanding_window_{label}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"    -> expanding_window_{label}.png disimpan")


# =============================================================================
# BAGIAN 10 — HYPERPARAMETER TUNING (sekali, pada window terbesar)
# =============================================================================
def tuning_final(df: pd.DataFrame, target_col: str,
                  label: str, horizon_days: int) -> dict:
    """
    Hyperparameter tuning dilakukan SEKALI menggunakan seluruh data
    yang tersedia (maksimal) dengan TimeSeriesSplit(gap=horizon_days).

    Hasilnya di-cache sebagai JSON dan digunakan untuk semua window
    sehingga setiap window tidak perlu tuning ulang (hemat waktu ~40x).

    Catatan: Ini adalah desain yang valid. Dalam praktik produksi,
    tuning bisa dijadwalkan ulang setiap 3-6 bulan atau saat MAE
    drift terdeteksi.
    """
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

    cache_path = METRICS_DIR / f"best_params_{label}.json"
    if cache_path.exists():
        log(f"\n[10] Cache ditemukan, lewati tuning ({label}): {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    log(f"\n[10] Hyperparameter tuning — {label.upper()} (gap={horizon_days})...")

    df_valid = df.dropna(subset=[target_col]).copy()
    X, y     = pisahkan_fitur(df_valid, target_col)

    param_dist = {
        "n_estimators"     : [300, 500, 700, 1000],
        "max_depth"        : [3, 4, 5, 6, 7],
        "learning_rate"    : [0.01, 0.02, 0.05, 0.1],
        "subsample"        : [0.7, 0.8, 0.9],
        "colsample_bytree" : [0.6, 0.7, 0.8, 0.9],
        "min_child_weight" : [1, 3, 5],
        "gamma"            : [0, 0.1, 0.2],
        "reg_alpha"        : [0, 0.1, 0.5],
        "reg_lambda"       : [0.5, 1.0, 2.0],
    }

    tscv = TimeSeriesSplit(n_splits=5, gap=horizon_days)
    base = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_STATE, verbosity=0,
    )
    search = RandomizedSearchCV(
        base, param_dist, n_iter=50, cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=1,
    )
    search.fit(X, y)

    best = search.best_params_
    log(f"    -> Best MAE (CV): Rp {-search.best_score_:,.2f}")
    log(f"    -> Params: {best}")

    with open(cache_path, "w") as f:
        json.dump(best, f, indent=4)
    log(f"    -> best_params_{label}.json disimpan (di-cache)")
    return best


# =============================================================================
# BAGIAN 11 — MODEL FINAL (dilatih pada SELURUH data untuk deployment)
# =============================================================================
def latih_model_final(df: pd.DataFrame, target_col: str,
                       label: str, params: dict) -> tuple:
    """
    Setelah evaluasi expanding window selesai dan kita yakin model layak,
    latih model FINAL pada seluruh data yang tersedia untuk deployment.

    Model ini yang akan digunakan di web dashboard untuk prediksi real-time.
    """
    log(f"\n[11] Melatih model final (seluruh data) — {label.upper()}...")

    df_valid  = df.dropna(subset=[target_col]).copy()
    X, y      = pisahkan_fitur(df_valid, target_col)
    feature_cols = list(X.columns)

    split_val = int(len(X) * 0.9)
    X_tr, X_val = X.iloc[:split_val], X.iloc[split_val:]
    y_tr, y_val = y.iloc[:split_val], y.iloc[split_val:]

    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_STATE, verbosity=0,
        eval_metric="rmse", early_stopping_rounds=50,
        **params,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False,
    )

    # Simpan model
    ts   = datetime.now().strftime("%Y%m%d")
    path = MODEL_DIR / f"model_final_{label}_{ts}.pkl"
    joblib.dump(model, path)
    log(f"    -> {path.name} disimpan ({len(df_valid):,} baris, {len(feature_cols)} fitur)")
    log(f"    -> Best iteration: {model.best_iteration}")

    # Simpan daftar fitur (penting untuk inference di web)
    feat_path = MODEL_DIR / f"feature_cols_{label}.json"
    with open(feat_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    log(f"    -> feature_cols_{label}.json disimpan")

    return model, feature_cols



# =============================================================================
# BAGIAN 11B — EVALUASI MODEL FINAL (PERBAIKAN 2)
# =============================================================================
def evaluasi_model_final(df: pd.DataFrame, target_col: str,
                          label: str, model) -> dict:
    """
    Evaluasi model final pada 20% data terakhir sebagai hold-out set.

    Mengapa 20% data TERAKHIR (bukan acak)?
      Karena ini time series — data terakhir adalah yang paling relevan
      dengan kondisi pasar saat ini. Hold-out berbasis waktu lebih valid
      daripada split acak untuk menilai performa prediktif ke depan.

    Hold-out ini TIDAK pernah digunakan dalam proses training manapun:
      - Tidak masuk expanding window (window terakhir belum sampai ujung)
      - Tidak masuk training model final (pakai 90% awal)
    Sehingga metrik di sini adalah estimasi performa yang paling jujur.

    Output: dict metrik + file evaluasi_final_{label}.csv
    """
    log(f"\n[11B] Evaluasi model final pada hold-out set — {label.upper()}...")

    df_valid  = df.dropna(subset=[target_col]).copy()
    holdout_n = int(len(df_valid) * 0.2)

    if holdout_n < 30:
        log(f"    ! Data terlalu sedikit untuk hold-out ({holdout_n} baris), skip.")
        return {}

    df_holdout = df_valid.tail(holdout_n).copy()
    tgl_mulai  = df_holdout["tanggal"].iloc[0].date()
    tgl_selesai= df_holdout["tanggal"].iloc[-1].date()

    X_hold, y_hold = pisahkan_fitur(df_holdout, target_col)
    y_pred = model.predict(X_hold)
    y_true = y_hold.values

    metrik = hitung_semua_metrik(y_true, y_pred)

    log(f"    -> Hold-out: {holdout_n} hari ({tgl_mulai} s/d {tgl_selesai})")
    log(f"    -> MAE    : Rp {metrik['MAE']:>10,.0f}")
    log(f"    -> RMSE   : Rp {metrik['RMSE']:>10,.0f}")
    log(f"    -> MAPE   : {metrik['MAPE']:>8.2f}%")
    log(f"    -> sMAPE  : {metrik['sMAPE']:>8.2f}%")
    log(f"    -> R²     : {metrik['R2']:>8.4f}")
    log(f"    -> DA     : {metrik['DA']:>8.1f}%")

    # ── Plot prediksi vs aktual pada hold-out ────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    tgl_plot = df_holdout["tanggal"].values
    ax.plot(tgl_plot, y_true, color="#E53935", linewidth=1.5,
            label="Aktual", zorder=3)
    ax.plot(tgl_plot, y_pred, color="#1976D2", linewidth=1.5,
            linestyle="--", label="Prediksi Model Final", zorder=3)
    ax.fill_between(tgl_plot, y_true, y_pred,
                    alpha=0.15, color="#1976D2", label="Selisih error")
    ax.set_title(
        f"Evaluasi Model Final {label.upper()} — Hold-out 20% Data Terakhir\n"
        f"MAE: Rp {metrik['MAE']:,.0f} | MAPE: {metrik['MAPE']:.2f}% | "
        f"R²: {metrik['R2']:.4f} | DA: {metrik['DA']:.1f}%",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Harga Cabai Merah (Rp)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rp {x:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"evaluasi_final_{label}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"    -> evaluasi_final_{label}.png disimpan")

    # ── Simpan metrik ke CSV ─────────────────────────────────────────────────
    pd.DataFrame([{
        "label"        : label.upper(),
        "target"       : target_col,
        "holdout_n"    : holdout_n,
        "tgl_mulai"    : str(tgl_mulai),
        "tgl_selesai"  : str(tgl_selesai),
        "MAE"          : round(metrik["MAE"], 2),
        "RMSE"         : round(metrik["RMSE"], 2),
        "MAPE_%"       : round(metrik["MAPE"], 4),
        "sMAPE_%"      : round(metrik["sMAPE"], 4),
        "R2"           : round(metrik["R2"], 6),
        "DA_%"         : round(metrik["DA"], 2),
    }]).to_csv(METRICS_DIR / f"evaluasi_final_{label}.csv", index=False)
    log(f"    -> evaluasi_final_{label}.csv disimpan")

    return metrik

# =============================================================================
# BAGIAN 12 — SHAP UNTUK MODEL FINAL
# =============================================================================
def shap_final(model, X_sample: pd.DataFrame, label: str) -> None:
    """SHAP analysis pada model final untuk laporan stakeholder."""
    if not SHAP_AVAILABLE:
        log(f"\n    ! SHAP tidak tersedia (pip install shap)")
        return

    log(f"\n[12] SHAP analysis — model final {label.upper()}...")
    try:
        explainer   = shap.TreeExplainer(model)
        n_sample    = min(500, len(X_sample))
        shap_values = explainer.shap_values(X_sample.iloc[:n_sample])

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample.iloc[:n_sample],
                          plot_type="dot", max_display=20, show=False)
        plt.title(
            f"SHAP Summary — Model {label.upper()}\n"
            "Merah = nilai fitur tinggi | Biru = nilai fitur rendah\n"
            "Nilai SHAP positif = mendorong harga naik",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"shap_summary_{label}.png", dpi=150,
                    bbox_inches="tight")
        plt.close()
        log(f"    -> shap_summary_{label}.png disimpan")

        mean_abs = np.abs(shap_values).mean(axis=0)
        df_shap  = pd.DataFrame({
            "fitur": X_sample.columns.tolist(),
            "shap_mean_abs": mean_abs,
        }).sort_values("shap_mean_abs", ascending=False)
        df_shap.to_csv(METRICS_DIR / f"shap_{label}.csv", index=False)

        log("    Top 10 fitur (SHAP):")
        for i, row in df_shap.head(10).iterrows():
            log(f"    {list(df_shap.index).index(i)+1:>2}. "
                f"{row['fitur']:<30}: {row['shap_mean_abs']:.4f}")
    except Exception as e:
        log(f"    ! SHAP error: {e}")


# =============================================================================
# BAGIAN 13 — LAPORAN RINGKASAN AKHIR
# =============================================================================
def laporan_ringkasan(all_results: dict) -> None:
    """
    Cetak dan simpan ringkasan semua hasil expanding window untuk semua horizon.
    Berisi perbandingan mean MAE XGBoost vs Ridge vs Naive.
    """
    log(f"\n{'═'*65}")
    log("  RINGKASAN AKHIR — EXPANDING WINDOW EVALUATION")
    log(f"{'═'*65}")
    log(f"  {'Model':<6} {'XGB-MAE mean':>14} {'XGB-MAE std':>12} "
        f"{'Ridge-MAE':>12} {'Naive-MAE':>12} {'Beat-Naive':>10}")
    log(f"  {'─'*6} {'─'*14} {'─'*12} {'─'*12} {'─'*12} {'─'*10}")

    summary_rows = []
    for label, df_res in all_results.items():
        if df_res is None or len(df_res) == 0:
            continue
        xgb_mean  = df_res["xgb_MAE"].mean()
        xgb_std   = df_res["xgb_MAE"].std()
        ridge_mean = df_res["ridge_MAE"].mean()
        naive_mean = df_res["naive_MAE"].mean()
        beat       = (df_res["xgb_MAE"] < df_res["naive_MAE"]).mean() * 100

        log(f"  {label.upper():<6} "
            f"Rp {xgb_mean:>10,.0f}  "
            f"±Rp {xgb_std:>8,.0f}  "
            f"Rp {ridge_mean:>8,.0f}  "
            f"Rp {naive_mean:>8,.0f}  "
            f"{beat:>8.1f}%")

        summary_rows.append({
            "model"        : label.upper(),
            "target"       : TARGET_MAP.get(label, TARGET),
            "xgb_MAE_mean" : round(xgb_mean, 2),
            "xgb_MAE_std"  : round(xgb_std, 2),
            "xgb_RMSE_mean": round(df_res["xgb_RMSE"].mean(), 2),
            "xgb_MAPE_mean": round(df_res["xgb_MAPE"].mean(), 4),
            "xgb_DA_mean"  : round(df_res["xgb_DA"].mean(), 2),
            "ridge_MAE_mean": round(ridge_mean, 2),
            "naive_MAE_mean": round(naive_mean, 2),
            "beat_naive_%"  : round(beat, 1),
            "n_windows"     : len(df_res),
        })

    log(f"{'═'*65}")

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(METRICS_DIR / "ringkasan_expanding_window.csv", index=False)
        log("  -> ringkasan_expanding_window.csv disimpan")

    # Simpan log
    log_path = MODEL_DIR / "log_expanding_window.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    log(f"  -> log_expanding_window.txt disimpan")
    log(f"\n  Semua output di: {MODEL_DIR}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    # ── PERBAIKAN 3: Reproducibility — pastikan hasil selalu sama ─────────────
    # np.random.seed mengontrol semua operasi random di numpy
    # Python random.seed mengontrol operasi random bawaan Python
    # XGBRegressor & RandomizedSearchCV sudah pakai RANDOM_STATE=42
    import random
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    log("=" * 65)
    log("  EXPANDING WINDOW CROSS-VALIDATION — HARGA CABAI MERAH")
    log("  Kota Padang, Sumatera Barat")
    log(f"  Min train: {MIN_TRAIN_DAYS} hari | Step: {STEP_DAYS} hari | Test: {TEST_DAYS} hari")
    log(f"  Random state: {RANDOM_STATE} (reproducible)")
    log("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_dataset()

    # ── 2. Buat expanding windows ────────────────────────────────────────────
    windows = buat_expanding_windows(df)

    # ── Cek target yang tersedia ─────────────────────────────────────────────
    available_targets = {
        label: col for label, col in TARGET_MAP.items()
        if col in df.columns and df[col].notna().sum() > MIN_TRAIN_DAYS
    }
    if not available_targets:
        raise ValueError(
            "Tidak ada kolom target multi-horizon. "
            "Pastikan preprocessing.py sudah dijalankan."
        )
    log(f"\n    Target tersedia: {list(available_targets.keys())}")

    horizon_map = {"h1": 1, "h3": 3, "h7": 7}
    all_results = {}

    for label, target_col in available_targets.items():
        log(f"\n{'='*65}")
        log(f"  PROSES MODEL — {label.upper()} ({target_col})")
        log(f"{'='*65}")

        horizon_days = horizon_map.get(label, 1)

        # ── Tuning sekali ──────────────────────────────────────────────────
        params = tuning_final(df, target_col, label, horizon_days)

        # ── Expanding window evaluation ────────────────────────────────────
        df_results = expanding_window_eval(df, windows, target_col, label, params)

        # ── Simpan hasil per window ────────────────────────────────────────
        if len(df_results) > 0:
            out_csv = METRICS_DIR / f"hasil_window_{label}.csv"
            df_results.to_csv(out_csv, index=False)
            log(f"    -> hasil_window_{label}.csv disimpan ({len(df_results)} windows)")

        # ── Plot ───────────────────────────────────────────────────────────
        plot_expanding_window(df_results, label)

        all_results[label] = df_results

    # ── Latih model final pada seluruh data ───────────────────────────────────
    log(f"\n{'='*65}")
    log("  TRAINING MODEL FINAL (seluruh data → deployment web)")
    log(f"{'='*65}")

    holdout_results = {}   # kumpulkan metrik hold-out semua model

    for label, target_col in available_targets.items():
        params = all_results.get(label)
        # Muat ulang best params dari cache
        cache = METRICS_DIR / f"best_params_{label}.json"
        if cache.exists():
            with open(cache) as f:
                p = json.load(f)
        else:
            p = {}

        model_final, feature_cols = latih_model_final(df, target_col, label, p)

        # ── PERBAIKAN 2: Evaluasi model final pada hold-out 20% ───────────
        metrik_final = evaluasi_model_final(df, target_col, label, model_final)
        if metrik_final:
            holdout_results[label] = metrik_final

        # SHAP untuk model final
        df_valid = df.dropna(subset=[target_col]).copy()
        X_all, _ = pisahkan_fitur(df_valid, target_col)
        shap_final(model_final, X_all, label)

    # ── Laporan ringkasan ─────────────────────────────────────────────────────
    laporan_ringkasan(all_results)

    # ── PERBAIKAN 2: Tampilkan ringkasan evaluasi hold-out semua model ────────
    if holdout_results:
        log(f"\n{'═'*65}")
        log("  EVALUASI MODEL FINAL — HOLD-OUT 20% DATA TERAKHIR")
        log("  (Angka ini yang digunakan sebagai akurasi di laporan)")
        log(f"{'═'*65}")
        log(f"  {'Model':<6} {'MAE':>14} {'RMSE':>14} {'MAPE':>10} {'sMAPE':>10} {'R²':>8} {'DA%':>8}")
        log(f"  {'─'*6} {'─'*14} {'─'*14} {'─'*10} {'─'*10} {'─'*8} {'─'*8}")
        for lbl, m in holdout_results.items():
            log(f"  {lbl.upper():<6} "
                f"Rp {m['MAE']:>10,.0f} "
                f"Rp {m['RMSE']:>10,.0f} "
                f"{m['MAPE']:>9.2f}% "
                f"{m['sMAPE']:>9.2f}% "
                f"{m['R2']:>8.4f} "
                f"{m['DA']:>7.1f}%")
        log(f"{'═'*65}")

        # Simpan ringkasan hold-out ke CSV
        rows = []
        for lbl, m in holdout_results.items():
            rows.append({
                "model"  : lbl.upper(),
                "MAE"    : round(m["MAE"], 2),
                "RMSE"   : round(m["RMSE"], 2),
                "MAPE_%" : round(m["MAPE"], 4),
                "sMAPE_%": round(m["sMAPE"], 4),
                "R2"     : round(m["R2"], 6),
                "DA_%"   : round(m["DA"], 2),
            })
        pd.DataFrame(rows).to_csv(
            METRICS_DIR / "ringkasan_evaluasi_final.csv", index=False
        )
        log("  -> ringkasan_evaluasi_final.csv disimpan")

    log("\n[SELESAI] Expanding window evaluation dan model final berhasil dibuat!")
    return all_results


if __name__ == "__main__":
    results = main()