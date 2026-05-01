import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "App" / "Output"
MODEL_DIR  = BASE_DIR / "machine learning" / "model XGBoost"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "harga_cabai_merah"
RANDOM_STATE = 42
LOG_LINES    = []


def log(msg: str):
    """Cetak ke terminal sekaligus simpan ke laporan."""
    print(msg)
    LOG_LINES.append(msg)


# =============================================================================
# BAGIAN 1 — LOAD DATA
# =============================================================================
def load_data():
    """
    Memuat train.csv dan test.csv dari folder Output.
    Catatan: fitur sudah di-scale [0,1] oleh preprocessing.py,
    tetapi TARGET (harga_cabai_merah) masih dalam Rupiah asli.
    Model langsung memprediksi skala Rupiah → tidak perlu inverse transform.
    """
    log("\n[1] Memuat data train dan test...")

    train_path = OUTPUT_DIR / "train.csv"
    test_path  = OUTPUT_DIR / "test.csv"

    for p in [train_path, test_path]:
        if not p.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {p}")

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    log(f"    -> Train : {df_train.shape[0]:,} baris, {df_train.shape[1]} kolom")
    log(f"    -> Test  : {df_test.shape[0]:,} baris, {df_test.shape[1]} kolom")

    # Verifikasi TARGET ada dan masih skala Rupiah
    if TARGET not in df_train.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan di train.csv")

    y_max = df_train[TARGET].max()
    y_min = df_train[TARGET].min()
    log(f"    -> Range TARGET (train) : Rp {y_min:,.0f} – Rp {y_max:,.0f}")
    if y_max <= 1.0:
        log("    ! PERINGATAN: TARGET tampaknya sudah di-scale. Periksa preprocessing.py.")

    return df_train, df_test


def pisahkan_fitur(df: pd.DataFrame):
    """
    Memisahkan fitur (X) dan target (y).
    Drop kolom 'tanggal', 'harga_cabai_rawit', dan TARGET dari fitur.
    """
    drop_cols = ["tanggal", "harga_cabai_rawit", TARGET]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[TARGET]
    return X, y


# =============================================================================
# BAGIAN 2 — HYPERPARAMETER TUNING
# =============================================================================
def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Mencari hyperparameter terbaik XGBoost menggunakan RandomizedSearchCV
    dengan TimeSeriesSplit (n_splits=5) dan scoring neg_mean_absolute_error.
    """
    log("\n[2] Hyperparameter tuning dengan RandomizedSearchCV...")

    param_dist = {
        "n_estimators"     : [300, 500, 700, 1000, 1500],
        "max_depth"        : [3, 4, 5, 6, 7, 8],
        "learning_rate"    : [0.01, 0.02, 0.05, 0.1, 0.2],
        "subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree" : [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight" : [1, 3, 5, 7, 10],
        "gamma"            : [0, 0.1, 0.2, 0.3, 0.5],
        "reg_alpha"        : [0, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda"       : [0.5, 1.0, 1.5, 2.0, 5.0],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    base_model = XGBRegressor(
        objective    = "reg:squarederror",
        tree_method  = "hist",
        random_state = RANDOM_STATE,
        verbosity    = 0,
    )

    search = RandomizedSearchCV(
        estimator           = base_model,
        param_distributions = param_dist,
        n_iter              = 50,
        cv                  = tscv,
        scoring             = "neg_mean_absolute_error",
        random_state        = RANDOM_STATE,
        n_jobs              = -1,
        verbose             = 1,
    )

    search.fit(X_train, y_train)

    best_params = search.best_params_
    # MAE dalam skala Rupiah (karena TARGET tidak di-scale)
    log(f"    -> Best CV MAE (Rupiah) : Rp {-search.best_score_:,.2f}")
    log(f"    -> Best params          : {best_params}")
    return best_params


# =============================================================================
# BAGIAN 3 — TRAINING MODEL
# =============================================================================
def train_model(X_train: pd.DataFrame, y_train: pd.Series, best_params: dict):
    """
    Melatih XGBRegressor dengan hyperparameter terbaik.
    Menggunakan early stopping dengan validasi internal 80/20.
    eval_metric='rmse' di-set eksplisit agar evals_result() berfungsi.
    """
    log("\n[3] Training model XGBoost...")

    # Split internal untuk early stopping (80/20 urutan waktu)
    split_val   = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_val], X_train.iloc[split_val:]
    y_tr, y_val = y_train.iloc[:split_val], y_train.iloc[split_val:]

    # Pisahkan n_estimators dari best_params agar tidak konflik
    # dengan early_stopping_rounds (XGBoost menggunakan n_estimators sebagai batas atas)
    model = XGBRegressor(
        objective             = "reg:squarederror",
        tree_method           = "hist",
        random_state          = RANDOM_STATE,
        verbosity             = 0,
        eval_metric           = "rmse",   # FIX: wajib di-set agar learning curve tersedia
        early_stopping_rounds = 50,
        **best_params,
    )

    model.fit(
        X_tr, y_tr,
        eval_set = [(X_tr, y_tr), (X_val, y_val)],
        verbose  = False,
    )

    best_iter = model.best_iteration
    log(f"    -> Best iteration (early stopping) : {best_iter}")
    log(f"    -> Train size (fit)                : {len(X_tr):,} baris")
    log(f"    -> Val size   (early stop)         : {len(X_val):,} baris")
    return model


# =============================================================================
# BAGIAN 4 — EVALUASI (skala Rupiah langsung)
# =============================================================================
def hitung_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Menghitung Mean Absolute Percentage Error (%)."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluasi_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Mengevaluasi model pada data test dalam skala Rupiah asli.

    FIX: TARGET tidak di-scale oleh preprocessing.py, sehingga prediksi
    model langsung dalam Rupiah — tidak diperlukan inverse transform.
    """
    log("\n[4] Evaluasi model...")

    y_pred = model.predict(X_test)
    y_true = y_test.values

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = hitung_mape(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    log(f"\n    METRIK EVALUASI (Skala Rupiah Asli):")
    log(f"    {'MAE':<6}: Rp {mae:>12,.2f}")
    log(f"    {'RMSE':<6}: Rp {rmse:>12,.2f}")
    log(f"    {'MAPE':<6}: {mape:.4f} %")
    log(f"    {'R²':<6}: {r2:.6f}")

    metrik = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
    return metrik, y_true, y_pred


# =============================================================================
# BAGIAN 5 — FEATURE IMPORTANCE
# =============================================================================
def feature_importance(model, feature_names: list):
    """
    Menampilkan, menyimpan, dan memplot top 20 fitur terpenting.
    """
    log("\n[5] Feature importance...")

    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        "fitur"      : feature_names,
        "importance" : importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    df_imp.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    log(f"    -> feature_importance.csv disimpan ({len(df_imp)} fitur)")

    top20 = df_imp.head(20)
    log("\n    Top 20 Fitur Terpenting:")
    for _, row in top20.iterrows():
        rank = df_imp[df_imp["fitur"] == row["fitur"]].index[0] + 1
        log(f"    {rank:>2}. {row['fitur']:<30} : {row['importance']:.6f}")

    # Plot bar horizontal
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top20)))
    bars = ax.barh(
        top20["fitur"][::-1].values,
        top20["importance"][::-1].values,
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(
        "Top 20 Fitur Terpenting — XGBoost\nPrediksi Harga Cabai Merah Kota Padang",
        fontsize=13, fontweight="bold", pad=15,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    for bar, val in zip(bars, top20["importance"][::-1].values):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=8, color="#333",
        )

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("    -> feature_importance.png disimpan")
    return df_imp


# =============================================================================
# BAGIAN 6 — VISUALISASI HASIL
# =============================================================================
def visualisasi(model, y_true_rp: np.ndarray, y_pred_rp: np.ndarray,
                df_test: pd.DataFrame):
    """
    Membuat dua plot:
    1. Prediksi vs Aktual pada data test (skala Rupiah)
    2. Learning curve (train RMSE vs validation RMSE)

    FIX: eval_metric='rmse' di-set saat training sehingga evals_result()
    selalu tersedia. Key dict adalah 'validation_0' dan 'validation_1'.
    """
    log("\n[6] Membuat visualisasi...")

    # Tanggal untuk sumbu x
    if "tanggal" in df_test.columns:
        tanggal = pd.to_datetime(df_test["tanggal"]).values
    else:
        tanggal = np.arange(len(y_true_rp))

    # ── Plot 1: Prediksi vs Aktual ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(tanggal, y_true_rp, label="Aktual",
            color="#2196F3", linewidth=1.5, alpha=0.9)
    ax.plot(tanggal, y_pred_rp, label="Prediksi",
            color="#FF5722", linewidth=1.5, linestyle="--", alpha=0.9)

    # Confidence band ±1 std error
    residuals = y_true_rp - y_pred_rp
    std_err   = residuals.std()
    ax.fill_between(
        tanggal,
        y_pred_rp - std_err, y_pred_rp + std_err,
        alpha=0.2, color="#FF5722", label="±1 Std Error",
    )

    # Trend line (prediksi)
    z = np.polyfit(np.arange(len(y_pred_rp)), y_pred_rp, 1)
    p = np.poly1d(z)
    ax.plot(tanggal, p(np.arange(len(y_pred_rp))),
            color="#9C27B0", linewidth=1.2, linestyle=":",
            label="Trend Prediksi", alpha=0.8)

    ax.set_title(
        "Prediksi vs Aktual Harga Cabai Merah — Kota Padang\n(Data Test)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Tanggal", fontsize=11)
    ax.set_ylabel("Harga (Rupiah)", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rp {x:,.0f}"))
    ax.legend(fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "prediksi_vs_aktual.png", dpi=150, bbox_inches="tight")
    plt.close()
    log("    -> prediksi_vs_aktual.png disimpan")

    # ── Plot 2: Learning Curve ───────────────────────────────────────────────
    # FIX: eval_metric='rmse' sudah di-set di train_model(), sehingga
    # evals_result() pasti berisi key 'rmse' di validation_0 dan validation_1.
    results = model.evals_result()
    train_loss = results.get("validation_0", {}).get("rmse", [])
    val_loss   = results.get("validation_1", {}).get("rmse", [])

    if train_loss and val_loss:
        epochs    = np.arange(1, len(train_loss) + 1)
        best_iter = model.best_iteration

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, train_loss, label="Train Loss (RMSE)",
                color="#2196F3", linewidth=1.5)
        ax.plot(epochs, val_loss,   label="Validation Loss (RMSE)",
                color="#FF5722", linewidth=1.5, linestyle="--")
        ax.axvline(x=best_iter, color="#4CAF50", linewidth=1.5,
                   linestyle=":", label=f"Best Iter = {best_iter}")

        ax.set_title(
            "Learning Curve — XGBoost\nTrain Loss vs Validation Loss",
            fontsize=13, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Iterasi (n_estimators)", fontsize=11)
        ax.set_ylabel("RMSE (Rupiah)", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rp {x:,.0f}"))
        ax.legend(fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(MODEL_DIR / "learning_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        log("    -> learning_curve.png disimpan")
    else:
        log("    ! Learning curve tidak tersedia (evals_result kosong)")


# =============================================================================
# BAGIAN 7 — SIMPAN MODEL & LAPORAN
# =============================================================================
def simpan_model(model, best_params: dict, metrik: dict, df_imp: pd.DataFrame):
    """
    Menyimpan model XGBoost, hyperparameter terbaik, dan laporan evaluasi.
    """
    log("\n[7] Menyimpan model dan laporan...")

    # Simpan model
    model_path = MODEL_DIR / "xgboost_cabai.pkl"
    joblib.dump(model, model_path)
    log("    + xgboost_cabai.pkl disimpan")

    # Simpan best params
    params_path = MODEL_DIR / "best_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4, ensure_ascii=False)
    log("    + best_params.json disimpan")

    # Simpan laporan evaluasi
    laporan_path = MODEL_DIR / "laporan_model.txt"
    lines = [
        "=" * 55,
        "  LAPORAN EVALUASI MODEL XGBOOST",
        "  Prediksi Harga Cabai Merah — Kota Padang",
        "=" * 55,
        "",
        "INFORMASI MODEL",
        "-" * 40,
        "  Algoritma   : XGBRegressor",
        "  Objective   : reg:squarederror",
        "  Tree Method : hist",
        f"  Random State: {RANDOM_STATE}",
        "  Catatan     : Fitur di-scale [0,1], TARGET dalam Rupiah asli",
        "",
        "HYPERPARAMETER TERBAIK",
        "-" * 40,
    ]
    for k, v in best_params.items():
        lines.append(f"  {k:<22}: {v}")

    lines += [
        "",
        "METRIK EVALUASI — SKALA RUPIAH ASLI",
        "-" * 40,
        f"  MAE  : Rp {metrik['MAE']:>12,.2f}",
        f"  RMSE : Rp {metrik['RMSE']:>12,.2f}",
        f"  MAPE : {metrik['MAPE']:.4f} %",
        f"  R²   : {metrik['R2']:.6f}",
        "",
        "TOP 10 FITUR TERPENTING",
        "-" * 40,
    ]
    for i, row in df_imp.head(10).iterrows():
        lines.append(f"  {i+1:>2}. {row['fitur']:<28} : {row['importance']:.6f}")

    lines += [
        "",
        "FILE OUTPUT",
        "-" * 40,
        f"  Model          : {MODEL_DIR / 'xgboost_cabai.pkl'}",
        f"  Best Params    : {MODEL_DIR / 'best_params.json'}",
        f"  Feature Imp.   : {MODEL_DIR / 'feature_importance.csv'}",
        f"  Plot Prediksi  : {MODEL_DIR / 'prediksi_vs_aktual.png'}",
        f"  Learning Curve : {MODEL_DIR / 'learning_curve.png'}",
        "=" * 55,
    ]

    with open(laporan_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log("    + laporan_model.txt disimpan")
    log(f"\n    Semua file tersimpan di: {MODEL_DIR}")


# =============================================================================
# BAGIAN 8 — LAPORAN AKHIR KE TERMINAL
# =============================================================================
def laporan_akhir(metrik: dict, df_imp: pd.DataFrame,
                  n_train: int, n_test: int, n_fitur: int):
    """Mencetak ringkasan hasil training ke terminal."""
    border = "═" * 47
    sep    = "─" * 47

    log(f"\n{border}")
    log("  HASIL TRAINING MODEL XGBOOST")
    log(border)
    log("  Model      : XGBRegressor")
    log(f"  Train size : {n_train:,} baris")
    log(f"  Test size  : {n_test:,} baris")
    log(f"  Fitur      : {n_fitur} fitur")
    log(sep)
    log("  METRIK EVALUASI (Skala Rupiah Asli):")
    log(f"    MAE  : Rp {metrik['MAE']:>12,.2f}")
    log(f"    RMSE : Rp {metrik['RMSE']:>12,.2f}")
    log(f"    MAPE : {metrik['MAPE']:.4f} %")
    log(f"    R²   : {metrik['R2']:.6f}")
    log(sep)
    log("  Top 5 Fitur Terpenting:")
    for i, row in df_imp.head(5).iterrows():
        log(f"    {i+1}. {row['fitur']:<28} : {row['importance']:.6f}")
    log("  Model disimpan: xgboost_cabai.pkl")
    log(border)


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("=" * 55)
    log("  TRAINING MODEL XGBOOST — HARGA CABAI MERAH")
    log("  Wilayah: Kota Padang, Sumatera Barat")
    log("=" * 55)

    try:
        # 1. Load data
        df_train, df_test = load_data()

        # Pisahkan fitur dan target
        X_train, y_train = pisahkan_fitur(df_train)
        X_test,  y_test  = pisahkan_fitur(df_test)
        feature_cols = list(X_train.columns)

        log(f"\n    -> Jumlah fitur : {len(feature_cols)}")
        log(f"    -> Fitur        : {feature_cols[:5]}... (dan seterusnya)")

        # 2. Hyperparameter tuning
        best_params = hyperparameter_tuning(X_train, y_train)

        # 3. Training model
        model = train_model(X_train, y_train, best_params)

        # 4. Evaluasi (skala Rupiah langsung, tidak perlu inverse transform)
        metrik, y_true_rp, y_pred_rp = evaluasi_model(model, X_test, y_test)

        # 5. Feature importance
        df_imp = feature_importance(model, feature_cols)

        # 6. Visualisasi
        visualisasi(model, y_true_rp, y_pred_rp, df_test)

        # 7. Simpan model & laporan
        simpan_model(model, best_params, metrik, df_imp)

        # 8. Laporan akhir
        laporan_akhir(metrik, df_imp, len(df_train), len(df_test), len(feature_cols))

        log("\n[SELESAI] Training model berhasil!")
        return model, metrik, df_imp

    except FileNotFoundError as e:
        print(f"\n[ERROR] File tidak ditemukan: {e}")
        print("Pastikan preprocessing.py sudah dijalankan terlebih dahulu.")
        return None, None, None
    except Exception as e:
        import traceback
        print(f"\n[ERROR] Terjadi kesalahan: {e}")
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    model, metrik, df_imp = main()