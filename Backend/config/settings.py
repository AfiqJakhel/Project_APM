"""
settings.py — Konfigurasi global project prediksi harga cabai
"""
from pathlib import Path

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# Model cabai MERAH → expanding_window_merah/
MODEL_DIR_MERAH = BASE_DIR / "machine_learning" / "output" / "expanding_window_merah"

# Model cabai RAWIT → expanding_window_rawit/
MODEL_DIR_RAWIT = BASE_DIR / "machine_learning" / "output" / "expanding_window_rawit"

# Backward compat: MODEL_DIR masih menunjuk ke merah
MODEL_DIR   = MODEL_DIR_MERAH

# Scaler (shared, dari preprocessing.py)
SCALER_DIR  = BASE_DIR / "machine_learning" / "output" / "xgboost_models"
DATA_DIR    = BASE_DIR / "data" / "processed"

# ── Target ───────────────────────────────────────────────────────────────────
TARGET       = "harga_cabai_merah"
TARGET_RAWIT = "harga_cabai_rawit"
TARGET_COLS  = ["target_h1", "target_h3", "target_h7"]
TARGET_COLS_RAWIT = ["target_rawit_h1", "target_rawit_h3", "target_rawit_h7"]

# ── Model files — pola glob untuk cari file terbaru ───────────────────────────
# Format: model_final_{label}_{YYYYMMDD}.pkl
MODEL_FILES = {
    # Cabai merah (di MODEL_DIR_MERAH)
    "h1"       : "model_final_h1_*.pkl",
    "h3"       : "model_final_h3_*.pkl",
    "h7"       : "model_final_h7_*.pkl",
    # Cabai rawit (di MODEL_DIR_RAWIT)
    "rawit_h1" : "model_final_rawit_h1_*.pkl",
    "rawit_h3" : "model_final_rawit_h3_*.pkl",
    "rawit_h7" : "model_final_rawit_h7_*.pkl",
}

FEATURE_FILES = {
    "h1"       : "feature_cols_h1.json",
    "h3"       : "feature_cols_h3.json",
    "h7"       : "feature_cols_h7.json",
    "rawit_h1" : "feature_cols_rawit_h1.json",
    "rawit_h3" : "feature_cols_rawit_h3.json",
    "rawit_h7" : "feature_cols_rawit_h7.json",
}
SCALER_FILE = "scaler.pkl"

# ── Threshold early warning inflasi ──────────────────────────────────────────
BATAS_HARGA_NORMAL  = 50_000
BATAS_HARGA_TINGGI  = 75_000
BATAS_HARGA_KRITIS  = 100_000

BATAS_RAWIT_NORMAL  = 60_000
BATAS_RAWIT_TINGGI  = 90_000
BATAS_RAWIT_KRITIS  = 120_000

# ── Horizon label ─────────────────────────────────────────────────────────────
HORIZON_LABEL = {
    "h1"      : "Prediksi Besok (H+1)",
    "h3"      : "Prediksi 3 Hari (H+3)",
    "h7"      : "Prediksi 7 Hari (H+7)",
    "rawit_h1": "Rawit — Prediksi Besok (H+1)",
    "rawit_h3": "Rawit — Prediksi 3 Hari (H+3)",
    "rawit_h7": "Rawit — Prediksi 7 Hari (H+7)",
}

# ── Expanding window config ───────────────────────────────────────────────────
MIN_TRAIN_DAYS = 365
STEP_DAYS      = 30
TEST_DAYS      = 30

# ── Real-time data settings (Dihapus) ─────────────────────────────────────────
