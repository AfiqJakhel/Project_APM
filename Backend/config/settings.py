"""
settings.py — Konfigurasi global project prediksi harga cabai
"""
from pathlib import Path

# ── Path ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
# MODEL_DIR: machine_learning/output/expanding_window/ (lokasi model hasil training)
# Model disimpan dengan format: model_final_{horizon}_YYYYMMDD.pkl
MODEL_DIR   = BASE_DIR / "machine_learning" / "output" / "expanding_window"
# SCALER_DIR: machine_learning/output/xgboost_models/ (lokasi scaler.pkl)
SCALER_DIR  = BASE_DIR / "machine_learning" / "output" / "xgboost_models"
DATA_DIR    = BASE_DIR / "data" / "processed"

# ── Target ───────────────────────────────────────────────────────────────────
TARGET       = "harga_cabai_merah"
TARGET_COLS  = ["target_h1", "target_h3", "target_h7"]

# ── Model files ───────────────────────────────────────────────────────────────
# Model files menggunakan pattern dengan timestamp: model_final_{horizon}_YYYYMMDD.pkl
# Sistem akan otomatis mencari file terbaru dengan glob pattern
MODEL_FILES = {
    "h1": "model_final_h1_*.pkl",
    "h3": "model_final_h3_*.pkl",
    "h7": "model_final_h7_*.pkl",
}
FEATURE_FILES = {
    "h1": "feature_cols_h1.json",
    "h3": "feature_cols_h3.json",
    "h7": "feature_cols_h7.json",
}
SCALER_FILE = "scaler.pkl"

# ── Threshold early warning inflasi ──────────────────────────────────────────
BATAS_HARGA_NORMAL  = 50_000   # Rp 50.000/kg (sesuaikan dengan kondisi Padang)
BATAS_HARGA_TINGGI  = 75_000   # Rp 75.000/kg → warning kuning
BATAS_HARGA_KRITIS  = 100_000  # Rp 100.000/kg → warning merah

# ── Horizon label ─────────────────────────────────────────────────────────────
HORIZON_LABEL = {
    "h1": "Prediksi Besok (H+1)",
    "h3": "Prediksi 3 Hari (H+3)",
    "h7": "Prediksi 7 Hari (H+7)",
}

# ── Expanding window config ───────────────────────────────────────────────────
MIN_TRAIN_DAYS = 365
STEP_DAYS      = 30
TEST_DAYS      = 30
