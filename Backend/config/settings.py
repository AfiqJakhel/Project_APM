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

# ── Real-time data settings ───────────────────────────────────────────────────

# PIHPS BI — Scraper settings
PIHPS_URL = "https://www.bi.go.id/hargapangan/TabelHarga/PasarTradisionalDaerah"

# Filter scraping — sesuaikan dengan filter di website PIHPS BI
PIHPS_PROVINSI  = "Sumatera Barat"
PIHPS_KOTA      = "Kota Padang"
PIHPS_KOMODITAS = ["Cabai Merah Keriting", "Cabai Rawit Hijau"]

# Mapping nama komoditas PIHPS ke nama kolom dataset
PIHPS_KOLOM_MAP = {
    "Cabai Merah Keriting": "harga_cabai_merah",
    "Cabai Rawit Hijau"   : "harga_cabai_rawit",
}

# Open-Meteo API — Koordinat Kota Padang
CUACA_LATITUDE   = -0.9471
CUACA_LONGITUDE  = 100.4172
CUACA_TIMEZONE   = "Asia/Jakarta"
CUACA_API_URL    = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&current=temperature_2m,relative_humidity_2m,precipitation"
    "&timezone={tz}"
)

# Scheduler settings
SCHEDULER_JAM   = 14   # jam 14.00 WIB
SCHEDULER_MENIT = 0

# File status real-time (JSON kecil untuk tracking update terakhir)
REALTIME_STATUS_FILE = DATA_DIR / "realtime_status.json"

# Timeout scraper (detik)
SCRAPER_TIMEOUT  = 30
SCRAPER_HEADLESS = True   # True = tidak buka jendela browser
