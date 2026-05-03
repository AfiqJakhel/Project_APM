"""
main.py — Entry point FastAPI
Prediksi Harga Cabai Kota Padang
Jalankan: uvicorn app.main:app --reload
"""
import sys
from pathlib import Path

# Tambah Backend/ ke sys.path agar semua import bisa ditemukan
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import predictor
from app.routes import predict, history, dashboard

# =============================================================================
# LIFESPAN EVENT — load model & dataset saat FastAPI pertama dijalankan
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load semua model XGBoost dan dataset saat server pertama kali start."""
    print("\n[Startup] Memuat ML artifacts dan dataset...")
    predictor.preload_artifacts()
    predictor.load_dataset_to_cache()
    
    info = predictor.get_cache_info()
    models_loaded = info.get("models_loaded", [])
    if models_loaded:
        print(f"[Startup] Model siap: {models_loaded}")
    else:
        print("[Startup] WARNING: Tidak ada model yang berhasil dimuat!")
        
    yield
    
    print("\n[Shutdown] Membersihkan resources...")
    predictor.clear_cache()

# =============================================================================
# INISIALISASI FASTAPI
# =============================================================================
app = FastAPI(
    title       = "API Prediksi Harga Cabai — Kota Padang",
    description = (
        "XGBoost multi-horizon prediction H+1, H+3, H+7\n\n"
        "Judul: Implementasi Machine Learning untuk Prediksi Fluktuasi "
        "Harga Cabai di Kota Padang Berbasis Web sebagai Upaya "
        "Pengendalian Inflasi Daerah Menggunakan Metode XGBoost"
    ),
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

# =============================================================================
# CORS — izinkan React frontend mengakses API ini
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins     = [
        "http://localhost:5173",   # Vite default
        "http://localhost:3000",   # Create React App
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)



# =============================================================================
# ROOT ENDPOINT
# =============================================================================
@app.get("/", tags=["Root"])
def root():
    """Cek status API dan model."""
    return {
        "status"      : "running",
        "message"     : "API Prediksi Harga Cabai Kota Padang",
        "model_ready" : len(predictor.get_cache_info().get("models_loaded", [])) > 0,
        "model_aktif" : predictor.get_cache_info().get("models_loaded", []),
        "endpoints"   : {
            "docs"      : "/docs",
            "prediksi"  : "/api/predict",
            "historis"  : "/api/history",
            "dashboard" : "/api/dashboard",
        }
    }

@app.get("/health", tags=["Root"])
def health():
    """Health check untuk monitoring."""
    return {
        "status"      : "ok",
        "model_ready" : len(predictor.get_cache_info().get("models_loaded", [])) > 0,
        "n_model"     : len(predictor.get_cache_info().get("models_loaded", [])),
    }

# =============================================================================
# REGISTER ROUTES
# =============================================================================
app.include_router(predict.router,   prefix="/api/predict",   tags=["Prediksi"])
app.include_router(history.router,   prefix="/api/history",   tags=["Historis"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])