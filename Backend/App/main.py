"""
main.py — Entry point FastAPI
Prediksi Harga Cabai Kota Padang
Jalankan: uvicorn app.main:app --reload
"""
import sys
import asyncio
import logging
from pathlib import Path

# Tambah Backend/ ke sys.path agar semua import bisa ditemukan
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from App.core import predictor
from App.core.scheduler import setup_scheduler, stop_scheduler
from App.routes import predict, history, dashboard, realtime

logger = logging.getLogger(__name__)

# =============================================================================
# INITIAL UPDATE — dijalankan saat startup, non-blocking
# =============================================================================
async def _initial_update():
    """Update data sekali saat startup agar data selalu fresh saat server restart."""
    await asyncio.sleep(5)   # Tunggu model selesai load dulu
    try:
        from App.core.scraper import jalankan_update_realtime
        logger.info("[Startup] Menjalankan initial update real-time...")
        await jalankan_update_realtime()
        logger.info("[Startup] Initial update selesai")
    except Exception as e:
        logger.warning(f"[Startup] Initial update gagal (tidak masalah): {e}")


# =============================================================================
# LIFESPAN EVENT — load model, dataset, dan scheduler saat startup
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load semua model XGBoost, dataset, dan setup scheduler saat server start."""
    print("\n[Startup] Memuat ML artifacts dan dataset...")
    predictor.preload_artifacts()
    predictor.load_dataset_to_cache()

    info = predictor.get_cache_info()
    models_loaded = info.get("models_loaded", [])
    if models_loaded:
        print(f"[Startup] Model siap: {models_loaded}")
    else:
        print("[Startup] WARNING: Tidak ada model yang berhasil dimuat!")

    # Setup scheduler harian otomatis
    try:
        setup_scheduler()
    except Exception as e:
        print(f"[Startup] Scheduler gagal start (tidak masalah): {e}")

    # Initial update — non-blocking background task
    asyncio.create_task(_initial_update())

    yield

    # Shutdown
    print("\n[Shutdown] Membersihkan resources...")
    try:
        stop_scheduler()
    except Exception:
        pass
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
    version     = "2.0.0",
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
            "realtime"  : "/api/realtime",
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
app.include_router(realtime.router,  prefix="/api/realtime",  tags=["Real-time Data"])