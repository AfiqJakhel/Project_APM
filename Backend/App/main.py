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
from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=True)

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core import predictor
from app.routes import predict, history, dashboard

logger = logging.getLogger(__name__)

# =============================================================================
# (Scraping dan scheduler realtime dimatikan sesuai permintaan user)
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
    models_merah = info.get("models_merah", [])
    models_rawit = info.get("models_rawit", [])
    if models_merah:
        print(f"[Startup] Model merah siap : {models_merah}")
    else:
        print("[Startup] WARNING: Tidak ada model merah yang dimuat!")
    if models_rawit:
        print(f"[Startup] Model rawit siap : {models_rawit}")
    else:
        print("[Startup] INFO: Model rawit belum tersedia (jalankan train_rawit.py)")

    # (Scheduler dan initial update dimatikan)

    yield

    # Shutdown
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
    info = predictor.get_cache_info()
    return {
        "status"       : "running",
        "message"      : "API Prediksi Harga Cabai Kota Padang (Merah + Rawit)",
        "model_merah"  : info.get("models_merah", []),
        "model_rawit"  : info.get("models_rawit", []),
        "merah_ready"  : len(info.get("models_merah", [])) > 0,
        "rawit_ready"  : len(info.get("models_rawit", [])) > 0,
        "endpoints"    : {
            "docs"              : "/docs",
            "prediksi_merah"    : "/api/predict/prediksi/{h1|h3|h7}",
            "prediksi_rawit"    : "/api/predict/prediksi/rawit/{h1|h3|h7}",
            "prediksi_semua"    : "/api/predict/prediksi/semua",
            "historis"          : "/api/predict/harga/historis",
            "dashboard"         : "/api/dashboard",
            "scheduler_status"  : "/api/v1/admin/scheduler",
        }
    }


@app.get("/health", tags=["Root"])
def health():
    """Health check untuk monitoring."""
    info = predictor.get_cache_info()
    n_merah = len(info.get("models_merah", []))
    n_rawit = len(info.get("models_rawit", []))
    return {
        "status"      : "ok",
        "merah_ready" : n_merah > 0,
        "rawit_ready" : n_rawit > 0,
        "n_model_merah": n_merah,
        "n_model_rawit": n_rawit,
    }


# =============================================================================
# REGISTER ROUTES
# =============================================================================
app.include_router(predict.router,   prefix="/api/predict",   tags=["Prediksi"])
app.include_router(history.router,   prefix="/api/history",   tags=["Historis"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])