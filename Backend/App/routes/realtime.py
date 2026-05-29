# app/routes/realtime.py
# Router FastAPI untuk endpoint data real-time (status, update manual, cuaca).

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime
from fastapi import APIRouter, HTTPException

from app.core.scraper import (
    jalankan_update_realtime,
    get_cuaca_realtime,
    baca_status_realtime,
)
from app.core.scheduler import get_next_run_time
from config.settings import CUACA_LATITUDE, CUACA_LONGITUDE

router = APIRouter()


# ---------------------------------------------------------------------------
# ENDPOINT 1: GET /status — Status sistem real-time
# ---------------------------------------------------------------------------

@router.get(
    "/status",
    summary="Status sistem real-time",
    response_description="Status data harga dan cuaca terkini beserta waktu update terakhir",
)
def get_realtime_status():
    """
    Kembalikan status lengkap sistem real-time:
    - Status data harga (live dari PIHPS atau fallback)
    - Status data cuaca (live dari Open-Meteo atau fallback)
    - Harga terkini dan cuaca terkini
    - Waktu update terakhir dan jadwal berikutnya

    Response ini dibaca dari `realtime_status.json` (tidak melakukan request baru).
    """
    try:
        status = baca_status_realtime()
        next_run = get_next_run_time()

        return {
            "status"                   : "ok",
            "data_harga_status"        : status.get("data_harga_status", "unknown"),
            "data_cuaca_status"        : status.get("data_cuaca_status", "unknown"),
            "tanggal_data_terkini"     : status.get("tanggal_data_terkini"),
            "waktu_update_terakhir"    : status.get("waktu_update_terakhir"),
            "harga_terkini"            : status.get("harga_terkini"),
            "cuaca_terkini"            : status.get("cuaca_terkini"),
            "jadwal_update_berikutnya" : next_run,
            "pesan"                    : status.get("pesan", ""),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal membaca status real-time: {str(e)}",
        )


# ---------------------------------------------------------------------------
# ENDPOINT 2: POST /update — Manual trigger update
# ---------------------------------------------------------------------------

@router.post(
    "/update",
    summary="Trigger update data real-time secara manual",
    response_description=(
        "Status update harga dan cuaca. "
        "CATATAN: Endpoint ini memakan waktu 15–60 detik karena menunggu Selenium scraper. "
        "Gunakan untuk demo presentasi atau testing."
    ),
)
async def trigger_update_manual():
    """
    **Trigger update data harga dan cuaca secara manual.**

    Berguna untuk demo presentasi — tidak perlu menunggu scheduler jam 14.00.
    Bisa dijalankan kapan saja sebelum atau saat presentasi.

    **Urutan proses:**
    1. Ambil cuaca terkini dari Open-Meteo (~1 detik)
    2. Scrape harga cabai dari PIHPS BI via Selenium (~15–60 detik)
    3. Update dataset CSV
    4. Reload cache predictor

    **Jika PIHPS BI tidak bisa diakses:**
    - Sistem otomatis fallback ke harga terakhir di CSV
    - Response tetap valid dengan `harga_status: "fallback"`
    - Prediksi tetap berjalan normal
    """
    try:
        hasil = await jalankan_update_realtime()
        return hasil
    except Exception as e:
        # Tidak pernah raise 500 — selalu return response yang valid
        return {
            "status"      : "error",
            "harga_status": "fallback",
            "cuaca_status": "fallback",
            "waktu_update": datetime.now().isoformat(),
            "pesan"       : f"Update gagal total: {str(e)}",
        }


# ---------------------------------------------------------------------------
# ENDPOINT 3: GET /cuaca — Cuaca Padang saat ini
# ---------------------------------------------------------------------------

@router.get(
    "/cuaca",
    summary="Cuaca Kota Padang saat ini",
    response_description="Data cuaca real-time dari Open-Meteo API (gratis, tanpa API key)",
)
async def get_cuaca():
    """
    **Ambil cuaca Kota Padang saat ini dari Open-Meteo API.**

    Tidak memperbarui dataset — hanya untuk widget cuaca di React frontend.
    Gratis, tidak perlu API key, tidak ada rate limit ketat.

    **Data yang dikembalikan:**
    - `suhu_rata`: Temperatur udara (°C)
    - `kelembaban`: Kelembaban relatif (%)
    - `curah_hujan`: Presipitasi saat ini (mm)
    """
    try:
        cuaca = await get_cuaca_realtime()
        return {
            **cuaca,
            "lokasi": "Kota Padang, Sumatera Barat",
            "sumber": "Open-Meteo API",
            "koordinat": {
                "latitude" : CUACA_LATITUDE,
                "longitude": CUACA_LONGITUDE,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil data cuaca: {str(e)}",
        )
