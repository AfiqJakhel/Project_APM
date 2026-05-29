# app/schemas/predict.py
# Pydantic v2 — Response schemas untuk endpoint prediksi dan dashboard.
# Schema request (PrediksiRequest) dan schema endpoint lama (PrediksiResponse,
# PrediksiSemuaResponse) telah dihapus bersama endpoint POST / yang sudah
# tidak digunakan.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# CUACA / REALTIME SCHEMAS
# ---------------------------------------------------------------------------

class CuacaInfo(BaseModel):
    """Info cuaca yang digunakan saat prediksi otomatis."""
    suhu_rata   : Optional[float] = None
    kelembaban  : Optional[float] = None
    curah_hujan : Optional[float] = None
    status      : str = "unknown"


class RealtimeStatus(BaseModel):
    """Status sistem real-time — response dari GET /api/realtime/status."""
    data_harga_status     : str
    data_cuaca_status     : str
    tanggal_data_terkini  : Optional[str]       = None
    waktu_update_terakhir : Optional[str]       = None
    harga_terkini         : Optional[float]     = None
    cuaca_terkini         : Optional[CuacaInfo] = None
    pesan                 : str


# ---------------------------------------------------------------------------
# RESPONSE SCHEMAS — Endpoint RESTful GET /api/predict/...
# ---------------------------------------------------------------------------

class PrediksiOtomatisResponse(BaseModel):
    """Response dari GET /api/predict/prediksi/{horizon}."""
    status           : str
    horizon          : str
    keterangan       : str
    tanggal_prediksi : str
    prediksi_rp      : float
    model_version    : str
    arah_prediksi    : Optional[str]       = None
    perubahan_persen : Optional[float]     = None
    # Real-time fields (opsional, backward-compatible)
    data_status      : Optional[str]       = None   # "live" | "fallback"
    data_tanggal     : Optional[str]       = None
    cuaca_digunakan  : Optional[CuacaInfo] = None


class DataHistorisResponse(BaseModel):
    """Response dari GET /api/predict/harga/historis."""
    status : str
    n_hari : int
    data   : list


class ModelMetrikResponse(BaseModel):
    """Response dari GET /api/predict/model/metrik."""
    status : str
    metrik : dict


class TanggalTersediaResponse(BaseModel):
    """Response dari GET /api/predict/tanggal-tersedia."""
    tanggal_min : str
    tanggal_max : str
    total_hari  : int


class FiturTerkiniResponse(BaseModel):
    """Response dari GET /api/predict/fitur-terkini."""
    status  : str
    data    : dict
    tanggal : str


class CacheInfoResponse(BaseModel):
    """Response dari GET /api/predict/cache-info."""
    status : str
    cache  : dict


# ---------------------------------------------------------------------------
# DASHBOARD SCHEMAS (dipakai oleh app/routes/dashboard.py)
# ---------------------------------------------------------------------------

class DashboardResponse(BaseModel):
    """Response dari GET /api/dashboard/."""
    tanggal_update    : str
    harga_hari_ini    : float
    harga_min_30hari  : float
    harga_max_30hari  : float
    harga_rata_30hari : float
    tren              : str               # "naik" | "turun" | "stabil"
    prediksi_h1       : Optional[float] = None
    prediksi_h3       : Optional[float] = None
    prediksi_h7       : Optional[float] = None
    status_model      : bool
    n_model_aktif     : int
    status_inflasi    : str               # "normal" | "waspada" | "kritis"
    realtime          : Optional[dict]  = None


class StatistikBulananItem(BaseModel):
    """Item dalam response GET /api/dashboard/statistik-bulanan."""
    bulan      : int
    nama_bulan : str
    rata_rata  : float
    min        : float
    max        : float
    n_hari     : int
