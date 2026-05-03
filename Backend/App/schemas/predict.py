# app/schemas/predict.py
# Pydantic v2 — Request & Response schemas untuk endpoint prediksi dan dashboard.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import date
from typing import Optional
from pydantic import BaseModel, ConfigDict, field_validator

from config.settings import HORIZON_LABEL


# ---------------------------------------------------------------------------
# REQUEST SCHEMA
# ---------------------------------------------------------------------------

class PrediksiRequest(BaseModel):
    """Body request untuk endpoint POST /api/predict/."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "tanggal": "2024-07-01",
            "horizon": "h1",
            "suhu_rata": 28.5,
            "curah_hujan": 12.3,
            "kelembaban": 80.0,
        }
    })

    tanggal: date
    horizon: str = "h1"
    suhu_rata: Optional[float] = None
    curah_hujan: Optional[float] = None
    kelembaban: Optional[float] = None

    @field_validator("horizon")
    @classmethod
    def validasi_horizon(cls, v: str) -> str:
        allowed = {"h1", "h3", "h7"}
        if v not in allowed:
            raise ValueError(f"Horizon harus salah satu dari: {allowed}. Diterima: '{v}'")
        return v


# ---------------------------------------------------------------------------
# RESPONSE SCHEMAS
# ---------------------------------------------------------------------------

class PrediksiResponse(BaseModel):
    """Response dari endpoint POST /api/predict/ dan GET /api/predict/semua."""

    tanggal_input: str
    horizon: str
    label_horizon: str      # contoh: "Prediksi Besok (H+1)"
    prediksi_harga: float   # Rupiah
    status_harga: str       # "normal" | "tinggi" | "kritis"
    pesan: str              # Penjelasan status dalam Bahasa Indonesia


class PrediksiSemuaResponse(BaseModel):
    """Response dari endpoint GET /api/predict/semua."""
    tanggal: str
    prediksi: dict


class TanggalTersediaResponse(BaseModel):
    """Response dari endpoint GET /api/predict/tanggal-tersedia."""
    tanggal_min: str
    tanggal_max: str
    total_hari: int


class FiturTerkiniResponse(BaseModel):
    """Response dari endpoint GET /api/predict/fitur-terkini."""
    status: str
    data: dict
    tanggal: str


class DataHistorisResponse(BaseModel):
    """Response dari endpoint GET /api/predict/historis."""
    status: str
    n_hari: int
    data: list


class ModelMetrikResponse(BaseModel):
    """Response dari endpoint GET /api/predict/metrik."""
    status: str
    metrik: dict


class CacheInfoResponse(BaseModel):
    """Response dari endpoint GET /api/predict/cache-info."""
    status: str
    cache: dict


class PrediksiOtomatisResponse(BaseModel):
    """Response dari endpoint GET /api/predict/prediksi/{horizon}."""
    status: str
    horizon: str
    keterangan: str
    tanggal_prediksi: str
    prediksi_rp: float
    model_version: str


class DashboardResponse(BaseModel):
    """Response dari endpoint GET /api/dashboard/."""

    tanggal_update: str
    harga_hari_ini: float
    harga_min_30hari: float
    harga_max_30hari: float
    harga_rata_30hari: float
    tren: str               # "naik" | "turun" | "stabil"
    prediksi_h1: Optional[float] = None
    prediksi_h3: Optional[float] = None
    prediksi_h7: Optional[float] = None
    status_model: bool
    n_model_aktif: int
    status_inflasi: str     # "normal" | "waspada" | "kritis"

class StatistikBulananItem(BaseModel):
    bulan: int
    nama_bulan: str
    rata_rata: float
    min: float
    max: float
    n_hari: int
