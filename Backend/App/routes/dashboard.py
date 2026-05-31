# app/routes/dashboard.py
# Router FastAPI untuk endpoint statistik dan ringkasan dashboard harga cabai.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from datetime import datetime
from typing import List
from fastapi import APIRouter, HTTPException, Query

from config.settings import (
    DATA_DIR,
    TARGET,
    BATAS_HARGA_TINGGI,
    BATAS_HARGA_KRITIS,
)
from app.core import predictor
from app.schemas.predict import DashboardResponse, StatistikBulananItem

router = APIRouter()

# Nama bulan dalam Bahasa Indonesia
NAMA_BULAN = [
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]


def _get_realtime_info() -> dict:
    """Info data telah diubah menjadi statis tanpa realtime."""
    return {
        "data_status"          : "statis",
        "waktu_update"         : None,
        "harga_hari_ini_sumber": "Data historis",
        "cuaca_sekarang"       : {
            "suhu_rata"  : 27.0,
            "kelembaban" : 80.0,
            "curah_hujan": 5.0,
            "status"     : "fallback",
        },
    }


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_dataset() -> pd.DataFrame:
    """Baca dataset dari cache via predictor."""
    return predictor.get_dataset()


def _detect_tren(df: pd.DataFrame, target_col: str = TARGET) -> str:
    """
    Deteksi tren harga berdasarkan perbandingan:
    - avg_7      : rata-rata harga 7 hari terakhir
    - avg_14_prev: rata-rata harga 7 hari sebelum 7 hari terakhir (hari ke-8 s/d ke-14)
    """
    if len(df) < 14 or target_col not in df.columns:
        return "stabil"

    recent_14 = df[target_col].iloc[-14:]
    avg_7      = recent_14.iloc[-7:].mean()
    avg_14_prev = recent_14.iloc[:7].mean()

    if avg_14_prev == 0:
        return "stabil"

    if avg_7 > avg_14_prev * 1.02:
        return "naik"
    elif avg_7 < avg_14_prev * 0.98:
        return "turun"
    else:
        return "stabil"


async def _predict_for_input(horizon: str, input_data: dict) -> float | None:
    """
    Jalankan prediksi untuk satu horizon dari input dictionary.
    Return None jika model tidak tersedia atau terjadi error.
    """
    try:
        hasil = await predictor.prediksi_harga(input_data, horizon)
        return float(hasil["prediksi_rp"])
    except Exception:
        return None


def _determine_status_inflasi(prediksi_h1: float | None) -> str:
    """
    Tentukan status inflasi berdasarkan prediksi H+1.
    - normal  : prediksi_h1 < BATAS_HARGA_TINGGI
    - waspada : BATAS_HARGA_TINGGI <= prediksi_h1 < BATAS_HARGA_KRITIS
    - kritis  : prediksi_h1 >= BATAS_HARGA_KRITIS
    """
    if prediksi_h1 is None:
        return "normal"
    if prediksi_h1 >= BATAS_HARGA_KRITIS:
        return "kritis"
    elif prediksi_h1 >= BATAS_HARGA_TINGGI:
        return "waspada"
    else:
        return "normal"


# ---------------------------------------------------------------------------
# ENDPOINT 1: GET / — Ringkasan dashboard
# ---------------------------------------------------------------------------

@router.get("/", response_model=DashboardResponse, summary="Ringkasan dashboard harga cabai")
async def get_dashboard():
    """
    Mengembalikan ringkasan data untuk dashboard utama:
    - Statistik 30 hari terakhir (min, max, rata-rata harga).
    - Harga cabai hari ini (baris terbaru dataset).
    - Deteksi tren (naik / turun / stabil).
    - Prediksi h1, h3, h7 berdasarkan baris terakhir dataset.
    - Status inflasi berdasarkan prediksi H+1 (normal / waspada / kritis).
    - Status model dan jumlah model yang aktif.
    """
    try:
        df = _load_dataset()

        if df.empty or TARGET not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Dataset kosong atau kolom '{TARGET}' tidak ditemukan.",
            )

        # ---- Statistik 30 hari terakhir (Merah) ----
        df_30 = df.tail(30)
        harga_min  = float(df_30[TARGET].min())
        harga_max  = float(df_30[TARGET].max())
        harga_rata = float(df_30[TARGET].mean())

        # ---- Statistik 30 hari terakhir (Rawit) ----
        harga_min_rawit = float(df_30["harga_cabai_rawit"].min()) if "harga_cabai_rawit" in df_30.columns else None
        harga_max_rawit = float(df_30["harga_cabai_rawit"].max()) if "harga_cabai_rawit" in df_30.columns else None
        harga_rata_rawit = float(df_30["harga_cabai_rawit"].mean()) if "harga_cabai_rawit" in df_30.columns else None

        # ---- Data hari ini (baris terakhir) ----
        baris_terakhir = df.iloc[-1].copy()
        harga_hari_ini = float(baris_terakhir[TARGET])
        harga_hari_ini_rawit = float(baris_terakhir["harga_cabai_rawit"]) if "harga_cabai_rawit" in baris_terakhir else None
        
        tanggal_update = (
            baris_terakhir["tanggal"].strftime("%Y-%m-%d")
            if hasattr(baris_terakhir["tanggal"], "strftime")
            else str(baris_terakhir["tanggal"])
        )

        # ---- Deteksi tren ----
        tren = _detect_tren(df, TARGET)
        tren_rawit = _detect_tren(df, "harga_cabai_rawit")

        # ---- Prediksi semua horizon menggunakan fitur terkini standar ----
        # Hal ini menjamin nilai prediksi sama persis dengan halaman /prediksi
        input_data_prediksi = predictor.get_fitur_terkini()
        prediksi_h1 = await _predict_for_input("h1", input_data_prediksi)
        prediksi_h3 = await _predict_for_input("h3", input_data_prediksi)
        prediksi_h7 = await _predict_for_input("h7", input_data_prediksi)
        
        prediksi_rawit_h1 = await _predict_for_input("rawit_h1", input_data_prediksi)
        prediksi_rawit_h3 = await _predict_for_input("rawit_h3", input_data_prediksi)
        prediksi_rawit_h7 = await _predict_for_input("rawit_h7", input_data_prediksi)

        # ---- Status inflasi berdasarkan prediksi H+1 ----
        status_inflasi = _determine_status_inflasi(prediksi_h1)
        status_inflasi_rawit = _determine_status_inflasi(prediksi_rawit_h1)

        # ---- Status model ----
        info = predictor.get_cache_info()
        models_loaded = info.get("models_loaded", [])
        status_model  = len(models_loaded) > 0
        n_model_aktif = len(models_loaded)

        # ---- Info real-time ----
        realtime_info = _get_realtime_info()

        return DashboardResponse(
            tanggal_update=tanggal_update,
            
            # Cabai Merah
            harga_hari_ini=round(harga_hari_ini, 2),
            harga_min_30hari=round(harga_min, 2),
            harga_max_30hari=round(harga_max, 2),
            harga_rata_30hari=round(harga_rata, 2),
            tren=tren,
            prediksi_h1=prediksi_h1,
            prediksi_h3=prediksi_h3,
            prediksi_h7=prediksi_h7,
            status_inflasi=status_inflasi,
            
            # Cabai Rawit
            harga_hari_ini_rawit=round(harga_hari_ini_rawit, 2) if harga_hari_ini_rawit is not None else None,
            harga_min_30hari_rawit=round(harga_min_rawit, 2) if harga_min_rawit is not None else None,
            harga_max_30hari_rawit=round(harga_max_rawit, 2) if harga_max_rawit is not None else None,
            harga_rata_30hari_rawit=round(harga_rata_rawit, 2) if harga_rata_rawit is not None else None,
            tren_rawit=tren_rawit,
            prediksi_rawit_h1=prediksi_rawit_h1,
            prediksi_rawit_h3=prediksi_rawit_h3,
            prediksi_rawit_h7=prediksi_rawit_h7,
            status_inflasi_rawit=status_inflasi_rawit,
            
            # Global
            status_model=status_model,
            n_model_aktif=n_model_aktif,
            realtime=realtime_info,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat memuat data dashboard: {str(e)}",
        )


# ---------------------------------------------------------------------------
# ENDPOINT 2: GET /statistik-bulanan — Rata-rata harga per bulan
# ---------------------------------------------------------------------------

@router.get("/statistik-bulanan", response_model=List[StatistikBulananItem], summary="Statistik harga cabai per bulan")
def statistik_bulanan(
    tahun: int = Query(
        default=datetime.now().year,
        description="Tahun yang ingin ditampilkan statistiknya (contoh: 2024)",
        ge=2022,
        le=2030,
    )
):
    """
    Kembalikan rata-rata, minimum, dan maksimum harga cabai per bulan
    untuk tahun yang dipilih. Berguna untuk grafik bar bulanan di React.

    Query param:
    - **tahun**: Tahun target (default: tahun sekarang)

    Return: List statistik per bulan (hanya bulan yang memiliki data).
    """
    try:
        df = _load_dataset()

        if TARGET not in df.columns:
            raise HTTPException(
                status_code=500,
                detail=f"Kolom '{TARGET}' tidak ditemukan di dataset.",
            )

        # Filter berdasarkan tahun
        df["tahun"]  = df["tanggal"].dt.year
        df["bulan"]  = df["tanggal"].dt.month
        df_tahun     = df[df["tahun"] == tahun]

        if df_tahun.empty:
            return []

        # Hitung statistik per bulan
        hasil: List[dict] = []
        for bulan_ke in range(1, 13):
            df_bulan = df_tahun[df_tahun["bulan"] == bulan_ke]
            if df_bulan.empty:
                continue  # Lewati bulan yang tidak ada datanya

            hasil.append({
                "bulan"      : bulan_ke,
                "nama_bulan" : NAMA_BULAN[bulan_ke],
                "rata_rata"  : round(float(df_bulan[TARGET].mean()), 2),
                "min"        : round(float(df_bulan[TARGET].min()), 2),
                "max"        : round(float(df_bulan[TARGET].max()), 2),
                "n_hari"     : int(len(df_bulan)),
            })

        return hasil

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat menghitung statistik bulanan: {str(e)}",
        )
