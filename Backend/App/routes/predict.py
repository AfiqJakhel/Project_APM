# app/routes/predict.py
# Router FastAPI untuk endpoint prediksi harga cabai menggunakan model XGBoost.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Path as PathParam

from config.settings import (
    DATA_DIR,
    TARGET,
    BATAS_HARGA_TINGGI,
    BATAS_HARGA_KRITIS,
    HORIZON_LABEL,
    MODEL_DIR,
)
from app.core import predictor
from app.schemas.predict import (
    PrediksiRequest, PrediksiResponse, PrediksiSemuaResponse,
    TanggalTersediaResponse, FiturTerkiniResponse, DataHistorisResponse,
    ModelMetrikResponse, CacheInfoResponse, PrediksiOtomatisResponse
)

router = APIRouter(tags=["Prediksi Harga Cabai"])

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _load_dataset() -> pd.DataFrame:
    """Baca dataset dari cache via predictor."""
    return predictor.get_dataset()


def _get_row_nearest(df: pd.DataFrame, target_date: date) -> pd.Series:
    """
    Cari baris dengan tanggal paling dekat dengan target_date.
    Menggunakan selisih hari absolut.
    """
    target_ts = pd.Timestamp(target_date)
    idx = (df["tanggal"] - target_ts).abs().idxmin()
    return df.loc[idx]


def _determine_status(harga: float) -> tuple[str, str]:
    """
    Tentukan status dan pesan berdasarkan threshold dari settings.py.
    Return: (status, pesan)
    """
    if harga >= BATAS_HARGA_KRITIS:
        status = "kritis"
        pesan = (
            f"Harga cabai diprediksi mencapai Rp {harga:,.0f} — "
            f"KRITIS (di atas Rp {BATAS_HARGA_KRITIS:,}). "
            "Perlu penanganan segera untuk mencegah inflasi."
        )
    elif harga >= BATAS_HARGA_TINGGI:
        status = "tinggi"
        pesan = (
            f"Harga cabai diprediksi Rp {harga:,.0f} — "
            f"TINGGI (di atas Rp {BATAS_HARGA_TINGGI:,}). "
            "Waspadai potensi kenaikan lebih lanjut."
        )
    else:
        status = "normal"
        pesan = (
            f"Harga cabai diprediksi Rp {harga:,.0f} — "
            "NORMAL. Harga masih dalam batas wajar."
        )
    return status, pesan

# ---------------------------------------------------------------------------
# ENDPOINT 1: POST / — Prediksi satu horizon
# ---------------------------------------------------------------------------

@router.post("/", response_model=PrediksiResponse, summary="Prediksi satu horizon")
async def prediksi_harga_endpoint(req: PrediksiRequest):
    """
    Prediksi harga cabai untuk satu horizon waktu (h1, h3, atau h7).

    - **tanggal**: Tanggal referensi prediksi (format YYYY-MM-DD).
    - **horizon**: Horizon waktu — `h1` (besok), `h3` (3 hari), `h7` (7 hari).
    - **suhu_rata / curah_hujan / kelembaban**: Override nilai cuaca (opsional).
    """
    try:
        # 1. Baca dataset dan cari baris terdekat
        df = _load_dataset()
        row = _get_row_nearest(df, req.tanggal).copy()

        # 2. Override kolom cuaca jika user menyediakan nilai
        if req.suhu_rata is not None and "suhu_rata" in row.index:
            row["suhu_rata"] = req.suhu_rata
        if req.curah_hujan is not None and "curah_hujan" in row.index:
            row["curah_hujan"] = req.curah_hujan
        if req.kelembaban is not None and "kelembaban" in row.index:
            row["kelembaban"] = req.kelembaban

        # 3. Convert row ke dict untuk predictor
        input_data = row.to_dict()
        input_data["tanggal"] = req.tanggal

        # 4. Prediksi menggunakan core predictor
        hasil = await predictor.prediksi_harga(input_data, req.horizon)
        harga = hasil["prediksi_rp"]

        # 5. Tentukan status
        status, pesan = _determine_status(harga)

        return PrediksiResponse(
            tanggal_input=str(req.tanggal),
            horizon=req.horizon,
            label_horizon=HORIZON_LABEL[req.horizon],
            prediksi_harga=round(harga, 2),
            status_harga=status,
            pesan=pesan,
            arah_prediksi=hasil.get("arah_prediksi"),
            confidence_arah=hasil.get("confidence_arah"),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat prediksi: {str(e)}",
        )

# ---------------------------------------------------------------------------
# ENDPOINT 2: GET /semua — Prediksi semua horizon sekaligus
# ---------------------------------------------------------------------------

@router.get("/semua", response_model=PrediksiSemuaResponse, summary="Prediksi semua horizon (h1, h3, h7)")
async def prediksi_semua(
    tanggal: str = Query(
        ...,
        description="Tanggal referensi dalam format YYYY-MM-DD",
        example="2024-07-01",
    )
):
    """
    Prediksi harga cabai untuk **semua** horizon (h1, h3, h7) sekaligus.
    Berguna untuk dashboard React agar cukup melakukan 1 request.

    Return: dict berisi hasil prediksi ketiga horizon.
    """
    try:
        tanggal_date = date.fromisoformat(tanggal)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Format tanggal tidak valid: '{tanggal}'. Gunakan format YYYY-MM-DD.",
        )

    try:
        df = _load_dataset()
        row = _get_row_nearest(df, tanggal_date).copy()
        
        # Convert row ke dict
        input_data = row.to_dict()
        input_data["tanggal"] = tanggal_date

        # Prediksi semua horizon menggunakan predictor
        hasil_prediksi = await predictor.prediksi_semua_horizon(input_data)
        
        # Format hasil untuk response
        hasil = {}
        for horizon, data in hasil_prediksi.items():
            if "error" in data:
                hasil[horizon] = {
                    "horizon": horizon,
                    "error": data["error"],
                }
            else:
                harga = data["prediksi_rp"]
                status, pesan = _determine_status(harga)
                hasil[horizon] = {
                    "tanggal_input": tanggal,
                    "horizon": horizon,
                    "label_horizon": HORIZON_LABEL[horizon],
                    "prediksi_harga": round(harga, 2),
                    "status_harga": status,
                    "pesan": pesan,
                    "model_version": data.get("model_version", "unknown"),
                    "arah_prediksi": data.get("arah_prediksi"),
                    "confidence_arah": data.get("confidence_arah"),
                }

        return {"tanggal": tanggal, "prediksi": hasil}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat prediksi semua horizon: {str(e)}",
        )


# ---------------------------------------------------------------------------
# ENDPOINT 3: GET /tanggal-tersedia — Rentang tanggal yang ada di dataset
# ---------------------------------------------------------------------------

@router.get("/tanggal-tersedia", response_model=TanggalTersediaResponse, summary="Rentang tanggal tersedia di dataset")
def tanggal_tersedia():
    """
    Kembalikan tanggal minimum dan maksimum yang tersedia di dataset.
    Berguna untuk validasi input tanggal di frontend React agar tidak
    meminta prediksi di luar jangkauan data historis.

    Return:
    - **tanggal_min**: Tanggal pertama dataset (YYYY-MM-DD)
    - **tanggal_max**: Tanggal terakhir dataset (YYYY-MM-DD)
    - **total_hari** : Jumlah baris / hari yang tersedia
    """
    try:
        df = _load_dataset()
        tanggal_min = df["tanggal"].min()
        tanggal_max = df["tanggal"].max()
        total_hari  = len(df)

        return {
            "tanggal_min": tanggal_min.strftime("%Y-%m-%d"),
            "tanggal_max": tanggal_max.strftime("%Y-%m-%d"),
            "total_hari" : total_hari,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat membaca rentang tanggal: {str(e)}",
        )


# ---------------------------------------------------------------------------
# ENDPOINT 4: GET /fitur-terkini — Data fitur terkini dari dataset
# ---------------------------------------------------------------------------

@router.get("/fitur-terkini", response_model=FiturTerkiniResponse, summary="Ambil data fitur terkini")
def fitur_terkini():
    """
    Ambil data fitur terkini dari dataset (baris terakhir).
    Berguna untuk pre-fill form prediksi di frontend.

    Return: Dict berisi semua fitur dengan nilai terkini
    """
    try:
        data = predictor.get_fitur_terkini()
        return {
            "status": "success",
            "data": data,
            "tanggal": data.get("tanggal", "unknown"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil fitur terkini: {str(e)}",
        )


# ---------------------------------------------------------------------------
# ENDPOINT 5: GET /metrik — Metrik evaluasi model
# ---------------------------------------------------------------------------

@router.get("/metrik", response_model=ModelMetrikResponse, summary="Metrik evaluasi model")
def metrik_model():
    """
    Ambil metrik evaluasi model (MAE, RMSE, MAPE, sMAPE, R², DA)
    untuk semua horizon (h1, h3, h7).

    Berguna untuk menampilkan akurasi model di dashboard.

    Return: Dict berisi metrik per horizon
    """
    try:
        metrik = predictor.get_metrik_model()
        return {
            "status": "success",
            "metrik": metrik,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil metrik model: {str(e)}",
        )


# ---------------------------------------------------------------------------
# ENDPOINT 6: GET /historis — Data historis untuk grafik
# ---------------------------------------------------------------------------

@router.get("/historis", response_model=DataHistorisResponse, summary="Data historis harga cabai")
def data_historis(
    n_hari: int = Query(
        90,
        description="Jumlah hari terakhir yang diambil",
        ge=1,
        le=365,
    )
):
    """
    Ambil data historis harga cabai untuk grafik dashboard.

    Args:
    - **n_hari**: Jumlah hari terakhir (default: 90, max: 365)

    Return: List of dict berisi tanggal dan harga
    """
    try:
        data = predictor.get_data_historis(n_hari)
        return {
            "status": "success",
            "n_hari": len(data),
            "data": data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil data historis: {str(e)}",
        )


# ---------------------------------------------------------------------------
# ENDPOINT 7: GET /cache-info — Informasi cache artifacts
# ---------------------------------------------------------------------------

@router.get("/cache-info", response_model=CacheInfoResponse, summary="Informasi cache artifacts")
def cache_info():
    """
    Ambil informasi tentang artifacts yang sudah di-cache.
    Berguna untuk debugging dan monitoring.

    Return: Dict berisi informasi cache
    """
    try:
        info = predictor.get_cache_info()
        return {
            "status": "success",
            "cache": info,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil info cache: {str(e)}",
        )


# ===========================================================================
# ENDPOINT BARU SESUAI SPESIFIKASI
# ===========================================================================

# ---------------------------------------------------------------------------
# ENDPOINT 1 — GET /prediksi/{horizon} — Prediksi otomatis (endpoint utama)
# ---------------------------------------------------------------------------

@router.get(
    "/prediksi/{horizon}",
    response_model=PrediksiOtomatisResponse,
    summary="Prediksi harga otomatis (data terkini)",
    response_description="Prediksi harga untuk horizon tertentu"
)
async def prediksi_otomatis(
    horizon: str = PathParam(
        ...,
        description="Horizon prediksi: h1 (besok), h3 (3 hari), h7 (7 hari)",
        regex="^(h1|h3|h7)$"
    )
):
    """
    **Prediksi harga cabai menggunakan data terkini secara otomatis.**
    
    Endpoint ini mengambil data fitur terkini dari dataset dan langsung
    melakukan prediksi tanpa perlu input manual dari user.
    
    **Horizon:**
    - `h1`: Prediksi harga besok (H+1)
    - `h3`: Prediksi harga 3 hari ke depan (H+3)
    - `h7`: Prediksi harga 7 hari ke depan (H+7)
    
    **Response:**
    ```json
    {
        "status": "success",
        "horizon": "h1",
        "keterangan": "Prediksi harga besok",
        "tanggal_prediksi": "2026-05-04",
        "prediksi_rp": 45000.0,
        "model_version": "model_final_h1_20260503.pkl"
    }
    ```
    """
    # Validasi horizon
    if not predictor.validate_horizon(horizon):
        raise HTTPException(
            status_code=422,
            detail=f"Horizon tidak valid: '{horizon}'. Gunakan 'h1', 'h3', atau 'h7'."
        )
    
    try:
        # Ambil data terkini otomatis
        input_data = predictor.get_fitur_terkini()
        
        # Prediksi
        hasil = await predictor.prediksi_harga(input_data, horizon)
        
        # Format response
        return {
            "status": "success",
            "horizon": horizon,
            "keterangan": HORIZON_LABEL.get(horizon, f"Prediksi {horizon}"),
            "tanggal_prediksi": hasil["tanggal_prediksi"],
            "prediksi_rp": hasil["prediksi_rp"],
            "model_version": hasil["model_version"],
            "arah_prediksi": hasil.get("arah_prediksi"),
            "confidence_arah": hasil.get("confidence_arah"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal melakukan prediksi: {str(e)}"
        )


# ---------------------------------------------------------------------------
# ENDPOINT 2 — GET /prediksi — Prediksi semua horizon sekaligus
# ---------------------------------------------------------------------------

@router.get(
    "/prediksi",
    summary="Prediksi semua horizon sekaligus",
    response_description="Array prediksi h1, h3, h7"
)
async def prediksi_semua_otomatis():
    """
    **Prediksi harga untuk semua horizon (h1, h3, h7) sekaligus.**
    
    Berguna untuk dashboard yang ingin menampilkan semua prediksi
    dalam satu request.
    
    **Response:**
    ```json
    {
        "status": "success",
        "tanggal_base": "2026-05-03",
        "prediksi": [
            {
                "horizon": "h1",
                "keterangan": "Prediksi harga besok",
                "tanggal_prediksi": "2026-05-04",
                "prediksi_rp": 45000.0,
                "model_version": "model_final_h1_20260503.pkl"
            },
            {
                "horizon": "h3",
                "keterangan": "Prediksi 3 hari ke depan",
                "tanggal_prediksi": "2026-05-06",
                "prediksi_rp": 46500.0,
                "model_version": "model_final_h3_20260503.pkl"
            },
            {
                "horizon": "h7",
                "keterangan": "Prediksi 7 hari ke depan",
                "tanggal_prediksi": "2026-05-10",
                "prediksi_rp": 48000.0,
                "model_version": "model_final_h7_20260503.pkl"
            }
        ]
    }
    ```
    """
    try:
        # Ambil data terkini
        input_data = predictor.get_fitur_terkini()
        tanggal_base = input_data.get("tanggal", datetime.now().strftime("%Y-%m-%d"))
        
        # Prediksi semua horizon
        hasil_prediksi = await predictor.prediksi_semua_horizon(input_data)
        
        # Format sebagai array
        prediksi_array = []
        for horizon in ["h1", "h3", "h7"]:
            data = hasil_prediksi.get(horizon, {})
            if "error" in data:
                prediksi_array.append({
                    "horizon": horizon,
                    "keterangan": HORIZON_LABEL.get(horizon, f"Prediksi {horizon}"),
                    "error": data["error"]
                })
            else:
                prediksi_array.append({
                    "horizon": horizon,
                    "keterangan": HORIZON_LABEL.get(horizon, f"Prediksi {horizon}"),
                    "tanggal_prediksi": data["tanggal_prediksi"],
                    "prediksi_rp": data["prediksi_rp"],
                    "model_version": data["model_version"],
                    "arah_prediksi": data.get("arah_prediksi"),
                    "confidence_arah": data.get("confidence_arah"),
                })
        
        return {
            "status": "success",
            "tanggal_base": tanggal_base,
            "prediksi": prediksi_array
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal melakukan prediksi: {str(e)}"
        )


# ---------------------------------------------------------------------------
# ENDPOINT 3 — GET /prediksi/arah/{horizon} — Prediksi khusus arah pergerakan
# ---------------------------------------------------------------------------

@router.get(
    "/prediksi/arah/{horizon}",
    summary="Prediksi arah pergerakan harga",
    response_description="Arah prediksi (naik/turun/stabil) dan confidence"
)
async def prediksi_khusus_arah(
    horizon: str = PathParam(
        ...,
        description="Horizon prediksi: h1 (besok), h3 (3 hari), h7 (7 hari)",
        regex="^(h1|h3|h7)$"
    )
):
    """
    **Prediksi HANYA arah pergerakan harga cabai.**
    
    Endpoint ini khusus digunakan oleh frontend jika hanya ingin
    mengetahui apakah harga akan naik, turun, atau stabil, tanpa
    memerlukan prediksi nilai harganya secara spesifik.
    
    **Response:**
    ```json
    {
        "status": "success",
        "horizon": "h1",
        "arah_prediksi": "naik",
        "confidence_arah": 85.5
    }
    ```
    """
    if not predictor.validate_horizon(horizon):
        raise HTTPException(
            status_code=422,
            detail=f"Horizon tidak valid: '{horizon}'. Gunakan 'h1', 'h3', atau 'h7'."
        )
    
    try:
        # Ambil data terkini otomatis
        input_data = predictor.get_fitur_terkini()
        
        # Kita panggil prediksi_harga karena di dalamnya sudah terintegrasi
        # pemanggilan prediksi_arah dan ekstraksi fiturnya sama.
        hasil = await predictor.prediksi_harga(input_data, horizon)
        
        return {
            "status": "success",
            "horizon": horizon,
            "arah_prediksi": hasil.get("arah_prediksi"),
            "confidence_arah": hasil.get("confidence_arah"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal memprediksi arah: {str(e)}"
        )


# ---------------------------------------------------------------------------
# ENDPOINT 4 — GET /harga/historis — Data historis untuk grafik
# ---------------------------------------------------------------------------

@router.get(
    "/harga/historis",
    response_model=DataHistorisResponse,
    summary="Data historis harga cabai untuk grafik",
    response_description="List data historis"
)
def harga_historis(
    n_hari: int = Query(
        90,
        description="Jumlah hari terakhir yang diambil",
        ge=1,
        le=365,
        example=90
    )
):
    """
    **Ambil data historis harga cabai untuk grafik dashboard.**
    
    Endpoint ini mengembalikan data historis harga cabai merah
    dalam format yang siap digunakan untuk Chart.js atau Recharts.
    
    **Query Parameters:**
    - `n_hari`: Jumlah hari terakhir (default: 90, max: 365)
    
    **Response:**
    ```json
    {
        "status": "success",
        "n_hari": 90,
        "data": [
            {
                "tanggal": "2026-02-03",
                "harga_cabai_merah": 42000.0
            },
            {
                "tanggal": "2026-02-04",
                "harga_cabai_merah": 43500.0
            },
            ...
        ]
    }
    ```
    """
    try:
        data = predictor.get_data_historis(n_hari)
        return {
            "status": "success",
            "n_hari": len(data),
            "data": data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil data historis: {str(e)}",
        )


# ---------------------------------------------------------------------------
# ENDPOINT 4 — GET /model/metrik — Metrik akurasi model
# ---------------------------------------------------------------------------

@router.get(
    "/model/metrik",
    response_model=ModelMetrikResponse,
    summary="Metrik akurasi model",
    response_description="Metrik evaluasi per horizon"
)
def model_metrik():
    """
    **Ambil metrik evaluasi model untuk semua horizon.**
    
    Menampilkan metrik akurasi model seperti MAE, RMSE, MAPE, sMAPE, R², dan DA
    untuk setiap horizon (h1, h3, h7).
    
    Berguna untuk ditampilkan di halaman "Tentang Model" di dashboard.
    
    **Metrik yang tersedia:**
    - **MAE** (Mean Absolute Error): Rata-rata kesalahan absolut dalam Rupiah
    - **RMSE** (Root Mean Squared Error): Akar rata-rata kuadrat kesalahan
    - **MAPE** (Mean Absolute Percentage Error): Persentase kesalahan rata-rata
    - **sMAPE** (Symmetric MAPE): MAPE yang lebih seimbang
    - **R²** (R-squared): Koefisien determinasi (0-1, semakin tinggi semakin baik)
    - **DA** (Directional Accuracy): Akurasi prediksi arah naik/turun (%)
    
    **Response:**
    ```json
    {
        "status": "success",
        "metrik": {
            "h1": {
                "MAE": 2500.0,
                "RMSE": 3200.0,
                "MAPE": 5.2,
                "sMAPE": 5.1,
                "R2": 0.92,
                "DA": 78.5
            },
            "h3": {...},
            "h7": {...}
        }
    }
    ```
    """
    try:
        metrik = predictor.get_metrik_model()
        return {
            "status": "success",
            "metrik": metrik,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil metrik model: {str(e)}",
        )


# ---------------------------------------------------------------------------
# ENDPOINT 5 — GET /health — Health check
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    summary="Health check sistem prediksi",
    response_description="Status kesehatan sistem"
)
def health_check():
    """
    **Cek kesehatan sistem prediksi.**
    
    Memeriksa apakah semua komponen yang diperlukan untuk prediksi
    tersedia dan dapat diakses:
    - Model XGBoost (h1, h3, h7)
    - Scaler (RobustScaler)
    - Dataset (dataset_preprocessed.csv)
    
    **Status:**
    - `healthy`: Semua komponen tersedia
    - `degraded`: Beberapa komponen tidak tersedia
    - `unhealthy`: Komponen kritis tidak tersedia
    
    **Response:**
    ```json
    {
        "status": "healthy",
        "timestamp": "2026-05-03T10:30:00",
        "models": {
            "h1": true,
            "h3": true,
            "h7": true
        },
        "scaler": true,
        "dataset": true,
        "message": "Semua komponen tersedia"
    }
    ```
    """
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "models": {},
        "scaler": False,
        "dataset": False,
    }
    
    # Cek models
    for horizon in ["h1", "h3", "h7"]:
        try:
            predictor.load_model(horizon)
            health_status["models"][horizon] = True
        except Exception:
            health_status["models"][horizon] = False
    
    # Cek scaler
    try:
        predictor.load_scaler()
        health_status["scaler"] = True
    except Exception:
        health_status["scaler"] = False
    
    # Cek dataset
    try:
        predictor.get_fitur_terkini()
        health_status["dataset"] = True
    except Exception:
        health_status["dataset"] = False
    
    # Tentukan status keseluruhan
    models_ok = all(health_status["models"].values())
    scaler_ok = health_status["scaler"]
    dataset_ok = health_status["dataset"]
    
    if models_ok and scaler_ok and dataset_ok:
        health_status["status"] = "healthy"
        health_status["message"] = "Semua komponen tersedia"
    elif scaler_ok and dataset_ok and any(health_status["models"].values()):
        health_status["status"] = "degraded"
        health_status["message"] = "Beberapa model tidak tersedia"
    else:
        health_status["status"] = "unhealthy"
        health_status["message"] = "Komponen kritis tidak tersedia"
        raise HTTPException(
            status_code=503,
            detail=health_status
        )
    
    return health_status


# ---------------------------------------------------------------------------
# ENDPOINT 6 — GET /model/info — Info model
# ---------------------------------------------------------------------------

@router.get(
    "/model/info",
    summary="Informasi detail model",
    response_description="Info model per horizon"
)
def model_info():
    """
    **Ambil informasi detail tentang model yang digunakan.**
    
    Menampilkan informasi seperti:
    - Versi file model (.pkl)
    - Tanggal training (dari nama file)
    - Jumlah fitur yang digunakan
    - Jumlah data training (estimasi dari dataset)
    
    **Response:**
    ```json
    {
        "status": "success",
        "info": {
            "h1": {
                "model_file": "model_final_h1_20260503.pkl",
                "tanggal_training": "2026-05-03",
                "jumlah_fitur": 42,
                "feature_cols_file": "feature_cols_h1.json"
            },
            "h3": {...},
            "h7": {...}
        },
        "dataset_info": {
            "total_baris": 1250,
            "tanggal_min": "2022-01-01",
            "tanggal_max": "2026-05-03"
        }
    }
    ```
    """
    try:
        info_models = {}
        
        # Info per horizon
        for horizon in ["h1", "h3", "h7"]:
            try:
                # Load artifacts untuk mendapatkan info
                predictor.load_model(horizon)
                feature_cols = predictor.load_feature_cols(horizon)
                
                # Ambil nama file model dari cache
                cache_info = predictor.get_cache_info()
                model_version = None
                for key in cache_info.get("cached_items", []):
                    if key == f"model_{horizon}_version":
                        model_version = predictor._CACHE.get(key, "unknown")
                        break
                
                # Extract tanggal dari nama file (format: model_final_h1_YYYYMMDD.pkl)
                tanggal_training = "unknown"
                if model_version and "_" in model_version:
                    parts = model_version.replace(".pkl", "").split("_")
                    if len(parts) >= 4 and parts[-1].isdigit():
                        date_str = parts[-1]
                        if len(date_str) == 8:
                            tanggal_training = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                
                info_models[horizon] = {
                    "model_file": model_version or "unknown",
                    "tanggal_training": tanggal_training,
                    "jumlah_fitur": len(feature_cols),
                    "feature_cols_file": f"feature_cols_{horizon}.json",
                }
            except Exception as e:
                info_models[horizon] = {
                    "error": str(e)
                }
        
        # Info dataset
        dataset_info = {}
        try:
            data_terkini = predictor.get_fitur_terkini()
            csv_path = DATA_DIR / "dataset_preprocessed.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=["tanggal"])
                dataset_info = {
                    "total_baris": len(df),
                    "tanggal_min": df["tanggal"].min().strftime("%Y-%m-%d"),
                    "tanggal_max": df["tanggal"].max().strftime("%Y-%m-%d"),
                }
        except Exception as e:
            dataset_info = {"error": str(e)}
        
        return {
            "status": "success",
            "info": info_models,
            "dataset_info": dataset_info,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil info model: {str(e)}"
        )


# ===========================================================================
# ENDPOINT LEGACY (TETAP DIPERTAHANKAN UNTUK BACKWARD COMPATIBILITY)
# ===========================================================================


# ---------------------------------------------------------------------------
# LEGACY: POST / — Prediksi satu horizon (dengan request body)
# ---------------------------------------------------------------------------

@router.post("/", response_model=PrediksiResponse, summary="[LEGACY] Prediksi dengan request body")
def prediksi_harga_endpoint(req: PrediksiRequest):
    """
    **[LEGACY] Prediksi harga cabai dengan request body.**
    
    Endpoint ini dipertahankan untuk backward compatibility.
    Untuk penggunaan baru, gunakan `GET /prediksi/{horizon}` sebagai gantinya.

    - **tanggal**: Tanggal referensi prediksi (format YYYY-MM-DD).
    - **horizon**: Horizon waktu — `h1` (besok), `h3` (3 hari), `h7` (7 hari).
    - **suhu_rata / curah_hujan / kelembaban**: Override nilai cuaca (opsional).
    """
    try:
        # 1. Baca dataset dan cari baris terdekat
        df = _load_dataset()
        row = _get_row_nearest(df, req.tanggal).copy()

        # 2. Override kolom cuaca jika user menyediakan nilai
        if req.suhu_rata is not None and "suhu_rata" in row.index:
            row["suhu_rata"] = req.suhu_rata
        if req.curah_hujan is not None and "curah_hujan" in row.index:
            row["curah_hujan"] = req.curah_hujan
        if req.kelembaban is not None and "kelembaban" in row.index:
            row["kelembaban"] = req.kelembaban

        # 3. Convert row ke dict untuk predictor
        input_data = row.to_dict()
        input_data["tanggal"] = req.tanggal

        # 4. Prediksi menggunakan core predictor
        hasil = predictor.prediksi_harga(input_data, req.horizon)
        harga = hasil["prediksi_rp"]

        # 5. Tentukan status
        status, pesan = _determine_status(harga)

        return PrediksiResponse(
            tanggal_input=str(req.tanggal),
            horizon=req.horizon,
            label_horizon=HORIZON_LABEL[req.horizon],
            prediksi_harga=round(harga, 2),
            status_harga=status,
            pesan=pesan,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat prediksi: {str(e)}",
        )


# ---------------------------------------------------------------------------
# LEGACY: GET /semua — Prediksi semua horizon dengan tanggal manual
# ---------------------------------------------------------------------------

@router.get("/semua", summary="[LEGACY] Prediksi semua horizon dengan tanggal")
def prediksi_semua_dengan_tanggal(
    tanggal: str = Query(
        ...,
        description="Tanggal referensi dalam format YYYY-MM-DD",
        example="2024-07-01",
    )
):
    """
    **[LEGACY] Prediksi harga cabai untuk semua horizon dengan tanggal manual.**
    
    Endpoint ini dipertahankan untuk backward compatibility.
    Untuk penggunaan baru, gunakan `GET /prediksi` sebagai gantinya.

    Return: dict berisi hasil prediksi ketiga horizon.
    """
    try:
        tanggal_date = date.fromisoformat(tanggal)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Format tanggal tidak valid: '{tanggal}'. Gunakan format YYYY-MM-DD.",
        )

    try:
        df = _load_dataset()
        row = _get_row_nearest(df, tanggal_date).copy()
        
        # Convert row ke dict
        input_data = row.to_dict()
        input_data["tanggal"] = tanggal_date

        # Prediksi semua horizon menggunakan predictor
        hasil_prediksi = predictor.prediksi_semua_horizon(input_data)
        
        # Format hasil untuk response
        hasil = {}
        for horizon, data in hasil_prediksi.items():
            if "error" in data:
                hasil[horizon] = {
                    "horizon": horizon,
                    "error": data["error"],
                }
            else:
                harga = data["prediksi_rp"]
                status, pesan = _determine_status(harga)
                hasil[horizon] = {
                    "tanggal_input": tanggal,
                    "horizon": horizon,
                    "label_horizon": HORIZON_LABEL[horizon],
                    "prediksi_harga": round(harga, 2),
                    "status_harga": status,
                    "pesan": pesan,
                    "model_version": data.get("model_version", "unknown"),
                }

        return {"tanggal": tanggal, "prediksi": hasil}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat prediksi semua horizon: {str(e)}",
        )


# ---------------------------------------------------------------------------
# UTILITY: GET /tanggal-tersedia — Rentang tanggal yang ada di dataset
# ---------------------------------------------------------------------------

@router.get("/tanggal-tersedia", summary="Rentang tanggal tersedia di dataset")
def tanggal_tersedia():
    """
    Kembalikan tanggal minimum dan maksimum yang tersedia di dataset.
    Berguna untuk validasi input tanggal di frontend React agar tidak
    meminta prediksi di luar jangkauan data historis.

    Return:
    - **tanggal_min**: Tanggal pertama dataset (YYYY-MM-DD)
    - **tanggal_max**: Tanggal terakhir dataset (YYYY-MM-DD)
    - **total_hari** : Jumlah baris / hari yang tersedia
    """
    try:
        df = _load_dataset()
        tanggal_min = df["tanggal"].min()
        tanggal_max = df["tanggal"].max()
        total_hari  = len(df)

        return {
            "tanggal_min": tanggal_min.strftime("%Y-%m-%d"),
            "tanggal_max": tanggal_max.strftime("%Y-%m-%d"),
            "total_hari" : total_hari,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat membaca rentang tanggal: {str(e)}",
        )


# ---------------------------------------------------------------------------
# UTILITY: GET /fitur-terkini — Data fitur terkini dari dataset
# ---------------------------------------------------------------------------

@router.get("/fitur-terkini", summary="Ambil data fitur terkini")
def fitur_terkini():
    """
    Ambil data fitur terkini dari dataset (baris terakhir).
    Berguna untuk pre-fill form prediksi di frontend.

    Return: Dict berisi semua fitur dengan nilai terkini
    """
    try:
        data = predictor.get_fitur_terkini()
        return {
            "status": "success",
            "data": data,
            "tanggal": data.get("tanggal", "unknown"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil fitur terkini: {str(e)}",
        )


# ---------------------------------------------------------------------------
# UTILITY: GET /cache-info — Informasi cache artifacts
# ---------------------------------------------------------------------------

@router.get("/cache-info", summary="Informasi cache artifacts")
def cache_info():
    """
    Ambil informasi tentang artifacts yang sudah di-cache.
    Berguna untuk debugging dan monitoring.

    Return: Dict berisi informasi cache
    """
    try:
        info = predictor.get_cache_info()
        return {
            "status": "success",
            "cache": info,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil info cache: {str(e)}",
        )


# ---------------------------------------------------------------------------
# UTILITY: POST /clear-cache — Hapus cache artifacts
# ---------------------------------------------------------------------------

@router.post("/clear-cache", summary="Hapus cache artifacts")
def clear_cache():
    """
    Hapus semua cache artifacts (model, scaler, feature_cols).
    Berguna saat model di-update dan perlu reload.

    Return: Status operasi
    """
    try:
        predictor.clear_cache()
        return {
            "status": "success",
            "message": "Cache berhasil dihapus. Artifacts akan di-reload pada request berikutnya.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal menghapus cache: {str(e)}",
        )
