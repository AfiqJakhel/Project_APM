# app/routes/predict.py
# Router FastAPI — endpoint prediksi harga cabai (XGBoost).
# Versi bersih: hanya satu set endpoint RESTful GET, tanpa duplikasi.

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Path as PathParam

from config.settings import (
    DATA_DIR,
    HORIZON_LABEL,
    REALTIME_STATUS_FILE,
)
from app.core import predictor
from app.schemas.predict import (
    TanggalTersediaResponse,
    FiturTerkiniResponse,
    DataHistorisResponse,
    ModelMetrikResponse,
    CacheInfoResponse,
    PrediksiOtomatisResponse,
    CuacaInfo,
)

router = APIRouter(tags=["Prediksi Harga Cabai"])


# ===========================================================================
# HELPERS (private)
# ===========================================================================

def _get_cuaca_untuk_prediksi() -> CuacaInfo:
    """
    Ambil cuaca terkini untuk digunakan di prediksi.
    Urutan: realtime_status.json (fresh < 6 jam) → rata-rata historis Kota Padang.
    """
    import json
    try:
        if REALTIME_STATUS_FILE.exists():
            with open(REALTIME_STATUS_FILE, "r", encoding="utf-8") as f:
                st = json.load(f)
            waktu_update_str = st.get("waktu_update")
            if waktu_update_str:
                waktu_update = datetime.fromisoformat(waktu_update_str)
                if datetime.now() - waktu_update < timedelta(hours=6):
                    return CuacaInfo(
                        suhu_rata   = st.get("suhu_rata"),
                        kelembaban  = st.get("kelembaban"),
                        curah_hujan = st.get("curah_hujan"),
                        status      = st.get("cuaca_status", "live"),
                    )
    except Exception:
        pass
    return CuacaInfo(
        suhu_rata   = 27.0,
        kelembaban  = 80.0,
        curah_hujan = 5.0,
        status      = "fallback",
    )


def _get_data_status() -> tuple[str, str]:
    """
    Ambil status dan tanggal data terkini dari realtime_status.json.
    Returns: (data_status, tanggal)
    """
    import json
    try:
        if REALTIME_STATUS_FILE.exists():
            with open(REALTIME_STATUS_FILE, "r", encoding="utf-8") as f:
                st = json.load(f)
            return st.get("harga_status", "fallback"), st.get("tanggal_data", "")
    except Exception:
        pass
    return "fallback", ""


def _load_dataset() -> pd.DataFrame:
    """Baca dataset dari cache via predictor."""
    return predictor.get_dataset()


# ===========================================================================
# ENDPOINTS
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /prediksi/{horizon} — Prediksi otomatis satu horizon
# ---------------------------------------------------------------------------

@router.get(
    "/prediksi/{horizon}",
    response_model=PrediksiOtomatisResponse,
    summary="Prediksi harga otomatis (data terkini)",
    response_description="Prediksi harga untuk horizon tertentu",
)
async def prediksi_otomatis(
    horizon: str = PathParam(
        ...,
        description="Horizon prediksi: h1 (besok), h3 (3 hari), h7 (7 hari)",
        regex="^(h1|h3|h7)$",
    )
):
    """
    **Prediksi harga cabai menggunakan data terkini secara otomatis.**

    Mengambil fitur terkini dari dataset dan langsung melakukan prediksi
    tanpa perlu input manual dari user.

    **Horizon:**
    - `h1`: Prediksi harga besok (H+1)
    - `h3`: Prediksi harga 3 hari ke depan (H+3)
    - `h7`: Prediksi harga 7 hari ke depan (H+7)
    """
    if not predictor.validate_horizon(horizon):
        raise HTTPException(
            status_code=422,
            detail=f"Horizon tidak valid: '{horizon}'. Gunakan 'h1', 'h3', atau 'h7'.",
        )

    try:
        input_data            = predictor.get_fitur_terkini()
        cuaca_info            = _get_cuaca_untuk_prediksi()
        data_status, data_tgl = _get_data_status()

        hasil = await predictor.prediksi_harga(input_data, horizon)

        return {
            "status"          : "success",
            "horizon"         : horizon,
            "keterangan"      : HORIZON_LABEL.get(horizon, f"Prediksi {horizon}"),
            "tanggal_prediksi": hasil["tanggal_prediksi"],
            "prediksi_rp"     : hasil["prediksi_rp"],
            "model_version"   : hasil["model_version"],
            "arah_prediksi"   : hasil.get("arah_prediksi"),
            "perubahan_persen": hasil.get("perubahan_persen"),
            "data_status"     : data_status,
            "data_tanggal"    : data_tgl,
            "cuaca_digunakan" : cuaca_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal melakukan prediksi: {str(e)}",
        )


# ---------------------------------------------------------------------------
# GET /prediksi — Prediksi semua horizon sekaligus
# ---------------------------------------------------------------------------

@router.get(
    "/prediksi",
    summary="Prediksi semua horizon sekaligus",
    response_description="Array prediksi h1, h3, h7",
)
async def prediksi_semua_otomatis():
    """
    **Prediksi harga untuk semua horizon (h1, h3, h7) dalam satu request.**

    Berguna untuk dashboard yang ingin menampilkan semua prediksi
    sekaligus — cukup 1 fetch, nol request tambahan saat ganti tab horizon.

    **Response:**
    ```json
    {
        "status": "success",
        "tanggal_base": "2026-05-29",
        "prediksi": [
            {"horizon": "h1", "prediksi_rp": 52488.0, ...},
            {"horizon": "h3", "prediksi_rp": 52661.0, ...},
            {"horizon": "h7", "prediksi_rp": 53893.0, ...}
        ]
    }
    ```
    """
    try:
        input_data   = predictor.get_fitur_terkini()
        tanggal_base = input_data.get("tanggal", datetime.now().strftime("%Y-%m-%d"))

        hasil_prediksi = await predictor.prediksi_semua_horizon(input_data)

        prediksi_array = []
        for horizon in ["h1", "h3", "h7"]:
            data = hasil_prediksi.get(horizon, {})
            if "error" in data:
                prediksi_array.append({
                    "horizon"    : horizon,
                    "keterangan" : HORIZON_LABEL.get(horizon, f"Prediksi {horizon}"),
                    "error"      : data["error"],
                })
            else:
                prediksi_array.append({
                    "horizon"         : horizon,
                    "keterangan"      : HORIZON_LABEL.get(horizon, f"Prediksi {horizon}"),
                    "tanggal_prediksi": data["tanggal_prediksi"],
                    "prediksi_rp"     : data["prediksi_rp"],
                    "model_version"   : data["model_version"],
                    "arah_prediksi"   : data.get("arah_prediksi"),
                    "perubahan_persen": data.get("perubahan_persen"),
                })

        return {
            "status"      : "success",
            "tanggal_base": tanggal_base,
            "prediksi"    : prediksi_array,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal melakukan prediksi semua horizon: {str(e)}",
        )


# ---------------------------------------------------------------------------
# GET /harga/historis — Data historis untuk grafik
# ---------------------------------------------------------------------------

@router.get(
    "/harga/historis",
    response_model=DataHistorisResponse,
    summary="Data historis harga cabai untuk grafik",
    response_description="List data historis harga",
)
def harga_historis(
    n_hari: int = Query(
        90,
        description="Jumlah hari terakhir yang diambil (default: 90, max: 365)",
        ge=1,
        le=365,
        example=90,
    )
):
    """
    **Ambil data historis harga cabai untuk grafik dashboard.**

    Mengembalikan data historis dalam format siap digunakan Chart.js / Recharts.

    - `n_hari`: Jumlah hari terakhir (default: 90, max: 365)
    """
    try:
        data = predictor.get_data_historis(n_hari)
        return {
            "status": "success",
            "n_hari": len(data),
            "data"  : data,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil data historis: {str(e)}",
        )


# ---------------------------------------------------------------------------
# GET /model/metrik — Metrik akurasi model
# ---------------------------------------------------------------------------

@router.get(
    "/model/metrik",
    response_model=ModelMetrikResponse,
    summary="Metrik akurasi model",
    response_description="Metrik evaluasi per horizon (MAE, RMSE, MAPE, R², DA)",
)
def model_metrik():
    """
    **Ambil metrik evaluasi model untuk semua horizon (h1, h3, h7).**

    Metrik yang tersedia: MAE, RMSE, MAPE, sMAPE, R², DA (Directional Accuracy).
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
# GET /health — Health check sistem prediksi
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    summary="Health check sistem prediksi",
    response_description="Status kesehatan sistem (healthy / degraded / unhealthy)",
)
def health_check():
    """
    **Cek kesehatan sistem prediksi.**

    Memeriksa ketersediaan: model XGBoost (h1, h3, h7), scaler, dan dataset.

    **Status:**
    - `healthy`  : Semua komponen tersedia
    - `degraded` : Beberapa model tidak tersedia
    - `unhealthy`: Komponen kritis tidak tersedia (returns 503)
    """
    health_status: dict = {
        "timestamp": datetime.now().isoformat(),
        "models"   : {},
        "scaler"   : False,
        "dataset"  : False,
    }

    for horizon in ["h1", "h3", "h7"]:
        try:
            predictor.load_model(horizon)
            health_status["models"][horizon] = True
        except Exception:
            health_status["models"][horizon] = False

    try:
        predictor.load_scaler()
        health_status["scaler"] = True
    except Exception:
        pass

    try:
        predictor.get_fitur_terkini()
        health_status["dataset"] = True
    except Exception:
        pass

    models_ok  = all(health_status["models"].values())
    scaler_ok  = health_status["scaler"]
    dataset_ok = health_status["dataset"]

    if models_ok and scaler_ok and dataset_ok:
        health_status["status"]  = "healthy"
        health_status["message"] = "Semua komponen tersedia"
    elif scaler_ok and dataset_ok and any(health_status["models"].values()):
        health_status["status"]  = "degraded"
        health_status["message"] = "Beberapa model tidak tersedia"
    else:
        health_status["status"]  = "unhealthy"
        health_status["message"] = "Komponen kritis tidak tersedia"
        raise HTTPException(status_code=503, detail=health_status)

    return health_status


# ---------------------------------------------------------------------------
# GET /model/info — Info detail model
# ---------------------------------------------------------------------------

@router.get(
    "/model/info",
    summary="Informasi detail model",
    response_description="Info model per horizon beserta info dataset",
)
def model_info():
    """
    **Ambil informasi detail tentang model yang digunakan.**

    Menampilkan versi file model (.pkl), tanggal training (dari nama file),
    jumlah fitur, dan info dataset (total baris, rentang tanggal).
    """
    try:
        info_models: dict = {}

        for horizon in ["h1", "h3", "h7"]:
            try:
                predictor.load_model(horizon)
                feature_cols = predictor.load_feature_cols(horizon)

                # Ambil versi model dari cache
                cache_info    = predictor.get_cache_info()
                model_version = None
                for key in cache_info.get("cached_items", []):
                    if key == f"model_{horizon}_version":
                        model_version = predictor._CACHE.get(key, "unknown")
                        break

                # Ekstrak tanggal training dari nama file (model_final_h1_YYYYMMDD.pkl)
                tanggal_training = "unknown"
                if model_version and "_" in model_version:
                    parts = model_version.replace(".pkl", "").split("_")
                    if len(parts) >= 4 and parts[-1].isdigit() and len(parts[-1]) == 8:
                        d = parts[-1]
                        tanggal_training = f"{d[:4]}-{d[4:6]}-{d[6:]}"

                info_models[horizon] = {
                    "model_file"       : model_version or "unknown",
                    "tanggal_training" : tanggal_training,
                    "jumlah_fitur"     : len(feature_cols),
                    "feature_cols_file": f"feature_cols_{horizon}.json",
                }
            except Exception as e:
                info_models[horizon] = {"error": str(e)}

        # Info dataset
        dataset_info: dict = {}
        try:
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
            "status"      : "success",
            "info"        : info_models,
            "dataset_info": dataset_info,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil info model: {str(e)}",
        )


# ---------------------------------------------------------------------------
# GET /tanggal-tersedia — Rentang tanggal yang ada di dataset
# ---------------------------------------------------------------------------

@router.get(
    "/tanggal-tersedia",
    response_model=TanggalTersediaResponse,
    summary="Rentang tanggal tersedia di dataset",
)
def tanggal_tersedia():
    """
    Kembalikan tanggal minimum dan maksimum yang tersedia di dataset.
    Berguna untuk validasi input tanggal di frontend.
    """
    try:
        df = _load_dataset()
        return {
            "tanggal_min": df["tanggal"].min().strftime("%Y-%m-%d"),
            "tanggal_max": df["tanggal"].max().strftime("%Y-%m-%d"),
            "total_hari" : len(df),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal membaca rentang tanggal: {str(e)}",
        )


# ---------------------------------------------------------------------------
# GET /fitur-terkini — Data fitur terkini dari dataset
# ---------------------------------------------------------------------------

@router.get(
    "/fitur-terkini",
    response_model=FiturTerkiniResponse,
    summary="Ambil data fitur terkini",
)
def fitur_terkini():
    """
    Ambil data fitur terkini dari dataset (baris terakhir).
    Berguna untuk pre-fill form prediksi di frontend.
    """
    try:
        data = predictor.get_fitur_terkini()
        return {
            "status" : "success",
            "data"   : data,
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
# GET /cache-info — Informasi cache artifacts (debugging)
# ---------------------------------------------------------------------------

@router.get(
    "/cache-info",
    response_model=CacheInfoResponse,
    summary="Informasi cache artifacts",
)
def cache_info():
    """
    Ambil informasi tentang artifacts yang sudah di-cache.
    Berguna untuk debugging dan monitoring.
    """
    try:
        info = predictor.get_cache_info()
        return {
            "status": "success",
            "cache" : info,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil info cache: {str(e)}",
        )


# ---------------------------------------------------------------------------
# POST /clear-cache — Hapus cache artifacts
# ---------------------------------------------------------------------------

@router.post(
    "/clear-cache",
    summary="Hapus cache artifacts",
)
def clear_cache():
    """
    Hapus semua cache artifacts (model, scaler, feature_cols).
    Berguna saat model di-update dan perlu reload.
    """
    try:
        predictor.clear_cache()
        return {
            "status" : "success",
            "message": "Cache berhasil dihapus. Artifacts akan di-reload pada request berikutnya.",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal menghapus cache: {str(e)}",
        )
