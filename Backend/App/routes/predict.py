# app/routes/predict.py
# Router FastAPI — endpoint prediksi harga cabai merah dan cabai rawit.

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
    return CuacaInfo(suhu_rata=27.0, kelembaban=80.0, curah_hujan=5.0, status="fallback")


def _get_data_status() -> tuple[str, str]:
    return "statis", "Dataset statis tanpa realtime"


def _load_dataset() -> pd.DataFrame:
    return predictor.get_dataset()


# ===========================================================================
# ENDPOINTS — CABAI RAWIT
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /prediksi/rawit/{horizon} — Prediksi rawit satu horizon
# ---------------------------------------------------------------------------
@router.get(
    "/prediksi/rawit/{horizon}",
    summary="Prediksi harga cabai rawit (otomatis)",
)
async def prediksi_rawit_otomatis(
    horizon: str = PathParam(
        ...,
        description="Horizon: h1 (besok), h3 (3 hari), h7 (7 hari)",
        regex="^(h1|h3|h7)$",
    )
):
    """
    Prediksi harga **cabai rawit** menggunakan data terkini secara otomatis.

    Model terpisah dari merah — memanfaatkan lag harga merah sebagai fitur eksogen.
    """
    label = f"rawit_{horizon}"
    try:
        input_data            = predictor.get_fitur_terkini()
        cuaca_info            = _get_cuaca_untuk_prediksi()
        data_status, data_tgl = _get_data_status()
        hasil                 = await predictor.prediksi_harga(input_data, label)

        return {
            "status"          : "success",
            "horizon"         : horizon,
            "label"           : label,
            "keterangan"      : HORIZON_LABEL.get(label, f"Rawit — Prediksi {horizon}"),
            "komoditas"       : "cabai_rawit",
            "tanggal_prediksi": hasil["tanggal_prediksi"],
            "prediksi_rp"     : hasil["prediksi_rp"],
            "harga_hari_ini"  : hasil.get("harga_hari_ini"),
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
        raise HTTPException(status_code=500, detail=f"Gagal prediksi rawit: {str(e)}")


# ---------------------------------------------------------------------------
# GET /prediksi/rawit — Semua horizon rawit
# ---------------------------------------------------------------------------
@router.get("/prediksi/rawit", summary="Prediksi rawit semua horizon (h1, h3, h7)")
async def prediksi_rawit_semua():
    """Prediksi harga **cabai rawit** untuk semua horizon dalam satu request."""
    try:
        input_data   = predictor.get_fitur_terkini()
        tanggal_base = input_data.get("tanggal", datetime.now().strftime("%Y-%m-%d"))
        hasil_all    = await predictor.prediksi_semua_rawit(input_data)

        prediksi_array = []
        for h in ["h1", "h3", "h7"]:
            label = f"rawit_{h}"
            data  = hasil_all.get(label, {})
            if "error" in data:
                prediksi_array.append({
                    "horizon"    : h,
                    "label"      : label,
                    "keterangan" : HORIZON_LABEL.get(label, f"Rawit — Prediksi {h}"),
                    "error"      : data["error"],
                })
            else:
                prediksi_array.append({
                    "horizon"         : h,
                    "label"           : label,
                    "keterangan"      : HORIZON_LABEL.get(label, f"Rawit — Prediksi {h}"),
                    "tanggal_prediksi": data["tanggal_prediksi"],
                    "prediksi_rp"     : data["prediksi_rp"],
                    "harga_hari_ini"  : data.get("harga_hari_ini"),
                    "model_version"   : data["model_version"],
                    "arah_prediksi"   : data.get("arah_prediksi"),
                    "perubahan_persen": data.get("perubahan_persen"),
                })

        return {"status": "success", "tanggal_base": tanggal_base,
                "komoditas": "cabai_rawit", "prediksi": prediksi_array}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal prediksi rawit semua: {str(e)}")


# ---------------------------------------------------------------------------
# GET /prediksi/semua — Merah + Rawit sekaligus
# ---------------------------------------------------------------------------
@router.get("/prediksi/semua", summary="Prediksi merah DAN rawit semua horizon sekaligus")
async def prediksi_semua_komoditas():
    """
    Mengembalikan prediksi **cabai merah** dan **cabai rawit**
    untuk semua horizon (h1, h3, h7) dalam satu request.

    Cocok untuk dashboard yang menampilkan kedua komoditas sekaligus.
    """
    try:
        input_data   = predictor.get_fitur_terkini()
        tanggal_base = input_data.get("tanggal", datetime.now().strftime("%Y-%m-%d"))

        merah_all = await predictor.prediksi_semua_horizon(input_data)
        rawit_all = await predictor.prediksi_semua_rawit(input_data)

        def _format(data_dict: dict, horizons: list, prefix: str = "") -> list:
            result = []
            for h in horizons:
                label = f"{prefix}{h}" if prefix else h
                data  = data_dict.get(label, {})
                if "error" in data:
                    result.append({"horizon": h, "label": label, "error": data["error"]})
                else:
                    result.append({
                        "horizon"         : h,
                        "label"           : label,
                        "keterangan"      : HORIZON_LABEL.get(label, label),
                        "tanggal_prediksi": data.get("tanggal_prediksi"),
                        "prediksi_rp"     : data.get("prediksi_rp"),
                        "arah_prediksi"   : data.get("arah_prediksi"),
                        "perubahan_persen": data.get("perubahan_persen"),
                        "model_version"   : data.get("model_version"),
                    })
            return result

        return {
            "status"      : "success",
            "tanggal_base": tanggal_base,
            "merah"       : _format(merah_all, ["h1", "h3", "h7"]),
            "rawit"       : _format(rawit_all, ["h1", "h3", "h7"], prefix="rawit_"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal prediksi semua komoditas: {str(e)}")


# ===========================================================================
# ENDPOINTS — CABAI MERAH
# ===========================================================================

# ---------------------------------------------------------------------------
# GET /prediksi/{horizon} — Prediksi merah satu horizon
# ---------------------------------------------------------------------------
@router.get(
    "/prediksi/{horizon}",
    response_model=PrediksiOtomatisResponse,
    summary="Prediksi harga cabai merah (otomatis)",
)
async def prediksi_otomatis(
    horizon: str = PathParam(
        ...,
        description="Horizon: h1 (besok), h3 (3 hari), h7 (7 hari)",
        regex="^(h1|h3|h7)$",
    )
):
    """
    Prediksi harga **cabai merah** menggunakan data terkini secara otomatis.

    Model memprediksi perubahan harga (delta) lalu dikonversi ke harga absolut.
    """
    if not predictor.validate_horizon(horizon):
        raise HTTPException(status_code=422, detail=f"Horizon tidak valid: '{horizon}'.")
    try:
        input_data            = predictor.get_fitur_terkini()
        cuaca_info            = _get_cuaca_untuk_prediksi()
        data_status, data_tgl = _get_data_status()
        hasil                 = await predictor.prediksi_harga(input_data, horizon)

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
        raise HTTPException(status_code=500, detail=f"Gagal prediksi: {str(e)}")


# ---------------------------------------------------------------------------
# GET /prediksi — Semua horizon merah
# ---------------------------------------------------------------------------
@router.get("/prediksi", summary="Prediksi merah semua horizon (h1, h3, h7)")
async def prediksi_semua_otomatis():
    """Prediksi harga **cabai merah** untuk semua horizon dalam satu request."""
    try:
        input_data   = predictor.get_fitur_terkini()
        tanggal_base = input_data.get("tanggal", datetime.now().strftime("%Y-%m-%d"))
        hasil_all    = await predictor.prediksi_semua_horizon(input_data)

        prediksi_array = []
        for h in ["h1", "h3", "h7"]:
            data = hasil_all.get(h, {})
            if "error" in data:
                prediksi_array.append({
                    "horizon"    : h,
                    "keterangan" : HORIZON_LABEL.get(h, f"Prediksi {h}"),
                    "error"      : data["error"],
                })
            else:
                prediksi_array.append({
                    "horizon"         : h,
                    "keterangan"      : HORIZON_LABEL.get(h, f"Prediksi {h}"),
                    "tanggal_prediksi": data["tanggal_prediksi"],
                    "prediksi_rp"     : data["prediksi_rp"],
                    "model_version"   : data["model_version"],
                    "arah_prediksi"   : data.get("arah_prediksi"),
                    "perubahan_persen": data.get("perubahan_persen"),
                })

        return {"status": "success", "tanggal_base": tanggal_base,
                "komoditas": "cabai_merah", "prediksi": prediksi_array}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal prediksi semua: {str(e)}")


# ===========================================================================
# ENDPOINTS — DATA HISTORIS, METRIK, DAN INFO
# ===========================================================================

@router.get(
    "/harga/historis",
    response_model=DataHistorisResponse,
    summary="Data historis harga cabai merah dan rawit",
)
def harga_historis(
    n_hari: int = Query(90, ge=1, le=365, description="Jumlah hari terakhir")
):
    """Ambil data historis harga cabai **merah dan rawit** untuk grafik."""
    try:
        data = predictor.get_data_historis(n_hari)
        return {"status": "success", "n_hari": len(data), "data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal data historis: {str(e)}")


@router.get(
    "/model/metrik",
    response_model=ModelMetrikResponse,
    summary="Metrik akurasi model merah dan rawit",
)
def model_metrik():
    """Metrik evaluasi (MAE, RMSE, MAPE, R², DA) untuk semua model."""
    try:
        metrik = predictor.get_metrik_model()
        return {"status": "success", "metrik": metrik}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal ambil metrik: {str(e)}")


@router.get("/health", summary="Health check sistem prediksi")
def health_check():
    """Cek ketersediaan semua model (merah + rawit), scaler, dan dataset."""
    health: dict = {
        "timestamp": datetime.now().isoformat(),
        "models"   : {},
        "scaler"   : False,
        "dataset"  : False,
    }

    for horizon in ["h1", "h3", "h7", "rawit_h1", "rawit_h3", "rawit_h7"]:
        try:
            predictor.load_model(horizon)
            health["models"][horizon] = True
        except Exception:
            health["models"][horizon] = False

    try:
        predictor.load_scaler()
        health["scaler"] = True
    except Exception:
        pass

    try:
        predictor.get_fitur_terkini()
        health["dataset"] = True
    except Exception:
        pass

    merah_ok   = all(health["models"].get(h, False) for h in ["h1", "h3", "h7"])
    rawit_ok   = all(health["models"].get(h, False) for h in ["rawit_h1", "rawit_h3", "rawit_h7"])
    dataset_ok = health["dataset"]

    if merah_ok and rawit_ok and dataset_ok:
        health["status"]  = "healthy"
        health["message"] = "Semua model merah & rawit tersedia"
    elif merah_ok and dataset_ok:
        health["status"]  = "degraded"
        health["message"] = "Model merah OK, model rawit belum tersedia"
    elif dataset_ok and any(health["models"].values()):
        health["status"]  = "degraded"
        health["message"] = "Sebagian model tidak tersedia"
    else:
        health["status"]  = "unhealthy"
        health["message"] = "Komponen kritis tidak tersedia"
        raise HTTPException(status_code=503, detail=health)

    return health


@router.get("/model/info", summary="Info detail model merah dan rawit")
def model_info():
    """Info file model, jumlah fitur, dan rentang dataset."""
    try:
        info_models: dict = {}
        for horizon in ["h1", "h3", "h7", "rawit_h1", "rawit_h3", "rawit_h7"]:
            try:
                predictor.load_model(horizon)
                feature_cols  = predictor.load_feature_cols(horizon)
                model_version = predictor._CACHE.get(f"model_{horizon}_version", "unknown")

                tanggal_training = "unknown"
                if model_version and "_" in model_version:
                    parts = model_version.replace(".pkl", "").split("_")
                    if parts[-1].isdigit() and len(parts[-1]) == 8:
                        d = parts[-1]
                        tanggal_training = f"{d[:4]}-{d[4:6]}-{d[6:]}"

                info_models[horizon] = {
                    "komoditas"       : "rawit" if horizon.startswith("rawit_") else "merah",
                    "model_file"      : model_version,
                    "tanggal_training": tanggal_training,
                    "jumlah_fitur"    : len(feature_cols),
                    "target_type"     : "delta (perubahan harga)",
                }
            except Exception as e:
                info_models[horizon] = {"error": str(e)}

        dataset_info: dict = {}
        try:
            csv_path = DATA_DIR / "dataset_preprocessed.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, parse_dates=["tanggal"])
                dataset_info = {
                    "total_baris"   : len(df),
                    "tanggal_min"   : df["tanggal"].min().strftime("%Y-%m-%d"),
                    "tanggal_max"   : df["tanggal"].max().strftime("%Y-%m-%d"),
                    "ada_rawit"     : "harga_cabai_rawit" in df.columns,
                    "ada_detrending": "harga_hari_ini" in df.columns,
                }
        except Exception as e:
            dataset_info = {"error": str(e)}

        return {"status": "success", "info": info_models, "dataset_info": dataset_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal ambil info model: {str(e)}")


@router.get(
    "/tanggal-tersedia",
    response_model=TanggalTersediaResponse,
    summary="Rentang tanggal dataset",
)
def tanggal_tersedia():
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
        raise HTTPException(status_code=500, detail=f"Gagal baca rentang tanggal: {str(e)}")


@router.get(
    "/fitur-terkini",
    response_model=FiturTerkiniResponse,
    summary="Data fitur terkini dari dataset",
)
def fitur_terkini():
    try:
        data = predictor.get_fitur_terkini()
        return {"status": "success", "data": data, "tanggal": data.get("tanggal", "unknown")}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal ambil fitur terkini: {str(e)}")


@router.get(
    "/cache-info",
    response_model=CacheInfoResponse,
    summary="Informasi cache artifacts",
)
def cache_info():
    try:
        info = predictor.get_cache_info()
        return {"status": "success", "cache": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal ambil info cache: {str(e)}")


@router.post("/clear-cache", summary="Hapus cache artifacts")
def clear_cache():
    try:
        predictor.clear_cache()
        return {"status": "success", "message": "Cache dihapus. Artifacts akan di-reload pada request berikutnya."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal hapus cache: {str(e)}")
