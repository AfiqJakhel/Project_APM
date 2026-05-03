"""
predictor.py
============
Core business logic untuk prediksi harga cabai menggunakan XGBoost.
Memisahkan logika prediksi dari routing layer (FastAPI).

Fungsi utama:
- load_artifacts(): Load model, scaler, feature_cols (dengan caching)
- prediksi_harga(): Prediksi harga untuk horizon tertentu
- get_fitur_terkini(): Ambil data terkini dari dataset
- get_metrik_model(): Ambil metrik evaluasi model
- get_data_historis(): Ambil data historis untuk grafik
"""

import json
import joblib
import logging
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "machine_learning" / "output" / "expanding_window"
SCALER_DIR = BASE_DIR / "machine_learning" / "output" / "xgboost_models"
SCALER_PATH = SCALER_DIR / "scaler.pkl"
DATA_DIR = BASE_DIR / "data" / "processed"
METRICS_DIR = MODEL_DIR / "metrics"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache untuk artifacts yang sudah di-load
_CACHE: Dict[str, any] = {
    "dataset": None,
    "dataset_last_updated": None
}


# =============================================================================
# BAGIAN 1 — LOAD ARTIFACTS (dengan caching)
# =============================================================================

def load_scaler():
    """
    Load RobustScaler dari scaler.pkl.
    Hasil di-cache agar tidak reload tiap request.
    
    Returns:
        RobustScaler object
        
    Raises:
        HTTPException(503): Jika scaler.pkl tidak ditemukan
    """
    if "scaler" in _CACHE:
        return _CACHE["scaler"]
    
    if not SCALER_PATH.exists():
        logger.error(f"scaler.pkl tidak ditemukan di {SCALER_PATH}")
        raise HTTPException(
            status_code=503,
            detail=f"Scaler tidak tersedia. File tidak ditemukan: {SCALER_PATH}"
        )
    
    try:
        scaler = joblib.load(SCALER_PATH)
        _CACHE["scaler"] = scaler
        logger.info(f"Scaler berhasil dimuat dari {SCALER_PATH}")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Gagal memuat scaler: {str(e)}"
        )


def load_model(label: str):
    """
    Load model XGBoost untuk horizon tertentu.
    Mencari file model_final_{label}_*.pkl terbaru (sort by filename).
    
    Args:
        label: Horizon label ("h1", "h3", atau "h7")
        
    Returns:
        XGBoost model object
        
    Raises:
        HTTPException(503): Jika model tidak ditemukan
    """
    cache_key = f"model_{label}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    
    # Cari file model terbaru dengan pattern model_final_{label}_*.pkl
    pattern = f"model_final_{label}_*.pkl"
    model_files = sorted(MODEL_DIR.glob(pattern), reverse=True)
    
    if not model_files:
        logger.error(f"Model {label} tidak ditemukan di {MODEL_DIR}")
        raise HTTPException(
            status_code=503,
            detail=f"Model {label} tidak tersedia. Pattern: {pattern}"
        )
    
    model_path = model_files[0]  # Ambil yang terbaru
    
    try:
        model = joblib.load(model_path)
        _CACHE[cache_key] = model
        _CACHE[f"model_{label}_version"] = model_path.name
        logger.info(f"Model {label} berhasil dimuat: {model_path.name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {label}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Gagal memuat model {label}: {str(e)}"
        )


def load_feature_cols(label: str) -> List[str]:
    """
    Load daftar nama kolom fitur untuk horizon tertentu.
    
    Args:
        label: Horizon label ("h1", "h3", atau "h7")
        
    Returns:
        List nama kolom fitur
        
    Raises:
        HTTPException(503): Jika feature_cols tidak ditemukan
    """
    cache_key = f"feature_cols_{label}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    
    feature_path = MODEL_DIR / f"feature_cols_{label}.json"
    
    if not feature_path.exists():
        logger.error(f"feature_cols_{label}.json tidak ditemukan di {MODEL_DIR}")
        raise HTTPException(
            status_code=503,
            detail=f"Feature columns {label} tidak tersedia: {feature_path}"
        )
    
    try:
        with open(feature_path, "r") as f:
            feature_cols = json.load(f)
        _CACHE[cache_key] = feature_cols
        logger.info(f"Feature cols {label} berhasil dimuat: {len(feature_cols)} fitur")
        return feature_cols
    except Exception as e:
        logger.error(f"Error loading feature_cols {label}: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Gagal memuat feature_cols {label}: {str(e)}"
        )


def load_all_artifacts(label: str) -> Tuple:
    """
    Load semua artifacts yang diperlukan untuk prediksi.
    
    Args:
        label: Horizon label ("h1", "h3", atau "h7")
        
    Returns:
        Tuple (model, scaler, feature_cols, model_version)
    """
    model = load_model(label)
    scaler = load_scaler()
    feature_cols = load_feature_cols(label)
    model_version = _CACHE.get(f"model_{label}_version", "unknown")
    
    return model, scaler, feature_cols, model_version

def preload_artifacts():
    """
    Memuat seluruh model, scaler, dan feature_cols saat startup
    agar tidak mengganggu request pertama.
    """
    logger.info("Mulai preloading ML artifacts...")
    try:
        load_scaler()
        for label in ["h1", "h3", "h7"]:
            load_model(label)
            load_feature_cols(label)
        logger.info("Preloading ML artifacts berhasil.")
    except Exception as e:
        logger.error(f"Gagal preloading ML artifacts: {e}")

def load_dataset_to_cache():
    """
    Memuat dataset CSV ke dalam memory cache.
    Dipanggil saat startup FastAPI.
    """
    csv_path = DATA_DIR / "dataset_preprocessed.csv"
    if not csv_path.exists():
        logger.error(f"Dataset tidak ditemukan: {csv_path}")
        return False
        
    try:
        df = pd.read_csv(csv_path, parse_dates=["tanggal"])
        df = df.sort_values("tanggal").reset_index(drop=True)
        _CACHE["dataset"] = df
        _CACHE["dataset_last_updated"] = datetime.now()
        logger.info(f"Dataset berhasil dimuat ke cache. Total baris: {len(df)}")
        return True
    except Exception as e:
        logger.error(f"Gagal memuat dataset: {e}")
        return False

def get_dataset() -> pd.DataFrame:
    """
    Mengambil dataset dari cache.
    Jika belum ada di cache, baca dari CSV.
    """
    if _CACHE.get("dataset") is None:
        logger.warning("Dataset belum di-cache, melakukan fallback load.")
        load_dataset_to_cache()
        
    df = _CACHE.get("dataset")
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset tidak tersedia di sistem.")
        
    return df



# =============================================================================
# BAGIAN 2 — FUNGSI PREDIKSI UTAMA
# =============================================================================

def prediksi_harga_sync(input_data: dict, label: str = "h1") -> dict:
    """
    Prediksi harga cabai untuk horizon tertentu.
    
    Args:
        input_data: Dict berisi nilai fitur terkini
        label: Horizon label ("h1", "h3", atau "h7")
        
    Returns:
        Dict berisi:
        - horizon: label horizon
        - prediksi_rp: nilai prediksi dalam Rupiah
        - tanggal_prediksi: tanggal target prediksi
        - model_version: nama file model yang dipakai
        - fitur_digunakan: jumlah fitur yang digunakan
        
    Raises:
        HTTPException(400): Jika input tidak valid
        HTTPException(503): Jika model tidak tersedia
    """
    # Validasi label
    if label not in ["h1", "h3", "h7"]:
        raise HTTPException(
            status_code=400,
            detail=f"Label tidak valid: {label}. Gunakan 'h1', 'h3', atau 'h7'"
        )
    
    try:
        # Load artifacts
        model, scaler, feature_cols, model_version = load_all_artifacts(label)
        
        # PENTING: Ambil hanya fitur yang diperlukan oleh model
        # Jika fitur tidak ada di input_data, isi dengan 0 (fallback aman)
        fitur_values = []
        missing_cols = []
        available_cols = []
        
        for col in feature_cols:
            if col in input_data and input_data[col] is not None:
                # Pastikan nilai numerik
                try:
                    val = float(input_data[col])
                    fitur_values.append(val)
                    available_cols.append(col)
                except (ValueError, TypeError):
                    fitur_values.append(0.0)
                    missing_cols.append(col)
            else:
                fitur_values.append(0.0)
                missing_cols.append(col)
        
        if missing_cols:
            logger.warning(f"Fitur tidak ditemukan/invalid di input (diisi 0): {len(missing_cols)} fitur")
            logger.debug(f"Missing: {missing_cols[:5]}...")
        
        # Buat DataFrame dengan urutan kolom yang benar
        df_input = pd.DataFrame([fitur_values], columns=feature_cols)
        
        # XGBoost dilatih di train.py menggunakan dataset_preprocessed.csv (UNSCALED)
        # sehingga tidak perlu melalui scaler.transform()
        prediksi = model.predict(df_input)
        prediksi_rp = float(prediksi[0])
        
        # Hitung tanggal prediksi
        horizon_days = int(label[1])  # "h1" -> 1, "h3" -> 3, "h7" -> 7
        tanggal_base = input_data.get("tanggal", datetime.now().date())
        if isinstance(tanggal_base, str):
            tanggal_base = datetime.fromisoformat(tanggal_base).date()
        elif isinstance(tanggal_base, pd.Timestamp):
            tanggal_base = tanggal_base.date()
        tanggal_prediksi = tanggal_base + timedelta(days=horizon_days)
        
        logger.info(f"Prediksi {label} berhasil: Rp {prediksi_rp:,.0f}")
        
        return {
            "horizon": label,
            "prediksi_rp": round(prediksi_rp, 2),
            "tanggal_prediksi": str(tanggal_prediksi),
            "model_version": model_version,
            "fitur_digunakan": len(feature_cols),
            "fitur_missing": len(missing_cols),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saat prediksi {label}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Gagal melakukan prediksi: {str(e)}"
        )

async def prediksi_harga(input_data: dict, label: str = "h1") -> dict:
    """
    Wrapper asinkron untuk prediksi_harga_sync agar tidak memblokir event loop.
    """
    return await run_in_threadpool(prediksi_harga_sync, input_data, label)



# =============================================================================
# BAGIAN 3 — FUNGSI AMBIL DATA TERKINI DARI CSV
# =============================================================================

def get_fitur_terkini() -> dict:
    """
    Ambil data fitur terkini dari dataset_preprocessed.csv.
    Mengambil baris terakhir yang tidak NaN.
    
    Returns:
        Dict berisi {nama_kolom: nilai} untuk semua fitur
        
    Raises:
        HTTPException(503): Jika dataset tidak ditemukan
    """
    try:
        df = get_dataset()
        
        # Ambil baris terakhir yang tidak NaN pada kolom penting
        # (harga_cabai_merah harus ada)
        df_valid = df.dropna(subset=["harga_cabai_merah"])
        
        if len(df_valid) == 0:
            raise HTTPException(
                status_code=503,
                detail="Dataset tidak memiliki data valid"
            )
        
        # Ambil baris terakhir
        row = df_valid.iloc[-1]
        
        # Convert ke dict, handle NaN
        data = row.to_dict()
        
        # Convert tanggal ke string
        if "tanggal" in data and pd.notna(data["tanggal"]):
            data["tanggal"] = data["tanggal"].strftime("%Y-%m-%d")
        
        # Replace NaN dengan 0 untuk fitur numerik (agar bisa digunakan untuk prediksi)
        for key, value in data.items():
            if pd.isna(value):
                # Untuk fitur numerik, isi dengan 0
                # Untuk fitur kategorikal/flag, isi dengan 0
                data[key] = 0.0
        
        logger.info(f"Data terkini berhasil diambil: {data.get('tanggal', 'unknown')}")
        return data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error membaca dataset: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Gagal membaca dataset: {str(e)}"
        )


# =============================================================================
# BAGIAN 4 — FUNGSI METRIK MODEL
# =============================================================================

def get_metrik_model() -> dict:
    """
    Ambil metrik evaluasi model dari ringkasan_evaluasi_final.csv.

    BUG FIX v2:
    1. Key "unknown" → karena kolom di CSV bernama "model" (huruf kecil)
       bukan "horizon" atau "label". Ditambahkan fallback ke kolom "model".
    2. MAPE = 0.0 dan sMAPE = 0.0 → karena nama kolom di CSV adalah
       "MAPE_%" dan "sMAPE_%" (dengan suffix %) bukan "MAPE" dan "sMAPE".
    3. DA = 0.0 → karena nama kolom di CSV adalah "DA_%" bukan "DA".

    Semua nama kolom dicek secara fleksibel (case-insensitive + cek suffix %).
    """
    # Coba beberapa lokasi file metrik yang mungkin
    kandidat_path = [
        METRICS_DIR / "ringkasan_evaluasi_final.csv",
        BASE_DIR / "machine_learning" / "output" / "expanding_window" / "metrics" / "ringkasan_evaluasi_final.csv",
        BASE_DIR / "machine_learning" / "reports" / "training" / "ringkasan_evaluasi_final.csv",
        BASE_DIR / "machine_learning" / "reports" / "validation" / "metrics" / "ringkasan_evaluasi_final.csv",
    ]

    metrik_path = None
    for p in kandidat_path:
        if p.exists():
            metrik_path = p
            break

    if metrik_path is None:
        logger.warning("File ringkasan_evaluasi_final.csv tidak ditemukan di semua lokasi")
        # Return metrik hardcoded dari hasil training aktual sebagai fallback
        # Nilai ini diambil dari hasil evaluasi hold-out 20% yang sudah dijalankan
        return {
            "h1": {"MAE": 1699.0,  "RMSE": 2706.0,  "MAPE": 3.27,  "sMAPE": 3.26,  "R2": 0.9813, "DA": 38.9},
            "h3": {"MAE": 3678.0,  "RMSE": 5162.0,  "MAPE": 7.15,  "sMAPE": 7.08,  "R2": 0.9315, "DA": 23.2},
            "h7": {"MAE": 6947.0,  "RMSE": 9877.0,  "MAPE": 12.95, "sMAPE": 12.91, "R2": 0.7453, "DA": 36.6},
            "source": "hardcoded_fallback",
            "warning": "File CSV tidak ditemukan, menggunakan nilai hasil training terakhir"
        }

    try:
        df = pd.read_csv(metrik_path)
        logger.info(f"Metrik dibaca dari: {metrik_path}")
        logger.info(f"Kolom CSV: {list(df.columns)}")

        # ── FIX 1: Deteksi nama kolom label/horizon secara fleksibel ──────────
        # CSV dari train.py mungkin punya kolom "model", "label", atau "horizon"
        label_col = None
        for candidate in ["model", "label", "horizon", "Model", "Label", "Horizon"]:
            if candidate in df.columns:
                label_col = candidate
                break

        if label_col is None:
            logger.error(f"Kolom label tidak ditemukan. Kolom tersedia: {list(df.columns)}")
            label_col = df.columns[0]  # Fallback ke kolom pertama

        # ── FIX 2 & 3: Deteksi nama kolom metrik secara fleksibel ─────────────
        # CSV mungkin punya "MAPE_%" atau "MAPE", "DA_%" atau "DA", dst.
        def get_col_value(row, names, default=0.0):
            """Cari nilai kolom dengan berbagai kemungkinan nama."""
            for name in names:
                if name in row.index and pd.notna(row[name]):
                    return float(row[name])
            return default

        metrik = {}
        for _, row in df.iterrows():
            # Ambil label horizon — normalisasi ke lowercase (H1 → h1)
            raw_label = str(row[label_col]).strip().lower()

            # Normalisasi: "H1", "h1", "model_h1" → "h1"
            if "h1" in raw_label:
                horizon = "h1"
            elif "h3" in raw_label:
                horizon = "h3"
            elif "h7" in raw_label:
                horizon = "h7"
            else:
                horizon = raw_label  # tetap pakai apa adanya jika tidak dikenal

            metrik[horizon] = {
                # MAE: cek berbagai nama kolom yang mungkin
                "MAE"  : get_col_value(row, ["MAE", "mae", "Mean_AE"]),
                "RMSE" : get_col_value(row, ["RMSE", "rmse", "Root_MSE"]),
                # FIX 2: MAPE bisa "MAPE_%", "MAPE_pct", atau "MAPE"
                "MAPE" : get_col_value(row, ["MAPE_%", "MAPE_pct", "MAPE", "mape"]),
                "sMAPE": get_col_value(row, ["sMAPE_%", "sMAPE_pct", "sMAPE", "smape"]),
                "R2"   : get_col_value(row, ["R2", "r2", "R_squared", "R²"]),
                # FIX 3: DA bisa "DA_%", "DA_pct", atau "DA"
                "DA"   : get_col_value(row, ["DA_%", "DA_pct", "DA", "da",
                                              "Directional_Accuracy"]),
            }

            logger.info(
                f"Metrik {horizon}: MAE={metrik[horizon]['MAE']:,.0f} | "
                f"MAPE={metrik[horizon]['MAPE']:.2f}% | "
                f"R2={metrik[horizon]['R2']:.4f} | "
                f"DA={metrik[horizon]['DA']:.1f}%"
            )

        metrik["source"] = str(metrik_path.name)
        logger.info(f"Metrik berhasil dimuat untuk {len(metrik)-1} horizon")
        return metrik

    except Exception as e:
        logger.error(f"Error membaca metrik: {e}")
        # Fallback ke nilai hardcoded jika CSV error
        return {
            "h1": {"MAE": 1699.0,  "RMSE": 2706.0,  "MAPE": 3.27,  "sMAPE": 3.26,  "R2": 0.9813, "DA": 38.9},
            "h3": {"MAE": 3678.0,  "RMSE": 5162.0,  "MAPE": 7.15,  "sMAPE": 7.08,  "R2": 0.9315, "DA": 23.2},
            "h7": {"MAE": 6947.0,  "RMSE": 9877.0,  "MAPE": 12.95, "sMAPE": 12.91, "R2": 0.7453, "DA": 36.6},
            "source": "hardcoded_fallback",
            "error": str(e)
        }


# =============================================================================
# BAGIAN 5 — FUNGSI DATA HISTORIS
# =============================================================================

def get_data_historis(n_hari: int = 90) -> List[dict]:
    """
    Ambil data historis harga cabai untuk grafik dashboard.
    
    Args:
        n_hari: Jumlah hari terakhir yang diambil (default: 90)
        
    Returns:
        List of dict berisi:
        [
            {"tanggal": "2024-01-01", "harga_cabai_merah": 45000},
            ...
        ]
        
    Raises:
        HTTPException(503): Jika dataset tidak ditemukan
    """
    try:
        df = get_dataset()
        
        # Ambil n_hari terakhir
        df_recent = df.tail(n_hari).copy()
        
        # Filter hanya kolom yang diperlukan
        df_recent = df_recent[["tanggal", "harga_cabai_merah"]].dropna()
        
        # Convert ke list of dict
        data = []
        for _, row in df_recent.iterrows():
            data.append({
                "tanggal": row["tanggal"].strftime("%Y-%m-%d"),
                "harga_cabai_merah": float(row["harga_cabai_merah"])
            })
        
        logger.info(f"Data historis berhasil diambil: {len(data)} hari")
        return data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error membaca data historis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Gagal membaca data historis: {str(e)}"
        )


# =============================================================================
# BAGIAN 6 — FUNGSI UTILITAS
# =============================================================================

def clear_cache():
    """
    Hapus semua cache artifacts.
    Berguna saat model di-update dan perlu reload.
    """
    global _CACHE
    _CACHE.clear()
    logger.info("Cache artifacts berhasil dihapus")


def get_cache_info() -> dict:
    """
    Ambil informasi tentang artifacts yang sudah di-cache.
    
    Returns:
        Dict berisi informasi cache
    """
    return {
        "cached_items": list(_CACHE.keys()),
        "cache_size": len(_CACHE),
        "scaler_loaded": "scaler" in _CACHE,
        "models_loaded": [k for k in _CACHE.keys() if k.startswith("model_")],
    }


def validate_horizon(label: str) -> bool:
    """
    Validasi apakah horizon label valid.
    
    Args:
        label: Horizon label
        
    Returns:
        True jika valid, False jika tidak
    """
    return label in ["h1", "h3", "h7"]


# =============================================================================
# BAGIAN 7 — FUNGSI PREDIKSI BATCH (BONUS)
# =============================================================================

async def prediksi_semua_horizon(input_data: dict) -> dict:
    """
    Prediksi harga untuk semua horizon sekaligus (h1, h3, h7).
    Berguna untuk dashboard yang ingin menampilkan semua prediksi.
    
    Args:
        input_data: Dict berisi nilai fitur terkini
        
    Returns:
        Dict berisi prediksi untuk semua horizon:
        {
            "h1": {...},
            "h3": {...},
            "h7": {...}
        }
    """
    hasil = {}
    
    for label in ["h1", "h3", "h7"]:
        try:
            hasil[label] = await prediksi_harga(input_data, label)
        except Exception as e:
            logger.error(f"Error prediksi {label}: {e}")
            hasil[label] = {
                "horizon": label,
                "error": str(e)
            }
    
    return hasil