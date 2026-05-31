"""
predictor.py
============
Core business logic prediksi harga cabai — cabai MERAH dan cabai RAWIT.

Perubahan v2:
  - Model memprediksi DELTA (perubahan harga), bukan harga absolut.
  - Inverse transform: prediksi_rp = harga_hari_ini + prediksi_delta
  - Support dual model: h1/h3/h7 (merah) dan rawit_h1/rawit_h3/rawit_h7 (rawit)
  - Folder model terpisah: expanding_window_merah/ dan expanding_window_rawit/

Fungsi utama:
  - load_artifacts()         : Load model, feature_cols (dengan caching)
  - prediksi_harga()         : Prediksi cabai merah untuk horizon tertentu
  - prediksi_rawit()         : Prediksi cabai rawit untuk horizon tertentu
  - prediksi_semua_horizon() : Semua horizon merah
  - prediksi_semua_rawit()   : Semua horizon rawit
  - get_fitur_terkini()      : Ambil baris terakhir dataset
  - get_metrik_model()       : Metrik dari kedua folder
  - get_data_historis()      : Data historis merah + rawit
"""

import json
import joblib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

# ─────────────────────────────────────────────────────────────────────────────
# PATH KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent.parent
MODEL_DIR_MERAH = BASE_DIR / "machine_learning" / "output" / "expanding_window_merah"
MODEL_DIR_RAWIT = BASE_DIR / "machine_learning" / "output" / "expanding_window_rawit"

# Backward compat — untuk kode lama yang masih pakai MODEL_DIR
MODEL_DIR = MODEL_DIR_MERAH

SCALER_DIR  = BASE_DIR / "machine_learning" / "output" / "xgboost_models"
SCALER_PATH = SCALER_DIR / "scaler.pkl"
DATA_DIR    = BASE_DIR / "data" / "processed"

METRICS_DIR_MERAH = MODEL_DIR_MERAH / "metrics"
METRICS_DIR_RAWIT = MODEL_DIR_RAWIT / "metrics"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache global
_CACHE: Dict[str, any] = {
    "dataset"             : None,
    "dataset_last_updated": None,
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — resolusi model dir berdasarkan label
# ─────────────────────────────────────────────────────────────────────────────
def _get_model_dir(label: str) -> Path:
    """Return folder model yang tepat berdasarkan label horizon."""
    return MODEL_DIR_RAWIT if label.startswith("rawit_") else MODEL_DIR_MERAH


def _is_rawit(label: str) -> bool:
    return label.startswith("rawit_")


# =============================================================================
# BAGIAN 1 — LOAD ARTIFACTS
# =============================================================================

def load_scaler():
    if "scaler" in _CACHE:
        return _CACHE["scaler"]
    if not SCALER_PATH.exists():
        logger.warning(f"scaler.pkl tidak ditemukan di {SCALER_PATH}. Model memakai data unscaled.")
        _CACHE["scaler"] = None
        return None
    try:
        scaler = joblib.load(SCALER_PATH)
        _CACHE["scaler"] = scaler
        logger.info(f"Scaler dimuat dari {SCALER_PATH}")
        return scaler
    except Exception as e:
        logger.warning(f"Gagal load scaler: {e}")
        _CACHE["scaler"] = None
        return None


def load_model(label: str):
    """
    Load model XGBoost untuk horizon tertentu.
    label: "h1", "h3", "h7" (merah) atau "rawit_h1", "rawit_h3", "rawit_h7"

    Model baru memprediksi DELTA (perubahan harga), bukan harga absolut.
    Inverse transform dilakukan di prediksi_harga_sync().
    """
    cache_key = f"model_{label}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    model_dir = _get_model_dir(label)
    pattern   = f"model_final_{label}_*.pkl"
    files     = sorted(model_dir.glob(pattern), reverse=True)

    if not files:
        # Fallback ke folder lama (expanding_window/) untuk backward compat merah
        if not _is_rawit(label):
            old_dir   = BASE_DIR / "machine_learning" / "output" / "expanding_window"
            old_files = sorted(old_dir.glob(pattern), reverse=True)
            if old_files:
                logger.warning(f"Model {label} tidak di expanding_window_merah/, fallback ke folder lama.")
                files = old_files
            else:
                raise HTTPException(
                    status_code=503,
                    detail=f"Model {label} tidak tersedia. Jalankan train_merah.py terlebih dahulu."
                )
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Model {label} tidak tersedia. Jalankan train_rawit.py terlebih dahulu."
            )

    model_path = files[0]
    try:
        model = joblib.load(model_path)
        _CACHE[cache_key]                   = model
        _CACHE[f"model_{label}_version"]    = model_path.name
        _CACHE[f"model_{label}_is_delta"]   = True   # semua model baru pakai delta
        logger.info(f"Model {label} dimuat: {model_path.name}")
        return model
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gagal load model {label}: {e}")


def load_feature_cols(label: str) -> List[str]:
    cache_key = f"feature_cols_{label}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    model_dir    = _get_model_dir(label)
    feature_path = model_dir / f"feature_cols_{label}.json"

    if not feature_path.exists():
        # Fallback ke folder lama untuk merah
        if not _is_rawit(label):
            old_path = BASE_DIR / "machine_learning" / "output" / "expanding_window" / f"feature_cols_{label}.json"
            if old_path.exists():
                feature_path = old_path
                logger.warning(f"feature_cols_{label}.json fallback ke folder lama.")
            else:
                raise HTTPException(status_code=503, detail=f"feature_cols_{label}.json tidak ditemukan.")
        else:
            raise HTTPException(status_code=503, detail=f"feature_cols_{label}.json tidak ditemukan.")

    try:
        with open(feature_path) as f:
            feature_cols = json.load(f)
        _CACHE[cache_key] = feature_cols
        logger.info(f"Feature cols {label}: {len(feature_cols)} fitur")
        return feature_cols
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Gagal load feature_cols {label}: {e}")


def load_all_artifacts(label: str) -> Tuple:
    model        = load_model(label)
    scaler       = load_scaler()
    feature_cols = load_feature_cols(label)
    model_version = _CACHE.get(f"model_{label}_version", "unknown")
    return model, scaler, feature_cols, model_version


def preload_artifacts():
    """Preload semua model (merah + rawit) saat startup."""
    logger.info("Preloading ML artifacts (merah + rawit)...")
    try:
        load_scaler()
        for label in ["h1", "h3", "h7", "rawit_h1", "rawit_h3", "rawit_h7"]:
            try:
                load_model(label)
                load_feature_cols(label)
            except HTTPException as e:
                logger.warning(f"Preload {label} skip: {e.detail}")
            except Exception as e:
                logger.warning(f"Preload {label} error: {e}")
        logger.info("Preloading selesai.")
    except Exception as e:
        logger.error(f"Gagal preloading: {e}")


def load_dataset_to_cache() -> bool:
    csv_path = DATA_DIR / "dataset_preprocessed.csv"
    if not csv_path.exists():
        logger.error(f"Dataset tidak ditemukan: {csv_path}")
        return False
    try:
        df = pd.read_csv(csv_path, parse_dates=["tanggal"])
        df = df.sort_values("tanggal").reset_index(drop=True)
        _CACHE["dataset"]              = df
        _CACHE["dataset_last_updated"] = datetime.now()
        logger.info(f"Dataset dimuat ke cache. Baris: {len(df)}")
        return True
    except Exception as e:
        logger.error(f"Gagal load dataset: {e}")
        return False


def get_dataset() -> pd.DataFrame:
    if _CACHE.get("dataset") is None:
        logger.warning("Dataset belum di-cache, fallback load.")
        load_dataset_to_cache()
    df = _CACHE.get("dataset")
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset tidak tersedia.")
    return df


# =============================================================================
# BAGIAN 2 — FUNGSI PREDIKSI (dengan Inverse Transform Delta → Absolut)
# =============================================================================

def prediksi_harga_sync(input_data: dict, label: str = "h1") -> dict:
    """
    Prediksi harga cabai untuk satu horizon.

    Model memprediksi DELTA (perubahan harga):
      prediksi_delta = model.predict(X)
      prediksi_rp    = harga_hari_ini + prediksi_delta   ← inverse transform

    Args:
        label: "h1", "h3", "h7" (merah) atau "rawit_h1", "rawit_h3", "rawit_h7"
    """
    valid_labels = ["h1", "h3", "h7", "rawit_h1", "rawit_h3", "rawit_h7"]
    if label not in valid_labels:
        raise HTTPException(status_code=400, detail=f"Label tidak valid: {label}.")

    try:
        model, scaler, feature_cols, model_version = load_all_artifacts(label)

        # ── Buat DataFrame fitur ─────────────────────────────────────────
        fitur_values  = []
        missing_cols  = []

        for col in feature_cols:
            if col in input_data and input_data[col] is not None:
                try:
                    fitur_values.append(float(input_data[col]))
                except (ValueError, TypeError):
                    fitur_values.append(0.0)
                    missing_cols.append(col)
            else:
                fitur_values.append(0.0)
                missing_cols.append(col)

        if missing_cols:
            logger.warning(f"Fitur missing/invalid (diisi 0): {len(missing_cols)}")

        df_input = pd.DataFrame([fitur_values], columns=feature_cols)

        # ── Prediksi (output = delta harga) ──────────────────────────────
        prediksi_delta = float(model.predict(df_input)[0])

        # ── Inverse Transform: delta → harga absolut ──────────────────────
        # Pilih referensi harga hari ini berdasarkan komoditas
        if _is_rawit(label):
            ref_key = "harga_cabai_rawit"
            hari_ini_ref_key = "harga_rawit_hari_ini"
        else:
            ref_key = "harga_cabai_merah"
            hari_ini_ref_key = "harga_hari_ini"

        # Prioritas: harga_hari_ini dari dataset (lebih akurat) → harga mentah
        harga_hari_ini = float(
            input_data.get(hari_ini_ref_key)
            or input_data.get(ref_key)
            or 0.0
        )

        # Rekonstruksi harga absolut dari log-return
        # target = ln(harga_besok / harga_hari_ini)
        # → prediksi_rp = harga_hari_ini * exp(prediksi_log_return)
        prediksi_rp = harga_hari_ini * np.exp(prediksi_delta)

        # Pastikan prediksi tidak negatif atau tidak wajar
        prediksi_rp = max(prediksi_rp, 1000.0)

        # ── Hitung arah dan perubahan persen ─────────────────────────────
        if harga_hari_ini > 0:
            perubahan_persen = ((prediksi_rp - harga_hari_ini) / harga_hari_ini) * 100
        else:
            perubahan_persen = 0.0

        batas_stabil = 0.25
        if perubahan_persen > batas_stabil:
            arah = "naik"
        elif perubahan_persen < -batas_stabil:
            arah = "turun"
        else:
            arah = "stabil"

        # ── Tanggal prediksi ─────────────────────────────────────────────
        # Label: "h1" → 1, "h3" → 3, "h7" → 7, "rawit_h1" → 1, dst
        horizon_part  = label.split("_")[-1]     # "h1", "h3", "h7"
        horizon_days  = int(horizon_part[1])     # 1, 3, atau 7

        tanggal_base = input_data.get("tanggal", datetime.now().date())
        if isinstance(tanggal_base, str):
            tanggal_base = datetime.fromisoformat(tanggal_base).date()
        elif isinstance(tanggal_base, pd.Timestamp):
            tanggal_base = tanggal_base.date()
        tanggal_prediksi = tanggal_base + timedelta(days=horizon_days)

        komoditas = "cabai_rawit" if _is_rawit(label) else "cabai_merah"

        logger.info(
            f"Prediksi {label} ({komoditas}): "
            f"delta={prediksi_delta:+,.0f} | "
            f"harga_hari_ini={harga_hari_ini:,.0f} | "
            f"prediksi_rp={prediksi_rp:,.0f} | "
            f"arah={arah} ({perubahan_persen:+.2f}%)"
        )

        return {
            "horizon"         : label,
            "komoditas"       : komoditas,
            "prediksi_rp"     : round(prediksi_rp, 2),
            "prediksi_delta"  : round(prediksi_delta, 2),   # informasi tambahan
            "harga_hari_ini"  : round(harga_hari_ini, 2),   # referensi untuk transparansi
            "tanggal_prediksi": str(tanggal_prediksi),
            "model_version"   : model_version,
            "fitur_digunakan" : len(feature_cols),
            "fitur_missing"   : len(missing_cols),
            "arah_prediksi"   : arah,
            "perubahan_persen": round(perubahan_persen, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error prediksi {label}: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal prediksi: {str(e)}")


async def prediksi_harga(input_data: dict, label: str = "h1") -> dict:
    """Wrapper async untuk prediksi_harga_sync."""
    return await run_in_threadpool(prediksi_harga_sync, input_data, label)


async def prediksi_semua_horizon(input_data: dict) -> dict:
    """Prediksi semua horizon cabai MERAH (h1, h3, h7)."""
    hasil = {}
    for label in ["h1", "h3", "h7"]:
        try:
            hasil[label] = await prediksi_harga(input_data, label)
        except Exception as e:
            logger.error(f"Error prediksi {label}: {e}")
            hasil[label] = {"horizon": label, "error": str(e)}
    return hasil


async def prediksi_semua_rawit(input_data: dict) -> dict:
    """Prediksi semua horizon cabai RAWIT (rawit_h1, rawit_h3, rawit_h7)."""
    hasil = {}
    for label in ["rawit_h1", "rawit_h3", "rawit_h7"]:
        try:
            hasil[label] = await prediksi_harga(input_data, label)
        except Exception as e:
            logger.error(f"Error prediksi rawit {label}: {e}")
            hasil[label] = {"horizon": label, "error": str(e)}
    return hasil


# =============================================================================
# BAGIAN 3 — DATA TERKINI
# =============================================================================

def get_fitur_terkini() -> dict:
    """Ambil baris terakhir dataset sebagai input prediksi."""
    try:
        df = get_dataset()
        df_valid = df.dropna(subset=["harga_cabai_merah"])
        if len(df_valid) == 0:
            raise HTTPException(status_code=503, detail="Dataset tidak memiliki data valid.")
        row  = df_valid.iloc[-1]
        data = row.to_dict()
        if "tanggal" in data and pd.notna(data["tanggal"]):
            data["tanggal"] = data["tanggal"].strftime("%Y-%m-%d")
        for key, value in data.items():
            if isinstance(value, float) and np.isnan(value):
                data[key] = 0.0
        logger.info(f"Data terkini: {data.get('tanggal', 'unknown')}")
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error membaca dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal membaca dataset: {e}")


# =============================================================================
# BAGIAN 4 — METRIK MODEL (dari kedua folder)
# =============================================================================

def get_metrik_model() -> dict:
    """
    Ambil metrik evaluasi model dari kedua folder expanding window.
    Mengembalikan metrik untuk h1/h3/h7 (merah) dan rawit_h1/h3/h7 (rawit).
    """

    def _get_col_value(row, names, default=0.0):
        for name in names:
            if name in row.index and pd.notna(row[name]):
                return float(row[name])
        return default

    def _baca_ringkasan_csv(csv_path: Path, prefix: str = "") -> dict:
        """Baca file ringkasan CSV dan normalkan nama horizon."""
        if not csv_path.exists():
            return {}
        try:
            df = pd.read_csv(csv_path)
            metrik = {}
            label_col = next(
                (c for c in ["model", "label", "horizon", "Model", "Label", "Horizon"]
                 if c in df.columns),
                df.columns[0]
            )
            for _, row in df.iterrows():
                raw_label = str(row[label_col]).strip().lower()

                # Normalisasi label
                if "rawit" in raw_label:
                    if "h1" in raw_label:
                        horizon = "rawit_h1"
                    elif "h3" in raw_label:
                        horizon = "rawit_h3"
                    elif "h7" in raw_label:
                        horizon = "rawit_h7"
                    else:
                        horizon = raw_label
                elif "h1" in raw_label:
                    horizon = "h1"
                elif "h3" in raw_label:
                    horizon = "h3"
                elif "h7" in raw_label:
                    horizon = "h7"
                else:
                    horizon = raw_label

                metrik[horizon] = {
                    "MAE"  : _get_col_value(row, ["xgb_MAE_mean", "MAE", "mae"]),
                    "RMSE" : _get_col_value(row, ["xgb_RMSE_mean", "RMSE", "rmse"]),
                    "MAPE" : _get_col_value(row, ["xgb_MAPE_mean", "MAPE_%", "MAPE", "mape"]),
                    "sMAPE": _get_col_value(row, ["xgb_sMAPE_mean", "sMAPE_%", "sMAPE", "smape"]),
                    "R2"   : _get_col_value(row, ["xgb_R2", "R2", "r2", "R_squared"]),
                    "DA"   : _get_col_value(row, ["xgb_DA_mean", "DA_%", "DA", "da"]),
                }
            return metrik
        except Exception as e:
            logger.warning(f"Gagal baca {csv_path}: {e}")
            return {}

    # ── Coba baca CSV ringkasan baru ──────────────────────────────────────────
    merah_csv = METRICS_DIR_MERAH / "ringkasan_expanding_window_merah.csv"
    rawit_csv = METRICS_DIR_RAWIT / "ringkasan_expanding_window_rawit.csv"

    metrik = {}
    metrik.update(_baca_ringkasan_csv(merah_csv))
    metrik.update(_baca_ringkasan_csv(rawit_csv))

    # Fallback ke file CSV lama (expanding_window/) jika belum ada model baru
    if not any(k in metrik for k in ["h1", "h3", "h7"]):
        old_csv_kandidat = [
            BASE_DIR / "machine_learning" / "output" / "expanding_window" / "metrics" / "ringkasan_evaluasi_final.csv",
            MODEL_DIR / "metrics" / "ringkasan_evaluasi_final.csv",
        ]
        for old_csv in old_csv_kandidat:
            if old_csv.exists():
                metrik.update(_baca_ringkasan_csv(old_csv))
                logger.info(f"Metrik merah fallback ke: {old_csv.name}")
                break

    # Hardcoded fallback jika sama sekali tidak ada file metrik
    if not metrik:
        logger.warning("Semua file metrik tidak ditemukan. Pakai fallback hardcoded.")
        return {
            "h1"       : {"MAE": 1699.0, "RMSE": 2706.0, "MAPE": 3.27, "sMAPE": 3.26, "R2": 0.9813, "DA": 38.9},
            "h3"       : {"MAE": 3678.0, "RMSE": 5162.0, "MAPE": 7.15, "sMAPE": 7.08, "R2": 0.9315, "DA": 23.2},
            "h7"       : {"MAE": 6947.0, "RMSE": 9877.0, "MAPE": 12.95,"sMAPE": 12.91,"R2": 0.7453, "DA": 36.6},
            "rawit_h1" : {"MAE": 0.0,    "RMSE": 0.0,    "MAPE": 0.0,  "sMAPE": 0.0,  "R2": 0.0,   "DA": 0.0},
            "rawit_h3" : {"MAE": 0.0,    "RMSE": 0.0,    "MAPE": 0.0,  "sMAPE": 0.0,  "R2": 0.0,   "DA": 0.0},
            "rawit_h7" : {"MAE": 0.0,    "RMSE": 0.0,    "MAPE": 0.0,  "sMAPE": 0.0,  "R2": 0.0,   "DA": 0.0},
            "source"   : "hardcoded_fallback",
        }

    metrik["source"] = "csv"
    return metrik


# =============================================================================
# BAGIAN 5 — DATA HISTORIS (merah + rawit)
# =============================================================================

def get_data_historis(n_hari: int = 90) -> List[dict]:
    """
    Ambil data historis harga cabai merah dan rawit untuk grafik.

    Returns:
        List of dict: [{"tanggal": ..., "harga_cabai_merah": ..., "harga_cabai_rawit": ...}]
    """
    try:
        df       = get_dataset()
        df_recent = df.tail(n_hari).copy()

        # Kolom yang diambil
        cols_ambil = ["tanggal", "harga_cabai_merah"]
        if "harga_cabai_rawit" in df.columns:
            cols_ambil.append("harga_cabai_rawit")

        df_recent = df_recent[cols_ambil].dropna(subset=["harga_cabai_merah"])

        data = []
        for _, row in df_recent.iterrows():
            entry = {
                "tanggal"         : row["tanggal"].strftime("%Y-%m-%d"),
                "harga_cabai_merah": float(row["harga_cabai_merah"]),
            }
            if "harga_cabai_rawit" in row and pd.notna(row["harga_cabai_rawit"]):
                entry["harga_cabai_rawit"] = float(row["harga_cabai_rawit"])
            data.append(entry)

        logger.info(f"Data historis: {len(data)} hari (merah + rawit)")
        return data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error data historis: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal data historis: {e}")


# =============================================================================
# BAGIAN 6 — UTILITIES
# =============================================================================

def clear_cache():
    global _CACHE
    _CACHE.clear()
    logger.info("Cache artifacts dihapus.")


def get_cache_info() -> dict:
    return {
        "cached_items": list(_CACHE.keys()),
        "cache_size"  : len(_CACHE),
        "scaler_loaded": "scaler" in _CACHE,
        "models_merah" : [k for k in _CACHE if k.startswith("model_") and "rawit" not in k],
        "models_rawit" : [k for k in _CACHE if k.startswith("model_rawit_")],
    }


def validate_horizon(label: str) -> bool:
    return label in ["h1", "h3", "h7", "rawit_h1", "rawit_h3", "rawit_h7"]