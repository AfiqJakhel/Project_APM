import json
from fastapi import APIRouter
from app.core.predictor import _CACHE
from config.settings import REALTIME_STATUS_FILE

router = APIRouter(tags=["Cache"])

@router.get("/status", summary="Mengecek status cache memori dan update scraping terakhir")
def cache_status():
    """
    Menampilkan status cache untuk memastikan dataset terbaru digunakan untuk prediksi.
    """
    dataset = _CACHE.get("dataset")
    dataset_last_updated = _CACHE.get("dataset_last_updated")
    
    jumlah_baris = len(dataset) if dataset is not None else 0
    waktu_load = dataset_last_updated.isoformat() if dataset_last_updated else None
    
    scraping_berhasil = None
    try:
        if REALTIME_STATUS_FILE.exists():
            with open(REALTIME_STATUS_FILE, "r", encoding="utf-8") as f:
                st = json.load(f)
                if st.get("harga_status") == "live":
                    scraping_berhasil = st.get("waktu_update")
    except Exception:
        pass

    return {
        "dataset_terakhir_diload": waktu_load,
        "jumlah_baris_di_cache": jumlah_baris,
        "scraping_terakhir_berhasil": scraping_berhasil
    }
