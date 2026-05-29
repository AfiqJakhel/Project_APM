import sys
import os
import asyncio
import logging
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import predictor

logger = logging.getLogger(__name__)

# Global state untuk melacak proses retrain
RETRAIN_STATE = {
    "is_running": False,
    "last_run": None,
    "last_status": "none",
    "last_error": None
}

async def jalankan_retrain():
    """
    Menjalankan machine_learning/train.py sebagai subprocess.
    """
    global RETRAIN_STATE
    
    if RETRAIN_STATE["is_running"]:
        logger.warning("[Retrain] Proses retrain sedang berjalan. Permintaan diabaikan.")
        return {"status": "skipped", "message": "Retrain is already in progress"}
        
    RETRAIN_STATE["is_running"] = True
    RETRAIN_STATE["last_status"] = "running"
    RETRAIN_STATE["last_error"] = None
    
    try:
        logger.info("[Retrain] Memulai proses retrain model XGBoost...")
        
        train_script = ROOT / "machine_learning" / "train.py"
        if not train_script.exists():
            raise FileNotFoundError(f"Script {train_script} tidak ditemukan!")

        # Gunakan executable python dari venv agar tidak salah pakai uvicorn.exe
        python_exe = str(ROOT / "venv" / "Scripts" / "python.exe")

        from starlette.concurrency import run_in_threadpool

        def jalankan_subprocess():
            return subprocess.run(
                [python_exe, str(train_script)],
                capture_output=True,
                cwd=str(ROOT),
                text=True, # Otomatis decode ke string
                encoding="utf-8",
                errors="replace" # Abaikan karakter yang gagal di-decode
            )

        # Jalankan di background thread agar tidak memblokir server
        process = await run_in_threadpool(jalankan_subprocess)
        
        if process.returncode == 0:
            logger.info("[Retrain] Script train.py selesai dengan sukses.")
            
            # 1. Hapus cache lama
            predictor.clear_cache()
            logger.info("[Retrain] Cache lama berhasil dibersihkan.")
            
            # 2. Preload model baru
            predictor.preload_artifacts()
            logger.info("[Retrain] Model baru berhasil di-preload ke cache.")
            
            # 3. Refresh dataset cache
            predictor.load_dataset_to_cache()
            logger.info("[Retrain] Dataset cache berhasil di-refresh.")
            
            RETRAIN_STATE["last_status"] = "success"
            RETRAIN_STATE["last_run"] = datetime.now().isoformat()
            
            logger.info("[Retrain] Retrain berhasil, model baru aktif.")
            return {"status": "success", "message": "Retrain berhasil, model baru aktif."}
        else:
            # Jika gagal
            out_str = process.stdout.strip() if process.stdout else ""
            err_str = process.stderr.strip() if process.stderr else ""
            error_msg = f"STDERR:\n{err_str}\n\nSTDOUT:\n{out_str}".strip()
            if not error_msg:
                error_msg = "Unknown error (No output produced by train.py)"
            
            logger.error(f"[Retrain] Proses train.py gagal dengan kode {process.returncode}:\n{error_msg}")
            
            RETRAIN_STATE["last_status"] = "failed"
            RETRAIN_STATE["last_error"] = error_msg
            RETRAIN_STATE["last_run"] = datetime.now().isoformat()
            
            return {"status": "error", "message": "Proses retrain gagal. Lihat log untuk detail."}
            
    except Exception as e:
        import traceback
        full_error = traceback.format_exc()
        logger.error(f"[Retrain] Terjadi exception saat menjalankan retrain:\n{full_error}")
        RETRAIN_STATE["last_status"] = "failed"
        RETRAIN_STATE["last_error"] = str(e) or "Exception without message (lihat terminal)"
        RETRAIN_STATE["last_run"] = datetime.now().isoformat()
        return {"status": "error", "message": f"Exception: {str(e)}"}
        
    finally:
        # Pastikan state is_running kembali False apa pun yang terjadi
        RETRAIN_STATE["is_running"] = False
