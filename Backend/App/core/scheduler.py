"""
scheduler.py
============
Singleton APScheduler untuk update harian otomatis jam 14.00 WIB.
"""

import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config.settings import SCHEDULER_JAM, SCHEDULER_MENIT

logger = logging.getLogger(__name__)

_scheduler: AsyncIOScheduler | None = None


def get_scheduler() -> AsyncIOScheduler:
    """Return singleton scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone="Asia/Jakarta")
    return _scheduler


def setup_scheduler():
    """
    Setup dan start scheduler. Dipanggil saat FastAPI startup via lifespan.
    Job: update_harian_realtime setiap hari SCHEDULER_JAM:SCHEDULER_MENIT WIB.
    """
    from App.core.scraper import jalankan_update_realtime

    scheduler = get_scheduler()

    scheduler.add_job(
        func              = jalankan_update_realtime,
        trigger           = CronTrigger(
            hour     = SCHEDULER_JAM,
            minute   = SCHEDULER_MENIT,
            timezone = "Asia/Jakarta",
        ),
        id                = "update_harian_realtime",
        name              = "Update Harga & Cuaca Harian",
        replace_existing  = True,
        misfire_grace_time= 3600,   # toleransi 1 jam jika komputer sleep
    )

    if not scheduler.running:
        scheduler.start()
        logger.info(
            f"[Scheduler] Scheduler started — update harian jam "
            f"{SCHEDULER_JAM:02d}:{SCHEDULER_MENIT:02d} WIB"
        )

    return scheduler


def stop_scheduler():
    """Stop scheduler saat FastAPI shutdown."""
    scheduler = get_scheduler()
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("[Scheduler] Scheduler stopped")


def get_next_run_time() -> str | None:
    """Ambil waktu run berikutnya dari scheduler."""
    try:
        scheduler = get_scheduler()
        job = scheduler.get_job("update_harian_realtime")
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
    except Exception:
        pass
    return None
