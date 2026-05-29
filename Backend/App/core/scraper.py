"""
scraper.py
==========
Scraper harga cabai dari PIHPS BI (Selenium) + cuaca dari Open-Meteo (httpx).
Semua fungsi dibungkus try/except — sistem tidak pernah crash.
Terintegrasi dengan sinkronisasi fitur 100% menggunakan mapper.
"""

import sys
import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime, date
from io import StringIO

import pandas as pd
import numpy as np
import httpx

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    PIHPS_URL, PIHPS_PROVINSI, PIHPS_KOTA,
    PIHPS_KOMODITAS, PIHPS_KOLOM_MAP,
    CUACA_API_URL, CUACA_LATITUDE, CUACA_LONGITUDE, CUACA_TIMEZONE,
    DATA_DIR, TARGET, REALTIME_STATUS_FILE,
    SCRAPER_TIMEOUT, SCRAPER_HEADLESS,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# 1. DRIVER INITIALIZATION (FIX WINERROR 193)
# ─────────────────────────────────────────────────────────────────────────────

def init_driver():
    """
    Inisialisasi ChromeDriver dengan cache_valid_range=0 dan fallback.
    Mencegah WinError 193 karena cache versi arsitektur yang salah.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.core.driver_cache import DriverCacheManager

    # Pastikan selalu download versi 64-bit untuk OS Windows 64-bit
    os.environ["WDM_ARCH"] = "64"

    options = Options()
    if SCRAPER_HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--log-level=3")

    # Coba gunakan chromedriver dari system PATH terlebih dahulu (lebih cepat)
    try:
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(SCRAPER_TIMEOUT)
        return driver
    except Exception as e_path:
        logger.info(f"[Scraper] ChromeDriver system PATH tidak ditemukan/error: {e_path}. Mencoba webdriver-manager...")

    # Jika gagal, baru gunakan webdriver-manager (dengan cache default, bukan valid_range=0)
    try:
        driver_path = ChromeDriverManager().install()
        service = Service(driver_path)
        
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(SCRAPER_TIMEOUT)
        return driver
    except OSError as e:
        logger.error(f"[Scraper] Gagal inisialisasi driver via WDM: {e}")
        return None
    except Exception as e:
        logger.error(f"[Scraper] Gagal inisialisasi driver: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 2. SCRAPER PIHPS BI
# ─────────────────────────────────────────────────────────────────────────────

def scrape_harga_pihps() -> dict:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait, Select
    from selenium.webdriver.support import expected_conditions as EC

    driver = init_driver()
    if not driver:
        raise RuntimeError("WebDriver gagal diinisialisasi (lihat log sebelumnya).")

    try:
        logger.info(f"[Scraper] Membuka {PIHPS_URL}")
        driver.get(PIHPS_URL)
        wait = WebDriverWait(driver, 15) # Kurangi timeout dari 30s ke 15s

        # ── Tunggu halaman siap ───────────────────────────────────────────────
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "select")))
        except Exception:
            logger.warning("[Scraper] Timeout tunggu select, coba parsing langsung...")

        # ── Pilih Provinsi ────────────────────────────────────────────────────
        try:
            sel_elements = driver.find_elements(By.TAG_NAME, "select")
            prov_sel = None
            for sel in sel_elements:
                opts = [o.text for o in sel.find_elements(By.TAG_NAME, "option")]
                if any(PIHPS_PROVINSI.lower() in o.lower() for o in opts):
                    prov_sel = sel
                    break

            if prov_sel:
                Select(prov_sel).select_by_visible_text(PIHPS_PROVINSI)
                import time; time.sleep(2)
        except Exception as e:
            logger.warning(f"[Scraper] Gagal pilih provinsi: {e}")

        # ── Pilih Kota ────────────────────────────────────────────────────────
        try:
            import time; time.sleep(1)
            sel_elements = driver.find_elements(By.TAG_NAME, "select")
            kota_sel = None
            for sel in sel_elements:
                opts = [o.text for o in sel.find_elements(By.TAG_NAME, "option")]
                if any(PIHPS_KOTA.lower() in o.lower() for o in opts):
                    kota_sel = sel
                    break

            if kota_sel:
                Select(kota_sel).select_by_visible_text(PIHPS_KOTA)
                time.sleep(2)
        except Exception as e:
            logger.warning(f"[Scraper] Gagal pilih kota: {e}")

        # ── Klik tombol "Lihat" / "Tampilkan" ────────────────────────────────
        try:
            import time
            btns = driver.find_elements(By.TAG_NAME, "button")
            btn_found = None
            for btn in btns:
                txt = btn.text.strip().lower()
                if any(kw in txt for kw in ["lihat", "tampil", "cari", "submit"]):
                    btn_found = btn
                    break
            if btn_found:
                driver.execute_script("arguments[0].click();", btn_found)
                time.sleep(3)
        except Exception as e:
            logger.warning(f"[Scraper] Gagal klik tombol: {e}")

        # ── Parse tabel hasil ─────────────────────────────────────────────────
        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        except Exception:
            pass

        page_source = driver.page_source
        tables = pd.read_html(StringIO(page_source))

        hasil = {}
        for tbl in tables:
            tbl.columns = [str(c).strip() for c in tbl.columns]
            for _, row in tbl.iterrows():
                for komoditas in PIHPS_KOMODITAS:
                    row_str = " ".join(str(v) for v in row.values)
                    if komoditas.lower() in row_str.lower():
                        for val in row.values:
                            try:
                                harga_str = str(val).replace(".", "").replace(",", "").strip()
                                harga = float(harga_str)
                                if 10_000 <= harga <= 500_000:
                                    kolom_dataset = PIHPS_KOLOM_MAP.get(komoditas)
                                    if kolom_dataset:
                                        hasil[kolom_dataset] = harga
                                    break
                            except (ValueError, TypeError):
                                continue

        if not hasil or "harga_cabai_merah" not in hasil:
            raise ValueError("Data harga_cabai_merah gagal diparse dari tabel PIHPS")

        return {
            **hasil,
            "tanggal": date.today().isoformat(),
            "sumber" : "PIHPS BI",
            "status" : "live",
        }

    finally:
        try:
            driver.quit()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPEN-METEO API (cuaca real-time)
# ─────────────────────────────────────────────────────────────────────────────

async def get_cuaca_realtime() -> dict:
    """
    Ambil cuaca Kota Padang dari Open-Meteo API.
    Termasuk angin dan lama penyinaran (daily).
    """
    _FALLBACK = {
        "suhu_rata"      : 27.0,
        "kelembaban"     : 80.0,
        "curah_hujan"    : 5.0,
        "kec_angin"      : 5.0,
        "lama_penyinaran": 6.0,
        "waktu"          : None,
        "status"         : "fallback",
    }

    try:
        # Request current & daily forecast
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={CUACA_LATITUDE}"
            f"&longitude={CUACA_LONGITUDE}&current=temperature_2m,relative_humidity_2m,"
            f"precipitation,wind_speed_10m&daily=sunshine_duration"
            f"&timezone={CUACA_TIMEZONE}&forecast_days=1"
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        current = data.get("current", {})
        daily = data.get("daily", {})

        suhu    = float(current.get("temperature_2m", 27.0))
        lembab  = float(current.get("relative_humidity_2m", 80.0))
        hujan   = float(current.get("precipitation", 0.0))
        angin   = float(current.get("wind_speed_10m", 5.0))
        waktu   = current.get("time", datetime.now().isoformat())
        
        # sunshine_duration dalam detik, ubah ke jam
        sunshine_sec = daily.get("sunshine_duration", [21600.0])[0]
        penyinaran = float(sunshine_sec) / 3600.0 if pd.notna(sunshine_sec) else 6.0

        logger.info(f"[Cuaca] Padang: {suhu}°C, {lembab}% RH, {hujan}mm ✓")
        return {
            "suhu_rata"      : suhu,
            "kelembaban"     : lembab,
            "curah_hujan"    : hujan,
            "kec_angin"      : angin,
            "lama_penyinaran": penyinaran,
            "waktu"          : waktu,
            "status"         : "live",
        }

    except Exception as e:
        logger.warning(f"[Cuaca] Open-Meteo gagal, pakai fallback: {e}")
        return _FALLBACK


# ─────────────────────────────────────────────────────────────────────────────
# 4. DAYOFF API (hari libur)
# ─────────────────────────────────────────────────────────────────────────────

async def get_daftar_libur(tahun: int) -> list:
    """
    Mengambil semua hari libur nasional dalam 1 tahun via github raw API.
    Mengembalikan list berisi object datetime.date.
    """
    try:
        url = f"https://raw.githubusercontent.com/gerinsp/dayoff-API/master/data/{tahun}.json"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            holidays = response.json()
        
        daftar_libur = []
        for h in holidays:
            # Parse tanggal dari JSON agar format konsisten
            h_date = datetime.strptime(h["tanggal"], "%Y-%m-%d").date()
            daftar_libur.append(h_date)
        return daftar_libur
    except Exception as e:
        logger.warning(f"[Libur] Gagal fetch DayOff API: {e}. Default ke kosong.")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATA TRANSFORMER / MAPPER (100% SINKRON DENGAN DATASET)
# ─────────────────────────────────────────────────────────────────────────────

def _engineering_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """
    Logika feature engineering IDENTIK dengan preprocessing.py.
    Menghasilkan 47 kolom yang persis sama strukturnya.
    """
    df = df.copy()

    # ── A. Kalender ──
    df["bulan"] = df["tanggal"].dt.month
    df["kuartal"] = df["tanggal"].dt.quarter
    df["is_weekend"] = (df["tanggal"].dt.dayofweek >= 5).astype(int)
    df["is_awal_bulan"] = (df["tanggal"].dt.day <= 7).astype(int)
    df["bulan_sin"] = np.sin(2 * np.pi * df["bulan"] / 12)
    df["bulan_cos"] = np.cos(2 * np.pi * df["bulan"] / 12)

    # ── B. Event khusus ──
    lebaran = pd.to_datetime(["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-30", "2026-03-20"])
    df["is_pra_lebaran"] = 0
    df["is_lebaran"] = 0
    df["is_pasca_lebaran"] = 0
    for d in lebaran:
        df.loc[(df["tanggal"] >= d - pd.Timedelta(14)) & (df["tanggal"] < d), "is_pra_lebaran"] = 1
        df.loc[(df["tanggal"] >= d) & (df["tanggal"] <= d + pd.Timedelta(2)), "is_lebaran"] = 1
        df.loc[(df["tanggal"] > d + pd.Timedelta(2)) & (df["tanggal"] <= d + pd.Timedelta(9)), "is_pasca_lebaran"] = 1
    
    df["is_natal_tahunbaru"] = (
        ((df["bulan"] == 12) & (df["tanggal"].dt.day >= 20)) | 
        ((df["bulan"] == 1) & (df["tanggal"].dt.day <= 7))
    ).astype(int)
    df["is_musim_panen"] = df["bulan"].isin([5, 6, 7, 11, 12, 1]).astype(int)

    # ── C. Lag harga ──
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    if "harga_cabai_rawit" in df.columns:
        df["rawit_lag_1"] = df["harga_cabai_rawit"].shift(1)
        df["rawit_lag_7"] = df["harga_cabai_rawit"].shift(7)

    # ── D. Rolling statistics ──
    for w in [7, 30]:
        s = df[TARGET].shift(1)
        df[f"roll_mean_{w}"] = s.rolling(w).mean()
        df[f"roll_std_{w}"] = s.rolling(w).std()

    # ── E. Momentum ──
    df["momentum_7"] = df[TARGET].shift(1) - df[TARGET].shift(7)

    # ── F. Fitur tren & volatilitas ──
    df["rasio_tren_7_30"] = (df["roll_mean_7"] / df["roll_mean_30"].replace(0, np.nan)).fillna(1.0)
    roll14_mean = df[TARGET].shift(1).rolling(14).mean().replace(0, np.nan)
    roll14_std = df[TARGET].shift(1).rolling(14).std()
    df["volatilitas_14"] = (roll14_std / roll14_mean).fillna(0)

    # ── G. Cuaca ──
    if "curah_hujan" in df.columns:
        df["roll_hujan_30"] = df["curah_hujan"].rolling(30, min_periods=1).sum()
        df["is_hari_hujan"] = (df["curah_hujan"] > 1.0).astype(int)

    # ── H. Diferensiasi harga ──
    df["selisih_harga_1"] = df[TARGET].diff(1)
    df["selisih_harga_7"] = df[TARGET].diff(7)
    df["arah_1"] = np.sign(df["selisih_harga_1"])
    df["arah_7"] = np.sign(df["selisih_harga_7"])
    df["streak_naik"] = df["arah_1"].groupby((df["arah_1"] != 1).cumsum()).cumcount()

    # ── I. Fitur konteks historis ──
    df["is_pasca_covid"] = (df["tanggal"].dt.year == 2022).astype(int)

    # ── J. Target multi-horizon ──
    df["target_h1"] = df[TARGET].shift(-1)
    df["target_h3"] = df[TARGET].shift(-3)
    df["target_h7"] = df[TARGET].shift(-7)
    df["arah_target_h1"] = (df["target_h1"] > df[TARGET]).astype(int)
    df["arah_target_h3"] = (df["target_h3"] > df[TARGET]).astype(int)
    df["arah_target_h7"] = (df["target_h7"] > df[TARGET]).astype(int)

    df.loc[df["target_h1"].isna(), "arah_target_h1"] = np.nan
    df.loc[df["target_h3"].isna(), "arah_target_h3"] = np.nan
    df.loc[df["target_h7"].isna(), "arah_target_h7"] = np.nan

    return df


def transform_data_ke_format_dataset(df_history: pd.DataFrame, harga_data: dict, cuaca_data: dict, daftar_libur: list) -> pd.DataFrame:
    """
    Fungsi Mapper: Menggabungkan data raw baru dengan history (tail 40 hari).
    Jika ada hari yang bolong (gap) antara data terakhir dengan hari ini, 
    sistem akan melakukan Forward Fill otomatis untuk menjaga integritas urutan waktu.
    """
    # 1. Pastikan history minimal 40 baris agar rolling window 30 dan lag 30 bisa dihitung
    df_tail = df_history.tail(40).copy()
    
    # 2. Ambil informasi tanggal terakhir di dataset
    last_date = df_tail["tanggal"].max().date()
    tgl_str = harga_data.get("tanggal", date.today().isoformat())
    target_date = date.fromisoformat(tgl_str) if isinstance(tgl_str, str) else tgl_str
    
    new_rows_raw = []
    
    # 3. Handle GAP (Bolong Hari): Generate baris Forward Fill
    from datetime import timedelta
    current_date = last_date + timedelta(days=1)
    
    # Ambil base features dari baris terakhir untuk ffill
    last_row = df_tail.iloc[-1]
    
    while current_date < target_date:
        ffill_row = {
            "tanggal": pd.Timestamp(current_date),
            TARGET: float(last_row.get(TARGET, 0)),
            "harga_cabai_rawit": float(last_row.get("harga_cabai_rawit", 0)),
            "suhu_rata": float(last_row.get("suhu_rata", np.nan)),
            "kelembaban": float(last_row.get("kelembaban", np.nan)),
            "curah_hujan": float(last_row.get("curah_hujan", np.nan)),
            "lama_penyinaran": float(last_row.get("lama_penyinaran", np.nan)),
            "kec_angin": float(last_row.get("kec_angin", np.nan)),
            "is_libur_nasional": 1 if current_date in daftar_libur else 0,
        }
        for col in df_tail.columns:
            if col not in ffill_row and not col.startswith(("lag_", "roll_", "selisih_", "arah_", "streak_", "target_", "is_", "bulan", "kuartal", "momentum_", "rasio_", "volatilitas_")):
                ffill_row[col] = np.nan
        new_rows_raw.append(ffill_row)
        current_date += timedelta(days=1)
        
    # 4. Siapkan baris hari ini (data asli hasil scrape)
    today_row = {
        "tanggal": pd.Timestamp(target_date),
        TARGET: float(harga_data.get("harga_cabai_merah", 0)),
        "harga_cabai_rawit": float(harga_data.get("harga_cabai_rawit", last_row.get("harga_cabai_rawit", 0))),
        "suhu_rata": float(cuaca_data.get("suhu_rata", np.nan)),
        "kelembaban": float(cuaca_data.get("kelembaban", np.nan)),
        "curah_hujan": float(cuaca_data.get("curah_hujan", np.nan)),
        "lama_penyinaran": float(cuaca_data.get("lama_penyinaran", np.nan)),
        "kec_angin": float(cuaca_data.get("kec_angin", np.nan)),
        "is_libur_nasional": 1 if target_date in daftar_libur else 0,
    }
    for col in df_tail.columns:
        if col not in today_row and not col.startswith(("lag_", "roll_", "selisih_", "arah_", "streak_", "target_", "is_", "bulan", "kuartal", "momentum_", "rasio_", "volatilitas_")):
            today_row[col] = np.nan
    new_rows_raw.append(today_row)

    # 5. Append ke tail dan hitung ulang semua fitur
    df_combined = pd.concat([df_tail, pd.DataFrame(new_rows_raw)], ignore_index=True)
    df_combined["tanggal"] = pd.to_datetime(df_combined["tanggal"])
    
    df_computed = _engineering_fitur(df_combined)
    
    # 6. Ambil HANYA baris-baris baru (jumlah ffill + 1 baris scrape)
    final_rows = df_computed.tail(len(new_rows_raw))
    
    # Pastikan urutan kolom sesuai dataset asli
    final_rows = final_rows.reindex(columns=df_history.columns)
    
    return final_rows


# ─────────────────────────────────────────────────────────────────────────────
# 6. UPDATE DATASET CSV
# ─────────────────────────────────────────────────────────────────────────────

def update_dataset(harga_data: dict, cuaca_data: dict, daftar_libur: list) -> dict:
    import portalocker

    csv_path = DATA_DIR / "dataset_preprocessed.csv"
    if not csv_path.exists():
        return {"status": "error", "pesan": "CSV tidak ditemukan"}

    try:
        with portalocker.Lock(str(csv_path) + ".lock", timeout=15):
            df = pd.read_csv(csv_path, parse_dates=["tanggal"])
            df = df.sort_values("tanggal").reset_index(drop=True)

            tgl_str = harga_data.get("tanggal", date.today().isoformat())
            tgl = pd.Timestamp(date.fromisoformat(tgl_str) if isinstance(tgl_str, str) else tgl_str)
            harga_baru = float(harga_data.get("harga_cabai_merah", 0))

            if harga_baru <= 0:
                raise ValueError(f"harga_cabai_merah tidak valid: {harga_baru}")

            # Hitung baris baru (termasuk ffill untuk hari yang bolong) via mapper
            new_df_rows = transform_data_ke_format_dataset(df, harga_data, cuaca_data, daftar_libur)
            
            # Cek apakah tanggal target sudah ada di dataset
            mask = df["tanggal"].dt.date == tgl.date()
            if mask.any():
                # Jika sudah ada, cukup update baris terakhir saja
                idx = df[mask].index[0]
                df.iloc[idx] = new_df_rows.iloc[-1]
                status_update = "already_exists"
                logger.info(f"[Dataset] Update baris existing tanggal {tgl.date()}")
            else:
                # Append semua baris baru (ffill + today)
                num_gap = len(new_df_rows) - 1
                df = pd.concat([df, new_df_rows], ignore_index=True)
                status_update = "updated"
                if num_gap > 0:
                    logger.info(f"[Dataset] Ditambahkan {num_gap} hari backfill otomatis + data hari ini: {tgl.date()} | Rp {harga_baru:,.0f}")
                else:
                    logger.info(f"[Dataset] Baris baru ditambahkan: {tgl.date()} | Rp {harga_baru:,.0f}")

            # Pastikan tipe datetime terjaga sebelum simpan
            df["tanggal"] = df["tanggal"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df.to_csv(csv_path, index=False)
            n_total = len(df)

        # Simpan status ke JSON
        _simpan_status(harga_data, cuaca_data, status_update)

        # Reload cache predictor
        try:
            from app.core import predictor
            predictor.load_dataset_to_cache()
            logger.info("[Dataset] Cache predictor di-reload")
        except Exception as e:
            logger.warning(f"[Dataset] Gagal reload cache: {e}")

        return {
            "status"       : status_update,
            "tanggal"      : tgl_str,
            "harga_baru"   : harga_baru,
            "n_baris_total": n_total,
        }

    except Exception as e:
        logger.error(f"[Dataset] Gagal update dataset: {e}")
        return {"status": "error", "pesan": str(e)}


def _simpan_status(harga_data: dict, cuaca_data: dict, update_status: str):
    """Simpan info update terakhir ke realtime_status.json."""
    try:
        status = {
            "waktu_update"      : datetime.now().isoformat(),
            "tanggal_data"      : harga_data.get("tanggal", date.today().isoformat()),
            "harga_cabai_merah" : harga_data.get("harga_cabai_merah"),
            "harga_status"      : harga_data.get("status", "unknown"),
            "sumber_harga"      : harga_data.get("sumber", "unknown"),
            "suhu_rata"         : cuaca_data.get("suhu_rata"),
            "kelembaban"        : cuaca_data.get("kelembaban"),
            "curah_hujan"       : cuaca_data.get("curah_hujan"),
            "cuaca_status"      : cuaca_data.get("status", "unknown"),
            "update_status"     : update_status,
        }
        REALTIME_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REALTIME_STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        logger.info(f"[Status] realtime_status.json diperbarui")
    except Exception as e:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 7. ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

async def jalankan_update_realtime() -> dict:
    logger.info("[Update] Memulai update real-time...")
    waktu_mulai = datetime.now()

    # 1. Fetch Cuaca & Libur (Async)
    tgl_hari_ini = date.today()
    cuaca_task = asyncio.create_task(get_cuaca_realtime())
    libur_task = asyncio.create_task(get_daftar_libur(tgl_hari_ini.year))
    
    cuaca_data, daftar_libur = await asyncio.gather(cuaca_task, libur_task)

    # 2. Fetch Harga PIHPS (Thread Pool)
    harga_data   = None
    harga_status = "fallback"

    try:
        loop = asyncio.get_event_loop()
        harga_data   = await loop.run_in_executor(None, scrape_harga_pihps)
        harga_status = "live"
        logger.info(f"[Scraper] Harga cabai merah: Rp {harga_data['harga_cabai_merah']:,.0f} ✓")
    except Exception as e:
        logger.warning(f"[Scraper] PIHPS gagal, pakai fallback harga terakhir: {e}")
        harga_data = _fallback_harga_terakhir()

    # 3. Update Dataset via Mapper (Otomatis Ffill Gap)
    dataset_result = update_dataset(harga_data, cuaca_data, daftar_libur)

    # Status Response
    if harga_status == "live" and cuaca_data["status"] == "live":
        overall_status = "success"
        pesan = f"Data berhasil diperbarui dari PIHPS BI pukul {waktu_mulai.strftime('%H:%M')} WIB"
    elif dataset_result.get("status") == "error":
        overall_status = "fallback"
        pesan = f"Update gagal: {dataset_result.get('pesan', 'unknown error')}"
    else:
        overall_status = "partial"
        pesan = "Data diperbarui dengan sebagian fallback"

    return {
        "status"           : overall_status,
        "harga_status"     : harga_status,
        "cuaca_status"     : cuaca_data["status"],
        "tanggal"          : harga_data.get("tanggal", date.today().isoformat()),
        "harga_cabai_merah": harga_data.get("harga_cabai_merah"),
        "suhu_rata"        : cuaca_data.get("suhu_rata"),
        "curah_hujan"      : cuaca_data.get("curah_hujan"),
        "kelembaban"       : cuaca_data.get("kelembaban"),
        "waktu_update"     : datetime.now().isoformat(),
        "pesan"            : pesan,
    }


def _fallback_harga_terakhir() -> dict:
    try:
        csv_path = DATA_DIR / "dataset_preprocessed.csv"
        df = pd.read_csv(csv_path, parse_dates=["tanggal"])
        df = df.sort_values("tanggal").reset_index(drop=True)
        last = df.dropna(subset=[TARGET]).iloc[-1]
        return {
            "harga_cabai_merah": float(last[TARGET]),
            "harga_cabai_rawit": float(last["harga_cabai_rawit"]) if "harga_cabai_rawit" in last and pd.notna(last.get("harga_cabai_rawit")) else 0.0,
            "tanggal"          : date.today().isoformat(), # Tetap gunakan tanggal hari ini untuk input baris baru
            "sumber"           : "CSV fallback (ffill)",
            "status"           : "fallback",
        }
    except Exception as e:
        return {
            "harga_cabai_merah": 50000.0,
            "tanggal"          : date.today().isoformat(),
            "sumber"           : "hardcoded fallback",
            "status"           : "fallback",
        }

# BACA STATUS (Dipertahankan dari sebelumnya)
def baca_status_realtime() -> dict:
    if not REALTIME_STATUS_FILE.exists():
        return {
            "data_harga_status"     : "fallback",
            "data_cuaca_status"     : "unknown",
            "tanggal_data_terkini"  : None,
            "waktu_update_terakhir" : None,
            "harga_terkini"         : None,
            "cuaca_terkini"         : None,
            "pesan"                 : "Belum pernah update — jalankan POST /api/realtime/update",
        }
    try:
        with open(REALTIME_STATUS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "data_harga_status"     : data.get("harga_status", "unknown"),
            "data_cuaca_status"     : data.get("cuaca_status", "unknown"),
            "tanggal_data_terkini"  : data.get("tanggal_data"),
            "waktu_update_terakhir" : data.get("waktu_update"),
            "harga_terkini"         : data.get("harga_cabai_merah"),
            "cuaca_terkini"         : {"suhu_rata": data.get("suhu_rata"), "kelembaban": data.get("kelembaban"), "curah_hujan": data.get("curah_hujan"), "status": data.get("cuaca_status", "unknown")},
            "pesan"                 : f"Data {'dari PIHPS BI' if data.get('harga_status') == 'live' else 'fallback'} pukul {data.get('waktu_update', '')[:16].replace('T', ' ')} WIB",
        }
    except Exception as e:
        return {
            "data_harga_status": "error", "data_cuaca_status": "error",
            "pesan": f"Error membaca status: {e}",
        }
