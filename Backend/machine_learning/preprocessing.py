import json
import glob
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CUACA_DIR = DATA_DIR / "raw" / "cuaca"
HARGA_DIR = DATA_DIR / "raw" / "harga_cabai"
LIBUR_DIR = DATA_DIR / "raw" / "hari_libur"
OUTPUT_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "machine_learning" / "output" / "xgboost_models"
# MODEL_DIR: machine_learning/output/xgboost_models/ (lokasi resmi .pkl)
# Jangan simpan .pkl di machine_learning/models/ — folder tersebut sudah dihapus
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "harga_cabai_merah"
LOG_LINES = []


def log(msg: str):
    """Cetak ke terminal sekaligus simpan ke laporan."""
    print(msg)
    LOG_LINES.append(msg)


# =============================================================================
# BAGIAN 1 — LOAD DATA CUACA BMKG
# =============================================================================
def load_cuaca() -> pd.DataFrame:
    log("\n[1] Memuat data cuaca BMKG...")

    all_files = sorted(glob.glob(str(CUACA_DIR / "**" / "*.xlsx"), recursive=True))
    log(f"    -> Ditemukan {len(all_files)} file cuaca")
    if not all_files:
        raise FileNotFoundError(f"Tidak ada file xlsx di: {CUACA_DIR}")

    frames = []
    for fp in all_files:
        try:
            df_raw = pd.read_excel(fp, header=None)
            header_idx = None
            for i, row in df_raw.iterrows():
                if "TANGGAL" in [str(v).strip().upper() for v in row.values]:
                    header_idx = i
                    break
            if header_idx is None:
                log(f"    x {Path(fp).name}: header tidak ditemukan")
                continue

            df = pd.read_excel(fp, header=header_idx)
            df.columns = [str(c).strip().upper() for c in df.columns]
            needed = ["TANGGAL", "TAVG", "RH_AVG", "RR", "SS", "FF_AVG"]
            available = [c for c in needed if c in df.columns]
            df = df[available].copy()
            df = df[df["TANGGAL"].astype(str).str.match(r"\d{2}-\d{2}-\d{4}")]
            frames.append(df)
            log(f"    + {Path(fp).name}: {len(df)} baris")
        except Exception as e:
            log(f"    x {Path(fp).name}: {e}")

    if not frames:
        raise ValueError("Tidak ada data cuaca yang berhasil dimuat!")

    df = pd.concat(frames, ignore_index=True)
    df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], format="%d-%m-%Y", errors="coerce")
    df.rename(
        columns={
            "TANGGAL": "tanggal",
            "TAVG": "suhu_rata",
            "RH_AVG": "kelembaban",
            "RR": "curah_hujan",
            "SS": "lama_penyinaran",
            "FF_AVG": "kec_angin",
        },
        inplace=True,
    )
    df.dropna(subset=["tanggal"], inplace=True)

    numeric_cols = [
        "suhu_rata",
        "kelembaban",
        "curah_hujan",
        "lama_penyinaran",
        "kec_angin",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].replace([8888, 9999, 8888.0, 9999.0], np.nan, inplace=True)

    miss_before = df[numeric_cols].isna().sum().to_dict()

    df = df.groupby("tanggal")[numeric_cols].mean().reset_index()
    df.sort_values("tanggal", inplace=True)

    date_range = pd.date_range(df["tanggal"].min(), df["tanggal"].max(), freq="D")
    df = df.set_index("tanggal").reindex(date_range).reset_index()
    df.rename(columns={"index": "tanggal"}, inplace=True)
    # PERBAIKAN LEAKAGE 1: Ganti interpolate(linear) → ffill murni
    # Alasan: interpolasi linear membutuhkan nilai t+1 untuk mengisi t (bocor ke depan).
    # ffill = "cuaca hari ini = cuaca terakhir yang tercatat" → kausal 100%.
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    log(f"    -> Missing sebelum imputasi (ffill) : {miss_before}")
    log(
        f"    -> Missing setelah imputasi   : {df[numeric_cols].isna().sum().to_dict()}"
    )
    log(f"    -> Shape final : {df.shape}")
    log(
        f"    -> Rentang     : {df['tanggal'].min().date()} s/d {df['tanggal'].max().date()}"
    )
    return df


# =============================================================================
# BAGIAN 2 — LOAD DATA HARGA CABAI
# =============================================================================
def load_harga_cabai() -> pd.DataFrame:
    log("\n[2] Memuat data harga cabai...")

    xlsx_files = sorted(glob.glob(str(HARGA_DIR / "*.xlsx")))
    log(f"    -> Ditemukan {len(xlsx_files)} file harga")
    if not xlsx_files:
        raise FileNotFoundError(f"Tidak ada file xlsx di: {HARGA_DIR}")

    frames = []
    for fp in xlsx_files:
        try:
            df_raw = pd.read_excel(fp)
            komoditas_col = None
            for col in df_raw.columns:
                if df_raw[col].astype(str).str.lower().str.contains("cabai|cabe").any():
                    komoditas_col = col
                    break
            if komoditas_col is None:
                log(f"    x {Path(fp).name}: kolom komoditas tidak ditemukan")
                continue

            df_raw["_ket"] = df_raw[komoditas_col].astype(str).str.strip()
            df_cabai = df_raw[
                df_raw["_ket"].str.lower().str.contains("cabai|cabe", na=False)
            ].copy()

            skip_cols = [komoditas_col, "_ket"] + list(df_raw.columns[:2])
            date_cols = [c for c in df_raw.columns if c not in skip_cols]

            for _, row in df_cabai.iterrows():
                nama = row["_ket"].lower()
                if "keriting" in nama or ("merah" in nama and "rawit" not in nama):
                    jenis = "harga_cabai_merah"
                elif "rawit" in nama:
                    jenis = "harga_cabai_rawit"
                else:
                    continue

                records = []
                for col in date_cols:
                    val = str(row[col]).strip()
                    harga = (
                        np.nan
                        if val in ["-", "nan", "", "None"]
                        else float(val.replace(",", "").replace(".", ""))
                    )
                    try:
                        tgl = pd.to_datetime(
                            str(col).strip(), dayfirst=True, errors="coerce"
                        )
                    except Exception:
                        tgl = pd.NaT
                    if pd.notna(tgl):
                        records.append({"tanggal": tgl, jenis: harga})
                if records:
                    frames.append(pd.DataFrame(records))
            log(f"    + {Path(fp).name}")
        except Exception as e:
            log(f"    x {Path(fp).name}: {e}")

    if not frames:
        raise ValueError("Tidak ada data harga yang berhasil dimuat!")

    df = pd.concat(frames, ignore_index=True)
    df = df.groupby("tanggal").mean().reset_index()
    df.sort_values("tanggal", inplace=True)

    harga_cols = ["harga_cabai_merah", "harga_cabai_rawit"]
    for col in harga_cols:
        if col in df.columns:
            n_neg = (df[col] <= 0).sum()
            if n_neg > 0:
                log(f"    ! {col}: {n_neg} nilai <= 0 diganti NaN")
                df.loc[df[col] <= 0, col] = np.nan

    for col in harga_cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            log(
                f"    ! Outlier '{col}': {outliers} nilai di luar "
                f"[{lower:,.0f}, {upper:,.0f}] -> diganti NaN"
            )
            df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan

    date_range = pd.date_range(df["tanggal"].min(), df["tanggal"].max(), freq="D")
    df_indexed = df.set_index("tanggal").reindex(date_range)
    for col in harga_cols:
        if col in df_indexed.columns:
            n_miss = df_indexed[col].isna().sum()
            # PERBAIKAN 1: Gunakan ffill (bukan interpolate) untuk hari libur/weekend
            # Alasan: pasar tutup di hari libur → harga tidak bergerak,
            # bukan naik/turun bertahap. ffill = "harga tetap seperti hari kerja terakhir"
            df_indexed[col] = df_indexed[col].ffill().bfill()
            log(
                f"    -> ffill '{col}': {n_miss} hari libur/weekend diisi harga terakhir"
            )
    df = df_indexed.reset_index().rename(columns={"index": "tanggal"})

    log(f"    -> Shape final : {df.shape}")
    log(
        f"    -> Rentang     : {df['tanggal'].min().date()} s/d {df['tanggal'].max().date()}"
    )
    return df


# =============================================================================
# BAGIAN 3 — LOAD DATA HARI LIBUR
# =============================================================================
def load_hari_libur() -> pd.DataFrame:
    log("\n[3] Memuat data hari libur...")

    json_files = sorted(glob.glob(str(LIBUR_DIR / "*.json")))
    log(f"    -> Ditemukan {len(json_files)} file JSON")
    if not json_files:
        raise FileNotFoundError(f"Tidak ada file JSON di: {LIBUR_DIR}")

    records = []
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                tgl = pd.to_datetime(item.get("tanggal", ""), errors="coerce")
                if pd.notna(tgl):
                    records.append(
                        {"tanggal": tgl, "keterangan_libur": item.get("keterangan", "")}
                    )
            log(f"    + {Path(fp).name}: {len(data)} hari libur")
        except Exception as e:
            log(f"    x {Path(fp).name}: {e}")

    df = pd.DataFrame(records)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df.drop_duplicates(subset=["tanggal"], inplace=True)
    df.sort_values("tanggal", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log(f"    -> Total hari libur : {len(df)}")
    log(f"    -> Tahun            : {sorted(df['tanggal'].dt.year.unique().tolist())}")
    return df


# =============================================================================
# BAGIAN 4 — MERGE
# =============================================================================
def merge_data(df_harga, df_cuaca, df_libur) -> pd.DataFrame:
    log("\n[4] Menggabungkan semua data...")

    start, end = df_harga["tanggal"].min(), df_harga["tanggal"].max()
    df_cuaca = df_cuaca[
        (df_cuaca["tanggal"] >= start) & (df_cuaca["tanggal"] <= end)
    ].copy()

    df = pd.merge(df_harga, df_cuaca, on="tanggal", how="left")

    df_flag = df_libur[["tanggal"]].copy()
    df_flag["is_libur_nasional"] = 1
    df = pd.merge(df, df_flag, on="tanggal", how="left")
    df["is_libur_nasional"] = df["is_libur_nasional"].fillna(0).astype(int)

    df.sort_values("tanggal", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # PERBAIKAN: Cuaca dari 2024 diback-fill ke 2022-2023 agar data harga 2022 tetap bisa dipakai
    cuaca_cols = ["suhu_rata", "kelembaban", "curah_hujan", "lama_penyinaran", "kec_angin"]
    for col in cuaca_cols:
        if col in df.columns:
            df[col] = df[col].bfill().ffill()

    for col in ["suhu_rata", "kelembaban", "curah_hujan"]:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            if pct > 30:
                log(f"    ! '{col}': {pct:.1f}% missing setelah merge (sudah di-bfill ke 2022)")

    log(f"    -> Shape setelah merge  : {df.shape}")
    log(f"    -> Hari libur ditandai  : {df['is_libur_nasional'].sum()} hari")
    return df


# =============================================================================
# BAGIAN 5 — FEATURE ENGINEERING (v2 — fitur selektif)
# =============================================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    log("\n[5] Feature engineering (v2 — fitur selektif)...")

    # ── A. Kalender (hanya yang informatif untuk harga komoditas) ──────────
    df["bulan"] = df["tanggal"].dt.month
    df["kuartal"] = df["tanggal"].dt.quarter
    df["is_weekend"] = (df["tanggal"].dt.dayofweek >= 5).astype(int)
    df["is_awal_bulan"] = (df["tanggal"].dt.day <= 7).astype(int)
    # Enkode siklus bulanan dengan sin/cos (menggantikan bulan mentah)
    df["bulan_sin"] = np.sin(2 * np.pi * df["bulan"] / 12)
    df["bulan_cos"] = np.cos(2 * np.pi * df["bulan"] / 12)
    
    # BARU: Enkode siklus mingguan (1-52) untuk H+3
    df["minggu_ke"] = df["tanggal"].dt.isocalendar().week.astype(int)
    df["minggu_sin"] = np.sin(2 * np.pi * df["minggu_ke"] / 52)
    df["minggu_cos"] = np.cos(2 * np.pi * df["minggu_ke"] / 52)
    df.drop(columns=["minggu_ke"], inplace=True)
    # Catatan: tahun, hari_bulan, hari_minggu, hari_sin/cos, hari_dlm_tahun,
    #          minggu_ke dihapus — tidak berpola signifikan untuk harga cabai

    # ── B. Event khusus (dipertahankan semua — domain knowledge kuat) ──────
    lebaran = pd.to_datetime(
        ["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-30", "2026-03-20"]
    )
    df["is_pra_lebaran"] = 0
    df["is_lebaran"] = 0
    df["is_pasca_lebaran"] = 0
    for d in lebaran:
        df.loc[
            (df["tanggal"] >= d - pd.Timedelta(14)) & (df["tanggal"] < d),
            "is_pra_lebaran",
        ] = 1
        df.loc[
            (df["tanggal"] >= d) & (df["tanggal"] <= d + pd.Timedelta(2)), "is_lebaran"
        ] = 1
        df.loc[
            (df["tanggal"] > d + pd.Timedelta(2))
            & (df["tanggal"] <= d + pd.Timedelta(9)),
            "is_pasca_lebaran",
        ] = 1
    df["is_natal_tahunbaru"] = (
        ((df["bulan"] == 12) & (df["tanggal"].dt.day >= 20))
        | ((df["bulan"] == 1) & (df["tanggal"].dt.day <= 7))
    ).astype(int)
    df["is_musim_panen"] = df["bulan"].isin([5, 6, 7, 11, 12, 1]).astype(int)

    # BARU: Jarak hari ke Lebaran terdekat (kontinu, 2-arah)
    # Lebih informatif dari is_pra_lebaran (biner) karena model bisa melihat
    # seberapa dekat dengan hari raya secara gradual
    _lebaran_dates = pd.to_datetime(
        ["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-30", "2026-03-20"]
    )
    df["days_to_lebaran"] = df["tanggal"].apply(
        lambda t: int(min(abs((t - d).days) for d in _lebaran_dates))
    )
    log(f"    -> days_to_lebaran ditambahkan ke preprocessing (konsistensi training-inference)")

    # ── C. Lag harga (selektif — hapus lag berkorelasi tinggi) ────────────
    # Simpan: lag_1 (kemarin), lag_7 (minggu lalu), lag_14 (dua minggu),
    #         lag_21 (tiga minggu), lag_30 (sebulan lalu). Hapus: lag_2, lag_3
    for lag in [1, 7, 14, 21, 30]:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)

    if "harga_cabai_rawit" in df.columns:
        # Rawit sebagai sinyal pasar substitusi jangka pendek
        df["rawit_lag_1"] = df["harga_cabai_rawit"].shift(1)
        df["rawit_lag_7"] = df["harga_cabai_rawit"].shift(7)

        # BARU: Lag rawit jangka menengah-panjang
        # SHAP menunjukkan rawit_lag_1 dan rawit_lag_7 dominan -> lag lebih jauh
        # memberi model perspektif tren historis substitusi
        df["rawit_lag_14"] = df["harga_cabai_rawit"].shift(14)
        df["rawit_lag_30"] = df["harga_cabai_rawit"].shift(30)

        # BARU: Rasio spread harga merah vs rawit
        # Jika rasio tinggi (merah relatif mahal), ekspektasi koreksi harga merah turun
        df["rasio_merah_rawit"] = (
            df[TARGET] / df["harga_cabai_rawit"].replace(0, np.nan)
        ).fillna(1.0)

    # ── D. Rolling statistics & EMA (hanya window 7 & 30) ────────
    # min/max di-drop: informasinya sudah tercakup oleh mean + std
    s_base = df[TARGET].shift(1)
    for w in [7, 30]:
        df[f"roll_mean_{w}"] = s_base.rolling(w).mean()
        df[f"roll_std_{w}"] = s_base.rolling(w).std()
        df[f"ema_{w}"] = s_base.ewm(span=w, adjust=False).mean()
        
    df["ema_crossover_7_30"] = df["ema_7"] - df["ema_30"]

    # ── E. Momentum (hanya jangka pendek) ─────────────────────────────────
    # momentum_30 dan pct_change di-drop: berkorelasi tinggi dengan momentum_7
    df["momentum_7"] = df[TARGET].shift(1) - df[TARGET].shift(7)
    
    # BARU: Momentum 3 hari (sangat penting untuk horizon H+3)
    df["momentum_3"] = df[TARGET].shift(1) - df[TARGET].shift(4)

    # ── F. Fitur tren & volatilitas (BARU) ────────────────────────────────
    # Rasio tren: apakah harga minggu ini naik lebih cepat dari tren bulanan?
    df["rasio_tren_7_30"] = (
        df["roll_mean_7"] / df["roll_mean_30"].replace(0, np.nan)
    ).fillna(1.0)

    # Koefisien variasi 14 hari: proxy ketidakstabilan / gejolak harga
    roll14_mean = df[TARGET].shift(1).rolling(14).mean().replace(0, np.nan)
    roll14_std = df[TARGET].shift(1).rolling(14).std()
    df["volatilitas_14"] = (roll14_std / roll14_mean).fillna(0)

    # ── G. Cuaca (fitur utama + akumulasi hujan jangka menengah-panjang) ──
    # Ditambahkan: lag_hujan_14, lag_hujan_21, roll_hujan_14, roll_hujan_21
    # Reasoning: dampak curah hujan terhadap harga cabai baru terasa 2-4 minggu
    # kemudian karena sesuai dengan siklus panen cabai di Sumatera Barat.
    # roll_hujan_30 dipertahankan untuk dampak jangka panjang (1 bulan).
    if "curah_hujan" in df.columns:
        df["lag_hujan_14"]  = df["curah_hujan"].shift(14)   # hujan 2 minggu lalu
        df["lag_hujan_21"]  = df["curah_hujan"].shift(21)   # hujan 3 minggu lalu
        df["roll_hujan_14"] = df["curah_hujan"].rolling(14, min_periods=1).sum()  # akumulasi 2 minggu
        df["roll_hujan_21"] = df["curah_hujan"].rolling(21, min_periods=1).sum()  # akumulasi 3 minggu
        df["roll_hujan_30"] = df["curah_hujan"].rolling(30, min_periods=1).sum()  # akumulasi 1 bulan

        # BARU: Fitur makro jangka panjang (siklus tanam ~2 bulan)
        # Reasoning: cabai butuh 60-90 hari dari tanam hingga panen.
        # Curah hujan selama periode ini menentukan total volume pasokan ke pasar.
        df["roll_hujan_60"] = df["curah_hujan"].rolling(60, min_periods=1).sum()  # akumulasi 2 bulan

        # BARU: Anomali cuaca ekstrem 3 hari terakhir
        # Reasoning: hujan > 50mm/hari menyebabkan cabai busuk & distribusi terganggu.
        # Dampaknya terasa dalam 1-3 hari, bukan 2 minggu.
        df["hujan_ekstrem_3hari"] = (
            (df["curah_hujan"] > 50).rolling(3, min_periods=1).sum().astype(int)
        )

        # BARU: Hujan ekstrem seketika (H+3)
        df["max_hujan_7"] = df["curah_hujan"].rolling(7, min_periods=1).max()

        df["is_hari_hujan"] = (df["curah_hujan"] > 1.0).astype(int)
    # suhu_rata dan curah_hujan dipertahankan langsung dari merge


    # ── H. Fitur Arah Pergerakan Harga (ANTI-LEAKAGE) ────────────────────
    # PENTING: Semua fitur arah menggunakan shift(1) agar hanya melihat masa lalu.
    # diff(n) tanpa shift menyertakan harga hari ini (t) → data leakage ke model.
    # Sebelumnya: selisih_harga_1/7, arah_1/7, streak_naik menggunakan harga[t] → BOCOR.
    harga_shift1 = df[TARGET].shift(1)  # harga kemarin sebagai basis semua fitur

    # Persentase perubahan (lebih informatif dari selisih Rp karena scale-free)
    df["pct_change_1"] = harga_shift1.pct_change(1) * 100   # % perubahan 1 hari lalu
    df["pct_change_3"] = harga_shift1.pct_change(3) * 100   # % perubahan 3 hari lalu
    df["pct_change_7"] = harga_shift1.pct_change(7) * 100   # % perubahan 7 hari lalu

    # Arah historis per hari (bebas leakage — semua pakai data masa lalu saja)
    df["arah_lag1"] = np.sign(df[TARGET].shift(1) - df[TARGET].shift(2))  # arah kemarin
    df["arah_lag2"] = np.sign(df[TARGET].shift(2) - df[TARGET].shift(3))  # arah 2 hari lalu
    df["arah_lag3"] = np.sign(df[TARGET].shift(3) - df[TARGET].shift(4))  # arah 3 hari lalu

    # Streak naik yang aman: berapa dari 3 hari terakhir harga naik? (nilai 0–3)
    # Berbeda dari versi lama (cumcount dari arah_1 yang bocor)
    df["streak_naik_3"] = (
        pd.DataFrame({
            "a1": df["arah_lag1"],
            "a2": df["arah_lag2"],
            "a3": df["arah_lag3"],
        }) == 1
    ).sum(axis=1)

    # Proporsi hari naik dalam 7 dan 14 hari terakhir (range 0.0–1.0)
    # min_periods mencegah NaN berlebihan di awal series
    diff_safe = df[TARGET].shift(1).diff()  # diff dari data kemarin → kausal
    df["prop_naik_7"]  = (diff_safe > 0).rolling(7,  min_periods=3).mean()
    df["prop_naik_14"] = (diff_safe > 0).rolling(14, min_periods=7).mean()

    # Pastikan kolom bocor lama tidak ada di dataset output
    _kolom_bocor = ["selisih_harga_1", "selisih_harga_7", "arah_1", "arah_7", "streak_naik"]
    df.drop(columns=[c for c in _kolom_bocor if c in df.columns], inplace=True)
    log(f"    -> Kolom bocor dihapus    : {[c for c in _kolom_bocor]}")
    log(f"    -> Fitur arah baru (aman) : pct_change_1/3/7, arah_lag1/2/3, "
        f"streak_naik_3 (0-3), prop_naik_7/14")

    # ── I. Fitur konteks historis (PERBAIKAN 4) ───────────────────────────
    # Tahun 2022 = masa pemulihan pasca-COVID, pola harga sangat tidak normal
    # Model perlu "diberitahu" bahwa data 2022 memiliki karakter berbeda
    df["is_pasca_covid"] = (df["tanggal"].dt.year == 2022).astype(int)

    # ── J. Target multi-horizon — DETRENDING (PERBAIKAN DOMINASI lag_1) ──
    # Menggunakan DELTA (perubahan harga) sebagai target, bukan harga absolut.
    # Tujuan: mengurangi dominasi lag_1 yang menyebabkan model berperilaku seperti
    # Naive Forecast. Dengan delta, model dipaksa belajar faktor perubahan, bukan
    # sekadar mengingat harga kemarin.
    #
    # Saat inference: prediksi_harga = harga_hari_ini + prediksi_delta
    # Kolom harga_hari_ini disimpan sebagai referensi inverse transform.
    #
    # CATATAN: Kolom ini TIDAK dimasukkan ke feature_cols, hanya dipakai training.

    # Simpan harga hari ini sebagai referensi untuk inverse transform
    df["harga_hari_ini"] = df[TARGET]

    # ── J. Target multi-horizon — LOG-RETURN (menggantikan delta absolut) ──
    # Log-return: target = ln(harga_besok / harga_hari_ini)
    # Keunggulan vs delta absolut:
    #   - Scale-free: Rp5000 dari Rp25000 (20%) ≠ Rp5000 dari Rp80000 (6%)
    #   - Distribusi lebih mendekati normal → lebih mudah dipelajari XGBoost
    #   - Simetris: naik 10% lalu turun 10% kembali ke titik awal
    #
    # Inverse transform saat inference: prediksi_rp = harga_hari_ini * exp(log_return)
    # CATATAN: Kolom ini TIDAK dimasukkan ke feature_cols, hanya dipakai training.

    # Hindari log(0) atau log(negatif) — ganti nilai non-positif dengan NaN
    harga_safe = df[TARGET].replace(0, np.nan)

    # Target merah: log-return H+1, H+3, H+7
    df["target_h1"] = np.log(df[TARGET].shift(-1) / harga_safe)  # log-return H+1
    df["target_h3"] = np.log(df[TARGET].shift(-3) / harga_safe)  # log-return H+3
    df["target_h7"] = np.log(df[TARGET].shift(-7) / harga_safe)  # log-return H+7

    # Target klasifikasi arah (tetap dipertahankan — 1=naik, 0=turun/tetap)
    df["arah_target_h1"] = (df["target_h1"] > 0).astype(int)
    df["arah_target_h3"] = (df["target_h3"] > 0).astype(int)
    df["arah_target_h7"] = (df["target_h7"] > 0).astype(int)

    # Pastikan target klasifikasi ikut NaN jika target regresi NaN
    df.loc[df["target_h1"].isna(), "arah_target_h1"] = np.nan
    df.loc[df["target_h3"].isna(), "arah_target_h3"] = np.nan
    df.loc[df["target_h7"].isna(), "arah_target_h7"] = np.nan

    # ── K. Target rawit multi-horizon — LOG-RETURN ─────────────────────────
    if "harga_cabai_rawit" in df.columns:
        df["harga_rawit_hari_ini"] = df["harga_cabai_rawit"]
        rawit_safe = df["harga_cabai_rawit"].replace(0, np.nan)
        df["target_rawit_h1"] = np.log(df["harga_cabai_rawit"].shift(-1) / rawit_safe)
        df["target_rawit_h3"] = np.log(df["harga_cabai_rawit"].shift(-3) / rawit_safe)
        df["target_rawit_h7"] = np.log(df["harga_cabai_rawit"].shift(-7) / rawit_safe)
        log(f"    -> Target rawit (log-return): target_rawit_h1, target_rawit_h3, target_rawit_h7")

    log(f"    -> Total fitur dibuat : {df.shape[1] - 2}")
    log(f"    -> Target merah (log-return): target_h1, target_h3, target_h7")
    log(f"    -> Referensi inverse transform: harga_hari_ini * exp(log_return)")
    return df


# =============================================================================
# BAGIAN 5B — SELEKSI FITUR OTOMATIS
# =============================================================================
def seleksi_fitur(df: pd.DataFrame, feature_cols: list) -> list:
    """
    Seleksi fitur otomatis dua tahap:
      1. Variance threshold  : hapus fitur hampir konstan (std < 0.01)
      2. Korelasi threshold  : hapus salah satu dari pasangan dengan r > 0.95
         (pilih yang lebih banyak berkorelasi dengan fitur lain)
    Mengembalikan list feature_cols yang sudah bersih.
    """
    log("\n[5B] Seleksi fitur otomatis...")
    n_awal = len(feature_cols)
    fitur_dihapus = []

    # ── Tahap 1: Variance threshold ───────────────────────────────────────
    low_var = [c for c in feature_cols if df[c].std() < 0.01]
    if low_var:
        log(f"    ! Hapus fitur hampir konstan (std < 0.01): {low_var}")
        fitur_dihapus.extend(low_var)
        feature_cols = [c for c in feature_cols if c not in low_var]

    # ── Tahap 2: Korelasi antar fitur (r > 0.95) ──────────────────────────
    corr_matrix = df[feature_cols].corr().abs()

    # Hitung jumlah korelasi tinggi tiap fitur (untuk memilih yg dihapus)
    high_corr_count = (corr_matrix > 0.95).sum() - 1  # kurangi diagonal

    # BARU: Daftar fitur yang kebal dari penghapusan korelasi otomatis (Fokus H+3)
    WHITELIST = {
        "lag_14", "lag_21", "rawit_lag_14", "rawit_lag_30", 
        "roll_hujan_14", "roll_hujan_21", "roll_hujan_60", "max_hujan_7",
        "ema_7", "ema_30", "ema_crossover_7_30"
    }

    to_drop = set()
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr_matrix.iloc[i, j] > 0.95:
                # Jangan hapus jika masuk whitelist
                if cols[i] in WHITELIST or cols[j] in WHITELIST:
                    continue
                    
                # Hapus fitur yang paling banyak berkorelasi dengan fitur lain
                if cols[i] not in to_drop and cols[j] not in to_drop:
                    if high_corr_count[cols[i]] >= high_corr_count[cols[j]]:
                        to_drop.add(cols[i])
                    else:
                        to_drop.add(cols[j])

    if to_drop:
        log(f"    ! Hapus fitur korelasi tinggi (r > 0.95): {sorted(to_drop)}")
        fitur_dihapus.extend(sorted(to_drop))
        feature_cols = [c for c in feature_cols if c not in to_drop]

    n_akhir = len(feature_cols)
    log(f"    -> Fitur sebelum seleksi : {n_awal}")
    log(f"    -> Fitur dihapus         : {n_awal - n_akhir} fitur")
    log(f"    -> Fitur setelah seleksi : {n_akhir} fitur")
    log(f"    -> Fitur final           : {feature_cols}")
    return feature_cols


# =============================================================================
# BAGIAN 6 — FINAL CLEANING & VALIDASI
# =============================================================================
def final_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    log("\n[6] Final cleaning & validasi...")

    n_awal = len(df)
    df.dropna(subset=[TARGET], inplace=True)

    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    df.dropna(subset=lag_cols, inplace=True)

    # PERBAIKAN LEAKAGE 2: Imputasi median hanya dari bagian train (80% pertama)
    # Alasan: median global menyertakan distribusi data test → bocor ke depan.
    # Solusi: hitung median dari df_train saja, terapkan ke seluruh dataset.
    split_idx_clean = int(len(df) * 0.8)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            median_train = df.iloc[:split_idx_clean][col].median()
            df[col] = df[col].fillna(median_train)

    df.reset_index(drop=True, inplace=True)

    total_missing = df.isna().sum().sum()
    n_negatif_target = (df[TARGET] <= 0).sum()

    log(f"    -> Baris dihapus (warm-up) : {n_awal - len(df)}")
    log(f"    -> Shape final             : {df.shape}")
    log(f"    -> Total missing values    : {total_missing}")
    log(f"    -> Nilai target <= 0       : {n_negatif_target}")

    if total_missing > 0:
        log(f"    ! Kolom masih NaN: {df.columns[df.isna().any()].tolist()}")
    if n_negatif_target > 0:
        log(f"    ! Ada harga negatif/nol pada target — cek data sumber!")

    log("\n" + "=" * 55)
    log("  RINGKASAN DATASET FINAL")
    log("=" * 55)
    log(f"  Jumlah baris    : {len(df):,}")
    log(f"  Jumlah kolom    : {df.shape[1]}")
    log(
        f"  Rentang tanggal : {df['tanggal'].min().date()} -> {df['tanggal'].max().date()}"
    )
    log(f"  Harga Cabai Merah:")
    log(f"    Min  : Rp {df[TARGET].min():>10,.0f}")
    log(f"    Max  : Rp {df[TARGET].max():>10,.0f}")
    log(f"    Mean : Rp {df[TARGET].mean():>10,.0f}")
    log(f"    Std  : Rp {df[TARGET].std():>10,.0f}")
    if "harga_cabai_rawit" in df.columns:
        log(f"  Harga Cabai Rawit:")
        log(f"    Min  : Rp {df['harga_cabai_rawit'].min():>10,.0f}")
        log(f"    Max  : Rp {df['harga_cabai_rawit'].max():>10,.0f}")
        log(f"    Mean : Rp {df['harga_cabai_rawit'].mean():>10,.0f}")
    log("=" * 55)
    return df


# =============================================================================
# BAGIAN 7 — NORMALISASI (MinMaxScaler)
# =============================================================================
def normalisasi(df: pd.DataFrame, feature_cols: list, target_cols: list):
    """
    Normalisasi fitur menggunakan RobustScaler (median + IQR).
    RobustScaler dipilih karena:
      - Tahan terhadap price shock / outlier harga komoditas
      - MinMaxScaler rentan: lonjakan harga baru akan menghasilkan nilai > 1.0
    Scaler di-fit HANYA pada data train untuk mencegah data leakage.
    target_cols (target_h1, h3, h7) ikut disimpan tapi TIDAK dinormalisasi
    karena nilainya harus tetap dalam satuan Rupiah untuk evaluasi model.
    """
    log("\n[7] Normalisasi fitur (RobustScaler — tahan price shock)...")

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx]

    scaler = RobustScaler()
    scaler.fit(df_train[feature_cols])

    # Kolom yang disimpan: tanggal + fitur + target utama + target multi-horizon
    # + rawit sebagai referensi (tidak dinormalisasi)
    keep_cols = ["tanggal"] + feature_cols + [TARGET] + target_cols
    if "harga_cabai_rawit" in df.columns:
        keep_cols.append("harga_cabai_rawit")

    df_scaled = df[keep_cols].copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])

    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    log(f"    -> Fitur dinormalisasi        : {len(feature_cols)}")
    log(f"    -> Target disimpan (asli Rp) : {[TARGET] + target_cols}")
    log(f"    -> Total kolom output         : {df_scaled.shape[1]}")
    log(f"    -> Scaler disimpan ke         : {scaler_path}")
    log(
        f"    -> Median fitur (center)      : min={df_train[feature_cols].median().min():.1f}, "
        f"max={df_train[feature_cols].median().max():.1f}"
    )
    return df_scaled, scaler, feature_cols


# =============================================================================
# BAGIAN 8 — SIMPAN OUTPUT
# =============================================================================
def simpan_output(
    df_asli: pd.DataFrame,
    df_scaled: pd.DataFrame,
    feature_cols: list,
    target_cols: list,
):
    log("\n[8] Menyimpan semua output...")

    split_idx = int(len(df_asli) * 0.8)

    # Dataset asli — simpan semua kolom untuk keperluan EDA & analisis
    df_asli.to_csv(OUTPUT_DIR / "dataset_preprocessed.csv", index=False)
    log(
        f"    + dataset_preprocessed.csv  ({len(df_asli):,} baris, "
        f"{df_asli.shape[1]} kolom, nilai asli)"
    )

    # Catatan: train.csv dan test.csv sudah dihapus karena pipeline XGBoost
    # menggunakan Expanding Window langsung dari dataset_preprocessed.csv

    # Baris paling akhir yang tidak punya target_h7 (7 hari terakhir) dibuang
    df_for_split = df_scaled.dropna(subset=target_cols).reset_index(drop=True)
    log(
        f"    + Baris dibuang (target_h7 NaN): " f"{len(df_scaled) - len(df_for_split)}"
    )

    # Info fitur lengkap (dari nilai asli)
    info = pd.DataFrame(
        {
            "fitur": feature_cols,
            "tipe": [str(df_asli[c].dtype) for c in feature_cols],
            "null_count": [int(df_asli[c].isna().sum()) for c in feature_cols],
            "min_asli": [round(df_asli[c].min(), 2) for c in feature_cols],
            "max_asli": [round(df_asli[c].max(), 2) for c in feature_cols],
            "mean_asli": [round(df_asli[c].mean(), 2) for c in feature_cols],
            "std_asli": [round(df_asli[c].std(), 2) for c in feature_cols],
        }
    )
    info.to_csv(OUTPUT_DIR / "info_fitur.csv", index=False)
    log(f"    + info_fitur.csv     ({len(feature_cols)} fitur setelah seleksi)")

    # Daftar target multi-horizon untuk referensi train.py
    target_info_data = []
    for col in [TARGET] + target_cols:
        if col == TARGET:
            desc = "Harga aktual hari ini (referensi)"
        elif "rawit" in col and "target" in col:
            desc = f"Prediksi Rawit {col.split('_')[-1]} (delta)"
        elif "arah_target" in col:
            desc = f"Arah Prediksi {col.split('_')[-1]} (1=Naik, 0=Turun/Tetap)"
        elif "target_h" in col:
            desc = f"Prediksi Merah {col.split('_')[-1]} (delta)"
        else:
            desc = f"Referensi: {col}"
        target_info_data.append({"target": col, "keterangan": desc})
        
    target_info = pd.DataFrame(target_info_data)
    target_info.to_csv(OUTPUT_DIR / "info_target.csv", index=False)
    log(f"    + info_target.csv    ({len(target_info)} target)")

    laporan_path = OUTPUT_DIR / "laporan_preprocessing.txt"
    with open(laporan_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
    log(f"    + laporan_preprocessing.txt")

    log(f"\n    Semua file tersimpan di: {OUTPUT_DIR}")
    return feature_cols


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("=" * 55)
    log("  PREPROCESSING DATA HARGA CABAI - KOTA PADANG v5")
    log("  (ffill causal | median-train | RobustScaler | multi-horizon)")
    log("=" * 55)

    errors = []
    for label, path in [
        ("cuaca", CUACA_DIR),
        ("harga_cabai", HARGA_DIR),
        ("hari_libur", LIBUR_DIR),
    ]:
        if not path.exists():
            errors.append(f"Folder tidak ada: {path}")
    if errors:
        for e in errors:
            log(f"  x {e}")
        return None, None, None

    df_cuaca = load_cuaca()
    df_harga = load_harga_cabai()
    df_libur = load_hari_libur()
    df_merged = merge_data(df_harga, df_cuaca, df_libur)
    df_feat = feature_engineering(df_merged)
    df_final = final_cleaning(df_feat)

    # Semua kolom yang merupakan target atau referensi inverse transform
    # harus di-exclude dari feature set — jangan pernah jadi input model.
    TARGET_COLS = [
        "target_h1", "target_h3", "target_h7",
        "target_rawit_h1", "target_rawit_h3", "target_rawit_h7",
        "arah_target_h1", "arah_target_h3", "arah_target_h7",
    ]
    # Kolom referensi inverse transform — bukan fitur, hanya dipakai saat training/eval
    REF_COLS = ["harga_hari_ini", "harga_rawit_hari_ini"]

    exclude = ["tanggal", TARGET, "harga_cabai_rawit"] + TARGET_COLS + REF_COLS
    feature_cols = [
        c
        for c in df_final.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df_final[c])
    ]

    log(f"\n[5C] Kandidat fitur sebelum seleksi: {len(feature_cols)} fitur")

    # Seleksi fitur otomatis (hapus redundan & berkorelasi tinggi)
    feature_cols = seleksi_fitur(df_final, feature_cols)

    df_scaled, scaler, feature_cols = normalisasi(df_final, feature_cols, TARGET_COLS + REF_COLS)
    simpan_output(df_final, df_scaled, feature_cols, TARGET_COLS + REF_COLS)

    log("\n[SELESAI] Preprocessing v5 berhasil!")
    log(f"  Target utama    : {TARGET}")
    log(f"  Target merah (delta) : target_h1, target_h3, target_h7")
    log(f"  Target rawit (delta) : target_rawit_h1, target_rawit_h3, target_rawit_h7")
    log(f"  Referensi inverse    : harga_hari_ini, harga_rawit_hari_ini")
    log(f"  Fitur baru cuaca     : lag_hujan_14, lag_hujan_21, roll_hujan_14, roll_hujan_21")
    log(f"  Fitur total          : {len(feature_cols)} fitur siap untuk XGBoost")
    log(f"  Output          : {OUTPUT_DIR}")
    log(f"  Scaler          : {MODEL_DIR / 'scaler.pkl'}")

    return df_final, df_scaled, feature_cols


if __name__ == "__main__":
    df_final, df_scaled, feature_cols = main()
