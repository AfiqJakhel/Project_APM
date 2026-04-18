import json
import glob
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
CUACA_DIR  = DATA_DIR / "cuaca"
HARGA_DIR  = DATA_DIR / "harga_cabai"
LIBUR_DIR  = DATA_DIR / "hari_libur"
OUTPUT_DIR = BASE_DIR / "App" / "output"
MODEL_DIR  = BASE_DIR / "machine learning" / "model XGBoost"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET    = "harga_cabai_merah"
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
            needed    = ["TANGGAL", "TAVG", "RH_AVG", "RR", "SS", "FF_AVG"]
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
    df.rename(columns={
        "TANGGAL": "tanggal", "TAVG": "suhu_rata",
        "RH_AVG": "kelembaban", "RR": "curah_hujan",
        "SS": "lama_penyinaran", "FF_AVG": "kec_angin"
    }, inplace=True)
    df.dropna(subset=["tanggal"], inplace=True)

    numeric_cols = ["suhu_rata", "kelembaban", "curah_hujan", "lama_penyinaran", "kec_angin"]
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
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear").ffill().bfill()

    log(f"    -> Missing sebelum interpolasi : {miss_before}")
    log(f"    -> Missing setelah interpolasi : {df[numeric_cols].isna().sum().to_dict()}")
    log(f"    -> Shape final : {df.shape}")
    log(f"    -> Rentang     : {df['tanggal'].min().date()} s/d {df['tanggal'].max().date()}")
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
                    harga = np.nan if val in ["-", "nan", "", "None"] else \
                        float(val.replace(",", "").replace(".", ""))
                    try:
                        tgl = pd.to_datetime(str(col).strip(), dayfirst=True, errors="coerce")
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

    # Validasi: hapus harga negatif/nol
    harga_cols = ["harga_cabai_merah", "harga_cabai_rawit"]
    for col in harga_cols:
        if col in df.columns:
            n_neg = (df[col] <= 0).sum()
            if n_neg > 0:
                log(f"    ! {col}: {n_neg} nilai <= 0 diganti NaN")
                df.loc[df[col] <= 0, col] = np.nan

    # Deteksi & tangani outlier dengan IQR
    for col in harga_cols:
        if col not in df.columns:
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            log(f"    ! Outlier '{col}': {outliers} nilai di luar "
                f"[{lower:,.0f}, {upper:,.0f}] -> diganti NaN")
            df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan

    # Buat index harian lengkap & interpolasi
    date_range = pd.date_range(df["tanggal"].min(), df["tanggal"].max(), freq="D")
    df_indexed = df.set_index("tanggal").reindex(date_range)
    for col in harga_cols:
        if col in df_indexed.columns:
            n_miss = df_indexed[col].isna().sum()
            df_indexed[col] = df_indexed[col].interpolate(method="time").ffill().bfill()
            log(f"    -> Interpolasi '{col}': {n_miss} hari kosong diisi")
    df = df_indexed.reset_index().rename(columns={"index": "tanggal"})

    log(f"    -> Shape final : {df.shape}")
    log(f"    -> Rentang     : {df['tanggal'].min().date()} s/d {df['tanggal'].max().date()}")
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
                    records.append({
                        "tanggal": tgl,
                        "keterangan_libur": item.get("keterangan", "")
                    })
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

    # Validasi hasil merge
    for col in ["suhu_rata", "kelembaban", "curah_hujan"]:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            if pct > 30:
                log(f"    ! '{col}': {pct:.1f}% missing setelah merge")

    log(f"    -> Shape setelah merge  : {df.shape}")
    log(f"    -> Hari libur ditandai  : {df['is_libur_nasional'].sum()} hari")
    return df


# =============================================================================
# BAGIAN 5 — FEATURE ENGINEERING
# =============================================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    log("\n[5] Feature engineering...")

    # A. Kalender
    df["tahun"]          = df["tanggal"].dt.year
    df["bulan"]          = df["tanggal"].dt.month
    df["hari_bulan"]     = df["tanggal"].dt.day
    df["hari_minggu"]    = df["tanggal"].dt.dayofweek
    df["minggu_ke"]      = df["tanggal"].dt.isocalendar().week.astype(int)
    df["kuartal"]        = df["tanggal"].dt.quarter
    df["is_weekend"]     = (df["hari_minggu"] >= 5).astype(int)
    df["hari_dlm_tahun"] = df["tanggal"].dt.dayofyear
    df["bulan_sin"]      = np.sin(2 * np.pi * df["bulan"] / 12)
    df["bulan_cos"]      = np.cos(2 * np.pi * df["bulan"] / 12)
    df["hari_sin"]       = np.sin(2 * np.pi * df["hari_minggu"] / 7)
    df["hari_cos"]       = np.cos(2 * np.pi * df["hari_minggu"] / 7)

    # B. Event
    lebaran = pd.to_datetime([
        "2022-05-02", "2023-04-22", "2024-04-10",
        "2025-03-30", "2026-03-20"
    ])
    df["is_pra_lebaran"]   = 0
    df["is_lebaran"]       = 0
    df["is_pasca_lebaran"] = 0
    for d in lebaran:
        df.loc[(df["tanggal"] >= d - pd.Timedelta(14)) & (df["tanggal"] < d),
               "is_pra_lebaran"] = 1
        df.loc[(df["tanggal"] >= d) & (df["tanggal"] <= d + pd.Timedelta(2)),
               "is_lebaran"] = 1
        df.loc[(df["tanggal"] > d + pd.Timedelta(2)) &
               (df["tanggal"] <= d + pd.Timedelta(9)),
               "is_pasca_lebaran"] = 1
    df["is_natal_tahunbaru"] = (
        ((df["bulan"] == 12) & (df["hari_bulan"] >= 20)) |
        ((df["bulan"] == 1)  & (df["hari_bulan"] <= 7))
    ).astype(int)
    df["is_musim_panen"] = df["bulan"].isin([5, 6, 7, 11, 12, 1]).astype(int)

    # C. Lag harga
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    if "harga_cabai_rawit" in df.columns:
        for lag in [1, 7]:
            df[f"rawit_lag_{lag}"] = df["harga_cabai_rawit"].shift(lag)

    # D. Rolling statistics
    for w in [3, 7, 14, 30]:
        s = df[TARGET].shift(1)
        df[f"roll_mean_{w}"] = s.rolling(w).mean()
        df[f"roll_std_{w}"]  = s.rolling(w).std()
        df[f"roll_min_{w}"]  = s.rolling(w).min()
        df[f"roll_max_{w}"]  = s.rolling(w).max()

    df["momentum_7"]   = df[TARGET].shift(1) - df[TARGET].shift(7)
    df["momentum_30"]  = df[TARGET].shift(1) - df[TARGET].shift(30)
    df["pct_change_1"] = df[TARGET].pct_change(1)
    df["pct_change_7"] = df[TARGET].pct_change(7)

    # E. Cuaca turunan
    if "suhu_rata" in df.columns:
        df["roll_suhu_7"]  = df["suhu_rata"].rolling(7,  min_periods=1).mean()
        df["roll_suhu_30"] = df["suhu_rata"].rolling(30, min_periods=1).mean()
    if "curah_hujan" in df.columns:
        df["roll_hujan_7"]     = df["curah_hujan"].rolling(7,  min_periods=1).sum()
        df["roll_hujan_30"]    = df["curah_hujan"].rolling(30, min_periods=1).sum()
        df["is_hari_hujan"]    = (df["curah_hujan"] > 1.0).astype(int)
        df["roll_hr_hujan_30"] = df["is_hari_hujan"].rolling(30, min_periods=1).sum()
    if "kelembaban" in df.columns:
        df["roll_kelembaban_7"] = df["kelembaban"].rolling(7, min_periods=1).mean()

    log(f"    -> Total fitur dibuat : {df.shape[1] - 2}")
    return df


# =============================================================================
# BAGIAN 6 — FINAL CLEANING & VALIDASI
# =============================================================================
def final_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    log("\n[6] Final cleaning & validasi...")

    n_awal = len(df)
    df.dropna(subset=[TARGET], inplace=True)

    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    df.dropna(subset=lag_cols, inplace=True)

    cuaca_cols = ["suhu_rata", "kelembaban", "curah_hujan", "lama_penyinaran", "kec_angin"]
    for col in cuaca_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df.reset_index(drop=True, inplace=True)

    total_missing     = df.isna().sum().sum()
    n_negatif_target  = (df[TARGET] <= 0).sum()

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
    log(f"  Jumlah fitur    : {df.shape[1] - 2}")
    log(f"  Rentang tanggal : {df['tanggal'].min().date()} -> {df['tanggal'].max().date()}")
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
def normalisasi(df: pd.DataFrame):
    """
    Normalisasi fitur ke [0,1] menggunakan MinMaxScaler.
    Scaler di-fit HANYA pada data train untuk mencegah data leakage.
    """
    log("\n[7] Normalisasi fitur (MinMaxScaler)...")

    exclude      = ["tanggal", TARGET, "harga_cabai_rawit"]
    feature_cols = [c for c in df.columns
                    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    split_idx = int(len(df) * 0.8)
    df_train  = df.iloc[:split_idx]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_train[feature_cols])

    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])

    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    log(f"    -> Jumlah fitur dinormalisasi : {len(feature_cols)}")
    log(f"    -> Scaler disimpan ke         : {scaler_path}")
    log(f"    -> Range setelah scaling      : "
        f"min={df_scaled[feature_cols].min().min():.3f}, "
        f"max={df_scaled[feature_cols].max().max():.3f}")
    return df_scaled, scaler, feature_cols


# =============================================================================
# BAGIAN 8 — SIMPAN OUTPUT
# =============================================================================
def simpan_output(df_asli: pd.DataFrame, df_scaled: pd.DataFrame, feature_cols: list):
    log("\n[8] Menyimpan semua output...")

    split_idx = int(len(df_asli) * 0.8)

    # Dataset nilai asli (untuk EDA & laporan)
    df_asli.to_csv(OUTPUT_DIR / "dataset_preprocessed.csv", index=False)
    log(f"    + dataset_preprocessed.csv  ({len(df_asli):,} baris, nilai asli)")

    # Dataset ternormalisasi (untuk training XGBoost)
    df_scaled.to_csv(OUTPUT_DIR / "dataset_scaled.csv", index=False)
    log(f"    + dataset_scaled.csv        ({len(df_scaled):,} baris, nilai [0,1])")

    # Train & test split dari data scaled
    df_scaled.iloc[:split_idx].to_csv(OUTPUT_DIR / "train.csv", index=False)
    df_scaled.iloc[split_idx:].to_csv(OUTPUT_DIR / "test.csv",  index=False)
    log(f"    + train.csv  ({split_idx:,} baris, "
        f"s/d {df_scaled.iloc[split_idx-1]['tanggal'].date()})")
    log(f"    + test.csv   ({len(df_scaled)-split_idx:,} baris, "
        f"mulai {df_scaled.iloc[split_idx]['tanggal'].date()})")

    # Info fitur lengkap (nilai asli)
    info = pd.DataFrame({
        "fitur"     : feature_cols,
        "tipe"      : [str(df_asli[c].dtype) for c in feature_cols],
        "null_count": [int(df_asli[c].isna().sum()) for c in feature_cols],
        "min_asli"  : [round(df_asli[c].min(), 2) for c in feature_cols],
        "max_asli"  : [round(df_asli[c].max(), 2) for c in feature_cols],
        "mean_asli" : [round(df_asli[c].mean(), 2) for c in feature_cols],
        "std_asli"  : [round(df_asli[c].std(), 2) for c in feature_cols],
    })
    info.to_csv(OUTPUT_DIR / "info_fitur.csv", index=False)
    log(f"    + info_fitur.csv     ({len(feature_cols)} fitur)")

    # Laporan teks lengkap
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
    log("  PREPROCESSING DATA HARGA CABAI - KOTA PADANG v2")
    log("=" * 55)

    errors = []
    for label, path in [
        ("cuaca", CUACA_DIR),
        ("harga_cabai", HARGA_DIR),
        ("hari_libur", LIBUR_DIR)
    ]:
        if not path.exists():
            errors.append(f"Folder tidak ada: {path}")
    if errors:
        for e in errors:
            log(f"  x {e}")
        return None, None, None

    df_cuaca  = load_cuaca()
    df_harga  = load_harga_cabai()
    df_libur  = load_hari_libur()
    df_merged = merge_data(df_harga, df_cuaca, df_libur)
    df_feat   = feature_engineering(df_merged)
    df_final  = final_cleaning(df_feat)
    df_scaled, scaler, feature_cols = normalisasi(df_final)
    simpan_output(df_final, df_scaled, feature_cols)

    log("\n[SELESAI] Preprocessing berhasil!")
    log(f"  Target   : {TARGET}")
    log(f"  Fitur    : {len(feature_cols)} fitur siap untuk XGBoost")
    log(f"  Output   : {OUTPUT_DIR}")
    log(f"  Scaler   : {MODEL_DIR / 'scaler.pkl'}")

    return df_final, df_scaled, feature_cols


if __name__ == "__main__":
    df_final, df_scaled, feature_cols = main()