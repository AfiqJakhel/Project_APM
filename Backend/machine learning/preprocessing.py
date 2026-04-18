import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI PATH
# ─────────────────────────────────────────────────────────────────────────────
# Path relatif dari lokasi file ini (Backend/machine learning/)
BASE_DIR    = Path(__file__).resolve().parent.parent   # → Backend/
DATA_DIR    = BASE_DIR / "data"
CUACA_DIR   = DATA_DIR / "cuaca"
HARGA_DIR   = DATA_DIR / "harga_cabai"
LIBUR_DIR   = DATA_DIR / "hari_libur"
OUTPUT_DIR  = BASE_DIR / "App" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BAGIAN 1 — LOAD DATA CUACA BMKG
# =============================================================================
def load_cuaca() -> pd.DataFrame:
    """
    Membaca semua file xlsx BMKG dari subfolder 2024/ dan 2025/.
    Format kolom: TANGGAL | TAVG | RH_AVG | RR | SS | FF_AVG
    Missing value BMKG: 8888 (tidak terukur), 9999 (tidak ada data), string kosong
    """
    print("\n[1] Memuat data cuaca BMKG...")

    # Cari semua file xlsx secara rekursif (termasuk subfolder 2024, 2025)
    all_files = sorted(glob.glob(str(CUACA_DIR / "**" / "*.xlsx"), recursive=True))
    print(f"    → Ditemukan {len(all_files)} file cuaca")

    if not all_files:
        raise FileNotFoundError(f"Tidak ada file xlsx di: {CUACA_DIR}")

    frames = []
    for fp in all_files:
        try:
            # Baca langsung — format BMKG: header baris ke-1 (setelah metadata)
            # Cari baris yang mengandung 'TANGGAL' sebagai header
            df_raw = pd.read_excel(fp, header=None)
            header_idx = None
            for i, row in df_raw.iterrows():
                vals = [str(v).strip().upper() for v in row.values]
                if "TANGGAL" in vals:
                    header_idx = i
                    break

            if header_idx is None:
                print(f"    ✗ {Path(fp).name}: header TANGGAL tidak ditemukan")
                continue

            df = pd.read_excel(fp, header=header_idx)
            df.columns = [str(c).strip().upper() for c in df.columns]

            # Pertahankan hanya kolom yang dibutuhkan
            needed = ["TANGGAL", "TAVG", "RH_AVG", "RR", "SS", "FF_AVG"]
            available = [c for c in needed if c in df.columns]
            df = df[available].copy()

            # Hapus baris keterangan di bawah data (baris yang TANGGAL-nya bukan tanggal)
            df = df[df["TANGGAL"].astype(str).str.match(r"\d{2}-\d{2}-\d{4}")]

            frames.append(df)
            print(f"    ✓ {Path(fp).name}: {len(df)} baris")

        except Exception as e:
            print(f"    ✗ {Path(fp).name}: error → {e}")

    if not frames:
        raise ValueError("Tidak ada data cuaca yang berhasil dimuat!")

    df = pd.concat(frames, ignore_index=True)

    # ── Parsing tanggal format DD-MM-YYYY ────────────────────────────────────
    df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], format="%d-%m-%Y", errors="coerce")
    df.rename(columns={
        "TANGGAL" : "tanggal",
        "TAVG"    : "suhu_rata",
        "RH_AVG"  : "kelembaban",
        "RR"      : "curah_hujan",
        "SS"      : "lama_penyinaran",
        "FF_AVG"  : "kec_angin"
    }, inplace=True)

    df.dropna(subset=["tanggal"], inplace=True)

    # ── Ganti kode missing BMKG (8888, 9999) dengan NaN ─────────────────────
    numeric_cols = ["suhu_rata", "kelembaban", "curah_hujan", "lama_penyinaran", "kec_angin"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].replace([8888, 9999, 8888.0, 9999.0], np.nan, inplace=True)

    # ── Deduplikasi: jika satu tanggal muncul dari beberapa stasiun, rata-rata
    df = df.groupby("tanggal")[numeric_cols].mean().reset_index()
    df.sort_values("tanggal", inplace=True)

    # ── Buat index harian lengkap, isi gap dengan interpolasi ────────────────
    date_range = pd.date_range(df["tanggal"].min(), df["tanggal"].max(), freq="D")
    df = df.set_index("tanggal").reindex(date_range).reset_index()
    df.rename(columns={"index": "tanggal"}, inplace=True)
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear").ffill().bfill()

    print(f"    → Shape final : {df.shape}")
    print(f"    → Rentang     : {df['tanggal'].min().date()} s/d {df['tanggal'].max().date()}")
    return df


# =============================================================================
# BAGIAN 2 — LOAD DATA HARGA CABAI
# =============================================================================
def load_harga_cabai() -> pd.DataFrame:
    """
    Membaca file Excel harga cabai format WIDE:
    - Baris  : komoditas (Cabai Merah Keriting, Cabai Rawit Hijau)
    - Kolom  : tanggal dalam format "DD/ MM/ YYYY" atau "DD/MM/YYYY"
    - Harga  : string dengan koma "42,500", missing = "-"

    Output: DataFrame panjang dengan kolom:
    tanggal | harga_cabai_merah | harga_cabai_rawit
    """
    print("\n[2] Memuat data harga cabai...")

    xlsx_files = sorted(glob.glob(str(HARGA_DIR / "*.xlsx")))
    print(f"    → Ditemukan {len(xlsx_files)} file harga")

    if not xlsx_files:
        raise FileNotFoundError(f"Tidak ada file xlsx di: {HARGA_DIR}")

    frames = []
    for fp in xlsx_files:
        try:
            df_raw = pd.read_excel(fp)

            # Temukan kolom komoditas (biasanya kolom ke-2)
            komoditas_col = None
            for col in df_raw.columns:
                sample = df_raw[col].astype(str).str.lower()
                if sample.str.contains("cabai|cabe").any():
                    komoditas_col = col
                    break

            if komoditas_col is None:
                print(f"    ✗ {Path(fp).name}: kolom komoditas tidak ditemukan")
                continue

            # Ambil hanya baris yang berisi komoditas cabai
            df_raw["_ket"] = df_raw[komoditas_col].astype(str).str.strip()
            df_cabai = df_raw[
                df_raw["_ket"].str.lower().str.contains("cabai|cabe", na=False)
            ].copy()

            # Kolom-kolom tanggal: semua kolom kecuali No, Komoditas, _ket
            skip_cols = [komoditas_col, "_ket"]
            # Tambahkan kolom No/index jika ada
            for col in df_raw.columns[:2]:
                if col not in skip_cols:
                    skip_cols.append(col)

            date_cols = [c for c in df_raw.columns if c not in skip_cols]

            # Proses tiap baris komoditas
            for _, row in df_cabai.iterrows():
                nama = row["_ket"].lower()

                # Tentukan jenis cabai
                if "keriting" in nama or ("merah" in nama and "keriting" not in nama):
                    jenis = "harga_cabai_merah"
                elif "rawit" in nama:
                    jenis = "harga_cabai_rawit"
                else:
                    continue

                # Buat series harga per tanggal
                records = []
                for col in date_cols:
                    val = str(row[col]).strip()
                    if val in ["-", "nan", "", "None"]:
                        harga = np.nan
                    else:
                        # Bersihkan format "42,500" → 42500
                        harga = float(val.replace(",", "").replace(".", ""))

                    # Parse tanggal dari nama kolom "03/ 01/ 2022" atau "2022-01-03"
                    try:
                        tgl = pd.to_datetime(str(col).strip(), dayfirst=True,
                                             errors="coerce")
                        if pd.isna(tgl):
                            tgl = pd.to_datetime(str(col).strip(), errors="coerce")
                    except Exception:
                        tgl = pd.NaT

                    if pd.notna(tgl):
                        records.append({"tanggal": tgl, jenis: harga})

                if records:
                    frames.append(pd.DataFrame(records))

            print(f"    ✓ {Path(fp).name}")

        except Exception as e:
            print(f"    ✗ {Path(fp).name}: error → {e}")

    if not frames:
        raise ValueError("Tidak ada data harga yang berhasil dimuat!")

    df = pd.concat(frames, ignore_index=True)

    # Gabungkan baris dengan tanggal yang sama
    df = df.groupby("tanggal").mean().reset_index()
    df.sort_values("tanggal", inplace=True)

    # ── Buat index harian lengkap ────────────────────────────────────────────
    date_range = pd.date_range(df["tanggal"].min(), df["tanggal"].max(), freq="D")
    df = df.set_index("tanggal").reindex(date_range).reset_index()
    df.rename(columns={"index": "tanggal"}, inplace=True)

    harga_cols = ["harga_cabai_merah", "harga_cabai_rawit"]
    df_indexed = df.set_index("tanggal")
    for col in harga_cols:
        if col in df_indexed.columns:
            n_miss = df_indexed[col].isna().sum()
            df_indexed[col] = df_indexed[col].interpolate(method="time").ffill().bfill()
            print(f"    → Interpolasi '{col}': {n_miss} hari kosong diisi")
    df = df_indexed.reset_index()
    print(f"    → Shape final : {df.shape}")
    print(f"    → Rentang     : {df['tanggal'].min().date()} s/d {df['tanggal'].max().date()}")
    return df


# =============================================================================
# BAGIAN 3 — LOAD DATA HARI LIBUR
# =============================================================================
def load_hari_libur() -> pd.DataFrame:
    """
    Membaca semua file JSON hari libur dari folder hari_libur/.
    Format: [{"tanggal": "2022-01-1", "keterangan": "Tahun Baru 2022"}]
    Tanggal tidak konsisten (bisa "2022-01-1" atau "2022-1-1") → pakai errors='coerce'
    """
    print("\n[3] Memuat data hari libur...")

    json_files = sorted(glob.glob(str(LIBUR_DIR / "*.json")))
    print(f"    → Ditemukan {len(json_files)} file JSON")

    if not json_files:
        raise FileNotFoundError(f"Tidak ada file JSON di: {LIBUR_DIR}")

    records = []
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                tgl_str = item.get("tanggal", "")
                ket     = item.get("keterangan", "")
                # pd.to_datetime menangani format tidak konsisten seperti "2022-01-1"
                tgl = pd.to_datetime(tgl_str, errors="coerce")
                if pd.notna(tgl):
                    records.append({"tanggal": tgl, "keterangan_libur": ket})

            print(f"    ✓ {Path(fp).name}: {len(data)} hari libur")

        except Exception as e:
            print(f"    ✗ {Path(fp).name}: error → {e}")

    df_libur = pd.DataFrame(records)
    df_libur["tanggal"] = pd.to_datetime(df_libur["tanggal"])
    df_libur.drop_duplicates(subset=["tanggal"], inplace=True)
    df_libur.sort_values("tanggal", inplace=True)
    df_libur.reset_index(drop=True, inplace=True)

    print(f"    → Total hari libur: {len(df_libur)}")
    print(f"    → Tahun: {sorted(df_libur['tanggal'].dt.year.unique().tolist())}")
    return df_libur

# =============================================================================
# BAGIAN 4 — MERGE SEMUA DATA
# =============================================================================
def merge_data(df_harga: pd.DataFrame,
               df_cuaca: pd.DataFrame,
               df_libur: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ketiga sumber data berdasarkan kolom tanggal.
    Harga cabai sebagai tabel utama (left join).
    """
    print("\n[4] Menggabungkan semua data...")

    # Filter cuaca sesuai rentang harga
    start, end = df_harga["tanggal"].min(), df_harga["tanggal"].max()
    df_cuaca = df_cuaca[
        (df_cuaca["tanggal"] >= start) & (df_cuaca["tanggal"] <= end)
    ].copy()

    # Merge harga ← cuaca
    df = pd.merge(df_harga, df_cuaca, on="tanggal", how="left")

    # Merge ← libur (tambahkan flag is_libur_nasional)
    df_libur_flag = df_libur[["tanggal"]].copy()
    df_libur_flag["is_libur_nasional"] = 1
    df = pd.merge(df, df_libur_flag, on="tanggal", how="left")
    df["is_libur_nasional"] = df["is_libur_nasional"].fillna(0).astype(int)

    df.sort_values("tanggal", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"    → Shape setelah merge : {df.shape}")
    print(f"    → Kolom               : {list(df.columns)}")
    return df

# =============================================================================
# BAGIAN 5 — FEATURE ENGINEERING
# =============================================================================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur-fitur turunan untuk XGBoost:
    - Fitur kalender & siklus
    - Fitur event khusus (Lebaran, Natal, musim panen)
    - Lag harga cabai merah (target utama)
    - Rolling statistics
    - Fitur cuaca turunan
    """
    print("\n[5] Feature engineering...")

    target = "harga_cabai_merah"   # target utama prediksi

    # ── A. Fitur Kalender ────────────────────────────────────────────────────
    df["tahun"]          = df["tanggal"].dt.year
    df["bulan"]          = df["tanggal"].dt.month
    df["hari_bulan"]     = df["tanggal"].dt.day
    df["hari_minggu"]    = df["tanggal"].dt.dayofweek      # 0=Senin, 6=Minggu
    df["minggu_ke"]      = df["tanggal"].dt.isocalendar().week.astype(int)
    df["kuartal"]        = df["tanggal"].dt.quarter
    df["is_weekend"]     = (df["hari_minggu"] >= 5).astype(int)
    df["hari_dlm_tahun"] = df["tanggal"].dt.dayofyear

    # Encoding siklus (sin/cos) — agar model tahu bulan 12 dekat dengan bulan 1
    df["bulan_sin"]  = np.sin(2 * np.pi * df["bulan"] / 12)
    df["bulan_cos"]  = np.cos(2 * np.pi * df["bulan"] / 12)
    df["hari_sin"]   = np.sin(2 * np.pi * df["hari_minggu"] / 7)
    df["hari_cos"]   = np.cos(2 * np.pi * df["hari_minggu"] / 7)

    # ── B. Fitur Event Khusus ────────────────────────────────────────────────
    # Tanggal Idul Fitri 2022-2026
    lebaran = pd.to_datetime([
        "2022-05-02", "2023-04-22", "2024-04-10",
        "2025-03-30", "2026-03-20"
    ])
    df["is_pra_lebaran"]   = 0
    df["is_lebaran"]       = 0
    df["is_pasca_lebaran"] = 0
    for d in lebaran:
        df.loc[(df["tanggal"] >= d - pd.Timedelta(14)) &
               (df["tanggal"] <  d), "is_pra_lebaran"] = 1
        df.loc[(df["tanggal"] >= d) &
               (df["tanggal"] <= d + pd.Timedelta(2)), "is_lebaran"] = 1
        df.loc[(df["tanggal"] >  d + pd.Timedelta(2)) &
               (df["tanggal"] <= d + pd.Timedelta(9)), "is_pasca_lebaran"] = 1

    # Natal & Tahun Baru
    df["is_natal_tahunbaru"] = (
        ((df["bulan"] == 12) & (df["hari_bulan"] >= 20)) |
        ((df["bulan"] == 1)  & (df["hari_bulan"] <= 7))
    ).astype(int)

    # Musim panen cabai di Sumatra Barat (Mei-Juli & Nov-Jan)
    df["is_musim_panen"] = df["bulan"].isin([5, 6, 7, 11, 12, 1]).astype(int)

    # ── C. Lag Harga Cabai Merah ─────────────────────────────────────────────
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f"lag_{lag}"] = df[target].shift(lag)

    # Lag harga cabai rawit (fitur tambahan)
    if "harga_cabai_rawit" in df.columns:
        for lag in [1, 7]:
            df[f"rawit_lag_{lag}"] = df["harga_cabai_rawit"].shift(lag)

    # ── D. Rolling Statistics Harga ──────────────────────────────────────────
    for w in [3, 7, 14, 30]:
        s = df[target].shift(1)
        df[f"roll_mean_{w}"]  = s.rolling(w).mean()
        df[f"roll_std_{w}"]   = s.rolling(w).std()
        df[f"roll_min_{w}"]   = s.rolling(w).min()
        df[f"roll_max_{w}"]   = s.rolling(w).max()

    # Momentum harga
    df["momentum_7"]   = df[target].shift(1) - df[target].shift(7)
    df["momentum_30"]  = df[target].shift(1) - df[target].shift(30)
    df["pct_change_1"] = df[target].pct_change(1)
    df["pct_change_7"] = df[target].pct_change(7)

    # ── E. Fitur Cuaca Turunan ───────────────────────────────────────────────
    if "suhu_rata" in df.columns:
        df["roll_suhu_7"]  = df["suhu_rata"].rolling(7,  min_periods=1).mean()
        df["roll_suhu_30"] = df["suhu_rata"].rolling(30, min_periods=1).mean()

    if "curah_hujan" in df.columns:
        df["roll_hujan_7"]      = df["curah_hujan"].rolling(7,  min_periods=1).sum()
        df["roll_hujan_30"]     = df["curah_hujan"].rolling(30, min_periods=1).sum()
        df["is_hari_hujan"]     = (df["curah_hujan"] > 1.0).astype(int)
        df["roll_hr_hujan_30"]  = df["is_hari_hujan"].rolling(30, min_periods=1).sum()

    if "kelembaban" in df.columns:
        df["roll_kelembaban_7"] = df["kelembaban"].rolling(7, min_periods=1).mean()

    total_fitur = df.shape[1] - 2   # kurangi tanggal dan target
    print(f"    → Total fitur dibuat : {total_fitur}")
    return df

# =============================================================================
# BAGIAN 6 — FINAL CLEANING
# =============================================================================
def final_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pembersihan akhir:
    - Hapus baris di awal yang belum punya cukup data lag (warm-up 30 hari)
    - Isi sisa NaN pada fitur cuaca
    - Validasi tidak ada NaN pada target dan lag
    """
    print("\n[6] Final cleaning...")

    target    = "harga_cabai_merah"
    n_awal    = len(df)

    # Hapus baris tanpa target
    df.dropna(subset=[target], inplace=True)

    # Hapus baris warm-up lag (biasanya 30 baris pertama)
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    df.dropna(subset=lag_cols, inplace=True)

    # Isi sisa NaN cuaca dengan forward-fill lalu backward-fill
    cuaca_cols = ["suhu_rata", "kelembaban", "curah_hujan",
                  "lama_penyinaran", "kec_angin"]
    for col in cuaca_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Isi sisa NaN rolling dengan median kolom
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    df.reset_index(drop=True, inplace=True)

    print(f"    → Baris dihapus (warm-up lag) : {n_awal - len(df)}")
    print(f"    → Shape final                 : {df.shape}")
    print(f"    → Sisa missing values         : {df.isna().sum().sum()}")

    # ── Ringkasan statistik target ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RINGKASAN DATASET FINAL")
    print("=" * 55)
    print(f"  Jumlah baris     : {len(df):,}")
    print(f"  Jumlah fitur     : {df.shape[1] - 2}")
    print(f"  Rentang tanggal  : {df['tanggal'].min().date()} → {df['tanggal'].max().date()}")
    print(f"  Harga Cabai Merah:")
    print(f"    Min  : Rp {df[target].min():>10,.0f}")
    print(f"    Max  : Rp {df[target].max():>10,.0f}")
    print(f"    Mean : Rp {df[target].mean():>10,.0f}")
    print(f"    Std  : Rp {df[target].std():>10,.0f}")
    if "harga_cabai_rawit" in df.columns:
        print(f"  Harga Cabai Rawit:")
        print(f"    Min  : Rp {df['harga_cabai_rawit'].min():>10,.0f}")
        print(f"    Max  : Rp {df['harga_cabai_rawit'].max():>10,.0f}")
        print(f"    Mean : Rp {df['harga_cabai_rawit'].mean():>10,.0f}")
    print("=" * 55)
    return df

# =============================================================================
# BAGIAN 7 — SIMPAN OUTPUT
# =============================================================================
def simpan_output(df: pd.DataFrame):
    """
    Menyimpan dataset ke folder App/output/:
    - dataset_preprocessed.csv  : dataset lengkap
    - train.csv                 : 80% data awal (time-based)
    - test.csv                  : 20% data akhir (time-based)
    - info_fitur.csv            : ringkasan semua fitur
    """
    print("\n[7] Menyimpan output...")

    target       = "harga_cabai_merah"
    feature_cols = [c for c in df.columns if c not in ["tanggal", target,
                                                        "harga_cabai_rawit"]]

    # Dataset lengkap
    df.to_csv(OUTPUT_DIR / "dataset_preprocessed.csv", index=False)
    print(f"    ✓ dataset_preprocessed.csv  ({len(df):,} baris)")

    # Train/test split berbasis waktu (80/20)
    split_idx = int(len(df) * 0.8)
    df.iloc[:split_idx].to_csv(OUTPUT_DIR / "train.csv", index=False)
    df.iloc[split_idx:].to_csv(OUTPUT_DIR / "test.csv",  index=False)
    print(f"    ✓ train.csv  ({split_idx:,} baris, "
          f"s/d {df.iloc[split_idx-1]['tanggal'].date()})")
    print(f"    ✓ test.csv   ({len(df)-split_idx:,} baris, "
          f"mulai {df.iloc[split_idx]['tanggal'].date()})")

    # Info fitur
    info = pd.DataFrame({
        "fitur"      : feature_cols,
        "tipe"       : [str(df[c].dtype) for c in feature_cols],
        "null_count" : [int(df[c].isna().sum()) for c in feature_cols],
        "min"        : [round(df[c].min(), 2) if pd.api.types.is_numeric_dtype(df[c])
                        else None for c in feature_cols],
        "max"        : [round(df[c].max(), 2) if pd.api.types.is_numeric_dtype(df[c])
                        else None for c in feature_cols],
        "mean"       : [round(df[c].mean(), 2) if pd.api.types.is_numeric_dtype(df[c])
                        else None for c in feature_cols],
    })
    info.to_csv(OUTPUT_DIR / "info_fitur.csv", index=False)
    print(f"    ✓ info_fitur.csv ({len(feature_cols)} fitur)")

    print(f"\n    Semua file tersimpan di: {OUTPUT_DIR}")
    return feature_cols

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 55)
    print("  PREPROCESSING DATA HARGA CABAI - KOTA PADANG")
    print("=" * 55)

    # Cek keberadaan folder data
    errors = []
    if not CUACA_DIR.exists():
        errors.append(f"Folder tidak ada: {CUACA_DIR}")
    if not HARGA_DIR.exists():
        errors.append(f"Folder tidak ada: {HARGA_DIR}")
    if not LIBUR_DIR.exists():
        errors.append(f"Folder tidak ada: {LIBUR_DIR}")
    if errors:
        print("\n[ERROR] Cek kembali struktur folder:")
        for e in errors:
            print(f"  ✗ {e}")
        return None, None

    # Jalankan semua tahap
    df_cuaca  = load_cuaca()
    df_harga  = load_harga_cabai()
    df_libur  = load_hari_libur()

    df_merged = merge_data(df_harga, df_cuaca, df_libur)
    df_feat   = feature_engineering(df_merged)
    df_final  = final_cleaning(df_feat)

    feature_cols = simpan_output(df_final)

    print("\n✅ Preprocessing selesai!")
    print(f"   Target   : harga_cabai_merah")
    print(f"   Fitur    : {len(feature_cols)} fitur siap untuk XGBoost")
    print(f"   Output   : Backend/App/output/")

    return df_final, feature_cols

if __name__ == "__main__":
    df_final, feature_cols = main()