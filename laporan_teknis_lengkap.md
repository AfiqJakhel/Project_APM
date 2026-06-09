# Laporan Teknis: Sistem Machine Learning Prediksi Harga Cabai di Kota Padang

**Judul Lengkap:** Implementasi Machine Learning untuk Prediksi Fluktuasi Harga Cabai di Kota Padang Berbasis Web sebagai Upaya Pengendalian Inflasi Daerah Menggunakan Metode XGBoost

**Versi Sistem:** v5 (Preprocessing), v6 Unified (Training), v2.0.0 (API)

**Teknologi:** Python 3.11 · XGBoost · FastAPI · Next.js (App Router) · Chart.js

---

## Gambaran Umum Sistem

Sistem ini merupakan platform prediksi harga pangan berbasis web yang dirancang khusus untuk memantau dan memperkirakan pergerakan harga **Cabai Merah Keriting (CMK)** dan **Cabai Rawit Merah (CRM)** di pasar tradisional Kota Padang, Sumatera Barat. Data harga bersumber dari Pusat Informasi Harga Pangan Strategis (PIHPS) Nasional, diperkaya dengan data cuaca harian dari Badan Meteorologi, Klimatologi, dan Geofisika (BMKG) Stasiun Padang Panjang.

Model menggunakan algoritma **XGBoost (eXtreme Gradient Boosting)** dengan pendekatan prediksi multi-horizon: **H+1** (esok hari), **H+3** (tiga hari ke depan), dan **H+7** (tujuh hari ke depan). Untuk setiap komoditas dan setiap horizon, dilatih satu model XGBoost yang independen — total 6 model aktif dalam sistem.

Target prediksi yang digunakan adalah **log-return** (bukan harga absolut), yaitu:

```
target = ln(harga[t+H] / harga[t])
```

Pendekatan ini membuat model belajar *perubahan relatif* harga, bukan sekadar mengingat nilai harga terakhir. Hasil prediksi kemudian dikonversi kembali ke satuan Rupiah melalui inverse transform:

```
P_prediksi = P_hari_ini × exp(log_return_prediksi)
```

---

# BAGIAN 1 — PREPROCESSING DATA

Pipeline preprocessing diimplementasikan dalam file `preprocessing.py` (Backend/machine_learning/). Seluruh alur dieksekusi secara sekuensial melalui fungsi `main()` dan menghasilkan dataset bersih yang siap digunakan untuk training.

---

## Step 1.1 — Pemuatan dan Inspeksi Data Awal

**Tujuan:**
Memuat seluruh data mentah dari tiga sumber berbeda (harga cabai, cuaca BMKG, hari libur nasional) ke dalam memori sebagai DataFrame Pandas, sekaligus melakukan inspeksi awal terhadap kualitas data yang diterima.

**Penjelasan Rinci:**
Sistem membaca data dari tiga direktori terpisah yang dikonfigurasi melalui konstanta path di awal skrip. Data cuaca BMKG tersimpan dalam format `.xlsx` dengan struktur yang tidak seragam antar-stasiun dan antar-tahun — header bisa berada di baris mana pun dalam file. Untuk menangani hal ini, sistem melakukan pencarian dinamis terhadap baris yang mengandung kata kunci `"TANGGAL"` sebagai penanda header. Data harga cabai juga berformat `.xlsx` dengan kolom tanggal tersebar horizontal (format wide/pivot), sehingga sistem harus melakukan *reshape* dari format wide ke format long (time-series harian). Data hari libur tersimpan dalam format JSON.

**Kode:**

```python
# preprocessing.py — Konfigurasi path dan konstanta global sistem

import json, glob, joblib, warnings
import numpy as np
import pandas as pd
from hijridate import Gregorian, Hijri
from pathlib import Path
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
CUACA_DIR = DATA_DIR / "raw" / "cuaca"
HARGA_DIR = DATA_DIR / "raw" / "harga_cabai"
LIBUR_DIR = DATA_DIR / "raw" / "hari_libur"
OUTPUT_DIR = DATA_DIR / "processed"
MODEL_DIR  = BASE_DIR / "machine_learning" / "output" / "xgboost_models"

TARGET = "harga_cabai_merah"
```

```python
# preprocessing.py — Fungsi load_cuaca(): deteksi header dinamis

def load_cuaca() -> pd.DataFrame:
    log("\n[1] Memuat data cuaca BMKG...")
    all_files = sorted(glob.glob(str(CUACA_DIR / "**" / "*.xlsx"), recursive=True))
    log(f"    -> Ditemukan {len(all_files)} file cuaca")

    frames = []
    for fp in all_files:
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

    df = pd.concat(frames, ignore_index=True)
    df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], format="%d-%m-%Y", errors="coerce")
    df.rename(columns={
        "TANGGAL": "tanggal", "TAVG": "suhu_rata", "RH_AVG": "kelembaban",
        "RR": "curah_hujan", "SS": "lama_penyinaran", "FF_AVG": "kec_angin",
    }, inplace=True)
    df.dropna(subset=["tanggal"], inplace=True)
    return df
```

**Catatan Teknis:**
Nilai kode `8888` dan `9999` pada data BMKG merupakan kode sentinel yang menandakan data tidak terukur atau rusak — nilainya secara eksplisit diganti dengan `NaN` sebelum pemrosesan lebih lanjut. Kode sentinel ini umum dijumpai dalam dataset cuaca BMKG Indonesia dan *tidak boleh* dibiarkan sebagai nilai numerik karena akan bias secara signifikan terhadap statistik fitur.

---

## Step 1.2 — Penanganan Missing Values

**Tujuan:**
Mengisi celah data (missing values) yang terjadi akibat hari libur pasar, weekend, data cuaca tidak terukur, atau ketidaktersediaan laporan harga pada hari tertentu, menggunakan metode imputasi yang bersifat kausal (tidak melanggar asumsi time-series).

**Penjelasan Rinci:**
Terdapat dua jenis missing values dalam sistem ini. Pertama, missing values pada data cuaca BMKG — terjadi karena gangguan alat ukur, hari tanpa pengamatan, atau format nilai sentinel (8888/9999). Kedua, missing values pada data harga cabai — terjadi karena pasar tutup di hari libur atau akhir pekan, sehingga tidak ada transaksi yang dicatat.

Untuk **kedua jenis missing values**, sistem secara konsisten menggunakan metode **forward-fill (`ffill`)** murni — nilai kosong diisi dengan nilai terakhir yang valid di masa lalu. Metode ini dipilih karena bersifat **kausal (anti-leakage)**: ia hanya menggunakan informasi yang sudah tersedia pada saat `t`, tanpa "melihat ke depan" ke nilai `t+1` atau seterusnya.

Alternatif yang umum — `interpolate(method='linear')` — secara implisit membutuhkan nilai `t+1` untuk menentukan nilai yang diinterpolasi di titik `t`, sehingga menyebabkan **data leakage** dalam konteks time-series. Oleh karena itu, interpolasi linear dinonaktifkan secara eksplisit.

**Kode:**

```python
# preprocessing.py — Imputasi ffill kausal pada data cuaca (anti-leakage)

numeric_cols = ["suhu_rata", "kelembaban", "curah_hujan", "lama_penyinaran", "kec_angin"]

# Ganti sentinel BMKG dengan NaN
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col].replace([8888, 9999, 8888.0, 9999.0], np.nan, inplace=True)

miss_before = df[numeric_cols].isna().sum().to_dict()

# Reindex ke kalender harian penuh — hari tanpa data jadi NaN
date_range = pd.date_range(df["tanggal"].min(), df["tanggal"].max(), freq="D")
df = df.set_index("tanggal").reindex(date_range).reset_index()
df.rename(columns={"index": "tanggal"}, inplace=True)

# PERBAIKAN LEAKAGE 1: Ganti interpolate(linear) -> ffill murni
# Alasan: interpolasi linear membutuhkan nilai t+1 untuk mengisi t (bocor ke depan).
# ffill = "cuaca hari ini = cuaca terakhir yang tercatat" -> kausal 100%.
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].ffill().bfill()
```

```python
# preprocessing.py — Imputasi ffill untuk harga cabai (hari libur/weekend)

for col in ["harga_cabai_merah", "harga_cabai_rawit"]:
    if col in df_indexed.columns:
        n_miss = df_indexed[col].isna().sum()
        # PERBAIKAN 1: Gunakan ffill (bukan interpolate) untuk hari libur/weekend
        # Alasan: pasar tutup di hari libur -> harga tidak bergerak,
        # bukan naik/turun bertahap. ffill = "harga tetap seperti hari kerja terakhir"
        df_indexed[col] = df_indexed[col].ffill().bfill()
        log(f"    -> ffill '{col}': {n_miss} hari libur/weekend diisi harga terakhir")
```

**Catatan Teknis:**
Penanganan outlier harga juga dilakukan pada tahap ini — harga dengan nilai `<= 0` dihapus karena tidak valid secara ekonomi, dan harga di luar batas `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]` diganti dengan NaN sebelum diimputasi. Ini mencegah harga yang salah catat (misalnya angka 0 akibat sel kosong di Excel sumber) masuk ke model.

---

## Step 1.3 — Feature Engineering

**Tujuan:**
Menghasilkan representasi fitur yang kaya dan informatif dari data mentah, sehingga model XGBoost mampu menangkap pola musiman, tren, momentum, dan dampak faktor eksternal (cuaca, hari libur) terhadap harga cabai.

**Penjelasan Rinci:**
Feature engineering dalam sistem ini mencakup delapan kelompok fitur yang dirancang berdasarkan pengetahuan domain harga pangan Indonesia:

- **A. Fitur Kalender** — menangkap pola musiman dan periodik
- **B. Fitur Event Khusus** — menangkap lonjakan harga di sekitar hari raya dan musim panen
- **C. Lag Harga** — nilai historis harga pada titik waktu tertentu di masa lalu
- **D. Rolling Statistics & EMA** — statistik pergerakan untuk menangkap tren dan volatilitas
- **E. Momentum** — laju perubahan harga dalam jangka pendek
- **F. Fitur Tren & Volatilitas** — rasio tren dan koefisien variasi
- **G. Fitur Cuaca** — dampak akumulasi curah hujan terhadap produksi dan distribusi cabai
- **H. Fitur Arah Pergerakan** — sinyal teknikal berbasis data historis, semua menggunakan `shift(1)` untuk anti-leakage

**Kode:**

```python
# preprocessing.py — Kelompok A: Fitur kalender dan siklus

df["bulan"]         = df["tanggal"].dt.month
df["day_of_week"]   = df["tanggal"].dt.dayofweek
df["day_of_month"]  = df["tanggal"].dt.day
df["week_of_year"]  = df["tanggal"].dt.isocalendar().week.astype(int)
df["kuartal"]       = df["tanggal"].dt.quarter
df["is_weekend"]    = (df["tanggal"].dt.dayofweek >= 5).astype(int)
df["is_awal_bulan"] = (df["tanggal"].dt.day <= 7).astype(int)

# Deteksi Ramadan menggunakan konversi Hijriyah
def check_ramadan(date):
    try:
        h = Gregorian(date.year, date.month, date.day).to_hijri()
        return 1 if h.month == 9 else 0
    except OverflowError:
        return 0
df["is_ramadan"] = df["tanggal"].apply(check_ramadan)

# Enkode siklus bulanan dengan sin/cos (menggantikan bulan mentah)
df["bulan_sin"] = np.sin(2 * np.pi * df["bulan"] / 12)
df["bulan_cos"] = np.cos(2 * np.pi * df["bulan"] / 12)

# Enkode siklus mingguan untuk horizon H+3
df["minggu_ke"]  = df["tanggal"].dt.isocalendar().week.astype(int)
df["minggu_sin"] = np.sin(2 * np.pi * df["minggu_ke"] / 52)
df["minggu_cos"] = np.cos(2 * np.pi * df["minggu_ke"] / 52)
df.drop(columns=["minggu_ke"], inplace=True)
```

```python
# preprocessing.py — Kelompok B: Fitur event khusus (Lebaran, musim panen)

lebaran = pd.to_datetime(
    ["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-30", "2026-03-20"]
)
df["is_pra_lebaran"]   = 0
df["is_lebaran"]       = 0
df["is_pasca_lebaran"] = 0
for d in lebaran:
    df.loc[(df["tanggal"] >= d - pd.Timedelta(14)) & (df["tanggal"] < d),
           "is_pra_lebaran"] = 1
    df.loc[(df["tanggal"] >= d) & (df["tanggal"] <= d + pd.Timedelta(2)),
           "is_lebaran"] = 1
    df.loc[(df["tanggal"] > d + pd.Timedelta(2)) &
           (df["tanggal"] <= d + pd.Timedelta(9)), "is_pasca_lebaran"] = 1

df["is_natal_tahunbaru"] = (
    ((df["bulan"] == 12) & (df["tanggal"].dt.day >= 20)) |
    ((df["bulan"] == 1)  & (df["tanggal"].dt.day <= 7))
).astype(int)

# Musim panen cabai Sumatera Barat: Mei, Jun, Jul, Nov, Des, Jan
df["is_musim_panen"] = df["bulan"].isin([5, 6, 7, 11, 12, 1]).astype(int)

# Jarak hari ke Lebaran terdekat (dinamis via hijridate)
df["days_to_lebaran"] = df["tanggal"].apply(get_days_to_lebaran)
```

```python
# preprocessing.py — Kelompok C & D: Lag harga dan rolling statistics

# Lag harga cabai merah (selektif)
for lag in [1, 7, 14, 21, 30]:
    df[f"lag_{lag}"] = df[TARGET].shift(lag)

# Lag harga rawit sebagai sinyal pasar substitusi
if "harga_cabai_rawit" in df.columns:
    df["rawit_lag_1"]  = df["harga_cabai_rawit"].shift(1)
    df["rawit_lag_7"]  = df["harga_cabai_rawit"].shift(7)
    df["rawit_lag_14"] = df["harga_cabai_rawit"].shift(14)
    df["rawit_lag_30"] = df["harga_cabai_rawit"].shift(30)
    # Rasio spread harga merah vs rawit (sinyal koreksi pasar)
    df["rasio_merah_rawit"] = (
        df[TARGET] / df["harga_cabai_rawit"].replace(0, np.nan)
    ).fillna(1.0)

# Rolling mean, std, dan EMA (window 7 dan 30 hari)
# s_base = shift(1) untuk mencegah harga hari ini masuk ke window
s_base = df[TARGET].shift(1)
for w in [7, 30]:
    df[f"roll_mean_{w}"] = s_base.rolling(w).mean()
    df[f"roll_std_{w}"]  = s_base.rolling(w).std()
    df[f"ema_{w}"]       = s_base.ewm(span=w, adjust=False).mean()
    # Sinyal Anomali Pasar (Z-Score)
    df[f"zscore_{w}"]    = (df[TARGET] - df[f"roll_mean_{w}"]) / (df[f"roll_std_{w}"] + 1e-8)

df["ema_crossover_7_30"] = df["ema_7"] - df["ema_30"]
```

```python
# preprocessing.py — Kelompok G: Fitur cuaca dengan lag akumulasi

if "curah_hujan" in df.columns:
    # Dampak curah hujan baru terasa 2-4 minggu kemudian (siklus panen cabai)
    df["lag_hujan_14"]  = df["curah_hujan"].shift(14)
    df["lag_hujan_21"]  = df["curah_hujan"].shift(21)
    df["roll_hujan_14"] = df["curah_hujan"].rolling(14, min_periods=1).sum()
    df["roll_hujan_21"] = df["curah_hujan"].rolling(21, min_periods=1).sum()
    df["roll_hujan_30"] = df["curah_hujan"].rolling(30, min_periods=1).sum()
    # Siklus tanam cabai ~60-90 hari (dari tanam hingga panen)
    df["roll_hujan_60"] = df["curah_hujan"].rolling(60, min_periods=1).sum()
    # Hujan >50mm/hari: cabai busuk & distribusi terganggu (dampak 1-3 hari)
    df["hujan_ekstrem_3hari"] = (
        (df["curah_hujan"] > 50).rolling(3, min_periods=1).sum().astype(int)
    )
    df["max_hujan_7"]   = df["curah_hujan"].rolling(7, min_periods=1).max()
    df["is_hari_hujan"] = (df["curah_hujan"] > 1.0).astype(int)
```

```python
# preprocessing.py — Kelompok H: Fitur arah pergerakan (ANTI-LEAKAGE)

# PENTING: Semua fitur arah menggunakan shift(1) — hanya melihat masa lalu.
harga_shift1 = df[TARGET].shift(1)  # harga kemarin sebagai basis

# Persentase perubahan (scale-free, lebih informatif dari selisih Rp)
df["pct_change_1"] = harga_shift1.pct_change(1) * 100
df["pct_change_3"] = harga_shift1.pct_change(3) * 100
df["pct_change_7"] = harga_shift1.pct_change(7) * 100

# Arah historis per hari (bebas leakage — semua pakai data masa lalu saja)
df["arah_lag1"] = np.sign(df[TARGET].shift(1) - df[TARGET].shift(2))
df["arah_lag2"] = np.sign(df[TARGET].shift(2) - df[TARGET].shift(3))
df["arah_lag3"] = np.sign(df[TARGET].shift(3) - df[TARGET].shift(4))

# Proporsi hari naik dalam 7 dan 14 hari terakhir
diff_safe = df[TARGET].shift(1).diff()
df["prop_naik_7"]  = (diff_safe > 0).rolling(7,  min_periods=3).mean()
df["prop_naik_14"] = (diff_safe > 0).rolling(14, min_periods=7).mean()
```

**Catatan Teknis:**
Penggunaan `shift(1)` secara konsisten pada seluruh kelompok fitur C, D, E, dan H adalah kunci pencegahan data leakage. Tanpa `shift(1)`, rolling window yang dihitung pada titik waktu `t` akan menyertakan nilai harga `t` itu sendiri — yang setara dengan memberi tahu model jawabannya sebelum ia harus membuat prediksi.

---

## Step 1.4 — Pembuatan Target Variabel (Multi-Horizon)

**Tujuan:**
Membuat variabel target (label) untuk tiga horizon prediksi (H+1, H+3, H+7) dalam format log-return, sehingga model belajar memprediksi *perubahan relatif* harga, bukan nilai harga absolut.

**Penjelasan Rinci:**
Sistem menggunakan **log-return** sebagai target, bukan delta harga absolut (Rp) maupun harga langsung. Keunggulan log-return:

1. **Scale-free** — Rp5.000 dari basis Rp25.000 (20%) berbeda signifikan dari Rp5.000 dari basis Rp80.000 (6%)
2. **Distribusi lebih normal** — memudahkan XGBoost mempelajari pola perubahan
3. **Simetris** — naik 10% kemudian turun 10% kembali mendekati titik awal
4. **Anti-dominasi lag_1** — model dipaksa belajar *faktor* perubahan, bukan sekadar mengingat harga kemarin

Target dibuat menggunakan operasi `shift(-k)` yang secara matematis berarti "harga k hari ke depan" dari perspektif baris saat ini.

**Kode:**

```python
# preprocessing.py — Pembuatan target log-return multi-horizon

# Simpan harga hari ini sebagai referensi untuk inverse transform saat inference
df["harga_hari_ini"] = df[TARGET]

# Hindari log(0) atau log(negatif) — ganti dengan NaN
harga_safe = df[TARGET].replace(0, np.nan)

# Target merah: log-return H+1, H+3, H+7
df["target_h1"] = np.log(df[TARGET].shift(-1) / harga_safe)  # log-return H+1
df["target_h3"] = np.log(df[TARGET].shift(-3) / harga_safe)  # log-return H+3
df["target_h7"] = np.log(df[TARGET].shift(-7) / harga_safe)  # log-return H+7

# Target klasifikasi arah (1=naik, 0=turun/tetap)
df["arah_target_h1"] = (df["target_h1"] > 0).astype(int)
df["arah_target_h3"] = (df["target_h3"] > 0).astype(int)
df["arah_target_h7"] = (df["target_h7"] > 0).astype(int)

# Pastikan target klasifikasi ikut NaN jika target regresi NaN
df.loc[df["target_h1"].isna(), "arah_target_h1"] = np.nan
df.loc[df["target_h3"].isna(), "arah_target_h3"] = np.nan
df.loc[df["target_h7"].isna(), "arah_target_h7"] = np.nan

# Target rawit: log-return H+1, H+3, H+7
if "harga_cabai_rawit" in df.columns:
    df["harga_rawit_hari_ini"] = df["harga_cabai_rawit"]
    rawit_safe = df["harga_cabai_rawit"].replace(0, np.nan)
    df["target_rawit_h1"] = np.log(df["harga_cabai_rawit"].shift(-1) / rawit_safe)
    df["target_rawit_h3"] = np.log(df["harga_cabai_rawit"].shift(-3) / rawit_safe)
    df["target_rawit_h7"] = np.log(df["harga_cabai_rawit"].shift(-7) / rawit_safe)
```

**Catatan Teknis:**
Kolom `harga_hari_ini` dan `harga_rawit_hari_ini` adalah **kolom referensi**, bukan fitur input model. Kolom ini wajib disertakan dalam dataset karena digunakan saat fase training untuk melakukan inverse transform prediksi log-return kembali ke satuan Rupiah. Kolom ini secara eksplisit dikecualikan dari `feature_cols` melalui daftar `non_features` pada kelas `KomoditasConfig` di `train.py`.

---

## Step 1.5 — Seleksi Fitur Otomatis

**Tujuan:**
Mengurangi dimensionalitas fitur secara otomatis dengan menghapus fitur yang redundan atau hampir tidak bervariasi, sehingga model terhindar dari *curse of dimensionality* dan waktu training dapat dipersingkat.

**Penjelasan Rinci:**
Seleksi fitur dilakukan dalam dua tahap:

1. **Variance Threshold** — fitur dengan standar deviasi < 0,01 dihapus karena hampir konstan dan tidak informatif
2. **Korelasi Threshold** — pasangan fitur dengan korelasi Pearson absolut > 0,95 dianggap redundan. Dari setiap pasangan, fitur yang memiliki korelasi tinggi lebih banyak dengan fitur lain yang dihapus

Terdapat **whitelist** fitur penting yang kebal dari penghapusan otomatis — misalnya `lag_14`, `lag_21`, `roll_hujan_14/21/60`, dan `ema_7/30`.

**Kode:**

```python
# preprocessing.py — Seleksi fitur otomatis dua tahap

def seleksi_fitur(df: pd.DataFrame, feature_cols: list) -> list:
    n_awal = len(feature_cols)

    # Tahap 1: Variance threshold
    low_var = [c for c in feature_cols if df[c].std() < 0.01]
    if low_var:
        log(f"    ! Hapus fitur hampir konstan (std < 0.01): {low_var}")
        feature_cols = [c for c in feature_cols if c not in low_var]

    # Tahap 2: Korelasi antar fitur (r > 0.95)
    corr_matrix     = df[feature_cols].corr().abs()
    high_corr_count = (corr_matrix > 0.95).sum() - 1  # kurangi diagonal

    # Fitur yang kebal dari penghapusan korelasi otomatis
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
                if cols[i] in WHITELIST or cols[j] in WHITELIST:
                    continue  # lindungi fitur whitelist
                if cols[i] not in to_drop and cols[j] not in to_drop:
                    if high_corr_count[cols[i]] >= high_corr_count[cols[j]]:
                        to_drop.add(cols[i])
                    else:
                        to_drop.add(cols[j])

    if to_drop:
        feature_cols = [c for c in feature_cols if c not in to_drop]

    log(f"    -> Fitur sebelum seleksi : {n_awal}")
    log(f"    -> Fitur setelah seleksi : {len(feature_cols)}")
    return feature_cols
```

**Catatan Teknis:**
Seleksi fitur dilakukan **setelah** feature engineering selesai dan **sebelum** normalisasi, agar korelasi yang dihitung mencerminkan data asli (belum di-scale). Seleksi berdasarkan korelasi pada data yang sudah di-scale bisa menghasilkan keputusan yang berbeda dan tidak konsisten.

---

## Step 1.6 — Scaling / Normalisasi

**Tujuan:**
Menstandarisasi skala seluruh fitur input menggunakan `RobustScaler` agar perbedaan satuan tidak mendominasi proses pembelajaran model, dengan menjamin bahwa scaler hanya dipelajari dari data training.

**Penjelasan Rinci:**
Sistem menggunakan **`RobustScaler`** dari scikit-learn, yang melakukan normalisasi berdasarkan **median dan IQR (Interquartile Range)**:

```
x_scaled = (x - median) / IQR
```

`RobustScaler` dipilih di atas `MinMaxScaler` atau `StandardScaler` karena ketahanannya terhadap outlier — yang sangat relevan untuk data harga komoditas pangan yang rentan terhadap *price shock*. Aturan anti-leakage diterapkan secara ketat: scaler di-fit **hanya pada 80% pertama data (training set)**.

**Kode:**

```python
# preprocessing.py — Normalisasi RobustScaler (anti-leakage)

def normalisasi(df: pd.DataFrame, feature_cols: list, target_cols: list):
    """
    Normalisasi fitur menggunakan RobustScaler (median + IQR).
    Scaler di-fit HANYA pada data train untuk mencegah data leakage.
    target_cols (target_h1, h3, h7) TIDAK dinormalisasi.
    """
    log("\n[7] Normalisasi fitur (RobustScaler — tahan price shock)...")

    # Anti-leakage: fit hanya pada 80% pertama (training set)
    split_idx = int(len(df) * 0.8)
    df_train  = df.iloc[:split_idx]

    scaler = RobustScaler()
    scaler.fit(df_train[feature_cols])  # <- HANYA dari training set

    keep_cols = ["tanggal"] + feature_cols + [TARGET] + target_cols
    if "harga_cabai_rawit" in df.columns:
        keep_cols.append("harga_cabai_rawit")

    df_scaled = df[keep_cols].copy()
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])  # transform semua

    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    log(f"    -> Fitur dinormalisasi        : {len(feature_cols)}")
    log(f"    -> Target disimpan (asli log) : {[TARGET] + target_cols}")
    log(f"    -> Scaler disimpan ke         : {scaler_path}")
    return df_scaled, scaler, feature_cols
```

**Catatan Teknis:**
Dalam pipeline training actual (`train.py`), model XGBoost dilatih langsung dari `dataset_preprocessed.csv` (nilai asli sebelum scaling) — karena XGBoost berbasis tree tidak sensitif terhadap skala fitur. Scaler `.pkl` yang disimpan tetap diperlukan jika di masa depan digunakan model lain yang sensitif terhadap skala.

---

## Step 1.7 — Final Cleaning dan Output Akhir Preprocessing

**Tujuan:**
Membersihkan baris-baris yang masih mengandung NaN akibat efek warm-up lag features, melakukan imputasi akhir anti-leakage, dan menyimpan seluruh artefak output ke disk.

**Penjelasan Rinci:**
Setelah feature engineering, baris-baris awal dataset akan mengandung NaN pada kolom lag (misalnya `lag_30` tidak memiliki nilai untuk 30 hari pertama). Baris-baris ini dihapus secara eksplisit. Imputasi median untuk kolom yang masih NaN dilakukan menggunakan **median dari training set saja** (80% pertama) — bukan median global — untuk menghindari bocornya informasi dari test set.

**Kode:**

```python
# preprocessing.py — Final cleaning dan imputasi median anti-leakage

def final_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    n_awal = len(df)
    df.dropna(subset=[TARGET], inplace=True)  # hapus baris tanpa harga target

    # Hapus baris warm-up lag (lag_1 hingga lag_30 butuh 30 hari data awal)
    lag_cols = [c for c in df.columns if c.startswith("lag_")]
    df.dropna(subset=lag_cols, inplace=True)

    # PERBAIKAN LEAKAGE 2: Imputasi median hanya dari bagian train (80% pertama)
    split_idx_clean = int(len(df) * 0.8)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            median_train = df.iloc[:split_idx_clean][col].median()
            df[col] = df[col].fillna(median_train)

    df.reset_index(drop=True, inplace=True)
    log(f"    -> Baris dihapus (warm-up) : {n_awal - len(df)}")
    log(f"    -> Shape final             : {df.shape}")
    return df
```

```python
# preprocessing.py — Penyimpanan output akhir preprocessing

def simpan_output(df_asli, df_scaled, feature_cols, target_cols):
    # Dataset asli — semua kolom untuk EDA dan training
    df_asli.to_csv(OUTPUT_DIR / "dataset_preprocessed.csv", index=False)

    # Info fitur lengkap
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

    laporan_path = OUTPUT_DIR / "laporan_preprocessing.txt"
    with open(laporan_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
```

**Catatan Teknis:**
`dataset_preprocessed.csv` adalah satu-satunya file perantara antara preprocessing dan training. Training script (`train.py`) membaca file ini secara langsung tanpa memerlukan file `train.csv` atau `test.csv` yang terpisah — pembagian train/test dilakukan secara dinamis di dalam loop expanding window.

---

## Ringkasan Alur Bagian 1 — Preprocessing Data

| Step | Nama | Input | Output/Artefak |
|------|------|-------|---------------|
| 1.1 | Pemuatan Data | File `.xlsx` BMKG, PIHPS, `.json` hari libur | 3 DataFrame mentah |
| 1.2 | Penanganan Missing Values | DataFrame mentah + kode sentinel | DataFrame bersih (ffill kausal) |
| 1.3 | Feature Engineering | DataFrame bersih | +50 kolom fitur baru (lag, rolling, cuaca, event) |
| 1.4 | Pembuatan Target | Kolom harga + `shift(-k)` | `target_h1`, `target_h3`, `target_h7` (log-return) |
| 1.5 | Seleksi Fitur | Semua kandidat fitur | Daftar `feature_cols` (fitur terpilih) |
| 1.6 | Scaling | `feature_cols` dari training set | `scaler.pkl` (RobustScaler) |
| 1.7 | Output Akhir | DataFrame final | `dataset_preprocessed.csv`, `info_fitur.csv` |

---

# BAGIAN 2 — TRAINING MODEL

Pipeline training diimplementasikan dalam file `train.py` (Backend/machine_learning/). Script ini melatih model XGBoost secara terpisah untuk setiap komoditas (merah dan rawit) dan setiap horizon (H+1, H+3, H+7), menghasilkan total 6 model final.

---

## Step 2.1 — Arsitektur Model: Satu Model per Horizon

**Tujuan:**
Memisahkan proses training untuk setiap horizon prediksi dan setiap komoditas, sehingga setiap model dapat dioptimalkan secara independen untuk karakteristik prediksi jangka pendek vs. jangka menengah yang berbeda.

**Penjelasan Rinci:**
Pendekatan **Direct Multi-Output** digunakan — yaitu membuat model terpisah untuk setiap horizon, berbeda dengan pendekatan *recursive*. Alasannya:

1. Model untuk H+7 membutuhkan fitur dengan lag lebih panjang yang tidak relevan untuk H+1
2. Kesalahan prediksi tidak terakumulasi (tidak ada error propagation)
3. Hyperparameter dapat di-tune secara independen per horizon

Seluruh konfigurasi komoditas diorganisir dalam kelas `KomoditasConfig`.

**Kode:**

```python
# train.py — Kelas konfigurasi komoditas dan inisialisasi pipeline

class KomoditasConfig:
    def __init__(self, nama, folder_output, target_utama, target_map,
                 non_features, ref_inverse):
        self.nama          = nama
        self.folder_output = folder_output
        self.model_dir     = BASE_DIR / "machine_learning" / "output" / folder_output
        self.plots_dir     = self.model_dir / "plots"
        self.metrics_dir   = self.model_dir / "metrics"
        self.target_utama  = target_utama
        self.target_map    = target_map    # {"h1": "target_h1", ...}
        self.non_features  = non_features  # kolom yang TIDAK boleh jadi fitur
        self.ref_inverse   = ref_inverse   # referensi inverse transform

CONFIGS = {
    "merah": KomoditasConfig(
        nama="Cabai Merah",
        folder_output="expanding_window_merah",
        target_utama="harga_cabai_merah",
        target_map={"h1": "target_h1", "h3": "target_h3", "h7": "target_h7"},
        non_features=[
            "tanggal", "harga_cabai_rawit", "harga_cabai_merah",
            "target_h1", "target_h3", "target_h7",
            "target_rawit_h1", "target_rawit_h3", "target_rawit_h7",
            "arah_target_h1", "arah_target_h3", "arah_target_h7",
            "harga_hari_ini", "harga_rawit_hari_ini",
        ],
        ref_inverse="harga_hari_ini",
    ),
    "rawit": KomoditasConfig(
        nama="Cabai Rawit",
        folder_output="expanding_window_rawit",
        target_utama="harga_cabai_rawit",
        target_map={
            "rawit_h1": "target_rawit_h1",
            "rawit_h3": "target_rawit_h3",
            "rawit_h7": "target_rawit_h7",
        },
        non_features=[
            "tanggal", "harga_cabai_rawit", "harga_cabai_merah",
            "target_h1", "target_h3", "target_h7",
            "target_rawit_h1", "target_rawit_h3", "target_rawit_h7",
            "arah_target_h1", "arah_target_h3", "arah_target_h7",
            "harga_hari_ini", "harga_rawit_hari_ini",
        ],
        ref_inverse="harga_rawit_hari_ini",
    )
}
```

**Catatan Teknis:**
Daftar `non_features` berperan kritis sebagai "gerbang keamanan" — memastikan bahwa kolom referensi inverse transform (`harga_hari_ini`) dan kolom target lainnya tidak pernah menjadi fitur input model. Kebocoran kolom-kolom ini ke fitur akan menyebabkan model memiliki performa yang sangat tinggi di training namun gagal total saat inference.

---

## Step 2.2 — Cross-Validation dengan Expanding Window

**Tujuan:**
Mengevaluasi kinerja model secara realistis menggunakan teknik validasi silang yang menghormati struktur temporal data time-series, sehingga model tidak pernah "melihat masa depan" saat dievaluasi.

**Penjelasan Rinci:**
**Expanding Window Cross-Validation** bekerja sebagai berikut: pada window ke-`k`, model dilatih pada data dari hari pertama hingga hari ke-`n` (expanding set), dan dievaluasi pada data hari ke-`n+1` hingga `n+30` (test window tetap 30 hari). Pada window berikutnya, batas training diperluas 30 hari ke depan.

Konfigurasi:
- **`MIN_TRAIN_DAYS = 365`** — minimum 1 tahun data untuk training pertama
- **`STEP_DAYS = 30`** — training set bertambah 30 hari setiap window
- **`TEST_DAYS = 30`** — setiap window dievaluasi pada 30 hari ke depan

**Kode:**

```python
# train.py — Pembentukan expanding windows

MIN_TRAIN_DAYS = 365  # minimum 1 tahun data untuk training pertama
STEP_DAYS      = 30   # training set bertambah 30 hari setiap window
TEST_DAYS      = 30   # setiap window dievaluasi pada 30 hari ke depan

def buat_expanding_windows(df: pd.DataFrame) -> list[dict]:
    log(f"\n[2] Membuat expanding windows ...")
    total     = len(df)
    windows   = []
    window_id = 1
    train_end = MIN_TRAIN_DAYS

    while True:
        test_end = train_end + TEST_DAYS
        if test_end > total:
            break
        windows.append({
            "window"               : window_id,
            "train_start"          : 0,           # selalu dari awal (expanding)
            "train_end"            : train_end,
            "test_start"           : train_end,
            "test_end"             : test_end,
            "tanggal_test_mulai"   : df["tanggal"].iloc[train_end],
            "tanggal_test_selesai" : df["tanggal"].iloc[test_end - 1],
        })
        train_end += STEP_DAYS
        window_id += 1

    log(f"    -> Total windows : {len(windows)}")
    return windows
```

```python
# train.py — Eksekusi evaluasi per window dengan inverse transform

def expanding_window_eval(df, windows, target_col, label, cfg, params) -> pd.DataFrame:
    records = []
    global_y_true, global_y_pred = [], []

    for w in windows:
        df_tr = df.iloc[w["train_start"]:w["train_end"]].dropna(subset=[target_col])
        df_te = df.iloc[w["test_start"]:w["test_end"]].dropna(subset=[target_col])
        if len(df_tr) < 100 or len(df_te) < 5: continue

        X_train, y_train = pisahkan_fitur(df_tr, target_col, cfg.non_features)
        X_test,  y_test  = pisahkan_fitur(df_te, target_col, cfg.non_features)

        # Inverse transform: log-return -> harga absolut
        ref_test   = df_te[cfg.ref_inverse].values       # harga hari ini (referensi)
        y_true_abs = ref_test * np.exp(y_test.values)    # harga aktual H+k

        # Prediksi XGBoost
        y_pred_log_xgb, _ = train_xgb_window(X_train, y_train, X_test, params)
        y_pred_abs_xgb    = ref_test * np.exp(y_pred_log_xgb)  # inverse transform
        y_pred_abs_xgb    = np.clip(y_pred_abs_xgb, 1000, None) # sanity check min Rp1000

        m_xgb = hitung_semua_metrik(y_true_abs, y_pred_abs_xgb, ref_test)
        global_y_true.extend(y_true_abs)
        global_y_pred.extend(y_pred_abs_xgb)
        records.append({"window": w["window"], **{f"xgb_{k}": v for k, v in m_xgb.items()}})

    df_res = pd.DataFrame(records)
    if len(df_res) > 0:
        # R2 Global — dihitung dari seluruh OOS predictions (bukan rata-rata per window)
        global_r2   = r2_score(global_y_true, global_y_pred)
        global_rmse = mean_squared_error(global_y_true, global_y_pred) ** 0.5
        df_res["xgb_R2"]   = global_r2
        df_res["xgb_RMSE"] = global_rmse
    return df_res
```

**Catatan Teknis:**
Metrik R² dan RMSE yang dilaporkan dihitung secara **global** dari seluruh prediksi out-of-sample, bukan rata-rata dari sub-window. Ini mencegah bias dari window dengan ukuran test set kecil yang mendominasi rata-rata per-window.

---

## Step 2.3 — Hyperparameter Tuning dengan Optuna

**Tujuan:**
Menemukan kombinasi hyperparameter XGBoost optimal untuk setiap label secara efisien menggunakan optimasi Bayesian berbasis TPE (Tree-structured Parzen Estimator).

**Penjelasan Rinci:**
Sistem menggunakan library **Optuna** dengan konfigurasi:
- **Sampler**: `TPESampler` (Bayesian, berbasis model probabilistik)
- **Pruner**: `MedianPruner` — menghentikan trial yang performanya buruk lebih awal
- **Jumlah Trial**: 200 trial, dengan timeout 300 detik
- **Validasi**: `TimeSeriesSplit` dengan `n_splits=10` dan `gap` sesuai horizon
- **Objective**: Meminimalkan rata-rata MSE di 10 fold validasi
- **Caching**: Hasil tuning disimpan ke `best_params_{label}.json`

**Kode:**

```python
# train.py — Hyperparameter tuning dengan Optuna

def tuning_final(df, target_col, label, cfg) -> dict:
    from sklearn.model_selection import TimeSeriesSplit
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    cache_path = cfg.metrics_dir / f"best_params_{label}.json"

    # Cek cache — lewati tuning jika sudah ada hasil sebelumnya
    USE_CACHE = "--no-cache" not in sys.argv
    if USE_CACHE and cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
            log(f"    -> Cache ditemukan, lewati tuning ({label})")
            return data.get("params", data)

    df_valid = df.dropna(subset=[target_col]).copy()
    X, y     = pisahkan_fitur(df_valid, target_col, cfg.non_features)

    # Gap sesuai horizon prediksi (mencegah overlap temporal)
    import re
    match = re.search(r'h(\d+)$', label)
    gap   = int(match.group(1)) if match else 1
    tscv  = TimeSeriesSplit(n_splits=10, gap=gap)

    def objective(trial):
        param = {
            "n_estimators"     : trial.suggest_int("n_estimators", 100, 300),
            "max_depth"        : trial.suggest_int("max_depth", 2, 4),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample"        : trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "min_child_weight" : trial.suggest_int("min_child_weight", 5, 15),
            "gamma"            : trial.suggest_float("gamma", 0.0, 0.01),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 0.0, 0.1),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 0.1, 5.0),
        }
        decay = trial.suggest_float("decay", 0.0005, 0.005)

        mses = []
        for step, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            w_tr       = buat_sample_weights(len(train_idx), decay)
            model      = XGBRegressor(objective="reg:squarederror",
                                       tree_method="hist", random_state=42, **param)
            model.fit(X_tr, y_tr, sample_weight=w_tr,
                      eval_set=[(X_va, y_va)], verbose=False)
            mse = mean_squared_error(y_va, model.predict(X_va))
            mses.append(mse)
            trial.report(mse, step)
            if trial.should_prune(): raise optuna.TrialPruned()
        return np.mean(mses)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42)
    study   = optuna.create_study(direction="minimize", sampler=sampler,
                                   pruner=MedianPruner())
    study.optimize(objective, n_trials=200, timeout=300)

    best   = study.best_params
    output = {
        "params"       : best,
        "best_cv_mse"  : round(study.best_value, 6),
        "label"        : label,
        "horizon_days" : gap,
        "n_splits"     : 10,
        "n_iter"       : 200,
        "tuning_date"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(cache_path, "w") as f:
        json.dump(output, f, indent=4)
    return best
```

**Catatan Teknis:**
Parameter `max_depth` dibatasi pada rentang 2–4 untuk mencegah *overfitting*. Ini relevan karena dataset harga pangan relatif kecil (~1.000 baris) dan pola harga cabai cenderung smooth/kontinu, bukan kompleks.

---

## Step 2.4 — Training Final Model

**Tujuan:**
Melatih model final menggunakan seluruh dataset yang tersedia dengan parameter terbaik hasil tuning, untuk menghasilkan model yang memaksimalkan informasi dari semua data historis.

**Penjelasan Rinci:**
Model final dilatih menggunakan **90% data terakhir** sebagai training set dan **10% terakhir** sebagai internal validation set. Sistem menggunakan **time-decay sample weighting** — data yang lebih baru mendapat bobot lebih tinggi.

Formula: `w(i) = exp(decay * (i - n + 1))`, di mana `decay` juga merupakan hyperparameter yang di-tune oleh Optuna. Nilai `decay = 0.001` berarti data 1000 hari lalu bobotnya ~0,37 dibanding data terbaru.

**Kode:**

```python
# train.py — Sample weighting berbasis time-decay

def buat_sample_weights(n: int, decay: float = 0.001) -> np.ndarray:
    """
    Data terbaru mendapat bobot mendekati 1.0,
    data terlama mendapat bobot mendekati exp(-decay * n).
    decay=0.001 -> data 1000 hari lalu bobotnya ~0.37
    """
    indices = np.arange(n)
    weights = np.exp(decay * (indices - n + 1))
    return weights / weights.mean()  # normalisasi agar rata-rata bobot = 1.0
```

```python
# train.py — Training model final dengan parameter terbaik

def latih_model_final(df, target_col, label, cfg, params):
    df_valid  = df.dropna(subset=[target_col]).copy()
    X, y      = pisahkan_fitur(df_valid, target_col, cfg.non_features)

    # 90% data untuk training, 10% untuk early stopping validation
    split_val  = int(len(X) * 0.9)
    X_tr, y_tr = X.iloc[:split_val], y.iloc[:split_val]

    decay      = params.get("decay", 0.001)
    w_tr       = buat_sample_weights(len(X_tr), decay)
    xgb_params = {k: v for k, v in params.items() if k != "decay"}

    model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=42,
        n_estimators      = xgb_params.get("n_estimators", 150),
        max_depth         = xgb_params.get("max_depth", 2),
        learning_rate     = xgb_params.get("learning_rate", 0.05),
        subsample         = xgb_params.get("subsample", 0.6),
        colsample_bytree  = xgb_params.get("colsample_bytree", 0.6),
        min_child_weight  = xgb_params.get("min_child_weight", 5),
        gamma             = xgb_params.get("gamma", 0.5),
        reg_alpha         = xgb_params.get("reg_alpha", 1.0),
        reg_lambda        = xgb_params.get("reg_lambda", 10.0),
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X.iloc[split_val:], y.iloc[split_val:])],
              verbose=False)

    # Simpan model dan daftar fitur
    ts         = datetime.now().strftime("%Y%m%d")
    model_path = cfg.model_dir / f"model_final_{label}_{ts}.pkl"
    joblib.dump(model, model_path)

    feat_path  = cfg.model_dir / f"feature_cols_{label}.json"
    with open(feat_path, "w") as f:
        json.dump(list(X.columns), f, indent=2)

    log(f"    -> Model Final: {model_path.name} | Fitur: {len(X.columns)}")
    return model, list(X.columns)
```

**Catatan Teknis:**
Model disimpan dalam format `.pkl` menggunakan `joblib`. Penamaan file `model_final_{label}_{tanggal}.pkl` memungkinkan versioning berdasarkan tanggal training — `predictor.py` selalu memilih file terbaru menggunakan `sorted(..., reverse=True)[0]`.

---

## Step 2.5 — Evaluasi Model

**Tujuan:**
Mengukur kinerja model secara komprehensif menggunakan berbagai metrik evaluasi yang mencakup akurasi prediksi nilai (regresi) maupun akurasi prediksi arah pergerakan harga.

**Penjelasan Rinci:**
Sistem menggunakan enam metrik evaluasi yang saling melengkapi:

| Metrik | Definisi | Keterangan |
|--------|----------|-----------|
| **MAE** | Mean Absolute Error (Rp) | Rata-rata selisih absolut prediksi vs aktual |
| **RMSE** | Root Mean Square Error (Rp) | Lebih sensitif terhadap outlier besar |
| **MAPE** | Mean Absolute Percentage Error (%) | Akurasi relatif |
| **sMAPE** | Symmetric MAPE (%) | Versi MAPE yang simetris |
| **R²** | Koefisien Determinasi | Seberapa besar varians data yang dijelaskan model |
| **DA** | Directional Accuracy (%) | Akurasi prediksi arah (naik/turun) harga |

**Kode:**

```python
# train.py — Perhitungan semua metrik evaluasi

def hitung_da(y_true: np.ndarray, y_pred: np.ndarray,
              ref_val: np.ndarray) -> float:
    """Directional Accuracy: proporsi prediksi yang benar arahnya."""
    diff_true = y_true - ref_val   # arah sebenarnya (+ = naik, - = turun)
    diff_pred = y_pred - ref_val   # arah prediksi
    mask = diff_true != 0
    if np.sum(mask) == 0:
        return float("nan")
    return float(np.mean(np.sign(diff_true[mask]) == np.sign(diff_pred[mask])) * 100)


def hitung_semua_metrik(y_true: np.ndarray, y_pred: np.ndarray,
                         ref_val: np.ndarray) -> dict:
    # sMAPE: symmetric MAPE (lebih stabil dari MAPE standar)
    denom  = np.abs(y_true) + np.abs(y_pred)
    mask   = denom != 0
    smape  = float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

    # MAPE standar (abaikan titik di mana y_true = 0)
    mask_mape = y_true != 0
    mape = float(np.mean(np.abs(
        (y_true[mask_mape] - y_pred[mask_mape]) / y_true[mask_mape]
    )) * 100)

    return {
        "MAE"  : float(mean_absolute_error(y_true, y_pred)),
        "RMSE" : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE" : mape,
        "sMAPE": smape,
        "R2"   : float(r2_score(y_true, y_pred)),
        "DA"   : hitung_da(y_true, y_pred, ref_val),
    }
```

**Catatan Teknis:**
Performa model yang telah divalidasi: H+1 (MAE Rp1.699, R²=0,98), H+3 (MAE Rp3.678, R²=0,93), H+7 (MAE Rp6.947, R²=0,75). Ini mencerminkan tren umum bahwa akurasi menurun seiring bertambahnya horizon prediksi — wajar secara statistik karena ketidakpastian semakin besar pada horizon yang lebih jauh.

---

## Step 2.6 — Analisis SHAP (Feature Importance)

**Tujuan:**
Memahami kontribusi setiap fitur terhadap prediksi model menggunakan SHAP (SHapley Additive exPlanations), untuk memastikan model belajar pola yang bermakna secara domain dan bukan sekadar mengingat noise.

**Penjelasan Rinci:**
`shap.TreeExplainer` digunakan karena secara khusus dioptimalkan untuk model berbasis tree (termasuk XGBoost). Ia menghitung SHAP values yang mewakili kontribusi marginal setiap fitur terhadap prediksi individual.

Sistem secara otomatis mendeteksi apakah `lag_1` masih mendominasi. Jika `lag_1` adalah fitur #1 dan rasionya terhadap fitur #2 lebih dari 2x, model dianggap masih berperilaku seperti *Naive Forecast*. Penggunaan target log-return dirancang untuk mengatasi masalah ini.

**Kode:**

```python
# train.py — Analisis SHAP TreeExplainer

def shap_final(model, X_sample: pd.DataFrame, label: str, cfg: KomoditasConfig):
    if not SHAP_AVAILABLE: return
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Simpan SHAP summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="dot",
                          max_display=15, show=False)
        plt.title(f"SHAP Summary — {cfg.nama} ({label.upper()})")
        plt.savefig(cfg.plots_dir / f"shap_summary_{label}.png", bbox_inches="tight")
        plt.close()

        # Hitung mean absolute SHAP values dan log top-10 fitur
        mean_abs = np.abs(shap_values).mean(axis=0)
        df_shap  = pd.DataFrame({
            "fitur"     : X_sample.columns,
            "shap_mean" : mean_abs
        }).sort_values("shap_mean", ascending=False)

        log(f"    -> SHAP Top 10 Fitur:")
        for i, row in df_shap.head(10).iterrows():
            log(f"         {row['fitur']:<20}: {row['shap_mean']:.4f}")

        # Deteksi dominasi lag_1 (Naive Forecast behavior)
        if df_shap.iloc[0]["fitur"] == "lag_1":
            ratio = df_shap.iloc[0]["shap_mean"] / df_shap.iloc[1]["shap_mean"]
            log(f"    -> SHAP: lag_1 masih dominan ({ratio:.1f}x dari fitur #2)")
        else:
            top_fitur = df_shap.iloc[0]["fitur"]
            log(f"    -> SHAP: lag_1 bukan #1 (Detrending berhasil. Fitur top: {top_fitur})")

    except Exception as e:
        log(f"    ! SHAP error: {e}")
```

**Catatan Teknis:**
SHAP values dihitung pada 500 baris terbaru dari dataset untuk efisiensi komputasi. 500 baris terakhir merepresentasikan periode terkini yang paling relevan untuk mengevaluasi perilaku model di kondisi pasar yang sedang terjadi.

---

## Step 2.7 — Penyimpanan Model dan Artefak

**Tujuan:**
Menyimpan seluruh artefak yang diperlukan untuk inference tanpa harus menjalankan ulang proses training.

**Penjelasan Rinci:**
Struktur direktori output:

```
machine_learning/output/
├── xgboost_models/
│   └── scaler.pkl
├── expanding_window_merah/
│   ├── model_final_h1_20260601.pkl
│   ├── model_final_h3_20260601.pkl
│   ├── model_final_h7_20260601.pkl
│   ├── feature_cols_h1.json
│   ├── feature_cols_h3.json
│   ├── feature_cols_h7.json
│   ├── plots/  (SHAP + expanding window charts)
│   └── metrics/
│       ├── best_params_h1.json
│       └── ringkasan_expanding_window_merah.csv
└── expanding_window_rawit/
    └── (struktur identik)
```

**Kode:**

```python
# train.py — Penyimpanan ringkasan metrik dan log

def laporan_ringkasan(all_results: dict, cfg: KomoditasConfig):
    """Simpan ringkasan metrik ke CSV dan log ke file teks."""
    rows = []
    for label, df_res in all_results.items():
        if len(df_res) == 0: continue
        xgb_m = df_res['xgb_MAE'].mean()
        na_m  = df_res['naive_MAE'].mean()
        beat  = (df_res['xgb_MAE'] < df_res['naive_MAE']).mean() * 100
        rows.append({
            "horizon"        : label,
            "xgb_MAE_mean"   : round(xgb_m, 2),
            "naive_MAE_mean" : round(na_m, 2),
            "xgb_RMSE_mean"  : round(df_res["xgb_RMSE"].mean(), 2),
            "xgb_MAPE_mean"  : round(df_res["xgb_MAPE"].mean(), 4),
            "xgb_sMAPE_mean" : round(df_res["xgb_sMAPE"].mean(), 4),
            "xgb_R2"         : round(df_res["xgb_R2"].mean(), 4),
            "xgb_DA_mean"    : round(df_res["xgb_DA"].mean(), 2),
            "beat_naive_%"   : round(beat, 1)
        })

    if rows:
        pd.DataFrame(rows).to_csv(
            cfg.metrics_dir / f"ringkasan_{cfg.folder_output}.csv", index=False
        )

    # Simpan log teks seluruh proses training
    with open(cfg.model_dir / f"log_{cfg.folder_output}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_LINES))
```

**Catatan Teknis:**
Format `.pkl` melalui `joblib` mempertahankan semua state internal model XGBoost, termasuk bobot tree, threshold split, dan konfigurasi early stopping. Penamaan file dengan timestamp memungkinkan versioning berdasarkan tanggal training — `predictor.py` selalu memilih file terbaru.

---

## Ringkasan Alur Bagian 2 — Training Model

| Step | Nama | Input | Output/Artefak |
|------|------|-------|---------------|
| 2.1 | Arsitektur Model | `KomoditasConfig` | 6 konfigurasi model (merah: h1/h3/h7, rawit: h1/h3/h7) |
| 2.2 | Expanding Window | Dataset + konfigurasi | Evaluasi OOS per window (MAE, RMSE, R², DA) |
| 2.3 | Hyperparameter Tuning | Optuna 200 trial | `best_params_{label}.json` |
| 2.4 | Training Final | Dataset penuh + best params | `model_final_{label}_{ts}.pkl` |
| 2.5 | Evaluasi Metrik | Prediksi OOS | `ringkasan_expanding_window_{komoditas}.csv` |
| 2.6 | Analisis SHAP | Model + 500 baris terbaru | `shap_summary_{label}.png`, log top-10 fitur |
| 2.7 | Penyimpanan | Semua artefak | `.pkl` model, `.json` feature_cols, `scaler.pkl` |

---

# BAGIAN 3 — PENGGUNAAN API DI FRONTEND

Sistem API dibangun menggunakan **FastAPI** di sisi backend dan **Next.js (App Router)** di sisi frontend. Komunikasi antara kedua komponen menggunakan protokol HTTP REST dengan format JSON.

---

## Step 3.1 — Endpoint FastAPI: Struktur dan Routing

**Tujuan:**
Menyediakan antarmuka HTTP yang terstruktur dan terdokumentasi untuk mengekspos fungsionalitas model prediksi, data historis, dan informasi sistem kepada frontend.

**Penjelasan Rinci:**
Aplikasi FastAPI diinisialisasi di `app/main.py` dengan deskripsi sistem, versioning, dan konfigurasi CORS. Response schema untuk setiap endpoint didefinisikan menggunakan **Pydantic v2 BaseModel** di `app/schemas/predict.py`.

**Daftar Endpoint Utama:**

| Method | Path | Fungsi |
|--------|------|--------|
| `GET` | `/api/predict/prediksi/{horizon}` | Prediksi cabai merah satu horizon |
| `GET` | `/api/predict/prediksi` | Prediksi cabai merah semua horizon |
| `GET` | `/api/predict/prediksi/rawit/{horizon}` | Prediksi cabai rawit satu horizon |
| `GET` | `/api/predict/prediksi/rawit` | Prediksi cabai rawit semua horizon |
| `GET` | `/api/predict/prediksi/semua` | Prediksi merah + rawit sekaligus |
| `GET` | `/api/predict/harga/historis` | Data historis untuk grafik |
| `GET` | `/api/predict/model/metrik` | Metrik akurasi semua model |
| `GET` | `/api/predict/health` | Health check sistem |

**Kode:**

```python
# app/main.py — Inisialisasi FastAPI dan konfigurasi CORS

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict, history, dashboard

app = FastAPI(
    title       = "API Prediksi Harga Cabai — Kota Padang",
    description = (
        "XGBoost multi-horizon prediction H+1, H+3, H+7\n\n"
        "Implementasi Machine Learning untuk Prediksi Fluktuasi "
        "Harga Cabai di Kota Padang Berbasis Web"
    ),
    version  = "2.0.0",
    docs_url = "/docs",
    lifespan = lifespan,
)

# CORS — izinkan frontend mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = [
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # Next.js default
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(predict.router,   prefix="/api/predict",   tags=["Prediksi"])
app.include_router(history.router,   prefix="/api/history",   tags=["Historis"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
```

```python
# app/schemas/predict.py — Pydantic response schemas

from typing import Optional
from pydantic import BaseModel

class PrediksiOtomatisResponse(BaseModel):
    """Response dari GET /api/predict/prediksi/{horizon}."""
    status           : str
    horizon          : str
    keterangan       : str
    tanggal_prediksi : str
    prediksi_rp      : float
    model_version    : str
    arah_prediksi    : Optional[str]   = None
    perubahan_persen : Optional[float] = None
    data_status      : Optional[str]   = None
    data_tanggal     : Optional[str]   = None
    cuaca_digunakan  : Optional[CuacaInfo] = None

class DashboardResponse(BaseModel):
    """Response dari GET /api/dashboard/."""
    tanggal_update    : str
    harga_hari_ini    : float
    harga_min_30hari  : float
    harga_max_30hari  : float
    harga_rata_30hari : float
    tren              : str               # "naik" | "turun" | "stabil"
    prediksi_h1       : Optional[float] = None
    prediksi_h3       : Optional[float] = None
    prediksi_h7       : Optional[float] = None
    status_inflasi    : str               # "normal" | "waspada" | "kritis"
    harga_hari_ini_rawit    : Optional[float] = None
    prediksi_rawit_h1       : Optional[float] = None
    prediksi_rawit_h3       : Optional[float] = None
    prediksi_rawit_h7       : Optional[float] = None
    status_inflasi_rawit    : Optional[str]   = None
    status_model      : bool
    n_model_aktif     : int
```

**Catatan Teknis:**
FastAPI secara otomatis menghasilkan dokumentasi interaktif (Swagger UI) di `/docs` berdasarkan schema Pydantic yang didefinisikan. Ini memungkinkan tim frontend dan dosen penguji untuk menjelajahi semua endpoint tanpa membaca kode secara langsung.

---

## Step 3.2 — Pipeline Prediksi di Backend

**Tujuan:**
Menjelaskan alur data lengkap dari saat request HTTP diterima hingga response prediksi harga dikembalikan — termasuk bagaimana model di-load ke memori dan di-cache untuk efisiensi.

**Penjelasan Rinci:**
Arsitektur `predictor.py` menggunakan **in-memory cache** (`_CACHE` dictionary global) untuk menyimpan model, scaler, feature_cols, dan dataset. Saat pertama kali server FastAPI dinyalakan, fungsi `preload_artifacts()` dipanggil melalui mekanisme `lifespan` asynccontextmanager.

Pipeline prediksi untuk satu request:
1. **Ambil fitur terkini** — `get_fitur_terkini()` mengambil baris terakhir dataset di cache
2. **Load artifacts** — `load_all_artifacts(label)` mengembalikan model, scaler, feature_cols dari cache
3. **Buat DataFrame input** — urutan kolom disesuaikan persis dengan `feature_cols` yang disimpan saat training
4. **Prediksi log-return** — `model.predict(df_input)` menghasilkan nilai log-return
5. **Inverse transform** — `prediksi_rp = harga_hari_ini * exp(log_return)`
6. **Hitung metadata** — arah pergerakan dan persentase perubahan

**Kode:**

```python
# app/main.py — Preloading model saat server startup (via lifespan)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load semua model XGBoost dan dataset saat server start."""
    print("\n[Startup] Memuat ML artifacts dan dataset...")
    predictor.preload_artifacts()       # load semua model + scaler ke cache
    predictor.load_dataset_to_cache()   # load dataset ke cache

    info = predictor.get_cache_info()
    if info.get("models_merah"):
        print(f"[Startup] Model merah siap : {info['models_merah']}")
    if info.get("models_rawit"):
        print(f"[Startup] Model rawit siap : {info['models_rawit']}")
    yield
    print("\n[Shutdown] Membersihkan resources...")
    predictor.clear_cache()
```

```python
# app/core/predictor.py — Pipeline prediksi inti (dengan inverse transform)

def prediksi_harga_sync(input_data: dict, label: str = "h1") -> dict:
    """
    Prediksi harga cabai untuk satu horizon.
    Model memprediksi log-return -> inverse transform -> harga absolut.
    """
    model, scaler, feature_cols, model_version = load_all_artifacts(label)

    # Buat DataFrame fitur dengan urutan kolom identik saat training
    fitur_values = []
    missing_cols = []
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

    df_input = pd.DataFrame([fitur_values], columns=feature_cols)

    # Prediksi log-return
    prediksi_log_return = float(model.predict(df_input)[0])

    # Inverse transform: prediksi_rp = harga_hari_ini * exp(prediksi_log_return)
    if _is_rawit(label):
        harga_hari_ini = float(
            input_data.get("harga_rawit_hari_ini") or
            input_data.get("harga_cabai_rawit") or 0.0
        )
    else:
        harga_hari_ini = float(
            input_data.get("harga_hari_ini") or
            input_data.get("harga_cabai_merah") or 0.0
        )

    prediksi_rp = harga_hari_ini * np.exp(prediksi_log_return)
    prediksi_rp = max(prediksi_rp, 1000.0)  # sanity check: minimal Rp 1.000

    # Hitung arah dan persentase perubahan
    if harga_hari_ini > 0:
        perubahan_persen = ((prediksi_rp - harga_hari_ini) / harga_hari_ini) * 100
    else:
        perubahan_persen = 0.0

    batas_stabil = 0.25  # perubahan < 0.25% dianggap stabil
    if   perubahan_persen >  batas_stabil: arah = "naik"
    elif perubahan_persen < -batas_stabil: arah = "turun"
    else:                                  arah = "stabil"

    horizon_days     = int(label.split("_")[-1][1])  # h1->1, h3->3, h7->7
    tanggal_base     = pd.Timestamp(input_data.get("tanggal", datetime.now().date()))
    tanggal_prediksi = tanggal_base + timedelta(days=horizon_days)

    return {
        "horizon"         : label,
        "prediksi_rp"     : round(prediksi_rp, 2),
        "prediksi_delta"  : round(prediksi_log_return, 2),
        "harga_hari_ini"  : round(harga_hari_ini, 2),
        "tanggal_prediksi": str(tanggal_prediksi.date()),
        "model_version"   : model_version,
        "arah_prediksi"   : arah,
        "perubahan_persen": round(perubahan_persen, 2),
    }


async def prediksi_harga(input_data: dict, label: str = "h1") -> dict:
    """Wrapper async — menjalankan prediksi_harga_sync di thread pool."""
    return await run_in_threadpool(prediksi_harga_sync, input_data, label)
```

**Catatan Teknis:**
Penggunaan `run_in_threadpool` dari `starlette.concurrency` adalah praktik terbaik untuk fungsi blocking (seperti `model.predict()`) di dalam endpoint FastAPI yang bersifat async. Tanpa ini, eksekusi model akan memblokir event loop asyncio dan mengurangi throughput server.

---

## Step 3.3 — Pengiriman Request dari Next.js ke FastAPI

**Tujuan:**
Mengimplementasikan layer komunikasi HTTP di sisi frontend yang menghubungkan antarmuka pengguna ke backend FastAPI, dengan penanganan timeout, error, dan pemisahan concerns yang bersih.

**Penjelasan Rinci:**
Seluruh komunikasi HTTP dari Next.js ke FastAPI dipusatkan di file `app/lib/api.ts`. File ini mengekspor interface TypeScript, fungsi helper `apiFetch<T>()` dengan fitur timeout 15 detik, dan fungsi fetch per endpoint. Endpoint prediksi merah dan rawit dipanggil **secara paralel** menggunakan `Promise.all()`.

**Kode:**

```typescript
// app/lib/api.ts — Konstanta base URL dan type definitions

const BASE_URL = "http://localhost:8000";

export interface PrediksiItem {
  horizon          : string;
  keterangan       : string;
  tanggal_prediksi : string;
  prediksi_rp      : number;
  model_version    : string;
  arah_prediksi?   : string | null;
  perubahan_persen?: number | null;
  error?           : string;
}

export interface MetrikHorizon {
  MAE  : number;
  RMSE : number;
  MAPE : number;
  sMAPE: number;
  R2   : number;
  DA   : number;
}

export interface DashboardResponse {
  tanggal_update    : string;
  harga_hari_ini    : number;
  harga_rata_30hari : number;
  prediksi_h1       : number | null;
  prediksi_h3       : number | null;
  prediksi_h7       : number | null;
  tren              : string;
  status_inflasi    : string;
  harga_hari_ini_rawit ?: number | null;
  prediksi_rawit_h1    ?: number | null;
  tren_rawit           ?: string | null;
  status_inflasi_rawit ?: string | null;
  status_model  : boolean;
  n_model_aktif : number;
}
```

```typescript
// app/lib/api.ts — Fungsi fetch helper generik dengan timeout

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const controller = new AbortController();
  // Timeout 15 detik — batalkan request jika backend tidak merespons
  const timeout = setTimeout(() => controller.abort(), 15000);

  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      ...options,
      signal : controller.signal,
      headers: {
        "Content-Type": "application/json",
        ...options?.headers,
      },
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => null);
      throw new Error(
        errorData?.detail || `API Error: ${res.status} ${res.statusText}`
      );
    }

    return await res.json();
  } catch (err: unknown) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Request timeout — backend tidak merespons dalam 15 detik.");
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}
```

```typescript
// app/lib/api.ts — Fetch prediksi merah dan rawit secara paralel

/** Fetch prediksi merah dan rawit secara paralel (mengurangi latensi total) */
export async function fetchPrediksiSemuaKomoditas(): Promise<PrediksiSemuaKomoditasResponse> {
  const [merah, rawit] = await Promise.all([
    apiFetch<PrediksiAllResponse>("/api/predict/prediksi"),
    apiFetch<PrediksiAllResponse>("/api/predict/prediksi/rawit")
  ]);
  return { merah, rawit };
}

/** Data historis untuk grafik */
export async function fetchHistoris(nHari: number = 90): Promise<HistorisResponse> {
  return apiFetch<HistorisResponse>(
    `/api/predict/harga/historis?n_hari=${nHari}`
  );
}

/** Metrik akurasi model (MAE, RMSE, MAPE, R², DA) */
export async function fetchMetrik(): Promise<MetrikResponse> {
  return apiFetch<MetrikResponse>("/api/predict/model/metrik");
}
```

**Catatan Teknis:**
`AbortController` dipilih di atas library pihak ketiga seperti `axios` untuk minimalisasi dependensi. Dengan Next.js App Router, seluruh fetch dilakukan di Client Component (`"use client"`) menggunakan `useEffect` — bukan Server-Side Rendering — karena data prediksi bersifat real-time dan bergantung pada state server yang aktif.

---

## Step 3.4 — Penanganan Response dan State Management

**Tujuan:**
Mengelola siklus data dari mulai pengiriman request hingga data siap ditampilkan di antarmuka pengguna, termasuk penanganan state loading dan berbagai skenario error.

**Penjelasan Rinci:**
State management di halaman utama (`app/page.tsx`) menggunakan React `useState`. Seluruh request API dieksekusi menggunakan `Promise.allSettled()` — yang tidak "gagal seluruhnya" jika satu request gagal, sehingga dashboard tetap menampilkan data yang berhasil di-fetch meskipun satu endpoint belum tersedia.

**Kode:**

```typescript
// app/page.tsx — State management dan data fetching dengan Promise.allSettled

export default function Home() {
  const [activeKomoditas, setActiveKomoditas] = useState<"merah" | "rawit">("merah");
  const [dashboard,     setDashboard]     = useState<DashboardResponse | null>(null);
  const [historis,      setHistoris]      = useState<HistorisDataPoint[]>([]);
  const [prediksi,      setPrediksi]      = useState<PrediksiItem[]>([]);
  const [prediksiRawit, setPrediksiRawit] = useState<PrediksiItem[]>([]);
  const [metrik,        setMetrik]        = useState<MetrikResponse | null>(null);
  const [loading,       setLoading]       = useState(true);
  const [error,         setError]         = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        // allSettled: lanjut meskipun salah satu request gagal
        const [dashData, histData, predAllData, metrikData] =
          await Promise.allSettled([
            fetchDashboard(),
            fetchHistoris(60),
            fetchPrediksiSemuaKomoditas(),
            fetchMetrik(),
          ]);

        if (dashData.status    === "fulfilled") setDashboard(dashData.value);
        if (histData.status    === "fulfilled") setHistoris(histData.value.data);
        if (predAllData.status === "fulfilled") {
          setPrediksi(predAllData.value.merah.prediksi);
          setPrediksiRawit(predAllData.value.rawit.prediksi);
        }
        if (metrikData.status  === "fulfilled") setMetrik(metrikData.value);

        // Error total hanya jika SEMUA request utama gagal
        if (dashData.status === "rejected" &&
            histData.status === "rejected" &&
            predAllData.status === "rejected") {
          setError(
            "Tidak bisa terhubung ke backend. " +
            "Pastikan server FastAPI berjalan di localhost:8000"
          );
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Terjadi kesalahan");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []); // [] = hanya dijalankan sekali saat komponen pertama dimount
```

**Catatan Teknis:**
`useEffect` dengan dependency array kosong `[]` berarti data hanya di-fetch sekali saat halaman pertama kali dibuka. Untuk aplikasi produksi yang membutuhkan data terkini, pendekatan polling dengan interval waktu tertentu atau library seperti `SWR`/`React Query` dapat dipertimbangkan.

---

## Step 3.5 — Visualisasi Data Prediksi di UI

**Tujuan:**
Menampilkan data historis harga dan prediksi model dalam bentuk grafik interaktif yang mudah dipahami oleh pengguna awam (pedagang, konsumen, pembuat kebijakan).

**Penjelasan Rinci:**
Komponen `PriceChart` (`app/components/PriceChart.tsx`) menggunakan library **Chart.js** untuk merender grafik garis. Grafik menampilkan:

1. **Garis Aktual** — harga historis 60 hari terakhir (solid line dengan gradient fill)
2. **Garis Prediksi** — nilai prediksi H+1, H+3, H+7 (dashed line dengan titik)

Fitur teknis: gradient fill, responsive, tooltip kustom format Rupiah, auto-scale Y-axis dengan padding 15%, dan cleanup pada unmount.

**Kode:**

```typescript
// app/components/PriceChart.tsx — Komponen Chart.js dengan prediksi overlay

"use client";

import { useEffect, useRef } from "react";
import { Chart, CategoryScale, LinearScale, PointElement,
         LineElement, Filler, Tooltip, Legend } from "chart.js";

Chart.register(CategoryScale, LinearScale, PointElement,
               LineElement, Filler, Tooltip, Legend);

interface PriceChartProps {
  labels?      : string[];
  data?        : number[];
  prediksiH1?  : number | null;
  prediksiH3?  : number | null;
  prediksiH7?  : number | null;
  lineColor?   : string;
  datasetLabel?: string;
}

export default function PriceChart({
  labels, data, prediksiH1, prediksiH3, prediksiH7,
  lineColor = "#0F6E56", datasetLabel = "Harga Aktual",
}: PriceChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartRef  = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    if (chartRef.current) chartRef.current.destroy(); // cleanup chart lama

    const ctx = canvasRef.current.getContext("2d");
    if (!ctx) return;

    let chartLabels    : string[];
    let actualPrices   : (number | null)[];
    let predictedPrices: (number | null)[];

    if (labels && data && data.length > 0) {
      chartLabels     = [...labels];
      actualPrices    = [...data];
      predictedPrices = new Array(data.length).fill(null);
      // Sambungkan garis prediksi dari titik aktual terakhir
      predictedPrices[predictedPrices.length - 1] = data[data.length - 1];

      if (prediksiH1 != null) { chartLabels.push("H+1"); actualPrices.push(null); predictedPrices.push(prediksiH1); }
      if (prediksiH3 != null) { chartLabels.push("H+3"); actualPrices.push(null); predictedPrices.push(prediksiH3); }
      if (prediksiH7 != null) { chartLabels.push("H+7"); actualPrices.push(null); predictedPrices.push(prediksiH7); }
    }

    // Gradient fill di bawah garis aktual
    const rgbaColor    = lineColor === "#0F6E56" ? "15, 110, 86" : "230, 81, 0";
    const lineGradient = ctx.createLinearGradient(0, 0, 0, 350);
    lineGradient.addColorStop(0, `rgba(${rgbaColor}, 0.15)`);
    lineGradient.addColorStop(1, `rgba(${rgbaColor}, 0.01)`);

    chartRef.current = new Chart(ctx, {
      type: "line",
      data: {
        labels  : chartLabels,
        datasets: [
          {
            label          : datasetLabel,
            data           : actualPrices,
            borderColor    : lineColor,
            backgroundColor: lineGradient,
            borderWidth    : 2.5,
            tension        : 0.4,
            fill           : true,
            pointRadius    : 0,
          },
          {
            label          : `Prediksi ${datasetLabel.replace("Aktual ", "")}`,
            data           : predictedPrices,
            borderColor    : lineColor,
            borderWidth    : 2,
            borderDash     : [6, 4],   // garis putus-putus untuk prediksi
            pointRadius    : 4,
            tension        : 0.4,
            fill           : false,
          },
        ],
      },
      options: {
        responsive        : true,
        maintainAspectRatio: false,
        plugins: {
          tooltip: {
            callbacks: {
              label: (ctx) =>
                ` ${ctx.dataset.label}: Rp ${ctx.parsed.y?.toLocaleString("id-ID")}`,
            },
          },
        },
        scales: {
          y: {
            ticks: {
              callback: (val) => `Rp ${(Number(val) / 1000).toFixed(0)}k`,
            },
          },
        },
      },
    });

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy(); // cleanup saat komponen unmount
        chartRef.current = null;
      }
    };
  }, [labels, data, prediksiH1, prediksiH3, prediksiH7, lineColor, datasetLabel]);

  return <canvas ref={canvasRef} />;
}
```

**Catatan Teknis:**
Dependency array `useEffect` menyertakan semua props yang mempengaruhi tampilan grafik. Setiap kali props berubah (misalnya pengguna beralih dari tampilan merah ke rawit), grafik lama di-destroy dan grafik baru dibuat ulang. Ini mencegah instance Chart.js berganda pada element `<canvas>` yang sama.

---

## Step 3.6 — Penanganan Error dan Validasi Input

**Tujuan:**
Memberikan umpan balik yang informatif kepada pengguna ketika terjadi kegagalan koneksi, model tidak tersedia, atau sistem dalam kondisi terdegradasi.

**Penjelasan Rinci:**
Sistem menerapkan error handling berlapis. **Di Sisi Backend**: HTTP 503 jika model tidak ditemukan, HTTP 422 jika parameter tidak valid, endpoint `/health` melaporkan status komponen secara granular. **Di Sisi Frontend**: pesan error yang ramah pengguna, skeleton UI animasi saat loading, dan badge "N/A" per item prediksi yang gagal.

**Kode:**

```python
# app/routes/predict.py — Validasi input dan HTTP error di sisi backend

@router.get("/prediksi/{horizon}", response_model=PrediksiOtomatisResponse)
async def prediksi_otomatis(
    horizon: str = PathParam(
        ...,
        description="Horizon: h1 (besok), h3 (3 hari), h7 (7 hari)",
        regex="^(h1|h3|h7)$",  # validasi otomatis oleh FastAPI
    )
):
    if not predictor.validate_horizon(horizon):
        raise HTTPException(status_code=422, detail=f"Horizon tidak valid: '{horizon}'.")
    try:
        input_data = predictor.get_fitur_terkini()
        hasil      = await predictor.prediksi_harga(input_data, horizon)
        return {"status": "success", "horizon": horizon, **hasil}
    except HTTPException:
        raise  # re-raise agar tidak terbungkus 500
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal prediksi: {str(e)}")


@router.get("/health")
def health_check():
    """Cek ketersediaan semua model (merah + rawit), scaler, dan dataset."""
    health = {"timestamp": datetime.now().isoformat(), "models": {},
              "scaler": False, "dataset": False}

    for horizon in ["h1", "h3", "h7", "rawit_h1", "rawit_h3", "rawit_h7"]:
        try:
            predictor.load_model(horizon)
            health["models"][horizon] = True
        except Exception:
            health["models"][horizon] = False

    merah_ok   = all(health["models"].get(h) for h in ["h1", "h3", "h7"])
    rawit_ok   = all(health["models"].get(h) for h in ["rawit_h1", "rawit_h3", "rawit_h7"])
    dataset_ok = health["dataset"]

    if merah_ok and rawit_ok and dataset_ok:
        health["status"]  = "healthy"
        health["message"] = "Semua model merah & rawit tersedia"
    elif merah_ok and dataset_ok:
        health["status"]  = "degraded"
        health["message"] = "Model merah OK, model rawit belum tersedia"
    else:
        health["status"]  = "unhealthy"
        raise HTTPException(status_code=503, detail=health)
    return health
```

```typescript
// app/page.tsx — Loading skeleton dan penanganan error di frontend

{/* Loading State — skeleton UI animasi */}
{loading && (
  <div className="dashboard-skeleton animate-in delay-1" aria-busy="true">
    <div className="metrics-grid" style={{ marginBottom: 20 }}>
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="metric-card">
          <div className="skeleton" style={{ width: "50%", height: 16, marginBottom: 12 }} />
          <div className="skeleton" style={{ width: "80%", height: 32, marginBottom: 12 }} />
          <div className="skeleton" style={{ width: "60%", height: 14 }} />
        </div>
      ))}
    </div>
  </div>
)}

{/* Error State — ditampilkan jika semua request gagal */}
{error && !loading && (
  <div className="alert-box animate-in delay-1"
       style={{ borderColor: "var(--red-muted)", background: "var(--red-light)" }}>
    <p className="alert-text">
      <strong>Error:</strong> {error}
    </p>
  </div>
)}

{/* Item prediksi yang error individual */}
{activePrediksi.map((p, i) => (
  <li key={i} className="prediction-item">
    <span className="prediction-date">{p.keterangan}</span>
    <span className="prediction-price">
      {p.error ? "Error" : formatRp(p.prediksi_rp)}
    </span>
    <span className={`prediction-badge badge ${p.error ? "badge-red" : "badge-green"}`}>
      {p.error ? "N/A" : p.tanggal_prediksi}
    </span>
  </li>
))}
```

**Catatan Teknis:**
Pesan error yang menghadap pengguna disederhanakan — tidak menampilkan stack trace atau detail teknis server. Pesan teknis tersimpan di log server FastAPI. Di sisi pengguna, pesan seperti "Pastikan server FastAPI berjalan di localhost:8000" jauh lebih actionable dibandingkan pesan HTTP mentah.

---

## Ringkasan Alur Bagian 3 — Penggunaan API di Frontend

| Step | Nama | Komponen | Output/Hasil |
|------|------|----------|-------------|
| 3.1 | Endpoint & Routing | `main.py`, `predict.py`, `schemas/predict.py` | 10+ endpoint REST + Swagger docs |
| 3.2 | Pipeline Prediksi | `predictor.py` (preload → cache → predict) | Respons JSON dengan `prediksi_rp`, `arah`, `perubahan_persen` |
| 3.3 | Fetch dari Next.js | `lib/api.ts` (apiFetch + Promise.all) | Data merah + rawit di-fetch paralel dalam <15 detik |
| 3.4 | State Management | `page.tsx` (useState + Promise.allSettled) | State: dashboard, historis, prediksi, metrik, loading, error |
| 3.5 | Visualisasi | `PriceChart.tsx` (Chart.js) | Grafik garis aktual + prediksi putus-putus |
| 3.6 | Error Handling | `predict.py` (HTTP 422/503/500) + `page.tsx` | Skeleton UI, pesan error, badge N/A per item prediksi |

---

## Penutup

Laporan ini telah menjelaskan secara rinci seluruh alur teknis sistem prediksi harga cabai di Kota Padang — dari data mentah BMKG dan PIHPS hingga dashboard interaktif yang dapat diakses melalui browser web. Beberapa keputusan desain kritis yang perlu disoroti:

1. **Anti-leakage secara konsisten** — penggunaan `ffill`, median-train, dan `shift(1)` di setiap tahap preprocessing memastikan model belajar dari data masa lalu saja
2. **Log-return sebagai target** — menghindari dominasi `lag_1` dan memaksa model belajar *faktor perubahan* harga
3. **Expanding window** — teknik validasi yang menghormati urutan waktu data
4. **In-memory caching** — model di-load sekali saat startup, menjamin latensi prediksi yang rendah
5. **Promise.allSettled** — dashboard tetap fungsional meskipun sebagian endpoint tidak tersedia

Sistem ini didesain untuk dapat dikembangkan lebih lanjut — misalnya menambahkan komoditas pangan lain (bawang merah, tomat, daging sapi), mengintegrasikan data PIHPS secara real-time, atau mengadopsi model yang lebih canggih seperti Temporal Fusion Transformer (TFT) untuk prediksi multi-step yang lebih akurat.
