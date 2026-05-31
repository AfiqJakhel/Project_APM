# Laporan Teknis Sistem Prediksi Harga Cabai (XGBoost Multi-Horizon)
**Lokasi:** Kota Padang, Sumatera Barat  
**Stack:** Python (Pandas, XGBoost, Optuna), FastAPI  
**Target:** Prediksi multi-horizon Harga Cabai Merah & Rawit (H+1, H+3, H+7)

---

## BAB 1: PIPELINE PREPROCESSING

### 1.1 Sumber dan Format Data
Proses pengambilan data (Data Ingestion) memuat dua format utama, yaitu `.xlsx` (excel) dari data pasar dan BMKG, serta `.json` untuk data kalender. File-file tersebut diload melalui fungsi iteratif pada direktori.

```python
# Potongan kode dari preprocessing.py: load_cuaca() & load_harga_cabai()
def load_cuaca() -> pd.DataFrame:
    all_files = sorted(glob.glob(str(CUACA_DIR / "**" / "*.xlsx"), recursive=True))
    # ... iterasi pembacaan pd.read_excel()
    # Menggabungkan semua frame dan membuat series harian yang rata
    df = pd.concat(frames, ignore_index=True)
    df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], format="%d-%m-%Y", errors="coerce")
    # ...
    # Handling missing values untuk deret waktu dengan forward-fill lalu backward-fill
    # Hal ini MENCEGAH data leakage dibandingkan metode interpolate()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
```
**Penjelasan:** 
Di dalam `load_cuaca()` dan `load_harga_cabai()`, modul `pandas` digunakan untuk merangkai (concatenate) dan menginisialisasi indeks _datetime_. Fungsi `ffill()` (forward-fill) amat krusial pada data deret waktu karena ia menarik harga/cuaca observasi terakhir hari kemarin untuk menutupi `NaN` hari ini, sehingga bebas dari intipan kebocoran data masa depan (_data leakage_) yang kerap timbul jika memakai _linear interpolation_.

### 1.2 Penambahan Fitur Kalender dan Hari Raya
Fluktuasi harga cabai amat sensitif dengan pola hari raya (Ramadan, Lebaran). Library kustom seperti `hijridate` (untuk kalender Islam) dan `holidays` digunakan di tahapan ini.

```python
# Potongan kode dari train.py: add_calendar_features() dan preprocessing.py
# 1. Mendeteksi hari raya dengan array statis dan menghitung days_to_lebaran
_lebaran_dates = pd.to_datetime(["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-30", "2026-03-20"])
df["days_to_lebaran"] = df["tanggal"].apply(
    lambda t: int(min(abs((t - d).days) for d in _lebaran_dates))
)

# 2. Menggunakan pustaka pihak ketiga untuk kelengkapan
import holidays
id_holidays = holidays.ID()
df['is_libur_nasional_lib'] = df['tanggal'].apply(lambda x: 1 if x in id_holidays else 0)

# Menggunakan isocalendar untuk mendapatkan urutan pekan dan akhir pekan
df['is_weekend'] = (df['tanggal'].dt.dayofweek >= 5).astype(int)
```
**Penjelasan:** 
Fitur `days_to_lebaran` adalah variabel kontinu (numerik absolut) yang menghitung delta/jarak secara dinamis (H-x atau H+x) menuju Idul Fitri. Ini sangat superior dibanding fitur biner klasikal 1/0 karena fluktuasi _demand_ logistik umumnya menanjak secara kurva seiring merapatnya hari lebaran. Penggunaan API kalender `holidays.ID()` menyempurnakan asimilasi cuti bersama.

### 1.3 Feature Engineering
Rekayasa fitur dikembangkan untuk mendeteksi _market memory_, volatilitas momentum, serta dampak kausal (seperti hujan berkepanjangan pada fase tanam).

```python
# Potongan kode dari preprocessing.py: feature_engineering()
# 1. Lagging: Ekstraksi Harga Historis Pasar
for lag in [1, 7, 14, 21, 30]:
    df[f"lag_{lag}"] = df[TARGET].shift(lag)

# 2. Momentum & Exponential Moving Average (EMA)
df["momentum_7"] = df[TARGET].shift(1) - df[TARGET].shift(7)
df[f"ema_7"] = s_base.ewm(span=7, adjust=False).mean()
df["ema_crossover_7_30"] = df["ema_7"] - df["ema_30"]

# 3. Cross-Commodity & Indikator Cuaca Makro (Fase Tanam)
df["rasio_merah_rawit"] = (df[TARGET] / df["harga_cabai_rawit"].replace(0, np.nan)).fillna(1.0)
df["roll_hujan_60"] = df["curah_hujan"].rolling(60, min_periods=1).sum()  
df["hujan_ekstrem_3hari"] = ((df["curah_hujan"] > 50).rolling(3, min_periods=1).sum().astype(int))
```
**Penjelasan:** 
*   Fungsi `shift(lag)` krusial digunakan di awal untuk memastikan kita tidak menggeser "hari ini", menghindari rekayasa melihat masa depan.
*   `ewm(span)` membangun EMA yang bereaksi lebih cepat dibanding Simple Moving Average (SMA). `ema_crossover_7_30` adalah proxy andal: membandingkan arus tren pendek 1 minggu vs 1 bulan untuk menemukan sinyal *breakout*.
*   `roll_hujan_60` menangkap rasio akumulasi air pada 2 bulan sebelum hari ini. Karena usia vegetatif cabai adalah ~60 hari sebelum fase panen, indikator curah hujan jangka panjang ini mengidentifikasi gangguan pasokan akibat gagal panen.

### 1.4 Feature Selection
Untuk menghapus redundansi dan mencegah _overfitting_ (curse of dimensionality), filter dua tahap diterapkan:

```python
# Potongan kode dari preprocessing.py: seleksi_fitur()
# Tahap 1: Variance Threshold (membuang fitur yang statis/tidak bergerak)
low_var = [c for c in feature_cols if df[c].std() < 0.01]

# Tahap 2: Correlation Threshold (Mencegah multikolinearitas yang tinggi)
corr_matrix = df[feature_cols].corr().abs()
high_corr_count = (corr_matrix > 0.95).sum() - 1

# Pengecualian / Whitelist
WHITELIST = {"lag_14", "lag_21", "roll_hujan_14", "roll_hujan_60", "max_hujan_7", "ema_7", "ema_crossover_7_30"}
to_drop = set()
cols = list(corr_matrix.columns)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if corr_matrix.iloc[i, j] > 0.95:
            if cols[i] in WHITELIST or cols[j] in WHITELIST:
                continue # Fitur penting hujan tetap disimpan
            if high_corr_count[cols[i]] >= high_corr_count[cols[j]]:
                to_drop.add(cols[i])
            else:
                to_drop.add(cols[j])
```
**Penjelasan:** 
Melalui perulangan _matrix threshold_, apabila dua pasang fitur berkorelasi `> 0.95` secara absolut (menandakan informasi ganda), salah satu dihapus. Namun algoritme menyediakan kumpulan `WHITELIST` seperti `roll_hujan_60`. Fitur cuaca dilindungi karena XGBoost (_Gradient Boosted Trees_) mahir mengeksploitasi interaksi relasional non-linear, yang mana metrik `corr()` (Pearson Linear) gagal menangkapnya. Sebanyak ~12 fitur berhasil direduksi.

### 1.5 Target Variable Construction
Transformasi data tidak menggunakan prediksi delta linier, melainkan **Log-Return**.

```python
# Potongan kode dari preprocessing.py: Konstruksi Target Multi-horizon
harga_safe = df[TARGET].replace(0, np.nan)

# Menghitung probabilitas rasio return eksponensial
df["target_h1"] = np.log(df[TARGET].shift(-1) / harga_safe)
df["target_h3"] = np.log(df[TARGET].shift(-3) / harga_safe)
df["target_h7"] = np.log(df[TARGET].shift(-7) / harga_safe)

# Klasifikasi biner Arah sebagai indikator bantuan
df["arah_target_h1"] = (df["target_h1"] > 0).astype(int)
```
**Penjelasan:** 
Di sinilah terobosan pemodelan terjadi. Menggunakan fungsi algoritma `np.log(harga_besok / harga_sekarang)`. Memodelkan prediksi harga mentah/absolut Rupiah (misal Rp 60.000) terbukti memicu **Naive-Forecast Benchmark**, model menjadi 'malas' (prediksi esok hanya fotokopi harga hari ini).
Dengan mengalibrasi target berwujud persentase return alamiah (logaritma normal yang asimtotik stabil dan _scale-free_), algoritme dibebaskan mendeteksi pola _rate of change_ (laju perubahan inflasi hariannya) secara sangat simetris.

---

## BAB 2: PIPELINE TRAINING MODEL

### 2.1 Arsitektur Expanding Window
Menghindari pemakaian K-Fold biasa karena deret waktu bersifat struktural kausal satu arah. Validasi yang tepat adalah _Expanding Window_ (CV OOS).

```python
# Potongan kode dari train.py: buat_expanding_windows()
MIN_TRAIN_DAYS = 365
STEP_DAYS      = 30
TEST_DAYS      = 30
windows = []
train_end = MIN_TRAIN_DAYS

while True:
    test_end = train_end + TEST_DAYS
    if test_end > total: break
    windows.append({
        "train_start"          : 0,
        "train_end"            : train_end,
        "test_start"           : train_end,
        "test_end"             : test_end,
    })
    train_end += STEP_DAYS
```
**Penjelasan:** 
Siklus perulangan mengalokasikan index data dimulai dari dasar 365 hari awal (memberikan siklus musiman tahunan utuh) lalu bergerak ekspansif ke kanan. Sebanyak 39 iterasi (_windows_) diciptakan. Skema murni menjamin pelarangan kebocoran memori (training tidak pernah menatap masa depan testing). _Train_ bermula dari 0 selalu bertambah lebar memori datanya mengumpulkan pengalaman historis iteratif untuk prediksi ke depannya.

### 2.2 Hyperparameter Tuning dengan Optuna
Eksplorasi pohon regresi dikelola asinkron menggunakan modul `optuna` dipadukan bobot eksponensial.

```python
# Potongan kode dari train.py: tuning_final() dan buat_sample_weights()
def buat_sample_weights(n: int, decay: float = 0.001) -> np.ndarray:
    indices = np.arange(n)
    weights = np.exp(decay * (indices - n + 1)) # Peluruhan eksponensial
    return weights / weights.mean()

# Dalam Optuna objective loop:
def objective(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 4),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
        # ...
    }
    model = XGBRegressor(objective="reg:squarederror", tree_method="hist", **param)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)
```
**Penjelasan:** 
Variabel `w_tr` hasil dari `buat_sample_weights()` memaksa model mementingkan fluktuasi pasar teranyar (_recent trend_) dan melupakan tren purba yang sudah lapuk secara eksponensial (parameter peluruhan `decay`). Konfigurasi regresi menggunakan `reg:squarederror` (MSE Regression loss) mempercepat asimptot kalkulasi error dan mengecilkan risiko penalti tebakan yang over-ambisius. Ruang pencarian terbatas parameter iteratif diselesaikan dengan `TPESampler`.

### 2.3 Training Model Final
Inisiasi permodelan menyeluruh (_retraining_) yang menyimpan versi mutakhir.

```python
# Potongan kode dari train.py: latih_model_final()
# Memecah validasi sekadar sebagai trigger early-stopping, mencegah model overkill
split_val = int(len(X) * 0.9)
model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X.iloc[split_val:], y.iloc[split_val:])], verbose=False)

# Membekukan konfigurasi obyek persisten 
ts = datetime.now().strftime("%Y%m%d")
model_path = cfg.model_dir / f"model_final_{label}_{ts}.pkl"
joblib.dump(model, model_path)
```
**Penjelasan:** 
Di skenario riil iterasi final, parameter terbaik disuntik untuk dieksekusi terhadap 100% sampel valid terkini (disisakan 10% sekuen belakangan hanya untuk regulasi penahan batas `early_stopping_rounds`). Algoritme membuang artefak akhir `.pkl` pada _model_path_ menggunakan `joblib.dump()` dihiasi stempel waktu/timestamp, yang mempertegas stabilitas identifikasi versi model saat sinkronisasi _Back-End API_ (CI/CD Deployment).

### 2.4 Evaluasi Model
Tidak sekadar error linier, rasio prediktif dikembangkan dalam akumulasi `Global Out-of-Sample`.

```python
# Potongan kode dari train.py: expanding_window_eval()
# Inverse log-return -> Harga prediksi Rupiah
y_true_abs = ref_test * np.exp(y_test.values)
y_pred_abs_xgb = ref_test * np.exp(y_pred_log_xgb)
y_pred_abs_xgb = np.clip(y_pred_abs_xgb, 1000, None) # Limit bawah asuransi (Sanity kliping)

# R2 Global akumulasi
global_y_true.extend(y_true_abs)
global_y_pred.extend(y_pred_abs_xgb)
global_r2 = r2_score(global_y_true, global_y_pred)

# Metrik Arah
def hitung_da(y_true, y_pred, ref_val):
    diff_true = y_true - ref_val
    diff_pred = y_pred - ref_val
    return np.mean(np.sign(diff_true[mask]) == np.sign(diff_pred[mask])) * 100
```
**Penjelasan:** 
Inversi harga Rupiah dinormalkan kembali dengan metrik probabilitas antilog `np.exp()`. Limit kliping (`np.clip(1000)`) merupakan batasan keamanan mekanis anti tebakan negatif. R-Square ($R^{2}$) dinilai berbasis Global Arrays (seluruh nilai agregat antar loop Window dijahit lalu dites di akhir secara komposit) daripada menengahi (_averaging_) nilai R-Square parsial sub-window yang rentan menghasilkan rasio eror kalkulus asimetris. Parameter tambahan Direction Accuracy (DA) mengunci pembuktian korelasi seberapa handal sinyal penunjukan trend ("Naik", "Turun").

### 2.5 SHAP Analysis
Buku panduan detrending regresi diinspeksi kebenarannya di fungsi ini.

```python
# Potongan kode dari train.py: shap_final()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
mean_abs = np.abs(shap_values).mean(axis=0)
df_shap = pd.DataFrame({"fitur": X_sample.columns, "shap_mean": mean_abs}).sort_values("shap_mean", ascending=False)

if df_shap.iloc[0]["fitur"] == "lag_1":
    log(f"    -> SHAP: lag_1 masih dominan")
else:
    log(f"    -> SHAP: lag_1 bukan #1 (Detrending berhasil. Fitur top: {df_shap.iloc[0]['fitur']})")
```
**Penjelasan:** 
Algoritma evaluasi `shap.TreeExplainer()` dipakai membongkar relasi bobot koefisien kontribusi game-theory Shapley. Pengecekan krusial "lag_1 bukan ranking #1" ini menegaskan perombakan _Log-Return Target_ (di BAB 1.5) sukses menumbangkan dominasi lag regresi autokorelasi satu hari ke belakang. Beban pembelajaran kini dipindahkan kepada ragam _momentum_, _moving average_, hingga indeks iklim hujan, mencerminkan pemodelan multivariat riil.

---

## BAB 3: INTEGRASI DENGAN WEB UI

### 3.1 Arsitektur Backend-Frontend
FastAPI `app/main.py` menggunakan event handler `lifespan` yang sinkronik menelan model prediktor di awal siklus _warm-up server_.

```python
# Potongan kode dari app/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load semua model XGBoost, dataset, dan setup scheduler saat server start."""
    print("\n[Startup] Memuat ML artifacts dan dataset...")
    predictor.preload_artifacts()
    predictor.load_dataset_to_cache()
    yield
    print("\n[Shutdown] Membersihkan resources...")
    predictor.clear_cache()

app = FastAPI(lifespan=lifespan, title="API Prediksi Harga Cabai")
```
**Penjelasan:** 
Dengan `preload_artifacts()`, seluruh ekstensi `model.pkl` dan dictionary konfigurasi _feature_cols_ diunggah di area volatilitas In-Memory (_RAM_). Kesiapsiagaan _cache pool_ ini bermakna fungsi tidak dibebani overhead _disk I/O fetching_ berulang kali tiap iterasi HTTP Request (Latency berkurang siginifikan saat Frontend mengeklik *Predict*).

### 3.2 Alur Inferensi/Prediksi
Skema `prediksi_harga_sync()` mengonversi input hari terakhir menjadi luaran rasio mata uang _real-time_.

```python
# Potongan kode dari app/core/predictor.py: prediksi_harga_sync()
df_input = pd.DataFrame([fitur_values], columns=feature_cols)

# Output log-return delta harga dari algoritme
prediksi_delta = float(model.predict(df_input)[0])

# Transformasi Kejut Balik
harga_hari_ini = float(input_data.get(hari_ini_ref_key) or 0.0)
prediksi_rp = harga_hari_ini * np.exp(prediksi_delta)
prediksi_rp = max(prediksi_rp, 1000.0) # Normalisasi Limit

# Sentimen arah 
perubahan_persen = ((prediksi_rp - harga_hari_ini) / harga_hari_ini) * 100
arah = "naik" if perubahan_persen > 0.25 else ("turun" if perubahan_persen < -0.25 else "stabil")
```
**Penjelasan:** 
Inti pengeksekusian _live-inference_ dijalankan asinkron per utas paralel `model.predict(X)` dengan arsitektur vektor matrix satu baris. Konversi balikan dikalkulasikan melalui perkalian Harga Patokan Hari Ini dikali eksponen (`np.exp()`) dari output prediksi _XGBoost_. Ambang limit fluktuasi stabil ("sideways") disetel logis di rentang simpangan deviasi 0.25% harian.

### 3.3 Penyajian Data di UI
Endpoint mengembalikan konvensi struktur metadata _JavaScript Object Notation_ (JSON).

```python
# Potongan kode dari app/routes/predict.py
return {
    "status"          : "success",
    "horizon"         : horizon, # h1 / h3 / h7
    "keterangan"      : HORIZON_LABEL.get(horizon, f"Prediksi {horizon}"),
    "tanggal_prediksi": hasil["tanggal_prediksi"],
    "prediksi_rp"     : hasil["prediksi_rp"],
    "arah_prediksi"   : hasil.get("arah_prediksi"),
    "perubahan_persen": hasil.get("perubahan_persen"),
    # ...
}
```
**Penjelasan:** 
Atribut ini langsung diproduksi ulang di ranah User-Interface (UI). Visualisasi seperti _bar-chart_ (Chart.js / Recharts) atau dashboard grafis interaktif menangkap representasi integer `prediksi_rp` dan parameter rasio volatilitas `perubahan_persen`. 

### 3.4 Scheduling dan Update Model
Rencana pembaharuan interval waktu real-time melalui rutinitas otomatis (_Cron Jobs_) ditunda penerapannya dan direkayasa manual.

```python
# Potongan kode komentar arsitektur dalam app/main.py
# =============================================================================
# (Scraping dan scheduler realtime dimatikan sesuai permintaan user)
# =============================================================================
```
**Penjelasan:** 
Di dalam direktori arsitektur, _scheduler_ latar belakang (seperti pustaka `APScheduler`) dinonaktifkan oleh administrator atas instruksi kustomisasi. Konsekuensinya, agregat pembaharuan injeksi _dataset CSV_ pasar (seperti operasi ETL mingguan / bulanan dari portal BMKG atau portal harga Disdag) bertumpu pada input otorisasi pemicu sistem _offline_ mandiri (_manual orchestration/pipeline running_) oleh admin sebelum dikalkulasi ulang pada mesin `lifespan` web _server_ saat di-_reboot_.
