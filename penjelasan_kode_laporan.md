# Penjelasan Detail Snippet Kode: Sistem Prediksi Harga Cabai

Dokumen ini adalah lampiran pendukung untuk **Laporan Teknis Lengkap**. Dokumen ini berisi penjelasan baris demi baris dari setiap snippet kode yang ada di dalam laporan utama, khusus disiapkan untuk membantu Anda menjawab pertanyaan dosen penguji terkait logika pemrograman.

---

## BAGIAN 1 — PREPROCESSING DATA (`preprocessing.py`)

### Step 1.1 — Pemuatan Data Awal
**Fokus Kode:** Konfigurasi *Path* dan Konstanta Global
*   `BASE_DIR = Path(__file__).resolve().parent.parent`: Menentukan direktori utama proyek secara dinamis. Ini *best practice* agar kode bisa dijalankan di komputer manapun tanpa harus mengubah path manual (hardcode).
*   `DATA_DIR, CUACA_DIR, HARGA_DIR`: Mengarahkan sistem ke lokasi penyimpanan data mentah (raw data).
*   `TARGET = "harga_cabai_merah"`: Menetapkan komoditas utama yang sedang diproses.

### Step 1.2 — Penanganan Missing Values (Data Kosong)
**Fokus Kode:** Imputasi dengan `ffill` (Forward Fill)
*   `df[col] = df[col].ffill().bfill()`: Fungsi `ffill()` mengambil nilai terakhir yang valid dan mengisinya ke baris kosong di bawahnya. Ini sangat krusial untuk data *time-series* agar tidak terjadi **Data Leakage** (kebocoran informasi dari masa depan). Jika menggunakan interpolasi linear, kita akan menggunakan data dari hari esok untuk menebak hari ini, yang mana itu melanggar aturan prediksi.

### Step 1.3 — Feature Engineering
**Fokus Kode:** Ekstraksi Fitur Cuaca, Waktu, dan Log-Return
*   `df["bulan_sin"] = np.sin(...)` & `df["bulan_cos"] = np.cos(...)`: Ini disebut *Cyclical Encoding*. Tujuannya memetakan bulan ke dalam koordinat lingkaran, agar model paham bahwa bulan 12 (Desember) itu sangat dekat jaraknya dengan bulan 1 (Januari), tidak melompat jauh.
*   `df["roll_mean_7"] = s_base.rolling(7).mean()`: Menghitung rata-rata bergerak (moving average) 7 hari ke belakang untuk menangkap tren jangka pendek mingguan.
*   `df["target_h1"] = np.log(df[TARGET].shift(-1) / harga_safe)`: Ini adalah baris paling penting. Menghitung **log-return** (persentase perubahan). `shift(-1)` artinya mengambil data harga *esok hari*.

### Step 1.6 — Normalisasi (RobustScaler)
**Fokus Kode:** Scaling yang Tahan Outlier
*   `split_idx = int(len(df) * 0.8)`: Memisahkan 80% data pertama sebagai *training set*.
*   `scaler.fit(df_train[feature_cols])`: *Sangat Penting!* Scaler hanya mempelajari distribusi dari data *training* (80%), BUKAN seluruh data. Jika memelajari seluruh data, informasi dari *test set* akan bocor ke *training set* (Data Leakage).
*   `RobustScaler()`: Dipilih karena kebal terhadap outlier/lonjakan harga ekstrem, berbeda dengan `MinMaxScaler` yang akan rusak jika ada satu harga yang melonjak tak masuk akal.

---

## BAGIAN 2 — TRAINING MODEL (`train.py`)

### Step 2.1 — Konfigurasi Model (Arsitektur Multi-Output)
**Fokus Kode:** Dictionary `KomoditasConfig`
*   `KomoditasConfig`: Sebuah *dataclass* atau struktur kamus yang menyimpan letak file input, target kolom, dan nama file output secara terpusat. Memudahkan penambahan komoditas baru (misal: Bawang Merah) di masa depan tanpa mengubah alur logika.

### Step 2.2 — Expanding Window Cross-Validation
**Fokus Kode:** Perulangan Evaluasi Time-Series
*   `for start_test in range(MIN_TRAIN_DAYS, total_days, TEST_STEP)`: Loop ini yang menjalankan *Expanding Window*. Ia memulai *training* dari 365 hari pertama (`MIN_TRAIN_DAYS`), lalu bergeser 30 hari (`TEST_STEP`) ke depan pada setiap iterasi.
*   `X_train = X.iloc[:start_test]`: Mengambil semua data dari hari pertama sampai tepat sebelum batas *test*.

### Step 2.3 — Hyperparameter Tuning (Optuna)
**Fokus Kode:** Pencarian Parameter Optimal
*   `trial.suggest_int("n_estimators", 100, 300)`: Menyuruh Optuna mencoba jumlah pohon dari 100 hingga 300.
*   `study.optimize(objective, n_trials=200)`: Mengulang proses tebak-tebakan parameter sebanyak 200 kali untuk menemukan kombinasi yang menghasilkan error (RMSE) paling kecil.

### Step 2.4 — Training Final
**Fokus Kode:** Melatih Model dengan 90% Data
*   `model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)`: Melatih model pada 90% data historis. `eval_set` digunakan sebagai "pengawas" 10% data terakhir agar model tahu kapan harus berhenti (*early stopping*) sebelum mengalami *overfitting*.

### Step 2.6 — Analisis SHAP (Feature Importance)
**Fokus Kode:** Mendeteksi Fitur Paling Berpengaruh
*   `explainer = shap.TreeExplainer(model)`: SHAP akan membongkar "otak" XGBoost dan melihat seberapa besar setiap fitur memengaruhi prediksi.
*   `shap_values = explainer.shap_values(X)`: Menghitung nilai kontribusi (positif/negatif) dari setiap variabel terhadap setiap baris tebakan.

---

## BAGIAN 3 — PENGGUNAAN API & FRONTEND (`predictor.py`, `api.ts`, `Page.tsx`)

### Step 3.1 — Pembuatan Endpoint FastAPI
**Fokus Kode:** Deklarasi Route HTTP GET
*   `@router.get("/prediksi")`: Mendeklarasikan bahwa jika ada permintaan (request) HTTP GET ke URL `/prediksi`, jalankan fungsi di bawahnya.

### Step 3.2 — Pipeline Inverse Transform (Backend)
**Fokus Kode:** Mengembalikan Log-Return ke Rupiah
*   `pred_log = model.predict(df_features)`: Model menebak persentase perubahannya.
*   `prediksi_rp = harga_hari_ini * np.exp(pred_log)`: Ini adalah operasi *Inverse Transform*. Menggunakan fungsi eksponensial (`exp`) untuk membatalkan proses logaritma sebelumnya (`ln`), lalu mengalikannya dengan harga hari ini agar kembali menjadi satuan Rupiah.

### Step 3.3 — Fungsi `apiFetch` (Frontend Next.js)
**Fokus Kode:** Pembatasan Waktu (Timeout)
*   `const controller = new AbortController()`: Membuat mekanisme pembatalan request.
*   `const timeoutId = setTimeout(() => controller.abort(), timeout)`: Jika server backend tidak membalas dalam waktu 15 detik (misalnya karena server mati/lemot), koneksi akan diputus secara paksa agar website tidak *loading* selamanya (*hanging*).

### Step 3.4 — Eksekusi Paralel (Frontend)
**Fokus Kode:** `Promise.allSettled`
*   `await Promise.allSettled([ fetchMerah, fetchRawit ])`: Alih-alih menunggu prediksi cabai merah selesai baru meminta prediksi cabai rawit (memakan waktu 2x lipat), perintah ini meminta keduanya dikerjakan secara bersamaan (paralel). `allSettled` memastikan jika salah satu gagal, yang lainnya tetap bisa ditampilkan.

### Step 3.5 — Visualisasi Chart.js
**Fokus Kode:** Garis Putus-putus Prediksi
*   `borderDash: [5, 5]`: Properti dari Chart.js yang membuat garis prediksi menjadi putus-putus, sehingga pengguna dapat membedakan secara visual antara data masa lalu (garis solid) dan data masa depan (garis putus-putus).

### Step 3.6 — Error Handling & Fallback UI
**Fokus Kode:** Conditional Rendering
*   `if (loading) return <Skeleton />`: Menampilkan efek bayangan *loading* saat data masih diambil dari backend, agar website terlihat profesional dan dinamis.
*   `if (error) return <Badge>N/A</Badge>`: Menangani skenario terburuk secara elegan. Jika backend mati, alih-alih menampilkan halaman putih error yang menakutkan, sistem menampilkan *badge* kecil bertuliskan "N/A" (Not Available).
