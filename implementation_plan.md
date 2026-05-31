# Peningkatan Evaluasi Model XGBoost

Dokumen ini merangkum rencana teknis untuk mengimplementasikan strategi peningkatan evaluasi model pemodelan harga cabai (Merah & Rawit).

## User Review Required

> [!IMPORTANT]
> **Modifikasi pada Model Training:**
> Kita akan menambahkan `optuna` ke dalam project untuk tuning model yang jauh lebih optimal, menggantikan `RandomizedSearchCV`. Optuna sangat efektif dalam menemukan parameter yang ideal (khususnya nilai L1/L2 regularization pada range yang sangat kecil).
> Proses install `optuna` akan ditambahkan di `requirements.txt`. Apakah Anda setuju dengan penambahan library ini?

## Proposed Changes

### 1. `Backend/requirements.txt`
Menambahkan dependensi baru untuk Bayesian Optimization.
#### [MODIFY] [requirements.txt](file:///c:/Data%20Afiq/Tugas/APM/Project%20TB%20APM/Backend/requirements.txt)
- Menambahkan baris `optuna==4.1.0` (atau versi stabil terbaru).

---

### 2. `Backend/machine_learning/preprocessing.py`
Fokus pada rekayasa fitur dengan menambahkan bobot harga kekinian. (Catatan: Fitur volatilitas dan rasio harga ternyata sudah diimplementasikan di versi `preprocessing.py` saat ini).
#### [MODIFY] [preprocessing.py](file:///c:/Data%20Afiq/Tugas/APM/Project%20TB%20APM/Backend/machine_learning/preprocessing.py)
- **Exponential Moving Average (EMA):** Menambahkan `ema_7` dan `ema_30` untuk memberikan bobot eksponensial pada tren harga terbaru (menggunakan fungsi `.ewm(span=7).mean()`). EMA lebih responsif terhadap guncangan harga (price shock) mendadak dibanding `roll_mean`.

---

### 3. `Backend/machine_learning/train.py`
Fokus pada pergantian algoritma hyperparameter tuning dan penambahan pembobotan data.
#### [MODIFY] [train.py](file:///c:/Data%20Afiq/Tugas/APM/Project%20TB%20APM/Backend/machine_learning/train.py)
- **Tuning via Optuna:** Memodifikasi `tuning_final()` agar menggunakan `optuna.create_study()`. Fungsi objektif Optuna akan melakukan evaluasi metrik MAE melintasi iterasi `TimeSeriesSplit`. Ini jauh lebih cepat dan akurat dibanding *Random Search*.
- **Time-Decay Sample Weights:** Pada `latih_model_final()` dan `train_xgb_window()`, membuat array `sample_weight` berbasis eksponensial untuk diberikan kepada XGBoost saat `.fit()`. Data yang lebih baru akan mendapat bobot `> 1.0` dan data lama mendapat bobot `< 1.0`. Ini membantu model agar beradaptasi dengan inflasi dan pergeseran pola konsumsi terbaru (Concept Drift).

## Verification Plan

### Automated / Metrik
- Menjalankan ulang `python machine_learning/preprocessing.py` untuk menghasilkan dataset dengan kolom fitur EMA baru.
- Menjalankan ulang `python machine_learning/train.py` tanpa *cache* (`--no-cache`) untuk memaksa Optuna melatih ulang hiperparameter terbaik.
- Membandingkan hasil di `ringkasan_evaluasi_final.csv` (Atau metrik terminal) dengan hasil run yang baru saja selesai untuk memastikan terdapat penurunan MAE/RMSE.
