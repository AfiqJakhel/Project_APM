# Perubahan Objektif Model: Optimasi R2 (MSE vs MAE)

Dokumen ini merangkum rencana teknis untuk mengubah objektif model dari meminimalkan *Mean Absolute Error* (MAE) menjadi *Mean Squared Error* (MSE) agar model secara matematis sejalan dengan metrik $R^2$.

## User Review Required

> [!WARNING]
> **Trade-off Kritis (Harap Dibaca):**
> Mengubah objektif dari `reg:pseudohubererror` (MAE-like) ke `reg:squarederror` (MSE) akan membuat model sangat reaktif terhadap *outlier* (lonjakan harga Lebaran/bencana). Secara metrik, nilai $R^2$ akan meroket (membaik mendekati nilai positif), **NAMUN** prediksi harga harian (dalam Rupiah) kemungkinan besar akan menjadi lebih fluktuatif dan MAE-nya memburuk. Jika Anda setuju dengan pengorbanan ini, beritahu saya untuk mulai!

## Proposed Changes

### `Backend/machine_learning/train.py`
Fokus pada fungsi `tuning_final()`, `train_xgb_window()`, dan `latih_model_final()`.

#### [MODIFY] [train.py](file:///c:/Data%20Afiq/Tugas/APM/Project%20TB%20APM/Backend/machine_learning/train.py)
1. **Fungsi `tuning_final`**:
   - Mengganti `from sklearn.metrics import mean_absolute_error` menjadi `mean_squared_error`.
   - Mengubah nama variabel `maes` menjadi `mses`.
   - Mengubah kalkulasi evaluasi Optuna menjadi `mean_squared_error(y_va, preds)`.
   - Mengubah format *logging* terminal dari `Best MAE (CV)` menjadi `Best MSE (CV)`.
   - Mengubah argumen XGBoost di dalam Optuna dari `objective="reg:pseudohubererror"` menjadi `objective="reg:squarederror"`.

2. **Fungsi `train_xgb_window`**:
   - Mengubah `objective="reg:pseudohubererror"` menjadi `objective="reg:squarederror"`.
   - Mengubah `eval_metric="mae"` menjadi `eval_metric="rmse"`.

3. **Fungsi `latih_model_final`**:
   - Mengubah `objective="reg:pseudohubererror"` menjadi `objective="reg:squarederror"`.

## Verification Plan

### Automated / Metrik
- Menjalankan ulang perintah: `python machine_learning/train.py --no-cache`.
- Mengawasi keluaran terminal Optuna: akan menampilkan `Best MSE (CV)` dengan nilai log-return yang sudah dikuadratkan.
- Mengecek nilai `xgb_R2` di terminal pada evaluasi *Expanding Window*. Target utamanya adalah R2 akan bergeser dari rentang negatif ekstrim ke angka yang lebih positif atau mendekati nol.
- Membandingkan nilai MAE terbaru dengan MAE versi Optuna sebelumnya (versi H1 merah = Rp1.456) untuk membuktikan efek *trade-off* outlier.
