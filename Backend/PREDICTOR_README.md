# Core Predictor Module

## Struktur Baru

```
Backend/app/core/
├── __init__.py
├── model_loader.py      # (legacy, masih digunakan di main.py untuk startup)
└── predictor.py         # ✨ BARU: Core business logic prediksi
```

## File: `predictor.py`

### Fungsi Utama

#### 1. **Load Artifacts (dengan caching)**

```python
load_scaler()                    # Load RobustScaler
load_model(label)                # Load model XGBoost (h1/h3/h7)
load_feature_cols(label)         # Load daftar fitur
load_all_artifacts(label)        # Load semua sekaligus
```

#### 2. **Prediksi**

```python
prediksi_harga(input_data, label="h1")
# Input:  dict berisi nilai fitur
# Output: {
#   "horizon": "h1",
#   "prediksi_rp": 45000.0,
#   "tanggal_prediksi": "2024-07-02",
#   "model_version": "model_final_h1_20240701.pkl",
#   "fitur_digunakan": 42,
#   "fitur_missing": 0
# }

prediksi_semua_horizon(input_data)
# Prediksi h1, h3, h7 sekaligus
```

#### 3. **Data Helpers**

```python
get_fitur_terkini()              # Ambil baris terakhir dataset
get_metrik_model()               # Ambil MAE, RMSE, MAPE, dll
get_data_historis(n_hari=90)     # Ambil data untuk grafik
```

#### 4. **Utilitas**

```python
clear_cache()                    # Hapus cache (reload model)
get_cache_info()                 # Info artifacts yang di-cache
validate_horizon(label)          # Validasi h1/h3/h7
```

---

## File: `routes/predict.py` (Updated)

### Endpoint Baru

| Method | Path                            | Deskripsi                   |
| ------ | ------------------------------- | --------------------------- |
| POST   | `/api/predict/`                 | Prediksi satu horizon       |
| GET    | `/api/predict/semua`            | Prediksi semua horizon      |
| GET    | `/api/predict/tanggal-tersedia` | Rentang tanggal dataset     |
| GET    | `/api/predict/fitur-terkini`    | Data terkini untuk pre-fill |
| GET    | `/api/predict/metrik`           | Metrik evaluasi model       |
| GET    | `/api/predict/historis`         | Data historis untuk grafik  |
| GET    | `/api/predict/cache-info`       | Info cache artifacts        |
| POST   | `/api/predict/clear-cache`      | Hapus cache (reload model)  |

---

## Keunggulan Arsitektur Baru

### ✅ Separation of Concerns

- **`predictor.py`**: Business logic (prediksi, load data, metrik)
- **`predict.py`**: HTTP routing (request/response handling)
- **`model_loader.py`**: Startup initialization (legacy, bisa di-migrate)

### ✅ Caching Otomatis

- Model, scaler, feature_cols di-cache di memory
- Tidak reload tiap request → performa tinggi
- Cache bisa di-clear manual via endpoint

### ✅ Error Handling Konsisten

- Semua fungsi raise `HTTPException` dengan status code yang tepat
- Logging otomatis untuk debugging
- Pesan error yang jelas dan informatif

### ✅ Type Hints & Docstrings

- Semua fungsi punya type hints
- Docstring menjelaskan input/output/raises
- Mudah di-maintain dan di-extend

### ✅ Testable

- Business logic terpisah dari FastAPI
- Bisa di-test tanpa perlu run server
- Mock-friendly (Path, joblib, pandas)

---

## Cara Pakai

### Di Routes (FastAPI)

```python
from app.core import predictor

# Prediksi
hasil = predictor.prediksi_harga(input_data, "h1")

# Ambil data terkini
data = predictor.get_fitur_terkini()

# Ambil metrik
metrik = predictor.get_metrik_model()
```

### Standalone (Testing)

```python
from Backend.app.core import predictor

# Load artifacts
predictor.load_scaler()
predictor.load_model("h1")

# Prediksi
input_data = {"lag_1": 45000, "suhu_rata": 27.5, ...}
hasil = predictor.prediksi_harga(input_data, "h1")
print(hasil["prediksi_rp"])
```

---

## Migration dari `model_loader.py`

### Sebelum (model_loader.py)

```python
model_store.load()
model, feature_cols = model_store.get_model("h1")
scaler = model_store.scaler
```

### Sesudah (predictor.py)

```python
hasil = predictor.prediksi_harga(input_data, "h1")
# Semua loading otomatis + caching
```

---

## Testing

### 1. Test Load Artifacts

```bash
cd Backend
python -c "from app.core import predictor; predictor.load_scaler(); print('OK')"
```

### 2. Test Prediksi

```bash
python -c "
from app.core import predictor
data = predictor.get_fitur_terkini()
hasil = predictor.prediksi_harga(data, 'h1')
print(f'Prediksi: Rp {hasil[\"prediksi_rp\"]:,.0f}')
"
```

### 3. Test Endpoint

```bash
# Start server
uvicorn app.main:app --reload

# Test endpoint
curl http://localhost:8000/api/predict/fitur-terkini
curl http://localhost:8000/api/predict/metrik
curl http://localhost:8000/api/predict/historis?n_hari=30
```

---

## Troubleshooting

### Error: "Scaler tidak tersedia"

```bash
# Jalankan preprocessing untuk generate scaler.pkl
cd Backend
python machine_learning/preprocessing.py
```

### Error: "Model h1 tidak tersedia"

```bash
# Jalankan training untuk generate model
cd Backend
python machine_learning/train.py
```

### Cache tidak update setelah retrain

```bash
# Clear cache via endpoint
curl -X POST http://localhost:8000/api/predict/clear-cache

# Atau restart server
```

---

## Next Steps

1. ✅ Migrate `model_loader.py` logic ke `predictor.py`
2. ✅ Update `predict.py` routes untuk pakai `predictor.py`
3. ⏳ Tambahkan unit tests untuk `predictor.py`
4. ⏳ Tambahkan endpoint untuk upload model baru
5. ⏳ Tambahkan monitoring metrics (latency, cache hit rate)

---

## File Changes Summary

| File                                        | Status     | Changes                                    |
| ------------------------------------------- | ---------- | ------------------------------------------ |
| `Backend/app/core/predictor.py`             | ✨ NEW     | Core business logic                        |
| `Backend/app/routes/predict.py`             | 🔄 UPDATED | Pakai predictor.py, tambah 5 endpoint baru |
| `Backend/app/core/model_loader.py`          | ⚠️ LEGACY  | Masih dipakai di main.py startup           |
| `Backend/machine_learning/preprocessing.py` | ✅ FIXED   | Hapus dataset_scaled.csv                   |
| `Backend/machine_learning/train.py`         | ✅ FIXED   | Dokumentasi dataset_preprocessed.csv       |

---

**Author**: Kiro AI  
**Date**: 2026-05-03  
**Version**: 1.0.0
