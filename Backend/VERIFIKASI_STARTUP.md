# Verifikasi Startup Backend

## ✅ Masalah Terselesaikan

### Masalah Awal

```
[Startup] WARNING: Tidak ada model yang berhasil dimuat!
! model_final_h1.pkl tidak ditemukan
! model_final_h3.pkl tidak ditemukan
! model_final_h7.pkl tidak ditemukan
```

### Penyebab

- `settings.py` mengarah ke folder `xgboost_models`
- Model sebenarnya ada di folder `expanding_window`
- Pattern nama file tidak sesuai (mencari `model_final_h1.pkl` padahal file bernama `model_final_h1_20260503.pkl`)

### Solusi

1. ✅ Update `settings.py`:
   - `MODEL_DIR` → `expanding_window`
   - `SCALER_DIR` → `xgboost_models` (terpisah)
   - Pattern file → `model_final_h1_*.pkl` (glob pattern)

2. ✅ Update `model_loader.py`:
   - Gunakan `glob()` untuk mencari file terbaru
   - Sort descending untuk ambil file terbaru
   - Pisahkan path scaler dan model

3. ✅ Update `predictor.py`:
   - Sesuaikan path SCALER_DIR

---

## 🚀 Cara Menjalankan

### 1. Start Server

```bash
cd Backend
uvicorn app.main:app --reload --port 8000
```

### Output yang Diharapkan

```
[Startup] Memuat model XGBoost...
[ModelLoader] Memuat model...
  + scaler.pkl dimuat dari .../xgboost_models/scaler.pkl
  + model_h1 dimuat: model_final_h1_20260503.pkl (35 fitur)
  + model_h3 dimuat: model_final_h3_20260503.pkl (35 fitur)
  + model_h7 dimuat: model_final_h7_20260503.pkl (35 fitur)
[ModelLoader] Selesai — 3 model aktif
[Startup] Model siap: ['h1', 'h3', 'h7']
INFO:     Application startup complete.
```

---

## 🧪 Testing

### Test 1: Health Check

```bash
curl http://localhost:8000/api/predict/health
```

**Expected Output:**

```json
{
  "status": "healthy",
  "timestamp": "2026-05-03T...",
  "models": {
    "h1": true,
    "h3": true,
    "h7": true
  },
  "scaler": true,
  "dataset": true,
  "message": "Semua komponen tersedia"
}
```

### Test 2: Prediksi H+1

```bash
curl http://localhost:8000/api/predict/prediksi/h1
```

**Expected Output:**

```json
{
  "status": "success",
  "horizon": "h1",
  "keterangan": "Prediksi Besok (H+1)",
  "tanggal_prediksi": "2026-05-04",
  "prediksi_rp": 45000.0,
  "model_version": "model_final_h1_20260503.pkl"
}
```

### Test 3: Prediksi Semua Horizon

```bash
curl http://localhost:8000/api/predict/prediksi
```

**Expected Output:**

```json
{
  "status": "success",
  "tanggal_base": "2026-05-03",
  "prediksi": [
    {
      "horizon": "h1",
      "keterangan": "Prediksi Besok (H+1)",
      "tanggal_prediksi": "2026-05-04",
      "prediksi_rp": 45000.0,
      "model_version": "model_final_h1_20260503.pkl"
    },
    ...
  ]
}
```

### Test 4: Model Info

```bash
curl http://localhost:8000/api/predict/model/info
```

**Expected Output:**

```json
{
  "status": "success",
  "info": {
    "h1": {
      "model_file": "model_final_h1_20260503.pkl",
      "tanggal_training": "2026-05-03",
      "jumlah_fitur": 35,
      "feature_cols_file": "feature_cols_h1.json"
    },
    ...
  },
  "dataset_info": {
    "total_baris": 1250,
    "tanggal_min": "2022-01-01",
    "tanggal_max": "2026-05-03"
  }
}
```

### Test 5: Swagger UI

Buka browser: http://localhost:8000/docs

---

## 📁 Struktur File

```
Backend/
├── machine_learning/
│   └── output/
│       ├── expanding_window/          ← MODEL_DIR (model .pkl)
│       │   ├── model_final_h1_20260503.pkl
│       │   ├── model_final_h3_20260503.pkl
│       │   ├── model_final_h7_20260503.pkl
│       │   ├── feature_cols_h1.json
│       │   ├── feature_cols_h3.json
│       │   └── feature_cols_h7.json
│       └── xgboost_models/            ← SCALER_DIR (scaler.pkl)
│           └── scaler.pkl
├── data/
│   └── processed/
│       └── dataset_preprocessed.csv
└── app/
    ├── core/
    │   ├── model_loader.py            ← Startup loader (legacy)
    │   └── predictor.py               ← Core business logic
    └── routes/
        └── predict.py                 ← FastAPI endpoints
```

---

## ⚠️ Troubleshooting

### Error: "scaler.pkl tidak ditemukan"

```bash
# Jalankan preprocessing
cd Backend
python machine_learning/preprocessing.py
```

### Error: "model*final_h1*\*.pkl tidak ditemukan"

```bash
# Jalankan training
cd Backend
python machine_learning/train.py
```

### Error: "dataset_preprocessed.csv tidak ditemukan"

```bash
# Jalankan preprocessing terlebih dahulu
cd Backend
python machine_learning/preprocessing.py
```

### Server tidak bisa start

```bash
# Cek dependencies
pip install -r requirements.txt

# Cek Python version (minimal 3.9)
python --version

# Cek port 8000 tidak dipakai
netstat -ano | findstr :8000
```

---

## 🔄 Update Model Tanpa Restart

Jika Anda melatih model baru:

1. **Training menghasilkan file baru:**

   ```
   model_final_h1_20260504.pkl  (baru)
   model_final_h1_20260503.pkl  (lama)
   ```

2. **Clear cache via API:**

   ```bash
   curl -X POST http://localhost:8000/api/predict/clear-cache
   ```

3. **Model baru otomatis diload** (file terbaru dipilih)

---

## 📊 Performance Check

### Response Time

```bash
# Install httpie
pip install httpie

# Test response time
time http GET http://localhost:8000/api/predict/prediksi/h1
```

**Target:**

- Health check: < 200ms
- Prediksi single: < 500ms
- Prediksi semua: < 800ms

### Memory Usage

```bash
# Windows
tasklist | findstr python

# Linux/Mac
ps aux | grep python
```

**Expected:** ~200-300 MB (dengan 3 model di-cache)

---

## ✅ Checklist Startup

- [ ] Scaler.pkl dimuat
- [ ] Model h1 dimuat
- [ ] Model h3 dimuat
- [ ] Model h7 dimuat
- [ ] Dataset tersedia
- [ ] Health check return "healthy"
- [ ] Prediksi h1 berhasil
- [ ] Swagger UI accessible

---

**Last Updated**: 2026-05-03  
**Status**: ✅ RESOLVED
