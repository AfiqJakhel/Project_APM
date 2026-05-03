# Troubleshooting: Feature Name Mismatch

## ❌ Error

```
The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- bulan
- is_pasca_lebaran
- is_pra_lebaran
- lag_14
- roll_mean_7
```

## 🔍 Root Cause

Model XGBoost dilatih dengan `enable_categorical=True` atau XGBoost versi >= 1.6 yang menyimpan feature names saat training. Saat inference, XGBoost memvalidasi bahwa feature names harus sama persis.

### Analisis

1. **Model expect**: 35 fitur (termasuk `bulan`, `is_pasca_lebaran`, dll)
2. **Dataset punya**: Semua 35 fitur ada di `dataset_preprocessed.csv`
3. **Masalah**: XGBoost strict validation mendeteksi feature names tidak match

### Penyebab Teknis

- Model dilatih dengan DataFrame yang punya feature names
- Saat inference, meskipun kita pass numpy array, XGBoost tetap validate
- XGBoost menyimpan feature names di model file (.pkl)

## ✅ Solusi

### Solusi 1: Retrain Model (RECOMMENDED)

Retrain model dengan setting yang benar:

```python
# Di train.py, tambahkan parameter ini saat create model:
model = XGBRegressor(
    ...
    enable_categorical=False,  # Disable categorical features
    # ATAU
    # Pastikan feature names konsisten
)

# Atau gunakan numpy array saat training:
X_train_np = X_train.values  # Convert ke numpy
model.fit(X_train_np, y_train)
```

**Langkah-langkah:**

```bash
cd Backend
python machine_learning/train.py
```

Model baru akan disimpan dengan timestamp terbaru dan otomatis diload oleh sistem.

### Solusi 2: Disable Feature Name Validation (Workaround)

Edit model yang sudah ada untuk disable validation:

```python
import joblib

# Load model
model = joblib.load('model_final_h1_20260503.pkl')

# Disable feature names
if hasattr(model, 'feature_names_in_'):
    delattr(model, 'feature_names_in_')

# Save kembali
joblib.dump(model, 'model_final_h1_20260503_fixed.pkl')
```

### Solusi 3: Downgrade XGBoost (Not Recommended)

```bash
pip install xgboost==1.5.2
```

Tapi ini tidak recommended karena kehilangan fitur baru XGBoost.

## 🔧 Quick Fix Script

Buat file `Backend/fix_model_features.py`:

```python
"""
fix_model_features.py
=====================
Script untuk menghapus feature_names_in_ dari model XGBoost
agar tidak strict validate feature names saat inference.
"""

import joblib
from pathlib import Path

MODEL_DIR = Path("machine_learning/output/expanding_window")

for horizon in ["h1", "h3", "h7"]:
    model_files = sorted(MODEL_DIR.glob(f"model_final_{horizon}_*.pkl"), reverse=True)

    if not model_files:
        print(f"❌ Model {horizon} tidak ditemukan")
        continue

    model_path = model_files[0]
    print(f"\n🔧 Processing: {model_path.name}")

    # Load model
    model = joblib.load(model_path)

    # Check if has feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        print(f"   Feature names found: {len(model.feature_names_in_)} features")
        print(f"   Removing feature_names_in_...")
        delattr(model, 'feature_names_in_')

    # Save dengan nama baru
    new_path = model_path.parent / f"{model_path.stem}_fixed.pkl"
    joblib.dump(model, new_path)
    print(f"   ✅ Saved to: {new_path.name}")

print("\n✅ Done! Restart server untuk load model baru.")
print("   Atau rename file *_fixed.pkl menjadi model_final_*.pkl")
```

Jalankan:

```bash
cd Backend
python fix_model_features.py
```

## 📋 Verification

### 1. Cek Feature Names di Model

```python
import joblib
model = joblib.load('model_final_h1_20260503.pkl')
print(hasattr(model, 'feature_names_in_'))  # Should be False after fix
```

### 2. Test Prediksi

```bash
curl http://localhost:8000/api/predict/prediksi/h1
```

Expected output:

```json
{
  "status": "success",
  "horizon": "h1",
  "prediksi_rp": 45000.0,
  ...
}
```

## 🎯 Best Practice untuk Training

Untuk menghindari masalah ini di masa depan:

```python
# train.py - Best practice
import numpy as np

# 1. Convert ke numpy array sebelum fit
X_train_np = X_train.values
X_val_np = X_val.values

# 2. Fit model dengan numpy array
model.fit(
    X_train_np, y_train,
    eval_set=[(X_train_np, y_train), (X_val_np, y_val)],
    verbose=False
)

# 3. Simpan feature_cols terpisah (sudah dilakukan)
with open(f"feature_cols_{label}.json", "w") as f:
    json.dump(list(X_train.columns), f)
```

## 📊 Status

- ✅ Health check: OK
- ✅ Data historis: OK
- ✅ Model info: OK
- ❌ Prediksi: FAIL (feature mismatch)

**Action Required**: Retrain model atau apply quick fix

---

**Last Updated**: 2026-05-03  
**Priority**: HIGH  
**Impact**: Prediksi endpoint tidak berfungsi
