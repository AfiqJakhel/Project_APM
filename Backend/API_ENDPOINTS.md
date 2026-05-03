# API Endpoints - Prediksi Harga Cabai Merah

## Base URL

```
http://localhost:8000/api/predict
```

---

## 📋 Daftar Endpoint

### **ENDPOINT UTAMA (Recommended)**

| Method | Path                  | Deskripsi                        | Response Time |
| ------ | --------------------- | -------------------------------- | ------------- |
| GET    | `/prediksi/{horizon}` | Prediksi otomatis (data terkini) | < 500ms       |
| GET    | `/prediksi`           | Prediksi semua horizon sekaligus | < 800ms       |
| GET    | `/harga/historis`     | Data historis untuk grafik       | < 300ms       |
| GET    | `/model/metrik`       | Metrik akurasi model             | < 100ms       |
| GET    | `/health`             | Health check sistem              | < 200ms       |
| GET    | `/model/info`         | Info detail model                | < 150ms       |

### **ENDPOINT LEGACY (Backward Compatibility)**

| Method | Path     | Deskripsi                               |
| ------ | -------- | --------------------------------------- |
| POST   | `/`      | Prediksi dengan request body            |
| GET    | `/semua` | Prediksi semua horizon (manual tanggal) |

### **ENDPOINT UTILITY**

| Method | Path                | Deskripsi                  |
| ------ | ------------------- | -------------------------- |
| GET    | `/tanggal-tersedia` | Rentang tanggal dataset    |
| GET    | `/fitur-terkini`    | Data fitur terkini         |
| GET    | `/cache-info`       | Info cache artifacts       |
| POST   | `/clear-cache`      | Hapus cache (reload model) |

---

## 📖 Detail Endpoint

### 1. GET `/prediksi/{horizon}` — Prediksi Otomatis ⭐

**Endpoint utama untuk prediksi harga.**

#### Request

```http
GET /api/predict/prediksi/h1
```

#### Path Parameters

- `horizon` (required): `h1` | `h3` | `h7`
  - `h1`: Prediksi harga besok (H+1)
  - `h3`: Prediksi harga 3 hari ke depan (H+3)
  - `h7`: Prediksi harga 7 hari ke depan (H+7)

#### Response (200 OK)

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

#### Error Responses

- **422 Unprocessable Entity**: Horizon tidak valid
  ```json
  {
    "detail": "Horizon tidak valid: 'h2'. Gunakan 'h1', 'h3', atau 'h7'."
  }
  ```
- **503 Service Unavailable**: Model tidak tersedia
  ```json
  {
    "detail": "Model h1 tidak tersedia. Pattern: model_final_h1_*.pkl"
  }
  ```

#### Contoh Penggunaan

```bash
# cURL
curl http://localhost:8000/api/predict/prediksi/h1

# JavaScript (Fetch)
fetch('http://localhost:8000/api/predict/prediksi/h1')
  .then(res => res.json())
  .then(data => console.log(data.prediksi_rp));

# Python (requests)
import requests
response = requests.get('http://localhost:8000/api/predict/prediksi/h1')
print(response.json()['prediksi_rp'])
```

---

### 2. GET `/prediksi` — Prediksi Semua Horizon ⭐

**Prediksi h1, h3, h7 sekaligus dalam satu request.**

#### Request

```http
GET /api/predict/prediksi
```

#### Response (200 OK)

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
    {
      "horizon": "h3",
      "keterangan": "Prediksi 3 Hari (H+3)",
      "tanggal_prediksi": "2026-05-06",
      "prediksi_rp": 46500.0,
      "model_version": "model_final_h3_20260503.pkl"
    },
    {
      "horizon": "h7",
      "keterangan": "Prediksi 7 Hari (H+7)",
      "tanggal_prediksi": "2026-05-10",
      "prediksi_rp": 48000.0,
      "model_version": "model_final_h7_20260503.pkl"
    }
  ]
}
```

#### Contoh Penggunaan (React)

```jsx
import { useEffect, useState } from "react";

function PrediksiDashboard() {
  const [prediksi, setPrediksi] = useState([]);

  useEffect(() => {
    fetch("http://localhost:8000/api/predict/prediksi")
      .then((res) => res.json())
      .then((data) => setPrediksi(data.prediksi));
  }, []);

  return (
    <div>
      {prediksi.map((p) => (
        <div key={p.horizon}>
          <h3>{p.keterangan}</h3>
          <p>Rp {p.prediksi_rp.toLocaleString()}</p>
          <small>{p.tanggal_prediksi}</small>
        </div>
      ))}
    </div>
  );
}
```

---

### 3. GET `/harga/historis` — Data Historis untuk Grafik ⭐

**Ambil data historis harga cabai untuk Chart.js / Recharts.**

#### Request

```http
GET /api/predict/harga/historis?n_hari=90
```

#### Query Parameters

- `n_hari` (optional): Jumlah hari terakhir (default: 90, max: 365)

#### Response (200 OK)

```json
{
  "status": "success",
  "n_hari": 90,
  "data": [
    {
      "tanggal": "2026-02-03",
      "harga_cabai_merah": 42000.0
    },
    {
      "tanggal": "2026-02-04",
      "harga_cabai_merah": 43500.0
    },
    ...
  ]
}
```

#### Contoh Penggunaan (Chart.js)

```javascript
fetch("http://localhost:8000/api/predict/harga/historis?n_hari=30")
  .then((res) => res.json())
  .then((response) => {
    const labels = response.data.map((d) => d.tanggal);
    const prices = response.data.map((d) => d.harga_cabai_merah);

    new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Harga Cabai Merah",
            data: prices,
            borderColor: "rgb(255, 99, 132)",
          },
        ],
      },
    });
  });
```

---

### 4. GET `/model/metrik` — Metrik Akurasi Model ⭐

**Ambil metrik evaluasi model untuk ditampilkan di dashboard.**

#### Request

```http
GET /api/predict/model/metrik
```

#### Response (200 OK)

```json
{
  "status": "success",
  "metrik": {
    "h1": {
      "MAE": 2500.0,
      "RMSE": 3200.0,
      "MAPE": 5.2,
      "sMAPE": 5.1,
      "R2": 0.92,
      "DA": 78.5
    },
    "h3": {
      "MAE": 3200.0,
      "RMSE": 4100.0,
      "MAPE": 6.8,
      "sMAPE": 6.5,
      "R2": 0.88,
      "DA": 72.3
    },
    "h7": {
      "MAE": 4500.0,
      "RMSE": 5800.0,
      "MAPE": 9.2,
      "sMAPE": 8.9,
      "R2": 0.82,
      "DA": 68.1
    }
  }
}
```

#### Penjelasan Metrik

- **MAE** (Mean Absolute Error): Rata-rata kesalahan dalam Rupiah. Semakin kecil semakin baik.
- **RMSE** (Root Mean Squared Error): Akar rata-rata kuadrat kesalahan. Lebih sensitif terhadap outlier.
- **MAPE** (Mean Absolute Percentage Error): Persentase kesalahan rata-rata.
- **sMAPE** (Symmetric MAPE): MAPE yang lebih seimbang untuk nilai naik/turun.
- **R²** (R-squared): Koefisien determinasi (0-1). Semakin tinggi semakin baik.
- **DA** (Directional Accuracy): Persentase prediksi arah naik/turun yang benar.

---

### 5. GET `/health` — Health Check ⭐

**Cek kesehatan sistem prediksi.**

#### Request

```http
GET /api/predict/health
```

#### Response (200 OK - Healthy)

```json
{
  "status": "healthy",
  "timestamp": "2026-05-03T10:30:00",
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

#### Response (200 OK - Degraded)

```json
{
  "status": "degraded",
  "timestamp": "2026-05-03T10:30:00",
  "models": {
    "h1": true,
    "h3": false,
    "h7": true
  },
  "scaler": true,
  "dataset": true,
  "message": "Beberapa model tidak tersedia"
}
```

#### Response (503 Service Unavailable - Unhealthy)

```json
{
  "status": "unhealthy",
  "timestamp": "2026-05-03T10:30:00",
  "models": {
    "h1": false,
    "h3": false,
    "h7": false
  },
  "scaler": false,
  "dataset": false,
  "message": "Komponen kritis tidak tersedia"
}
```

---

### 6. GET `/model/info` — Info Detail Model ⭐

**Ambil informasi detail tentang model yang digunakan.**

#### Request

```http
GET /api/predict/model/info
```

#### Response (200 OK)

```json
{
  "status": "success",
  "info": {
    "h1": {
      "model_file": "model_final_h1_20260503.pkl",
      "tanggal_training": "2026-05-03",
      "jumlah_fitur": 42,
      "feature_cols_file": "feature_cols_h1.json"
    },
    "h3": {
      "model_file": "model_final_h3_20260503.pkl",
      "tanggal_training": "2026-05-03",
      "jumlah_fitur": 42,
      "feature_cols_file": "feature_cols_h3.json"
    },
    "h7": {
      "model_file": "model_final_h7_20260503.pkl",
      "tanggal_training": "2026-05-03",
      "jumlah_fitur": 42,
      "feature_cols_file": "feature_cols_h7.json"
    }
  },
  "dataset_info": {
    "total_baris": 1250,
    "tanggal_min": "2022-01-01",
    "tanggal_max": "2026-05-03"
  }
}
```

---

## 🔧 Endpoint Utility

### GET `/tanggal-tersedia`

Rentang tanggal yang tersedia di dataset.

### GET `/fitur-terkini`

Data fitur terkini (baris terakhir dataset).

### GET `/cache-info`

Informasi artifacts yang di-cache di memory.

### POST `/clear-cache`

Hapus cache dan reload model (tanpa restart server).

---

## 🚀 Quick Start

### 1. Start Server

```bash
cd Backend
uvicorn app.main:app --reload --port 8000
```

### 2. Test Endpoint

```bash
# Health check
curl http://localhost:8000/api/predict/health

# Prediksi H+1
curl http://localhost:8000/api/predict/prediksi/h1

# Prediksi semua horizon
curl http://localhost:8000/api/predict/prediksi

# Data historis 30 hari
curl http://localhost:8000/api/predict/harga/historis?n_hari=30

# Metrik model
curl http://localhost:8000/api/predict/model/metrik
```

### 3. Swagger UI

Buka browser: http://localhost:8000/docs

---

## 📊 Response Time Target

| Endpoint              | Target  | Typical |
| --------------------- | ------- | ------- |
| `/prediksi/{horizon}` | < 500ms | ~200ms  |
| `/prediksi`           | < 800ms | ~400ms  |
| `/harga/historis`     | < 300ms | ~150ms  |
| `/model/metrik`       | < 100ms | ~50ms   |
| `/health`             | < 200ms | ~80ms   |
| `/model/info`         | < 150ms | ~70ms   |

---

## ⚠️ Error Handling

### Status Codes

- **200 OK**: Request berhasil
- **400 Bad Request**: Input tidak valid
- **422 Unprocessable Entity**: Validasi gagal (horizon tidak valid)
- **500 Internal Server Error**: Error server
- **503 Service Unavailable**: Model/scaler tidak tersedia

### Error Response Format

```json
{
  "detail": "Pesan error yang jelas dan informatif"
}
```

---

## 🔐 CORS Configuration

CORS sudah dikonfigurasi di `main.py` untuk:

- `http://localhost:5173` (Vite)
- `http://localhost:3000` (Create React App)
- `http://127.0.0.1:5173`
- `http://127.0.0.1:3000`

---

## 📝 Notes

1. **Caching**: Model, scaler, dan feature_cols di-cache di memory untuk performa optimal
2. **Auto-reload**: Gunakan `/clear-cache` untuk reload model tanpa restart server
3. **Monitoring**: Gunakan `/health` untuk monitoring sistem
4. **Backward Compatibility**: Endpoint legacy tetap tersedia untuk aplikasi lama

---

**Last Updated**: 2026-05-03  
**API Version**: 1.0.0  
**Framework**: FastAPI 0.104+
