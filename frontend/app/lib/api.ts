/**
 * api.ts — Centralized API client for CabaiWatch
 * Connects Next.js frontend to FastAPI backend at localhost:8000
 */

const BASE_URL = "http://localhost:8000";

// ─── Types ───────────────────────────────────────────────────────────────────

export interface PrediksiItem {
  horizon: string;
  keterangan: string;
  tanggal_prediksi: string;
  prediksi_rp: number;
  model_version: string;
  arah_prediksi?: string | null;
  confidence_arah?: number | null;
  error?: string;
}

export interface PrediksiAllResponse {
  status: string;
  tanggal_base: string;
  prediksi: PrediksiItem[];
}

export interface PrediksiSingleResponse {
  status: string;
  horizon: string;
  keterangan: string;
  tanggal_prediksi: string;
  prediksi_rp: number;
  model_version: string;
  arah_prediksi?: string | null;
  confidence_arah?: number | null;
}

export interface HistorisDataPoint {
  tanggal: string;
  harga_cabai_merah: number;
}

export interface HistorisResponse {
  status: string;
  n_hari: number;
  data: HistorisDataPoint[];
}

export interface MetrikHorizon {
  MAE: number;
  RMSE: number;
  MAPE: number;
  sMAPE: number;
  R2: number;
  DA: number;
}

export interface MetrikResponse {
  status: string;
  metrik: {
    h1: MetrikHorizon;
    h3: MetrikHorizon;
    h7: MetrikHorizon;
  };
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  models: {
    h1: boolean;
    h3: boolean;
    h7: boolean;
  };
  scaler: boolean;
  dataset: boolean;
  message: string;
}

export interface ModelInfoHorizon {
  model_file: string;
  tanggal_training: string;
  jumlah_fitur: number;
  feature_cols_file: string;
  error?: string;
}

export interface ModelInfoResponse {
  status: string;
  info: {
    h1: ModelInfoHorizon;
    h3: ModelInfoHorizon;
    h7: ModelInfoHorizon;
  };
  dataset_info: {
    total_baris: number;
    tanggal_min: string;
    tanggal_max: string;
    error?: string;
  };
}

export interface DashboardResponse {
  tanggal_update: string;
  harga_hari_ini: number;
  harga_min_30hari: number;
  harga_max_30hari: number;
  harga_rata_30hari: number;
  tren: string;
  prediksi_h1: number | null;
  prediksi_h3: number | null;
  prediksi_h7: number | null;
  status_model: boolean;
  n_model_aktif: number;
  status_inflasi: string;
}

export interface HistoryRecord {
  tanggal: string;
  harga_cabai_merah: number;
  harga_cabai_rawit?: number;
}

// ─── Fetch Helper ────────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);

  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      ...options,
      signal: controller.signal,
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

// ─── API Functions ───────────────────────────────────────────────────────────

/** Dashboard summary (harga hari ini, tren, prediksi) */
export async function fetchDashboard(): Promise<DashboardResponse> {
  return apiFetch<DashboardResponse>("/api/dashboard/");
}

/** Prediksi semua horizon sekaligus */
export async function fetchPrediksiAll(): Promise<PrediksiAllResponse> {
  return apiFetch<PrediksiAllResponse>("/api/predict/prediksi");
}

/** Prediksi satu horizon */
export async function fetchPrediksi(
  horizon: "h1" | "h3" | "h7"
): Promise<PrediksiSingleResponse> {
  return apiFetch<PrediksiSingleResponse>(
    `/api/predict/prediksi/${horizon}`
  );
}

/** Data historis untuk grafik (dari /api/predict/harga/historis) */
export async function fetchHistoris(
  nHari: number = 90
): Promise<HistorisResponse> {
  return apiFetch<HistorisResponse>(
    `/api/predict/harga/historis?n_hari=${nHari}`
  );
}

/** Riwayat harga dengan filter tanggal (dari /api/history/) */
export async function fetchHistory(params?: {
  start?: string;
  end?: string;
  limit?: number;
}): Promise<HistoryRecord[]> {
  const searchParams = new URLSearchParams();
  if (params?.start) searchParams.set("start", params.start);
  if (params?.end) searchParams.set("end", params.end);
  if (params?.limit) searchParams.set("limit", String(params.limit));

  const query = searchParams.toString();
  return apiFetch<HistoryRecord[]>(
    `/api/history/${query ? `?${query}` : ""}`
  );
}

/** Metrik akurasi model (MAE, RMSE, MAPE, R², DA) */
export async function fetchMetrik(): Promise<MetrikResponse> {
  return apiFetch<MetrikResponse>("/api/predict/model/metrik");
}

/** Health check sistem */
export async function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/api/predict/health");
}

/** Info detail model */
export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  return apiFetch<ModelInfoResponse>("/api/predict/model/info");
}
