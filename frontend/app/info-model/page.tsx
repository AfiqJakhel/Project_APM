"use client";

import { useEffect, useState } from "react";
import {
  fetchModelInfo,
  fetchHealth,
  type ModelInfoResponse,
  type HealthResponse,
} from "../lib/api";

/* ══════════════════════════════════════════
   INFO MODEL PAGE
   ══════════════════════════════════════════ */
export default function InfoModelPage() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const [infoData, healthData] = await Promise.allSettled([
          fetchModelInfo(),
          fetchHealth(),
        ]);
        if (infoData.status === "fulfilled") setInfo(infoData.value);
        if (healthData.status === "fulfilled") setHealth(healthData.value);
        if (infoData.status === "rejected" && healthData.status === "rejected") {
          setError("Tidak bisa terhubung ke backend");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Gagal memuat info model");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const statusColor = (ok: boolean) => (ok ? "var(--green)" : "var(--red)");
  const statusText = (ok: boolean) => (ok ? "✅ Aktif" : "❌ Tidak tersedia");

  const horizonColors: Record<string, string> = {
    h1: "#0F6E56",
    h3: "#3B82F6",
    h7: "#7C3AED",
  };

  return (
    <>
      <header className="topbar">
        <div className="topbar-left">
          <h1>Informasi Model</h1>
          <p>Detail teknis model XGBoost dan status sistem</p>
        </div>
        <div className="topbar-right">
          {health && (
            <span
              className={`badge ${
                health.status === "healthy"
                  ? "badge-green"
                  : health.status === "degraded"
                  ? "badge-yellow"
                  : "badge-red"
              }`}
              style={{ fontSize: 11, padding: "5px 12px" }}
            >
              {health.status.toUpperCase()}
            </span>
          )}
        </div>
      </header>

      <div className="content-area">
        {loading && (
          <div className="loading-container animate-in delay-1">
            <div className="loading-spinner" />
            <p>Memuat informasi model...</p>
          </div>
        )}

        {error && (
          <div className="alert-box" style={{ borderColor: "var(--red-muted)", background: "var(--red-light)" }}>
            <p className="alert-text"><strong>Error:</strong> {error}</p>
          </div>
        )}

        {!loading && (
          <>
            {/* System Health */}
            {health && (
              <div className="card animate-in delay-1">
                <div className="card-header">
                  <h3>Status Sistem</h3>
                  <span
                    className={`badge ${
                      health.status === "healthy"
                        ? "badge-green"
                        : health.status === "degraded"
                        ? "badge-yellow"
                        : "badge-red"
                    }`}
                  >
                    {health.message}
                  </span>
                </div>
                <div className="card-body">
                  <div className="metrics-grid" style={{ gridTemplateColumns: "repeat(5, 1fr)" }}>
                    {(["h1", "h3", "h7"] as const).map((h) => (
                      <div key={h} className="metric-card" style={{ border: `2px solid ${health.models[h] ? "var(--green-muted)" : "var(--red-muted)"}` }}>
                        <div className="metric-label">Model {h.toUpperCase()}</div>
                        <div className="metric-value" style={{ fontSize: 16, color: statusColor(health.models[h]) }}>
                          {statusText(health.models[h])}
                        </div>
                      </div>
                    ))}
                    <div className="metric-card" style={{ border: `2px solid ${health.scaler ? "var(--green-muted)" : "var(--red-muted)"}` }}>
                      <div className="metric-label">Scaler</div>
                      <div className="metric-value" style={{ fontSize: 16, color: statusColor(health.scaler) }}>
                        {statusText(health.scaler)}
                      </div>
                    </div>
                    <div className="metric-card" style={{ border: `2px solid ${health.dataset ? "var(--green-muted)" : "var(--red-muted)"}` }}>
                      <div className="metric-label">Dataset</div>
                      <div className="metric-value" style={{ fontSize: 16, color: statusColor(health.dataset) }}>
                        {statusText(health.dataset)}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Model Details */}
            {info && (
              <>
                <div className="metrik-cards-grid animate-in delay-2">
                  {(["h1", "h3", "h7"] as const).map((h) => {
                    const m = info.info[h];
                    if (!m || m.error) {
                      return (
                        <div key={h} className="card">
                          <div className="card-header">
                            <h3>Model {h.toUpperCase()}</h3>
                            <span className="badge badge-red">Error</span>
                          </div>
                          <div className="card-body">
                            <p style={{ color: "var(--red)" }}>{m?.error || "Tidak tersedia"}</p>
                          </div>
                        </div>
                      );
                    }
                    return (
                      <div key={h} className="card">
                        <div className="card-header">
                          <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ width: 8, height: 8, borderRadius: "50%", background: horizonColors[h], display: "inline-block" }}></span>
                            Model {h.toUpperCase()}
                          </h3>
                          <span className="badge badge-green">Aktif</span>
                        </div>
                        <div className="card-body">
                          <div className="info-list" style={{ gap: 10 }}>
                            <div className="info-detail-row">
                              <span className="info-detail-label">File Model</span>
                              <span className="info-detail-value">{m.model_file}</span>
                            </div>
                            <div className="info-detail-row">
                              <span className="info-detail-label">Tanggal Training</span>
                              <span className="info-detail-value">{m.tanggal_training}</span>
                            </div>
                            <div className="info-detail-row">
                              <span className="info-detail-label">Jumlah Fitur</span>
                              <span className="info-detail-value">{m.jumlah_fitur} fitur</span>
                            </div>
                            <div className="info-detail-row">
                              <span className="info-detail-label">Feature Cols</span>
                              <span className="info-detail-value">{m.feature_cols_file}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Dataset Info */}
                {info.dataset_info && !info.dataset_info.error && (
                  <div className="card animate-in delay-3">
                    <div className="card-header">
                      <h3>Informasi Dataset</h3>
                      <span className="badge badge-green">Preprocessed</span>
                    </div>
                    <div className="card-body">
                      <div className="metrics-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                        <div className="metric-card">
                          <div className="metric-label">Total Data</div>
                          <div className="metric-value" style={{ fontSize: 22 }}>
                            {info.dataset_info.total_baris.toLocaleString("id-ID")}
                          </div>
                          <div className="metric-sub neutral">baris data</div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-label">Data Mulai</div>
                          <div className="metric-value" style={{ fontSize: 18 }}>
                            {new Date(info.dataset_info.tanggal_min).toLocaleDateString("id-ID", { day: "numeric", month: "long", year: "numeric" })}
                          </div>
                          <div className="metric-sub neutral">Tanggal pertama</div>
                        </div>
                        <div className="metric-card">
                          <div className="metric-label">Data Akhir</div>
                          <div className="metric-value" style={{ fontSize: 18 }}>
                            {new Date(info.dataset_info.tanggal_max).toLocaleDateString("id-ID", { day: "numeric", month: "long", year: "numeric" })}
                          </div>
                          <div className="metric-sub neutral">Tanggal terakhir</div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* About Section */}
            <div className="card animate-in delay-4">
              <div className="card-header">
                <h3>Tentang Sistem</h3>
              </div>
              <div className="card-body">
                <div className="info-list">
                  <div className="info-item">
                    <div>
                      <strong>Algoritma: XGBoost (Extreme Gradient Boosting)</strong>
                      <p>Model ensemble berbasis decision tree yang menggunakan gradient boosting untuk prediksi harga</p>
                    </div>
                  </div>
                  <div className="info-item">
                    <div>
                      <strong>3 Horizon Prediksi: H+1, H+3, H+7</strong>
                      <p>Masing-masing horizon menggunakan model terpisah yang dioptimalkan khusus</p>
                    </div>
                  </div>
                  <div className="info-item">
                    <div>
                      <strong>Sumber Data: PIHPS BI, BMKG, SKB 3 Menteri</strong>
                      <p>Data harga dari Panel Informasi Harga Pangan Strategis Bank Indonesia, cuaca dari BMKG</p>
                    </div>
                  </div>
                  <div className="info-item">
                    <div>
                      <strong>Stack: FastAPI + Next.js + Chart.js</strong>
                      <p>Backend Python FastAPI, frontend Next.js TypeScript, visualisasi Chart.js</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}
