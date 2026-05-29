"use client";

import { useEffect, useState } from "react";
import {
  fetchDashboard,
  fetchPrediksiAll,
  fetchHealth,
  type DashboardResponse,
  type PrediksiItem,
  type HealthResponse,
} from "../lib/api";

function formatRp(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `Rp ${value.toLocaleString("id-ID", { maximumFractionDigits: 0 })}`;
}

export default function RekapPage() {
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [prediksi, setPrediksi] = useState<PrediksiItem[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const [d, p, h] = await Promise.allSettled([
          fetchDashboard(),
          fetchPrediksiAll(),
          fetchHealth(),
        ]);
        if (d.status === "fulfilled") setDashboard(d.value);
        if (p.status === "fulfilled") setPrediksi(p.value.prediksi);
        if (h.status === "fulfilled") setHealth(h.value);
        if (d.status === "rejected" && p.status === "rejected") {
          setError("Tidak bisa terhubung ke backend");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Terjadi kesalahan");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const today = new Date().toLocaleDateString("id-ID", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  });

  return (
    <>
      <header className="topbar">
        <div className="topbar-left">
          <h1>Rekap Harian</h1>
          <p>{today}</p>
        </div>
      </header>

      <div className="content-area">
        {loading && (
          <div className="rekap-skeleton animate-in delay-1" aria-busy="true">
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-header">
                <div className="skeleton" style={{ width: 150, height: 20 }} />
              </div>
              <div className="card-body">
                <div className="metrics-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="metric-card">
                      <div className="skeleton" style={{ width: "50%", height: 16, marginBottom: 12 }} />
                      <div className="skeleton" style={{ width: "80%", height: 32, marginBottom: 12 }} />
                      <div className="skeleton" style={{ width: "60%", height: 14 }} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-header">
                <div className="skeleton" style={{ width: 150, height: 20 }} />
              </div>
              <div className="card-body" style={{ padding: 20 }}>
                {[1, 2, 3].map((i) => (
                  <div key={i} style={{ marginBottom: 16 }}>
                     <div className="skeleton" style={{ width: "100%", height: 24 }} />
                  </div>
                ))}
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <div className="skeleton" style={{ width: 150, height: 20 }} />
              </div>
              <div className="card-body" style={{ padding: 20 }}>
                 <div className="skeleton" style={{ width: "60%", height: 16 }} />
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="alert-box" style={{ borderColor: "var(--red-muted)", background: "var(--red-light)" }}>
            <p className="alert-text"><strong>Error:</strong> {error}</p>
          </div>
        )}

        {!loading && dashboard && (
          <>
            {/* Daily Summary */}
            <div className="card animate-in delay-1">
              <div className="card-header">
                <h3>Ringkasan Hari Ini</h3>
                <span className={`badge ${
                  dashboard.status_inflasi === "kritis" ? "badge-red" :
                  dashboard.status_inflasi === "waspada" ? "badge-yellow" : "badge-green"
                }`}>
                  {dashboard.status_inflasi.toUpperCase()}
                </span>
              </div>
              <div className="card-body">
                <div className="metrics-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
                  <div className="metric-card">
                    <div className="metric-label">Harga Hari Ini</div>
                    <div className="metric-value" style={{ fontSize: 24 }}>{formatRp(dashboard.harga_hari_ini)}</div>
                    <div className={`metric-sub ${dashboard.tren === "naik" ? "up" : "neutral"}`}>
                      Tren: {dashboard.tren}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Rata-rata 30 Hari</div>
                    <div className="metric-value" style={{ fontSize: 24 }}>{formatRp(dashboard.harga_rata_30hari)}</div>
                    <div className="metric-sub neutral">
                      Min: {formatRp(dashboard.harga_min_30hari)}
                    </div>
                  </div>
                  <div className="metric-card">
                    <div className="metric-label">Maks 30 Hari</div>
                    <div className="metric-value" style={{ fontSize: 24 }}>{formatRp(dashboard.harga_max_30hari)}</div>
                    <div className="metric-sub neutral">
                      {dashboard.n_model_aktif} model aktif
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Predictions */}
            <div className="card animate-in delay-2">
              <div className="card-header">
                <h3>Prediksi Hari Ini</h3>
                <span className="badge badge-purple">XGBoost</span>
              </div>
              <div className="card-body" style={{ padding: 0 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Horizon</th>
                      <th>Keterangan</th>
                      <th>Tanggal Prediksi</th>
                      <th>Harga Prediksi</th>
                      <th>Arah</th>
                    </tr>
                  </thead>
                  <tbody>
                    {prediksi.length > 0 ? prediksi.map((p, i) => (
                      <tr key={i}>
                        <td><strong>{p.horizon.toUpperCase()}</strong></td>
                        <td>{p.keterangan}</td>
                        <td>{p.error ? "—" : p.tanggal_prediksi}</td>
                        <td style={{ fontWeight: 700 }}>{p.error ? "Error" : formatRp(p.prediksi_rp)}</td>
                        <td>
                          {p.arah_prediksi === "naik" ? "Naik" :
                           p.arah_prediksi === "turun" ? "Turun" : "Stabil"}
                        </td>
                      </tr>
                    )) : (
                      <tr>
                        <td colSpan={5} style={{ textAlign: "center", color: "var(--text-muted)" }}>
                          Tidak ada data prediksi
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Health */}
            {health && (
              <div className="card animate-in delay-3">
                <div className="card-header">
                  <h3>Status Sistem</h3>
                  <span className={`badge ${health.status === "healthy" ? "badge-green" : "badge-yellow"}`}>
                    {health.status}
                  </span>
                </div>
                <div className="card-body">
                  <p style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                    {health.message} · Timestamp: {new Date(health.timestamp).toLocaleString("id-ID")}
                  </p>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
