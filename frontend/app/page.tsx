"use client";

import { useEffect, useState } from "react";
import PriceChart from "./components/PriceChart";
import {
  fetchDashboard,
  fetchHistoris,
  fetchPrediksiAll,
  fetchMetrik,
  type DashboardResponse,
  type HistorisDataPoint,
  type PrediksiItem,
  type MetrikResponse,
} from "./lib/api";

/* ──────────── Feature importance (static — from model) ──────────── */
const features = [
  { label: "Lag harga 1 hari", pct: 34.2, barClass: "bar-green" },
  { label: "Lag harga 7 hari", pct: 22.7, barClass: "bar-teal" },
  { label: "Moving avg 7 hari", pct: 18.4, barClass: "bar-blue" },
  { label: "Curah hujan harian", pct: 12.1, barClass: "bar-orange" },
  { label: "Hari libur nasional", pct: 7.2, barClass: "bar-purple" },
  { label: "Bulan (musiman)", pct: 5.4, barClass: "bar-pink" },
];

/* ──────────── Helpers ──────────── */
function formatRp(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `Rp ${value.toLocaleString("id-ID", { maximumFractionDigits: 0 })}`;
}

function formatDate(dateStr: string): string {
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString("id-ID", {
      day: "numeric",
      month: "long",
      year: "numeric",
    });
  } catch {
    return dateStr;
  }
}

function trendLabel(tren: string): string {
  if (tren === "naik") return "Naik";
  if (tren === "turun") return "Turun";
  return "Stabil";
}

function statusInflasiColor(status: string): string {
  if (status === "kritis") return "badge-red";
  if (status === "waspada") return "badge-yellow";
  return "badge-green";
}

/* ══════════════════════════════════════════
   DASHBOARD PAGE
   ══════════════════════════════════════════ */
export default function Home() {
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [historis, setHistoris] = useState<HistorisDataPoint[]>([]);
  const [prediksi, setPrediksi] = useState<PrediksiItem[]>([]);
  const [metrik, setMetrik] = useState<MetrikResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const [dashData, histData, predData, metrikData] = await Promise.allSettled([
          fetchDashboard(),
          fetchHistoris(60),
          fetchPrediksiAll(),
          fetchMetrik(),
        ]);

        if (dashData.status === "fulfilled") setDashboard(dashData.value);
        if (histData.status === "fulfilled") setHistoris(histData.value.data);
        if (predData.status === "fulfilled") setPrediksi(predData.value.prediksi);
        if (metrikData.status === "fulfilled") setMetrik(metrikData.value);

        // If all failed, show error
        if (
          dashData.status === "rejected" &&
          histData.status === "rejected" &&
          predData.status === "rejected"
        ) {
          setError("Tidak bisa terhubung ke backend. Pastikan server FastAPI berjalan di localhost:8000");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Terjadi kesalahan");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Chart data from historis
  const chartLabels = historis.map((d) => {
    const dt = new Date(d.tanggal);
    return `${dt.getDate()}/${dt.getMonth() + 1}`;
  });
  const chartData = historis.map((d) => d.harga_cabai_merah);

  // Get R² from metrik
  const r2Value = metrik?.metrik?.h1?.R2;

  return (
    <>
      {/* Top Bar */}
      <header className="topbar">
        <div className="topbar-left">
          <h1>Dashboard monitoring harga cabai</h1>
          <p>
            {dashboard
              ? `Diperbarui: ${formatDate(dashboard.tanggal_update)}, 08.00 WIB · Pasar Tradisional · Kota Padang`
              : "Memuat data..."}
          </p>
        </div>
        <div className="topbar-right">
          {dashboard && (
            <span className={`badge ${statusInflasiColor(dashboard.status_inflasi)}`} style={{ fontSize: 11, padding: "5px 12px" }}>
              Status: {dashboard.status_inflasi.toUpperCase()}
            </span>
          )}
        </div>
      </header>

      <div className="content-area">
        {/* Loading State */}
        {loading && (
          <div className="dashboard-skeleton animate-in delay-1" aria-busy="true">
            <div className="metrics-grid" style={{ marginBottom: 20 }}>
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="metric-card">
                  <div className="skeleton" style={{ width: "50%", height: 16, marginBottom: 12 }} />
                  <div className="skeleton" style={{ width: "80%", height: 32, marginBottom: 12 }} />
                  <div className="skeleton" style={{ width: "60%", height: 14 }} />
                </div>
              ))}
            </div>
            <div className="main-grid" style={{ marginBottom: 20 }}>
              <div className="card">
                <div className="card-header">
                  <div className="skeleton" style={{ width: 150, height: 20 }} />
                </div>
                <div className="card-body" style={{ height: 320, padding: 20 }}>
                  <div className="skeleton" style={{ width: "100%", height: "100%" }} />
                </div>
              </div>
              <div className="card">
                <div className="card-header">
                  <div className="skeleton" style={{ width: 150, height: 20 }} />
                </div>
                <div className="card-body" style={{ padding: 20 }}>
                  {[1, 2, 3].map((i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 20, paddingBottom: 15, borderBottom: '1px solid var(--border-light)' }}>
                       <div className="skeleton" style={{ width: 60, height: 16 }} />
                       <div className="skeleton" style={{ width: 100, height: 20 }} />
                       <div className="skeleton" style={{ width: 50, height: 18, borderRadius: 20 }} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="alert-box animate-in delay-1" style={{ borderColor: "var(--red-muted)", background: "var(--red-light)" }}>
            <p className="alert-text">
              <strong>Error:</strong> {error}
            </p>
          </div>
        )}

        {!loading && (
          <>
            {/* Alert Box */}
            {dashboard && dashboard.status_inflasi !== "normal" && (
              <div className="alert-box animate-in delay-1">
                <p className="alert-text">
                  <strong>Peringatan dini:</strong> Prediksi menunjukkan harga cabai
                  dalam status <strong>{dashboard.status_inflasi}</strong>. Tren harga{" "}
                  <strong>{dashboard.tren}</strong> — perlu perhatian.
                </p>
                <span className="badge-siaga">
                  {dashboard.status_inflasi === "kritis" ? "Kritis" : "Siaga"}
                </span>
              </div>
            )}

            {/* Metrics Grid */}
            <div className="metrics-grid">
              <div className="metric-card metric-card--cmk animate-in delay-2">
                <div className="metric-label">
                  <span className="dot dot-cmk"></span>Harga hari ini — CMK
                </div>
                <div className="metric-value">
                  {dashboard ? formatRp(dashboard.harga_hari_ini) : "—"}
                </div>
                <div className={`metric-sub ${dashboard?.tren === "naik" ? "up" : dashboard?.tren === "turun" ? "down" : "neutral"}`}>
                  {dashboard ? trendLabel(dashboard.tren) : "—"}
                </div>
              </div>
              <div className="metric-card metric-card--avg30 animate-in delay-3">
                <div className="metric-label">
                  <span className="dot dot-crm"></span>Rata-rata 30 hari
                </div>
                <div className="metric-value">
                  {dashboard ? formatRp(dashboard.harga_rata_30hari) : "—"}
                </div>
                <div className="metric-sub neutral">
                  Min: {dashboard ? formatRp(dashboard.harga_min_30hari) : "—"} · Max: {dashboard ? formatRp(dashboard.harga_max_30hari) : "—"}
                </div>
              </div>
              <div className="metric-card metric-card--prediksi animate-in delay-4">
                <div className="metric-label">
                  <span className="dot dot-pred"></span>Prediksi H+7
                </div>
                <div className="metric-value">
                  {dashboard ? formatRp(dashboard.prediksi_h7) : "—"}
                </div>
                <div className="metric-sub neutral">
                  H+1: {dashboard ? formatRp(dashboard.prediksi_h1) : "—"} · H+3: {dashboard ? formatRp(dashboard.prediksi_h3) : "—"}
                </div>
              </div>
              <div className="metric-card metric-card--akurasi animate-in delay-5">
                <div className="metric-label">
                  <span className="dot dot-acc"></span>Akurasi model (R²)
                </div>
                <div className="metric-value">
                  {r2Value !== undefined ? `${(r2Value * 100).toFixed(1)}%` : "—"}
                </div>
                <div className="metric-sub up">
                  {metrik?.metrik?.h1?.MAPE !== undefined
                    ? `MAPE ${metrik.metrik.h1.MAPE.toFixed(1)}% · data uji`
                    : "—"}
                </div>
              </div>
            </div>

            {/* Main Grid: Chart + Predictions */}
            <div className="main-grid">
              {/* Chart Card */}
              <div className="card animate-in delay-5">
                <div className="card-header">
                  <h3>Tren harga &amp; prediksi</h3>
                  <span className="badge badge-green">XGBoost aktif</span>
                </div>
                <div className="chart-legend">
                  <div className="legend-item">
                    <span className="legend-line" style={{ background: "#0F6E56" }}></span>
                    Aktual CMK
                  </div>
                  <div className="legend-item">
                    <span className="legend-line dashed" style={{ color: "#0F6E56" }}></span>
                    Prediksi CMK
                  </div>
                </div>
                <div className="chart-container" style={{ height: "320px" }}>
                  <PriceChart
                    labels={chartLabels.length > 0 ? chartLabels : undefined}
                    data={chartData.length > 0 ? chartData : undefined}
                    prediksiH1={dashboard?.prediksi_h1}
                    prediksiH3={dashboard?.prediksi_h3}
                    prediksiH7={dashboard?.prediksi_h7}
                  />
                </div>
              </div>

              {/* Predictions List Card */}
              <div className="card animate-in delay-6">
                <div className="card-header">
                  <h3>Prediksi per horizon</h3>
                  <span className="badge badge-purple">XGBoost</span>
                </div>
                <div className="card-body">
                  {prediksi.length > 0 ? (
                    <ul className="prediction-list">
                      {prediksi.map((p, i) => (
                        <li key={i} className="prediction-item">
                          <span className="prediction-date">{p.keterangan}</span>
                          <span className="prediction-price">
                            {p.error ? "Error" : formatRp(p.prediksi_rp)}
                          </span>
                          <span className={`prediction-badge badge ${p.error ? "badge-red" : "badge-green"}`}>
                            {p.error ? "N/A" : p.tanggal_prediksi}
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
                      Data prediksi belum tersedia
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Bottom Grid: Feature Importance + Recent History */}
            <div className="bottom-grid">
              {/* Feature Importance */}
              <div className="card animate-in delay-7">
                <div className="card-header">
                  <h3>Feature importance</h3>
                  <span className="badge badge-green">XGBoost</span>
                </div>
                <div className="card-body">
                  <div className="feature-list">
                    {features.map((f, i) => (
                      <div key={i} className="feature-item">
                        <div className="feature-label">
                          <span>{f.label}</span>
                          <span>{f.pct}%</span>
                        </div>
                        <div className="feature-bar-bg">
                          <div
                            className={`feature-bar-fill ${f.barClass}`}
                            style={{ width: `${(f.pct / 34.2) * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Recent Prices Table */}
              <div className="card animate-in delay-8">
                <div className="card-header">
                  <h3>Riwayat harga terbaru</h3>
                  <span className="badge badge-green">PIHPS</span>
                </div>
                <div className="card-body" style={{ padding: 0 }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Tanggal</th>
                        <th>Harga CMK</th>
                        <th>Perubahan</th>
                      </tr>
                    </thead>
                    <tbody>
                      {historis.length > 0 ? (
                        historis.slice(-7).reverse().map((row, i) => {
                          const idx = historis.length - 1 - i;
                          const prev = idx > 0 ? historis[idx - 1]?.harga_cabai_merah : null;
                          const change = prev ? ((row.harga_cabai_merah - prev) / prev * 100) : 0;
                          const statusClass = change > 0 ? "status-naik" : change < 0 ? "status-stabil" : "status-plus";
                          return (
                            <tr key={i}>
                              <td>{new Date(row.tanggal).toLocaleDateString("id-ID", { day: "numeric", month: "short" })}</td>
                              <td>{formatRp(row.harga_cabai_merah)}</td>
                              <td>
                                <span className={`status-badge ${statusClass}`}>
                                  {change > 0 ? "+" : ""}{change.toFixed(1)}%
                                </span>
                              </td>
                            </tr>
                          );
                        })
                      ) : (
                        <tr>
                          <td colSpan={3} style={{ textAlign: "center", color: "var(--text-muted)" }}>
                            Belum ada data
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}
