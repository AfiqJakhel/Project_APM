"use client";

import { useEffect, useState, useCallback } from "react";
import PriceChart from "../components/PriceChart";
import { fetchHistory, type HistoryRecord } from "../lib/api";

/* ──────────── Helpers ──────────── */
function formatRp(value: number): string {
  return `Rp ${value.toLocaleString("id-ID", { maximumFractionDigits: 0 })}`;
}

/* ══════════════════════════════════════════
   HISTORIS PAGE
   ══════════════════════════════════════════ */
export default function HistorisPage() {
  const [data, setData] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [limit, setLimit] = useState(90);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchHistory({
        limit,
      });
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Gagal memuat data historis");
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Chart data
  const chartLabels = data.map((d) => {
    const dt = new Date(d.tanggal);
    return `${dt.getDate()}/${dt.getMonth() + 1}`;
  });
  const chartPrices = data.map((d) => d.harga_cabai_merah);

  // Stats
  const prices = data.map((d) => d.harga_cabai_merah).filter((v) => v > 0);
  const avg = prices.length > 0 ? prices.reduce((a, b) => a + b, 0) / prices.length : 0;
  const min = prices.length > 0 ? Math.min(...prices) : 0;
  const max = prices.length > 0 ? Math.max(...prices) : 0;
  const volatility =
    prices.length > 1
      ? Math.sqrt(
          prices.reduce((sum, p) => sum + Math.pow(p - avg, 2), 0) / (prices.length - 1)
        )
      : 0;

  return (
    <>
      <header className="topbar">
        <div className="topbar-left">
          <h1>Data Historis Harga Cabai</h1>
          <p>Grafik interaktif dan tabel data harga historis</p>
        </div>
      </header>

      <div className="content-area">
        {/* Filters */}
        <div className="card animate-in delay-1">
          <div className="card-header">
            <h3>Filter Data</h3>
            <span className="badge badge-green">{data.length} data</span>
          </div>
          <div className="card-body">
            <div className="filter-row" style={{ display: 'flex', gap: '16px', alignItems: 'flex-end' }}>
              <div className="filter-group">
                <label className="filter-label">Rentang Waktu</label>
                <select
                  className="filter-input"
                  value={limit}
                  onChange={(e) => setLimit(Number(e.target.value))}
                >
                  <option value={30}>30 hari</option>
                  <option value={60}>60 hari</option>
                  <option value={90}>90 hari</option>
                  <option value={180}>180 hari</option>
                  <option value={365}>365 hari</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Stats cards */}
        <div className="metrics-grid animate-in delay-2">
          <div className="metric-card">
            <div className="metric-label">Rata-rata</div>
            <div className="metric-value" style={{ fontSize: 22 }}>{formatRp(avg)}</div>
            <div className="metric-sub neutral">{data.length} hari data</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Terendah</div>
            <div className="metric-value" style={{ fontSize: 22, color: "var(--green)" }}>{formatRp(min)}</div>
            <div className="metric-sub up">Harga minimum</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Tertinggi</div>
            <div className="metric-value" style={{ fontSize: 22, color: "var(--red)" }}>{formatRp(max)}</div>
            <div className="metric-sub neutral">Harga maksimum</div>
          </div>
          <div className="metric-card">
            <div className="metric-label">Volatilitas (σ)</div>
            <div className="metric-value" style={{ fontSize: 22 }}>{formatRp(volatility)}</div>
            <div className="metric-sub neutral">Standar deviasi</div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="alert-box" style={{ borderColor: "var(--red-muted)", background: "var(--red-light)" }}>
            <p className="alert-text"><strong>Error:</strong> {error}</p>
          </div>
        )}

        {/* Chart */}
        <div className="card animate-in delay-3">
          <div className="card-header">
            <h3>Grafik Harga Historis</h3>
            <span className="badge badge-green">CMK</span>
          </div>
          <div className="chart-container" style={{ height: "380px" }}>
            {loading ? (
              <div className="loading-container">
                <div className="loading-spinner" />
                <p>Memuat grafik...</p>
              </div>
            ) : (
              <PriceChart
                labels={chartLabels}
                data={chartPrices}
              />
            )}
          </div>
        </div>

        {/* Table */}
        <div className="card animate-in delay-4">
          <div className="card-header">
            <h3>Tabel Data Historis</h3>
            <span className="badge badge-purple">{data.length} baris</span>
          </div>
          <div className="card-body" style={{ padding: 0, maxHeight: 400, overflowY: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>No</th>
                  <th>Tanggal</th>
                  <th>Harga CMK (Rp/kg)</th>
                  {data.some((d) => d.harga_cabai_rawit) && <th>Harga CRM (Rp/kg)</th>}
                  <th>Perubahan</th>
                </tr>
              </thead>
              <tbody>
                {data.length > 0 ? (
                  [...data].reverse().map((row, i) => {
                    const idx = data.length - 1 - i;
                    const prev = idx > 0 ? data[idx - 1]?.harga_cabai_merah : null;
                    const change = prev
                      ? ((row.harga_cabai_merah - prev) / prev) * 100
                      : 0;
                    const statusClass =
                      change > 1 ? "status-naik" : change < -1 ? "status-stabil" : "status-plus";
                    return (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        <td>{new Date(row.tanggal).toLocaleDateString("id-ID", { day: "numeric", month: "short", year: "numeric" })}</td>
                        <td>{formatRp(row.harga_cabai_merah)}</td>
                        {data.some((d) => d.harga_cabai_rawit) && (
                          <td>{row.harga_cabai_rawit ? formatRp(row.harga_cabai_rawit) : "—"}</td>
                        )}
                        <td>
                          <span className={`status-badge ${statusClass}`}>
                            {change > 0 ? "+" : ""}
                            {change.toFixed(1)}%
                          </span>
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={5} style={{ textAlign: "center", color: "var(--text-muted)", padding: 24 }}>
                      {loading ? "Memuat..." : "Tidak ada data"}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  );
}
