"use client";

import { useState, useEffect } from "react";
import { fetchPrediksi, type PrediksiSingleResponse } from "../lib/api";

/* ──────────── Helpers ──────────── */
function formatRp(value: number): string {
  return `Rp ${value.toLocaleString("id-ID", { maximumFractionDigits: 0 })}`;
}

function statusBadge(harga: number): { label: string; cls: string } {
  if (harga >= 80000) return { label: "Kritis", cls: "badge-red" };
  if (harga >= 60000) return { label: "Tinggi", cls: "badge-yellow" };
  return { label: "Normal", cls: "badge-green" };
}



const horizonInfo = {
  h1: { label: "H+1 (Besok)", desc: "Prediksi harga 1 hari ke depan", color: "#0F6E56" },
  h3: { label: "H+3 (3 Hari)", desc: "Prediksi harga 3 hari ke depan", color: "#3B82F6" },
  h7: { label: "H+7 (7 Hari)", desc: "Prediksi harga 7 hari ke depan", color: "#7C3AED" },
};

/* ══════════════════════════════════════════
   PREDIKSI PAGE
   ══════════════════════════════════════════ */
export default function PrediksiPage() {
  const [selectedHorizon, setSelectedHorizon] = useState<"h1" | "h3" | "h7">("h1");
  const [result, setResult] = useState<PrediksiSingleResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function runPredict() {
      setLoading(true);
      setError(null);
      setResult(null);
      try {
        const data = await fetchPrediksi(selectedHorizon);
        setResult(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Gagal memuat prediksi");
      } finally {
        setLoading(false);
      }
    }
    runPredict();
  }, [selectedHorizon]);

  const info = horizonInfo[selectedHorizon];
  const status = result ? statusBadge(result.prediksi_rp) : null;

  return (
    <>
      <header className="topbar">
        <div className="topbar-left">
          <h1>Prediksi Harga Cabai</h1>
          <p>Pilih horizon waktu dan lihat prediksi dari model XGBoost</p>
        </div>
      </header>

      <div className="content-area">
        {/* Horizon Selection */}
        <div className="card animate-in delay-1">
          <div className="card-header">
            <h3>Pilih Horizon Prediksi</h3>
            <span className="badge badge-green">XGBoost</span>
          </div>
          <div className="card-body">
            <div className="horizon-grid">
              {(["h1", "h3", "h7"] as const).map((h) => {
                const hi = horizonInfo[h];
                const isActive = selectedHorizon === h;
                return (
                  <button
                    key={h}
                    className={`horizon-card ${isActive ? "horizon-card-active" : ""}`}
                    onClick={() => setSelectedHorizon(h)}
                    style={{ "--accent": hi.color } as React.CSSProperties}
                  >
                    <div className="horizon-card-label">{hi.label}</div>
                    <div className="horizon-card-desc">{hi.desc}</div>
                  </button>
                );
              })}
            </div>
            
            {loading && (
              <div style={{ marginTop: 16, display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, color: 'var(--text-secondary)' }}>
                <span className="loading-spinner-sm" style={{ borderColor: 'rgba(15,110,86,0.3)', borderTopColor: 'var(--primary)' }} />
                Memproses prediksi {info.label}...
              </div>
            )}
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="alert-box animate-in delay-2" style={{ borderColor: "var(--red-muted)", background: "var(--red-light)" }}>
            <p className="alert-text">
              <strong>Error:</strong> {error}
            </p>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="prediksi-result animate-in delay-2">
            <div className="metrics-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
              {/* Main prediction */}
              <div className="metric-card" style={{ gridColumn: "1 / -1" }}>
                <div className="metric-label">
                  <span className="dot" style={{ background: info.color }}></span>
                  {result.keterangan}
                </div>
                <div className="metric-value" style={{ fontSize: 36, color: info.color }}>
                  {formatRp(result.prediksi_rp)}
                </div>
                <div className="metric-sub neutral" style={{ marginTop: 8, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
                  <span className={`badge ${status!.cls}`} style={{ fontSize: 11 }}>
                    {status!.label}
                  </span>
                  <span>Tanggal prediksi: <strong>{result.tanggal_prediksi}</strong></span>
                  {result.arah_prediksi && (
                    <span>
                      Arah: <strong>{result.arah_prediksi}</strong>
                      {result.confidence_arah && ` (${result.confidence_arah.toFixed(1)}%)`}
                    </span>
                  )}
                </div>
              </div>

              {/* Detail cards */}
              <div className="metric-card">
                <div className="metric-label">Horizon</div>
                <div className="metric-value" style={{ fontSize: 20 }}>{result.horizon.toUpperCase()}</div>
                <div className="metric-sub neutral">{result.keterangan}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Versi Model</div>
                <div className="metric-value" style={{ fontSize: 14, wordBreak: "break-all" }}>{result.model_version}</div>
                <div className="metric-sub neutral">File .pkl</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Arah Pergerakan</div>
                <div className="metric-value" style={{ fontSize: 20 }}>
                  {result.arah_prediksi || "—"}
                </div>
                <div className="metric-sub neutral">
                  Confidence: {result.confidence_arah ? `${result.confidence_arah.toFixed(1)}%` : "—"}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Info box */}
        <div className="card animate-in delay-3">
          <div className="card-header">
            <h3>Cara Kerja Prediksi</h3>
          </div>
          <div className="card-body">
            <div className="info-list">
              <div className="info-item">
                <span className="info-number">1</span>
                <div>
                  <strong>Ambil data terkini</strong>
                  <p>Sistem mengambil data harga dan cuaca terakhir dari dataset</p>
                </div>
              </div>
              <div className="info-item">
                <span className="info-number">2</span>
                <div>
                  <strong>Ekstraksi fitur</strong>
                  <p>Data diproses menjadi 42+ fitur termasuk lag, moving average, dan faktor musiman</p>
                </div>
              </div>
              <div className="info-item">
                <span className="info-number">3</span>
                <div>
                  <strong>Prediksi XGBoost</strong>
                  <p>Model XGBoost melakukan prediksi berdasarkan fitur yang diekstrak</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
