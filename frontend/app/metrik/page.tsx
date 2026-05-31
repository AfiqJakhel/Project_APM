"use client";

import { useEffect, useState } from "react";
import { fetchMetrik, type MetrikResponse, type MetrikHorizon } from "../lib/api";

/* ──────────── Helpers ──────────── */
function ratingLabel(r2: number): { label: string; cls: string } {
  if (r2 >= 0.9) return { label: "Sangat Baik", cls: "badge-green" };
  if (r2 >= 0.8) return { label: "Baik", cls: "badge-green" };
  if (r2 >= 0.7) return { label: "Cukup", cls: "badge-yellow" };
  return { label: "Perlu Perbaikan", cls: "badge-red" };
}

function mapeRating(mape: number): { label: string; cls: string } {
  if (mape < 10) return { label: "Akurat", cls: "badge-green" };
  if (mape < 20) return { label: "Cukup", cls: "badge-yellow" };
  return { label: "Kurang", cls: "badge-red" };
}

const horizonLabels: Record<string, { name: string; desc: string; color: string }> = {
  h1: { name: "H+1 (Besok)", desc: "Prediksi 1 hari ke depan", color: "#0F6E56" },
  h3: { name: "H+3 (3 Hari)", desc: "Prediksi 3 hari ke depan", color: "#3B82F6" },
  h7: { name: "H+7 (7 Hari)", desc: "Prediksi 7 hari ke depan", color: "#7C3AED" },
};

const metrikExplanations: Record<string, { name: string; desc: string; unit: string; better: string }> = {
  MAE: { name: "MAE", desc: "Mean Absolute Error — Rata-rata kesalahan absolut", unit: "Rp", better: "Semakin kecil semakin baik" },
  RMSE: { name: "RMSE", desc: "Root Mean Squared Error — Lebih sensitif terhadap outlier", unit: "Rp", better: "Semakin kecil semakin baik" },
  MAPE: { name: "MAPE", desc: "Mean Absolute Percentage Error — Persentase error rata-rata", unit: "%", better: "< 10% = akurat" },
  sMAPE: { name: "sMAPE", desc: "Symmetric MAPE — MAPE yang lebih seimbang", unit: "%", better: "< 10% = akurat" },
  R2: { name: "R²", desc: "Koefisien Determinasi — Seberapa baik model menjelaskan variansi data", unit: "", better: "Mendekati 1.0 = sempurna" },
  DA: { name: "DA", desc: "Directional Accuracy — Akurasi prediksi arah naik/turun", unit: "%", better: "> 70% = baik" },
};

/* ══════════════════════════════════════════
   METRIK PAGE
   ══════════════════════════════════════════ */
export default function MetrikPage() {
  const [activeKomoditas, setActiveKomoditas] = useState<"merah" | "rawit">("merah");
  const [data, setData] = useState<MetrikResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      setLoading(true);
      try {
        const result = await fetchMetrik();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Gagal memuat metrik");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  function renderMetrikCard(horizon: string, metrik: MetrikHorizon) {
    const info = horizonLabels[horizon];
    const rating = ratingLabel(metrik.R2);
    const mape = mapeRating(metrik.MAPE);

    return (
      <div key={horizon} className="card animate-in delay-2">
        <div className="card-header">
          <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="dot" style={{ background: info.color, width: 8, height: 8, borderRadius: "50%", display: "inline-block" }}></span>
            {info.name}
          </h3>
          <span className={`badge ${rating.cls}`}>{rating.label}</span>
        </div>
        <div className="card-body">
          <div className="metrik-grid">
            {/* R² — hero metric */}
            <div className="metrik-hero">
              <div className="metrik-hero-label">R² Score</div>
              <div className="metrik-hero-value" style={{ color: info.color }}>
                {(metrik.R2 * 100).toFixed(1)}%
              </div>
              <div className="metrik-hero-bar">
                <div
                  className="metrik-hero-bar-fill"
                  style={{ width: `${metrik.R2 * 100}%`, background: info.color }}
                />
              </div>
            </div>

            {/* Other metrics */}
            <div className="metrik-details">
              <div className="metrik-detail-item">
                <span className="metrik-detail-label">MAE</span>
                <span className="metrik-detail-value">Rp {metrik.MAE.toLocaleString("id-ID")}</span>
              </div>
              <div className="metrik-detail-item">
                <span className="metrik-detail-label">RMSE</span>
                <span className="metrik-detail-value">Rp {metrik.RMSE.toLocaleString("id-ID")}</span>
              </div>
              <div className="metrik-detail-item">
                <span className="metrik-detail-label">MAPE</span>
                <span className="metrik-detail-value">
                  {metrik.MAPE.toFixed(1)}%
                  <span className={`badge ${mape.cls}`} style={{ marginLeft: 6, fontSize: 9 }}>{mape.label}</span>
                </span>
              </div>
              <div className="metrik-detail-item">
                <span className="metrik-detail-label">sMAPE</span>
                <span className="metrik-detail-value">{metrik.sMAPE.toFixed(1)}%</span>
              </div>
              <div className="metrik-detail-item">
                <span className="metrik-detail-label">DA</span>
                <span className="metrik-detail-value">{metrik.DA.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <>
      <header className="topbar">
        <div className="topbar-left">
          <h1>Metrik Akurasi Model</h1>
          <p>Evaluasi performa model XGBoost per horizon prediksi</p>
        </div>
      </header>

      {/* ── Komoditas Selector (Sliding Tab) ─────────────────────────── */}
      <div style={{ padding: "0 40px", marginTop: "10px" }}>
        <div className="sliding-tabs-container">
          <div 
            className="slider-bg" 
            style={{
              width: "50%",
              left: activeKomoditas === "merah" ? "4px" : "calc(50% - 4px)",
              background: "linear-gradient(90deg, #0F3E39 0%, #3F9E96 100%)"
            }}
          />
          <button
            className={`sliding-tab ${activeKomoditas === "merah" ? "active" : ""}`}
            style={{ width: "160px", justifyContent: "center" }}
            onClick={() => setActiveKomoditas("merah")}
            disabled={loading}
          >
            Cabai Merah
          </button>
          <button
            className={`sliding-tab ${activeKomoditas === "rawit" ? "active" : ""}`}
            style={{ width: "160px", justifyContent: "center" }}
            onClick={() => setActiveKomoditas("rawit")}
            disabled={loading}
          >
            Cabai Rawit
          </button>
        </div>
      </div>

      <div className="content-area">
        {loading && (
          <div className="metrik-skeleton animate-in delay-1" aria-busy="true">
            <div className="metrik-cards-grid" style={{ marginBottom: 20 }}>
              {[1, 2, 3].map((i) => (
                <div key={i} className="card">
                  <div className="card-header">
                     <div className="skeleton" style={{ width: 120, height: 20 }} />
                  </div>
                  <div className="card-body">
                    <div className="metrik-grid">
                      <div className="metrik-hero">
                        <div className="skeleton" style={{ width: "60%", height: 16, marginBottom: 12 }} />
                        <div className="skeleton" style={{ width: "80%", height: 36, marginBottom: 16 }} />
                        <div className="skeleton" style={{ width: "100%", height: 8 }} />
                      </div>
                      <div className="metrik-details" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                        {[1, 2, 3, 4, 5].map((j) => (
                          <div key={j} style={{ display: 'flex', justifyContent: 'space-between' }}>
                             <div className="skeleton" style={{ width: "40%", height: 14 }} />
                             <div className="skeleton" style={{ width: "30%", height: 16 }} />
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="card" style={{ marginBottom: 20 }}>
              <div className="card-header">
                 <div className="skeleton" style={{ width: 200, height: 20 }} />
              </div>
              <div className="card-body" style={{ padding: 20 }}>
                 {[1, 2, 3, 4, 5].map((i) => (
                   <div key={i} style={{ marginBottom: 16 }}>
                      <div className="skeleton" style={{ width: "100%", height: 24 }} />
                   </div>
                 ))}
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="alert-box" style={{ borderColor: "var(--red-muted)", background: "var(--red-light)" }}>
            <p className="alert-text"><strong>Error:</strong> {error}</p>
          </div>
        )}

        {data && !loading && (
          <>
            {/* Metric cards per horizon */}
            <div className="metrik-cards-grid">
              {(["h1", "h3", "h7"] as const).map((h) => {
                const key = (activeKomoditas === "rawit" ? `rawit_${h}` : h) as keyof typeof data.metrik;
                return data.metrik[key] ? renderMetrikCard(h, data.metrik[key] as MetrikHorizon) : null;
              })}
            </div>

            {/* Comparison table */}
            <div className="card animate-in delay-4">
              <div className="card-header">
                <h3>Perbandingan Antar Horizon</h3>
                <span className="badge badge-purple">Tabel</span>
              </div>
              <div className="card-body" style={{ padding: 0 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Metrik</th>
                      <th>H+1 (Besok)</th>
                      <th>H+3 (3 Hari)</th>
                      <th>H+7 (7 Hari)</th>
                      <th>Keterangan</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(metrikExplanations).map(([key, info]) => {
                      const k = key as keyof MetrikHorizon;
                      const getH = (h: "h1"|"h3"|"h7") => data.metrik[(activeKomoditas === "rawit" ? `rawit_${h}` : h) as keyof typeof data.metrik] as MetrikHorizon | undefined;
                      const vals = [getH("h1")?.[k], getH("h3")?.[k], getH("h7")?.[k]];
                      const format = (v: number | undefined) => {
                        if (v === undefined) return "—";
                        if (info.unit === "Rp") return `Rp ${v.toLocaleString("id-ID")}`;
                        if (info.unit === "%") return `${v.toFixed(1)}%`;
                        return k === "R2" ? `${(v * 100).toFixed(1)}%` : v.toFixed(2);
                      };

                      // Highlight best value
                      const isBetter = (a: number, b: number) => {
                        if (k === "R2" || k === "DA") return a > b;
                        return a < b;
                      };
                      const validVals = vals.filter((v): v is number => v !== undefined);
                      const bestIdx = validVals.length > 0
                        ? vals.indexOf(validVals.reduce((best, v) => isBetter(v, best) ? v : best))
                        : -1;

                      return (
                        <tr key={key}>
                          <td><strong>{info.name}</strong></td>
                          {vals.map((v, i) => (
                            <td key={i} style={{ fontWeight: i === bestIdx ? 700 : 400, color: i === bestIdx ? "var(--green)" : undefined }}>
                              {format(v)}
                            </td>
                          ))}
                          <td style={{ fontSize: 11, color: "var(--text-muted)" }}>{info.better}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Metric Descriptions */}
            <div className="card animate-in delay-5">
              <div className="card-header">
                <h3>Penjelasan Metrik</h3>
              </div>
              <div className="card-body">
                <div className="info-list">
                  {Object.values(metrikExplanations).map((info, i) => (
                    <div key={i} className="info-item">
                      <span className="info-number" style={{ fontSize: 11, width: 44 }}>{info.name}</span>
                      <div>
                        <strong>{info.desc}</strong>
                        <p>{info.better}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  );
}
