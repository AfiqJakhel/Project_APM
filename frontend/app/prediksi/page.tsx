// ════════════════════════════════════════════════════════
//  PENDEKATAN A — Fetch All Horizons at Mount (Optimal)
//  Satu request ke /api/predict/prediksi saat pertama kali.
//  Ganti horizon = hanya switch display, ZERO request tambahan.
// ════════════════════════════════════════════════════════
"use client";

import { useState, useEffect, useCallback } from "react";
import { fetchPrediksiAll, type PrediksiItem } from "../lib/api";

// ─── Types ────────────────────────────────────────────────────────────────────

type HorizonKey = "h1" | "h3" | "h7";

/** Semua data horizon tersimpan di sini setelah satu fetch */
type PrediksiCache = Record<HorizonKey, PrediksiItem>;

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatRp(value: number): string {
  return `Rp ${value.toLocaleString("id-ID", { maximumFractionDigits: 0 })}`;
}

function statusBadge(harga: number): { label: string; cls: string } {
  if (harga >= 80000) return { label: "Kritis", cls: "badge-red" };
  if (harga >= 60000) return { label: "Tinggi", cls: "badge-yellow" };
  return { label: "Normal", cls: "badge-green" };
}

const horizonInfo: Record<HorizonKey, { label: string; desc: string; color: string }> = {
  h1: { label: "H+1 (Besok)",  desc: "Prediksi harga 1 hari ke depan", color: "#0F6E56" },
  h3: { label: "H+3 (3 Hari)", desc: "Prediksi harga 3 hari ke depan", color: "#3B82F6" },
  h7: { label: "H+7 (7 Hari)", desc: "Prediksi harga 7 hari ke depan", color: "#7C3AED" },
};

// ─── Skeleton ─────────────────────────────────────────────────────────────────

function ResultSkeleton() {
  return (
    <div className="prediksi-result animate-in delay-2" aria-busy="true" aria-label="Memuat data...">
      <div className="metrics-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
        {/* Main skeleton */}
        <div className="metric-card" style={{ gridColumn: "1 / -1" }}>
          <div className="skeleton" style={{ width: "60%", height: 16, marginBottom: 12 }} />
          <div className="skeleton" style={{ width: "45%", height: 36, marginBottom: 16 }} />
          <div style={{ display: "flex", gap: 12 }}>
            <div className="skeleton" style={{ width: 70, height: 22, borderRadius: 20 }} />
            <div className="skeleton" style={{ width: 160, height: 22 }} />
          </div>
        </div>
        {/* Detail skeletons */}
        {[1, 2, 3].map((i) => (
          <div key={i} className="metric-card">
            <div className="skeleton" style={{ width: "50%", height: 13, marginBottom: 10 }} />
            <div className="skeleton" style={{ width: "70%", height: 22, marginBottom: 8 }} />
            <div className="skeleton" style={{ width: "40%", height: 13 }} />
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function PrediksiPage() {
  const [selectedHorizon, setSelectedHorizon] = useState<HorizonKey>("h1");
  const [cache, setCache] = useState<PrediksiCache | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch SEMUA horizon sekaligus — hanya dipanggil saat mount (dan retry)
  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPrediksiAll();

      // Ubah array menjadi Record agar akses O(1) per horizon
      const mapped = data.prediksi.reduce<Partial<PrediksiCache>>((acc, item) => {
        const key = item.horizon as HorizonKey;
        acc[key] = item;
        return acc;
      }, {});

      // Pastikan ketiga horizon ada (guard jika backend partial)
      if (!mapped.h1 || !mapped.h3 || !mapped.h7) {
        throw new Error("Data horizon tidak lengkap dari server.");
      }

      setCache(mapped as PrediksiCache);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Gagal memuat prediksi.");
    } finally {
      setLoading(false);
    }
  }, []);

  // Satu kali saat mount — dependency array kosong []
  useEffect(() => {
    loadAll();
  }, [loadAll]);

  // Data aktif diambil langsung dari cache — NO fetch
  const result: PrediksiItem | null = cache ? cache[selectedHorizon] : null;
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
        {/* ── Horizon Selector ─────────────────────────────────────────── */}
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
                    onClick={() => setSelectedHorizon(h)} // Zero network request
                    disabled={loading}
                    style={{ "--accent": hi.color } as React.CSSProperties}
                    aria-pressed={isActive}
                  >
                    <div className="horizon-card-label">{hi.label}</div>
                    <div className="horizon-card-desc">{hi.desc}</div>
                  </button>
                );
              })}
            </div>

            {/* Loading indicator saat fetch pertama kali */}
            {loading && (
              <div
                style={{
                  marginTop: 16,
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  fontSize: 13,
                  color: "var(--text-secondary)",
                }}
              >
                <span
                  className="loading-spinner-sm"
                  style={{
                    borderColor: "rgba(15,110,86,0.3)",
                    borderTopColor: "var(--primary)",
                  }}
                />
                Memuat semua data prediksi...
              </div>
            )}
          </div>
        </div>

        {/* ── Error State ──────────────────────────────────────────────── */}
        {error && !loading && (
          <div
            className="alert-box animate-in delay-2"
            style={{ borderColor: "var(--red-muted)", background: "var(--red-light)" }}
            role="alert"
          >
            <p className="alert-text">
              <strong>Error:</strong> {error}
            </p>
            <button
              className="btn btn-sm"
              style={{ marginTop: 10 }}
              onClick={loadAll}
            >
              🔄 Coba Lagi
            </button>
          </div>
        )}

        {/* ── Skeleton saat loading ─────────────────────────────────────── */}
        {loading && <ResultSkeleton />}

        {/* ── Result (dari cache — no fetch) ───────────────────────────── */}
        {!loading && result && (
          <div className="prediksi-result animate-in delay-2">
            <div className="metrics-grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
              {/* Main prediction */}
              <div className="metric-card" style={{ gridColumn: "1 / -1" }}>
                <div className="metric-label">
                  <span className="dot" style={{ background: info.color }} />
                  {result.keterangan}
                </div>
                <div className="metric-value" style={{ fontSize: 36, color: info.color }}>
                  {formatRp(result.prediksi_rp)}
                </div>
                <div
                  className="metric-sub neutral"
                  style={{
                    marginTop: 8,
                    display: "flex",
                    gap: 12,
                    alignItems: "center",
                    flexWrap: "wrap",
                  }}
                >
                  <span className={`badge ${status!.cls}`} style={{ fontSize: 11 }}>
                    {status!.label}
                  </span>
                  <span>
                    Tanggal prediksi: <strong>{result.tanggal_prediksi}</strong>
                  </span>
                  {result.arah_prediksi && (
                    <span>
                      Arah: <strong>{result.arah_prediksi}</strong>
                      {result.perubahan_persen !== undefined && result.perubahan_persen !== null && ` (${result.perubahan_persen > 0 ? '+' : ''}${result.perubahan_persen.toFixed(1)}%)`}
                    </span>
                  )}
                </div>
              </div>

              {/* Detail cards */}
              <div className="metric-card">
                <div className="metric-label">Horizon</div>
                <div className="metric-value" style={{ fontSize: 20 }}>
                  {result.horizon.toUpperCase()}
                </div>
                <div className="metric-sub neutral">{result.keterangan}</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Versi Model</div>
                <div className="metric-value" style={{ fontSize: 14, wordBreak: "break-all" }}>
                  {result.model_version}
                </div>
                <div className="metric-sub neutral">File .pkl</div>
              </div>
              <div className="metric-card">
                <div className="metric-label">Arah Pergerakan</div>
                <div className="metric-value" style={{ fontSize: 20 }}>
                  {result.arah_prediksi || "—"}
                </div>
                <div className="metric-sub neutral">
                  Perubahan:{" "}
                  {result.perubahan_persen !== undefined && result.perubahan_persen !== null ? `${result.perubahan_persen > 0 ? '+' : ''}${result.perubahan_persen.toFixed(1)}%` : "—"}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Info Box ─────────────────────────────────────────────────── */}
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
