import PriceChart from "./components/PriceChart";

/* ──────────── SVG Icon helpers ──────────── */
const IconDashboard = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="7" height="7" rx="1" />
    <rect x="14" y="3" width="7" height="7" rx="1" />
    <rect x="3" y="14" width="7" height="7" rx="1" />
    <rect x="14" y="14" width="7" height="7" rx="1" />
  </svg>
);
const IconTrend = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
    <polyline points="16 7 22 7 22 13" />
  </svg>
);
const IconHistory = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </svg>
);
const IconCompare = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="20" x2="18" y2="10" />
    <line x1="12" y1="20" x2="12" y2="4" />
    <line x1="6" y1="20" x2="6" y2="14" />
  </svg>
);
const IconReport = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" y1="13" x2="8" y2="13" />
    <line x1="16" y1="17" x2="8" y2="17" />
  </svg>
);
const IconBell = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
    <path d="M13.73 21a2 2 0 0 1-3.46 0" />
  </svg>
);
const IconAlert = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);
const IconLeaf = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 20A7 7 0 0 1 9.8 6.9C15.5 4.9 17 3.5 19 2c1 2 2 4.5 2 8 0 5.5-4.8 10-10 10Z" />
    <path d="M2 21c0-3 1.9-5.5 4.5-6.5" />
  </svg>
);

/* ──────────── Prediction data ──────────── */
const predictions = [
  { date: "28 Apr", price: "Rp 43.800", pct: "+3%", color: "badge-yellow" },
  { date: "29 Apr", price: "Rp 44.500", pct: "+5%", color: "badge-yellow" },
  { date: "30 Apr", price: "Rp 45.900", pct: "+8%", color: "badge-red" },
  { date: "1 Mei", price: "Rp 47.200", pct: "+11%", color: "badge-red" },
  { date: "2 Mei", price: "Rp 48.600", pct: "+14%", color: "badge-red" },
  { date: "3 Mei", price: "Rp 49.300", pct: "+16%", color: "badge-red" },
  { date: "4 Mei", price: "Rp 50.100", pct: "+18%", color: "badge-red" },
];

/* ──────────── Feature importance data ──────────── */
const features = [
  { label: "Lag harga 1 hari", pct: 34.2, barClass: "bar-green" },
  { label: "Lag harga 7 hari", pct: 22.7, barClass: "bar-teal" },
  { label: "Moving avg 7 hari", pct: 18.4, barClass: "bar-blue" },
  { label: "Curah hujan harian", pct: 12.1, barClass: "bar-orange" },
  { label: "Hari libur nasional", pct: 7.2, barClass: "bar-purple" },
  { label: "Bulan (musiman)", pct: 5.4, barClass: "bar-pink" },
];

/* ──────────── Table data ──────────── */
const tableData = [
  { tanggal: "27 Apr", komoditas: "CMK", harga: "42.500", status: "Naik", statusClass: "status-naik" },
  { tanggal: "27 Apr", komoditas: "CRM", harga: "38.000", status: "Naik", statusClass: "status-naik" },
  { tanggal: "26 Apr", komoditas: "CMK", harga: "38.500", status: "+2%", statusClass: "status-plus" },
  { tanggal: "26 Apr", komoditas: "CRM", harga: "35.500", status: "Stabil", statusClass: "status-stabil" },
  { tanggal: "25 Apr", komoditas: "CMK", harga: "36.100", status: "Stabil", statusClass: "status-stabil" },
  { tanggal: "25 Apr", komoditas: "CRM", harga: "33.000", status: "Stabil", statusClass: "status-stabil" },
];

/* ══════════════════════════════════════════
   PAGE COMPONENT
   ══════════════════════════════════════════ */
export default function Home() {
  return (
    <div className="dashboard-shell">
      {/* ── SIDEBAR ── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <IconLeaf />
          </div>
          <div className="sidebar-brand">
            <span className="sidebar-brand-name">CabaiWatch</span>
            <span className="sidebar-brand-sub">Padang · 2026</span>
          </div>
        </div>

        <nav className="sidebar-section">
          <div className="sidebar-section-label">Menu Utama</div>
          <ul className="sidebar-menu">
            <li className="sidebar-menu-item active">
              <IconDashboard /> Dashboard
            </li>
            <li className="sidebar-menu-item">
              <IconTrend /> Prediksi harga
            </li>
            <li className="sidebar-menu-item">
              <IconHistory /> Data historis
            </li>
            <li className="sidebar-menu-item">
              <IconCompare /> Perbandingan model
            </li>
          </ul>
        </nav>

        <nav className="sidebar-section">
          <div className="sidebar-section-label">Laporan</div>
          <ul className="sidebar-menu">
            <li className="sidebar-menu-item">
              <IconReport /> Rekap harian
            </li>
            <li className="sidebar-menu-item">
              <IconBell /> Riwayat alert
            </li>
          </ul>
        </nav>

        <div className="sidebar-footer">
          <p className="sidebar-footer-text">
            Sumber data: PIHPS BI · BMKG · SKB 3 Menteri
          </p>
        </div>
      </aside>

      {/* ── MAIN CONTENT ── */}
      <main className="main-content">
        {/* Top Bar */}
        <header className="topbar">
          <div className="topbar-left">
            <h1>Dashboard monitoring harga cabai</h1>
            <p>
              Diperbarui: 27 April 2026, 08.00 WIB · Pasar Tradisional · Kota Padang
            </p>
          </div>
          <div className="topbar-right">
            <div className="pill-toggle">
              <button className="pill-btn active">CMK</button>
              <button className="pill-btn">CRM</button>
              <button className="pill-btn">Keduanya</button>
            </div>
          </div>
        </header>

        <div className="content-area">
          {/* Alert Box */}
          <div className="alert-box animate-in delay-1">
            <span className="alert-icon" style={{ color: "#E24B4A" }}>
              <IconAlert />
            </span>
            <p className="alert-text">
              <strong>Peringatan dini:</strong> Prediksi kenaikan CMK 18–24%
              dalam 7 hari ke depan — curah hujan tinggi + akhir bulan.
            </p>
            <span className="badge-siaga">Siaga</span>
          </div>

          {/* Metrics Grid */}
          <div className="metrics-grid">
            <div className="metric-card animate-in delay-2">
              <div className="metric-label">
                <span className="dot dot-cmk"></span>Harga hari ini — CMK
              </div>
              <div className="metric-value">Rp 42.500</div>
              <div className="metric-sub up">+12,3% vs minggu lalu</div>
            </div>
            <div className="metric-card animate-in delay-3">
              <div className="metric-label">
                <span className="dot dot-crm"></span>Harga hari ini — CRM
              </div>
              <div className="metric-value">Rp 38.000</div>
              <div className="metric-sub up">+8,6% vs minggu lalu</div>
            </div>
            <div className="metric-card animate-in delay-4">
              <div className="metric-label">
                <span className="dot dot-pred"></span>Prediksi 7 hari — CMK
              </div>
              <div className="metric-value">Rp 50.100</div>
              <div className="metric-sub neutral">Interval: 47.200–53.000</div>
            </div>
            <div className="metric-card animate-in delay-5">
              <div className="metric-label">
                <span className="dot dot-acc"></span>Akurasi model (R²)
              </div>
              <div className="metric-value">92,4%</div>
              <div className="metric-sub up">MAPE 6,8% · data uji</div>
            </div>
          </div>

          {/* Main Grid: Chart + Predictions */}
          <div className="main-grid">
            {/* Chart Card */}
            <div className="card animate-in delay-5">
              <div className="card-header">
                <h3>Tren harga &amp; prediksi 14 hari ke depan</h3>
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
                <div className="legend-item">
                  <span className="legend-line" style={{ background: "#E67E22" }}></span>
                  Aktual CRM
                </div>
                <div className="legend-item">
                  <span className="legend-bar" style={{ background: "rgba(56, 189, 248, 0.4)" }}></span>
                  Curah hujan
                </div>
              </div>
              <div className="chart-container" style={{ height: "320px" }}>
                <PriceChart />
              </div>
            </div>

            {/* Predictions List Card */}
            <div className="card animate-in delay-6">
              <div className="card-header">
                <h3>Prediksi 7 hari ke depan</h3>
                <span className="badge badge-purple">Interval 95%</span>
              </div>
              <div className="card-body">
                <ul className="prediction-list">
                  {predictions.map((p, i) => (
                    <li key={i} className="prediction-item">
                      <span className="prediction-date">{p.date}</span>
                      <span className="prediction-price">{p.price}</span>
                      <span className={`prediction-badge badge ${p.color}`}>
                        {p.pct}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>

          {/* Bottom Grid: Feature Importance + Table */}
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
                      <th>Komoditas</th>
                      <th>Rp/kg</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tableData.map((row, i) => (
                      <tr key={i}>
                        <td>{row.tanggal}</td>
                        <td>{row.komoditas}</td>
                        <td>{row.harga}</td>
                        <td>
                          <span className={`status-badge ${row.statusClass}`}>
                            {row.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
