"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/* ──────────── SVG Icon helpers ──────────── */
const IconLeaf = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 20A7 7 0 0 1 9.8 6.9C15.5 4.9 17 3.5 19 2c1 2 2 4.5 2 8 0 5.5-4.8 10-10 10Z" />
    <path d="M2 21c0-3 1.9-5.5 4.5-6.5" />
  </svg>
);
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
const IconCpu = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <rect x="4" y="4" width="16" height="16" rx="2" />
    <rect x="9" y="9" width="6" height="6" />
    <line x1="9" y1="1" x2="9" y2="4" />
    <line x1="15" y1="1" x2="15" y2="4" />
    <line x1="9" y1="20" x2="9" y2="23" />
    <line x1="15" y1="20" x2="15" y2="23" />
    <line x1="20" y1="9" x2="23" y2="9" />
    <line x1="20" y1="14" x2="23" y2="14" />
    <line x1="1" y1="9" x2="4" y2="9" />
    <line x1="1" y1="14" x2="4" y2="14" />
  </svg>
);

interface NavItem {
  href: string;
  label: string;
  icon: React.ReactNode;
}

const menuUtama: NavItem[] = [
  { href: "/",          label: "Dashboard",      icon: <IconDashboard /> },
  { href: "/prediksi",  label: "Prediksi harga", icon: <IconTrend /> },
  { href: "/historis",  label: "Data historis",   icon: <IconHistory /> },
  { href: "/metrik",    label: "Metrik model",    icon: <IconCompare /> },
];

const menuLaporan: NavItem[] = [
  { href: "/info-model", label: "Info model",  icon: <IconCpu /> },
  { href: "/rekap",      label: "Rekap harian", icon: <IconReport /> },
];

export default function Sidebar() {
  const pathname = usePathname();

  const renderItem = (item: NavItem) => {
    const isActive = pathname === item.href;
    return (
      <li key={item.href}>
        <Link
          href={item.href}
          className={`sidebar-menu-item${isActive ? " active" : ""}`}
        >
          {item.icon} {item.label}
        </Link>
      </li>
    );
  };

  return (
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
          {menuUtama.map(renderItem)}
        </ul>
      </nav>

      <nav className="sidebar-section">
        <div className="sidebar-section-label">Laporan</div>
        <ul className="sidebar-menu">
          {menuLaporan.map(renderItem)}
        </ul>
      </nav>

      <div className="sidebar-footer">
        <p className="sidebar-footer-text">
          Sumber data: PIHPS BI · BMKG · SKB 3 Menteri
        </p>
      </div>
    </aside>
  );
}
