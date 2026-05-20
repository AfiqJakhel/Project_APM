"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";

/* ──────────── SVG Icon helpers ──────────── */
const IconLeaf = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <path d="M11 20A7 7 0 0 1 9.8 6.9C15.5 4.9 17 3.5 19 2c1 2 2 4.5 2 8 0 5.5-4.8 10-10 10Z" />
    <path d="M2 21c0-3 1.9-5.5 4.5-6.5" />
  </svg>
);

interface NavItem {
  href: string;
  label: string;
}

const menuUtama: NavItem[] = [
  { href: "/",          label: "Dashboard" },
  { href: "/prediksi",  label: "Prediksi harga" },
  { href: "/historis",  label: "Data historis" },
  { href: "/metrik",    label: "Metrik model" },
];

const menuLaporan: NavItem[] = [
  { href: "/info-model", label: "Info model" },
  { href: "/rekap",      label: "Rekap harian" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(false);

  // Tutup sidebar saat rute berubah di mobile
  useEffect(() => {
    setIsOpen(false);
  }, [pathname]);

  const renderItem = (item: NavItem) => {
    const isActive = pathname === item.href;
    return (
      <li key={item.href}>
        <Link
          href={item.href}
          className={`sidebar-menu-item${isActive ? " active" : ""}`}
        >
          {item.label}
        </Link>
      </li>
    );
  };

  return (
    <>
      <button className="mobile-toggle" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? "✕" : "☰"}
      </button>

      {isOpen && (
        <div className="sidebar-overlay" onClick={() => setIsOpen(false)} />
      )}

      <aside className={`sidebar ${isOpen ? "open" : ""}`}>
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
    </>
  );
}
