import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CabaiWatch — Dashboard Monitoring Harga Cabai Kota Padang",
  description:
    "Dashboard pemantauan dan prediksi harga cabai merah keriting (CMK) dan cabai rawit merah (CRM) untuk wilayah Kota Padang. Menggunakan model XGBoost dengan data PIHPS BI, BMKG, dan SKB 3 Menteri.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="id">
      <body>{children}</body>
    </html>
  );
}
