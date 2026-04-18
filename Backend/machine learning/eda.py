"""
=============================================================================
EDA (Exploratory Data Analysis) - PREDIKSI HARGA CABAI KOTA PADANG
=============================================================================
Jalankan SETELAH preprocessing.py selesai.
Input  : App/output/dataset_preprocessed.csv
Output : App/output/eda/
  - 01_tren_harga_cabai.png
  - 02_distribusi_harga.png
  - 03_harga_per_bulan.png
  - 04_harga_vs_cuaca.png
  - 05_heatmap_korelasi.png
  - 06_missing_value.png
  - 07_outlier_boxplot.png
  - laporan_eda.txt
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "App" / "output"
EDA_DIR    = OUTPUT_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi"      : 120,
    "font.size"       : 10,
    "axes.titlesize"  : 12,
    "axes.titleweight": "bold",
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "axes.spines.top" : False,
    "axes.spines.right": False,
})

WARNA_MERAH = "#E74C3C"
WARNA_RAWIT = "#E67E22"
WARNA_BIRU  = "#2E86AB"
LOG_EDA     = []


def log(msg):
    print(msg)
    LOG_EDA.append(msg)


# =============================================================================
# LOAD DATA
# =============================================================================
def load_data() -> pd.DataFrame:
    path = OUTPUT_DIR / "dataset_preprocessed.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {path}\n"
            "Pastikan preprocessing.py sudah dijalankan terlebih dahulu!"
        )
    df = pd.read_csv(path, parse_dates=["tanggal"])
    df.sort_values("tanggal", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log(f"Data dimuat: {len(df):,} baris, {df.shape[1]} kolom")
    log(f"Rentang: {df['tanggal'].min().date()} s/d {df['tanggal'].max().date()}")
    return df


# =============================================================================
# PLOT 1 — TREN HARGA CABAI
# =============================================================================
def plot_tren_harga(df: pd.DataFrame):
    log("\n[1] Membuat grafik tren harga cabai...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Tren Harga Cabai di Kota Padang (2022–2026)", fontsize=14, fontweight="bold")

    # Cabai Merah
    axes[0].plot(df["tanggal"], df["harga_cabai_merah"],
                 color=WARNA_MERAH, linewidth=0.8, label="Harga harian")
    roll7 = df["harga_cabai_merah"].rolling(7).mean()
    axes[0].plot(df["tanggal"], roll7,
                 color="#922B21", linewidth=1.8, label="MA-7 hari", linestyle="--")
    axes[0].set_ylabel("Harga (Rp)")
    axes[0].set_title("Cabai Merah Keriting")
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"Rp {x:,.0f}")
    )
    axes[0].legend(fontsize=9)

    # Cabai Rawit
    if "harga_cabai_rawit" in df.columns:
        axes[1].plot(df["tanggal"], df["harga_cabai_rawit"],
                     color=WARNA_RAWIT, linewidth=0.8, label="Harga harian")
        roll7r = df["harga_cabai_rawit"].rolling(7).mean()
        axes[1].plot(df["tanggal"], roll7r,
                     color="#784212", linewidth=1.8, label="MA-7 hari", linestyle="--")
        axes[1].set_ylabel("Harga (Rp)")
        axes[1].set_title("Cabai Rawit Hijau")
        axes[1].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"Rp {x:,.0f}")
        )
        axes[1].legend(fontsize=9)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    out = EDA_DIR / "01_tren_harga_cabai.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"    -> Disimpan: {out.name}")


# =============================================================================
# PLOT 2 — DISTRIBUSI HARGA
# =============================================================================
def plot_distribusi(df: pd.DataFrame):
    log("\n[2] Membuat grafik distribusi harga...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribusi Harga Cabai", fontsize=14, fontweight="bold")

    for ax, col, warna, label in [
        (axes[0], "harga_cabai_merah", WARNA_MERAH, "Cabai Merah Keriting"),
        (axes[1], "harga_cabai_rawit", WARNA_RAWIT, "Cabai Rawit Hijau"),
    ]:
        if col not in df.columns:
            continue
        data = df[col].dropna()
        ax.hist(data, bins=40, color=warna, alpha=0.7, edgecolor="white")
        ax.axvline(data.mean(),   color="#2C3E50", linestyle="--",
                   linewidth=1.5, label=f"Mean: Rp {data.mean():,.0f}")
        ax.axvline(data.median(), color="#7F8C8D", linestyle=":",
                   linewidth=1.5, label=f"Median: Rp {data.median():,.0f}")
        ax.set_title(label)
        ax.set_xlabel("Harga (Rp)")
        ax.set_ylabel("Frekuensi")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
        ax.legend(fontsize=8)

        log(f"    {label}:")
        log(f"      Mean   : Rp {data.mean():>10,.0f}")
        log(f"      Median : Rp {data.median():>10,.0f}")
        log(f"      Std    : Rp {data.std():>10,.0f}")
        log(f"      Skew   : {data.skew():.3f}")

    plt.tight_layout()
    out = EDA_DIR / "02_distribusi_harga.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"    -> Disimpan: {out.name}")


# =============================================================================
# PLOT 3 — RATA-RATA HARGA PER BULAN (MUSIMAN)
# =============================================================================
def plot_musiman(df: pd.DataFrame):
    log("\n[3] Membuat grafik pola musiman...")

    nama_bulan = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
                  "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Rata-rata Harga Cabai per Bulan (Pola Musiman)", fontsize=14, fontweight="bold")

    for ax, col, warna, label in [
        (axes[0], "harga_cabai_merah", WARNA_MERAH, "Cabai Merah Keriting"),
        (axes[1], "harga_cabai_rawit", WARNA_RAWIT, "Cabai Rawit Hijau"),
    ]:
        if col not in df.columns:
            continue
        bulanan = df.groupby("bulan")[col].agg(["mean", "std"]).reset_index()
        ax.bar(bulanan["bulan"], bulanan["mean"],
               color=warna, alpha=0.75, edgecolor="white")
        ax.errorbar(bulanan["bulan"], bulanan["mean"],
                    yerr=bulanan["std"], fmt="none",
                    color="#2C3E50", capsize=4, linewidth=1)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(nama_bulan)
        ax.set_title(label)
        ax.set_ylabel("Rata-rata Harga (Rp)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rp {x:,.0f}"))

    plt.tight_layout()
    out = EDA_DIR / "03_harga_per_bulan.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"    -> Disimpan: {out.name}")


# =============================================================================
# PLOT 4 — HARGA VS CUACA (SCATTER)
# =============================================================================
def plot_harga_vs_cuaca(df: pd.DataFrame):
    log("\n[4] Membuat grafik harga vs cuaca...")

    cuaca_vars = [
        ("suhu_rata",    "Suhu Rata-rata (°C)"),
        ("curah_hujan",  "Curah Hujan (mm)"),
        ("kelembaban",   "Kelembaban (%)"),
    ]
    available = [(c, l) for c, l in cuaca_vars if c in df.columns]
    if not available:
        log("    ! Tidak ada kolom cuaca — skip")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]
    fig.suptitle("Hubungan Harga Cabai Merah vs Variabel Cuaca",
                 fontsize=14, fontweight="bold")

    for ax, (col, xlabel) in zip(axes, available):
        sample = df[["harga_cabai_merah", col]].dropna().sample(
            min(500, len(df)), random_state=42
        )
        ax.scatter(sample[col], sample["harga_cabai_merah"],
                   alpha=0.3, s=15, color=WARNA_MERAH)
        # Garis tren
        z = np.polyfit(sample[col], sample["harga_cabai_merah"], 1)
        p = np.poly1d(z)
        xs = np.linspace(sample[col].min(), sample[col].max(), 100)
        ax.plot(xs, p(xs), color="#2C3E50", linewidth=1.5, linestyle="--")
        corr = sample["harga_cabai_merah"].corr(sample[col])
        ax.set_title(f"Korelasi: {corr:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Harga Cabai Merah (Rp)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"Rp {x:,.0f}"))
        log(f"    Korelasi harga_cabai_merah vs {col}: {corr:.3f}")

    plt.tight_layout()
    out = EDA_DIR / "04_harga_vs_cuaca.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"    -> Disimpan: {out.name}")


# =============================================================================
# PLOT 5 — HEATMAP KORELASI FITUR
# =============================================================================
def plot_korelasi(df: pd.DataFrame):
    log("\n[5] Membuat heatmap korelasi...")

    # Pilih fitur paling relevan untuk heatmap (tidak semua — terlalu banyak)
    kolom_pilihan = [
        "harga_cabai_merah", "harga_cabai_rawit",
        "suhu_rata", "kelembaban", "curah_hujan", "lama_penyinaran", "kec_angin",
        "lag_1", "lag_7", "lag_14", "lag_30",
        "roll_mean_7", "roll_mean_30", "roll_std_7",
        "momentum_7", "is_libur_nasional", "is_pra_lebaran",
        "is_lebaran", "is_musim_panen", "bulan_sin", "bulan_cos",
    ]
    cols = [c for c in kolom_pilihan if c in df.columns]
    corr_matrix = df[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.3, ax=ax, annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Heatmap Korelasi Fitur", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    out = EDA_DIR / "05_heatmap_korelasi.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    # Log top korelasi dengan target
    log("    Top 10 fitur paling berkorelasi dengan harga_cabai_merah:")
    top = corr_matrix["harga_cabai_merah"].drop("harga_cabai_merah").abs().sort_values(ascending=False).head(10)
    for feat, val in top.items():
        log(f"      {feat:<30} : {val:.3f}")
    log(f"    -> Disimpan: {out.name}")


# =============================================================================
# PLOT 6 — MISSING VALUE
# =============================================================================
def plot_missing_value(df: pd.DataFrame):
    log("\n[6] Membuat grafik missing value...")

    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)

    if len(miss) == 0:
        log("    -> Tidak ada missing value — grafik dilewati")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(miss) * 0.4)))
    colors = [WARNA_MERAH if v / len(df) > 0.1 else WARNA_BIRU for v in miss.values]
    bars = ax.barh(miss.index, miss.values, color=colors, alpha=0.8)
    ax.bar_label(bars, labels=[f"{v} ({v/len(df)*100:.1f}%)" for v in miss.values],
                 padding=4, fontsize=8)
    ax.set_title("Jumlah Missing Value per Kolom", fontsize=14, fontweight="bold")
    ax.set_xlabel("Jumlah Missing")

    plt.tight_layout()
    out = EDA_DIR / "06_missing_value.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"    -> Disimpan: {out.name}")


# =============================================================================
# PLOT 7 — OUTLIER BOXPLOT
# =============================================================================
def plot_outlier(df: pd.DataFrame):
    log("\n[7] Membuat boxplot outlier...")

    cols_plot = [c for c in ["harga_cabai_merah", "harga_cabai_rawit",
                              "suhu_rata", "curah_hujan", "kelembaban"]
                 if c in df.columns]

    fig, axes = plt.subplots(1, len(cols_plot), figsize=(3 * len(cols_plot), 5))
    if len(cols_plot) == 1:
        axes = [axes]
    fig.suptitle("Deteksi Outlier (Boxplot)", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, cols_plot):
        data  = df[col].dropna()
        Q1    = data.quantile(0.25)
        Q3    = data.quantile(0.75)
        IQR   = Q3 - Q1
        n_out = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=WARNA_BIRU, alpha=0.6),
                   medianprops=dict(color=WARNA_MERAH, linewidth=2))
        ax.set_title(f"{col}\n({n_out} outlier)", fontsize=9)
        ax.set_xticklabels([])
        log(f"    {col}: {n_out} outlier (IQR method)")

    plt.tight_layout()
    out = EDA_DIR / "07_outlier_boxplot.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    log(f"    -> Disimpan: {out.name}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("=" * 55)
    log("  EDA - DATA HARGA CABAI KOTA PADANG")
    log("=" * 55)

    # Cek dependensi
    try:
        import seaborn
    except ImportError:
        print("ERROR: seaborn belum terinstall. Jalankan:")
        print("  pip install seaborn matplotlib")
        return

    df = load_data()

    plot_tren_harga(df)
    plot_distribusi(df)
    plot_musiman(df)
    plot_harga_vs_cuaca(df)
    plot_korelasi(df)
    plot_missing_value(df)
    plot_outlier(df)

    # Simpan laporan EDA
    laporan_path = EDA_DIR / "laporan_eda.txt"
    with open(laporan_path, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG_EDA))

    log(f"\n[SELESAI] EDA selesai! Semua grafik tersimpan di: {EDA_DIR}")
    log(f"  Total grafik : 7 file PNG")
    log(f"  Laporan      : {laporan_path.name}")


if __name__ == "__main__":
    main()