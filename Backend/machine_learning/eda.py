import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import ccf

# Setup direktori
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_preprocessed.csv"
OUTPUT_DIR = BASE_DIR / "data" / "eda_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def main():
    print("=" * 60)
    print("  MEMULAI EXPLORATORY DATA ANALYSIS (EDA) LENGKAP  ")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset tidak ditemukan di {DATA_PATH}")
        sys.exit(1)

    print(f"Membaca data dari: {DATA_PATH.name}...")
    df = pd.read_csv(DATA_PATH, parse_dates=["tanggal"])
    df = df.sort_values("tanggal").reset_index(drop=True)
    
    # -------------------------------------------------------------------
    # 1. RINGKASAN STATISTIK
    # -------------------------------------------------------------------
    print("\n[1/10] Membuat ringkasan statistik...")
    stats_text = []
    stats_text.append("=== RINGKASAN STATISTIK EDA ===")
    stats_text.append(f"Total baris: {len(df)}")
    stats_text.append(f"Rentang waktu: {df['tanggal'].min().date()} s/d {df['tanggal'].max().date()}")
    
    for target in ['harga_cabai_merah', 'harga_cabai_rawit']:
        if target in df.columns:
            stats_text.append(f"\n--- {target.upper()} ---")
            stats_text.append(f"Min  : Rp {df[target].min():,.0f}")
            stats_text.append(f"Max  : Rp {df[target].max():,.0f}")
            stats_text.append(f"Mean : Rp {df[target].mean():,.0f}")
            stats_text.append(f"Std  : Rp {df[target].std():,.0f}")
            stats_text.append(f"Median: Rp {df[target].median():,.0f}")

    with open(OUTPUT_DIR / "00_ringkasan_statistik.txt", "w") as f:
        f.write("\n".join(stats_text))

    # -------------------------------------------------------------------
    # 2. ANALISIS DISTRIBUSI HARGA
    # -------------------------------------------------------------------
    print("[2/10] Memplot Distribusi Harga...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(df['harga_cabai_merah'], kde=True, ax=axes[0], color='red', bins=30)
    axes[0].set_title('Distribusi Harga Cabai Merah')
    axes[0].set_xlabel('Harga (Rp)')
    
    if 'harga_cabai_rawit' in df.columns:
        sns.histplot(df['harga_cabai_rawit'], kde=True, ax=axes[1], color='green', bins=30)
        axes[1].set_title('Distribusi Harga Cabai Rawit')
        axes[1].set_xlabel('Harga (Rp)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_distribusi_harga.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------
    # 3. ANALISIS TIME SERIES
    # -------------------------------------------------------------------
    print("[3/10] Memplot Pergerakan Waktu (Time Series)...")
    plt.figure(figsize=(16, 6))
    plt.plot(df['tanggal'], df['harga_cabai_merah'], label='Cabai Merah', color='red', alpha=0.8)
    if 'harga_cabai_rawit' in df.columns:
        plt.plot(df['tanggal'], df['harga_cabai_rawit'], label='Cabai Rawit', color='green', alpha=0.8)
    plt.title('Pergerakan Harga Cabai (2022 - Saat Ini)')
    plt.xlabel('Tahun')
    plt.ylabel('Harga (Rp)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_timeseries_harga.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------
    # 4. ANALISIS SEASONALITY
    # -------------------------------------------------------------------
    print("[4/10] Memplot Musiman (Seasonality)...")
    if 'month' not in df.columns:
        df['month'] = df['tanggal'].dt.month
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='month', y='harga_cabai_merah', data=df, color='salmon')
    plt.title('Variasi Harga Cabai Merah per Bulan (Seasonality Check)')
    plt.xlabel('Bulan (1=Jan, 12=Des)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_seasonality_bulan.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------------
    # 5. FEATURE IMPORTANCE (KORELASI TOP 20)
    # -------------------------------------------------------------------
    print("[5/10] Memplot Top 20 Feature Importance...")
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df_num = df.select_dtypes(include=numerics)
    target_col = 'target_h1' if 'target_h1' in df_num.columns else 'harga_cabai_merah'
    if target_col in df_num.columns:
        correlations = df_num.corrwith(df_num[target_col]).abs().sort_values(ascending=False)
        bocor_cols = ['harga_cabai_merah', 'target_h1', 'target_h3', 'target_h7', 
                      'target_rawit_h1', 'target_rawit_h3', 'target_rawit_h7',
                      'arah_target_h1', 'arah_target_h3', 'arah_target_h7']
        correlations = correlations.drop(labels=[c for c in bocor_cols if c in correlations.index], errors='ignore')
        top_20 = correlations.head(20)
        plt.figure(figsize=(12, 10))
        sns.barplot(x=top_20.values, y=top_20.index, hue=top_20.index, palette='viridis', dodge=False)
        plt.title(f'Top 20 Fitur Paling Berkorelasi dengan {target_col}')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "04_top20_feature_importance.png", dpi=150)
        plt.close()

    # -------------------------------------------------------------------
    # ADVANCED 6: CCF Rawit vs Merah
    # -------------------------------------------------------------------
    print("[6/10] Memplot Cross-Correlation (CCF)...")
    if 'harga_cabai_merah' in df.columns and 'harga_cabai_rawit' in df.columns:
        plt.figure(figsize=(10, 5))
        temp_df = df[['harga_cabai_rawit', 'harga_cabai_merah']].dropna()
        ccf_vals = ccf(temp_df['harga_cabai_rawit'], temp_df['harga_cabai_merah'], adjusted=False)[:30]
        plt.stem(range(30), ccf_vals, basefmt=" ")
        plt.title('Cross-Correlation: Cabai Rawit (Leading) vs Cabai Merah')
        plt.xlabel('Lag (Hari)')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.axhline(y=1.96/np.sqrt(len(temp_df)), color='r', linestyle='--')
        plt.axhline(y=-1.96/np.sqrt(len(temp_df)), color='r', linestyle='--')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "05_ccf_rawit_merah.png", dpi=150)
        plt.close()

    # -------------------------------------------------------------------
    # ADVANCED 7: Heatmap Korelasi Cuaca dengan Lags
    # -------------------------------------------------------------------
    print("[7/10] Memplot Heatmap Korelasi Cuaca-Harga (Multi-Lag)...")
    weather_cols = ['suhu_rata', 'curah_hujan', 'lag_hujan_14', 'lag_hujan_21', 'roll_hujan_14', 'roll_hujan_30']
    target_cols = ['harga_cabai_merah', 'target_h1', 'target_h7']
    available_weather = [c for c in weather_cols if c in df.columns]
    available_targets = [c for c in target_cols if c in df.columns]
    if available_weather and available_targets:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[available_weather + available_targets].corr().loc[available_weather, available_targets]
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Heatmap Korelasi: Variabel Cuaca & Lag vs Target Harga')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "06_heatmap_cuaca_lag.png", dpi=150)
        plt.close()

    # -------------------------------------------------------------------
    # ADVANCED 8: Dekomposisi STL
    # -------------------------------------------------------------------
    print("[8/10] Memplot Dekomposisi STL...")
    if 'harga_cabai_merah' in df.columns:
        temp_ts = df.set_index('tanggal')['harga_cabai_merah'].dropna()
        try:
            res = STL(temp_ts, period=30).fit()
            fig = res.plot()
            fig.set_size_inches(12, 10)
            plt.suptitle('Dekomposisi STL: Harga Cabai Merah (Period=30 Hari)', fontsize=16)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "07_stl_decomposition.png", dpi=150)
            plt.close()
        except Exception as e:
            print(f"  -> Gagal memplot STL: {e}")

    # -------------------------------------------------------------------
    # ADVANCED 9: Analisis Spike Harga
    # -------------------------------------------------------------------
    print("[9/10] Memplot Analisis Lonjakan Harga (Spike Analysis)...")
    if 'harga_cabai_merah' in df.columns:
        plt.figure(figsize=(16, 6))
        price = df['harga_cabai_merah']
        Q1 = price.quantile(0.25)
        Q3 = price.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        plt.plot(df['tanggal'], price, label='Harga Merah', color='blue', alpha=0.6)
        plt.axhline(upper_bound, color='red', linestyle='--', label=f'Batas Spikes (Q3 + 1.5 IQR) = Rp {upper_bound:,.0f}')
        spikes = df[df['harga_cabai_merah'] > upper_bound]
        plt.scatter(spikes['tanggal'], spikes['harga_cabai_merah'], color='red', s=50, zorder=5, label='Anomali/Spike')
        plt.title('Deteksi Periode Lonjakan Harga (Spike Analysis)')
        plt.xlabel('Tahun')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "08_spike_analysis.png", dpi=150)
        plt.close()

    # -------------------------------------------------------------------
    # ADVANCED 10: Autocorrelation (ACF & PACF)
    # -------------------------------------------------------------------
    print("[10/10] Memplot Autocorrelation (ACF & PACF)...")
    if 'harga_cabai_merah' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ts_data = df['harga_cabai_merah'].dropna()
        plot_acf(ts_data, lags=40, ax=axes[0], title='Autocorrelation (ACF) - Harga Merah')
        plot_pacf(ts_data, lags=40, ax=axes[1], title='Partial Autocorrelation (PACF) - Harga Merah')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "09_acf_pacf.png", dpi=150)
        plt.close()

    print("\n=======================================================")
    print(f"  SELESAI! Semua grafik disimpan di: {OUTPUT_DIR}")
    print("=======================================================")

if __name__ == "__main__":
    main()
