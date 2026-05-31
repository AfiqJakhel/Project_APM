import pandas as pd
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CSV_PATH = DATA_DIR / "dataset_preprocessed.csv"

def run_sanity_check():
    print("=" * 60)
    print(" SANITY CHECK FITUR H+3 BARU")
    print("=" * 60)
    
    if not CSV_PATH.exists():
        print(f"❌ Error: File {CSV_PATH} tidak ditemukan.")
        print("   Pastikan preprocessing.py sudah selesai dijalankan.")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Fitur yang baru ditambahkan/dimodifikasi
    target_features = [
        "lag_21", "momentum_3", "streak_naik_3", 
        "max_hujan_7", "minggu_sin", "minggu_cos",
        "roll_hujan_60", "rawit_lag_30"
    ]
    
    # Cari yang beneran ada di dataset (setelah diseleksi/whitelist)
    found_features = [f for f in target_features if f in df.columns]
    missing_features = [f for f in target_features if f not in df.columns]
    
    if missing_features:
        print(f"⚠️ PERINGATAN: Fitur berikut terhapus/hilang: {missing_features}")
    
    print("\n1. Statistik Dasar Fitur Baru:")
    if found_features:
        print(df[found_features].describe().loc[["count", "mean", "min", "max"]].to_string())
    
    print("\n2. Pengecekan Missing Values (NaN):")
    if found_features:
        na_counts = df[found_features].isna().sum()
        print(na_counts.to_string())
        
        # Analisis jika ada NaN
        if na_counts.sum() > 0:
            print("\n   ℹ️ Catatan: Fitur lag/momentum wajar jika ada sedikit NaN di awal baris")
            print("   karena pergeseran waktu (shift). Pastikan jumlah NaN tidak mendominasi.")
    
    print("\n" + "=" * 60)
    print(" ✅ Sanity check selesai.")
    print("=" * 60)

if __name__ == "__main__":
    run_sanity_check()
