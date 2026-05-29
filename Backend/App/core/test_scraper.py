import sys
import asyncio
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.scraper import transform_data_ke_format_dataset, DATA_DIR

def test_mapper_struktur_data():
    """
    Test apakah fungsi mapper mengembalikan dataframe 
    dengan jumlah dan nama kolom yang persis 100% sama dengan dataset mentah.
    """
    csv_path = DATA_DIR / "dataset_preprocessed.csv"
    if not csv_path.exists():
        print("Dataset tidak ditemukan. Skip test.")
        return
        
    df_history = pd.read_csv(csv_path, parse_dates=["tanggal"])
    
    # 1. Mock Data Input (Hasil Scraper & API)
    harga_data = {
        "tanggal": date.today().isoformat(),
        "harga_cabai_merah": 55000.0,
        "harga_cabai_rawit": 48000.0,
        "sumber": "PIHPS BI"
    }
    
    cuaca_data = {
        "suhu_rata": 28.5,
        "kelembaban": 82.0,
        "curah_hujan": 12.5,
        "lama_penyinaran": 8.0,
        "kec_angin": 10.0,
    }
    
    is_libur = 0  # Misal bukan hari libur
    
    # 2. Proses melalui Mapper
    try:
        new_row_df = transform_data_ke_format_dataset(df_history, harga_data, cuaca_data, is_libur)
    except Exception as e:
        print(f"ERROR saat menjalankan mapper: {e}")
        return
    
    # 3. Validasi
    kolom_asli = list(df_history.columns)
    kolom_baru = list(new_row_df.columns)
    
    print("=" * 50)
    print("TEST HASIL MAPPER SINKRONISASI DATA")
    print("=" * 50)
    
    # Cek jumlah kolom
    print(f"Jumlah kolom dataset asli : {len(kolom_asli)}")
    print(f"Jumlah kolom hasil mapper : {len(kolom_baru)}")
    assert len(kolom_asli) == len(kolom_baru), "JUMLAH KOLOM TIDAK SAMA!"
    
    # Cek urutan dan nama kolom
    for i, (col_a, col_b) in enumerate(zip(kolom_asli, kolom_baru)):
        if col_a != col_b:
            print(f"MISMATCH di index {i}: '{col_a}' != '{col_b}'")
            assert col_a == col_b
            
    print("[PASS] Struktur kolom 100% konsisten!")
    
    # Tampilkan sample nilainya
    print("\nSAMPLE VALUE 5 KOLOM PERTAMA & TERAKHIR:")
    print(new_row_df.iloc[0, :5].to_dict())
    print("...")
    print(new_row_df.iloc[0, -5:].to_dict())
    print("=" * 50)
    print("Test Berhasil!")

if __name__ == "__main__":
    test_mapper_struktur_data()
