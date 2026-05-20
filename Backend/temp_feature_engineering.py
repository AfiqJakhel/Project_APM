def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    log("\n[5] Feature engineering (v2 — fitur selektif)...")

    # ── A. Kalender (hanya yang informatif untuk harga komoditas) ──────────
    df["bulan"] = df["tanggal"].dt.month
    df["kuartal"] = df["tanggal"].dt.quarter
    df["is_weekend"] = (df["tanggal"].dt.dayofweek >= 5).astype(int)
    df["is_awal_bulan"] = (df["tanggal"].dt.day <= 7).astype(int)
    # Enkode siklus bulanan dengan sin/cos (menggantikan bulan mentah)
    df["bulan_sin"] = np.sin(2 * np.pi * df["bulan"] / 12)
    df["bulan_cos"] = np.cos(2 * np.pi * df["bulan"] / 12)
    # Catatan: tahun, hari_bulan, hari_minggu, hari_sin/cos, hari_dlm_tahun,
    #          minggu_ke dihapus — tidak berpola signifikan untuk harga cabai

    # ── B. Event khusus (dipertahankan semua — domain knowledge kuat) ──────
    lebaran = pd.to_datetime(
        ["2022-05-02", "2023-04-22", "2024-04-10", "2025-03-30", "2026-03-20"]
    )
    df["is_pra_lebaran"] = 0
    df["is_lebaran"] = 0
    df["is_pasca_lebaran"] = 0
    for d in lebaran:
        df.loc[
            (df["tanggal"] >= d - pd.Timedelta(14)) & (df["tanggal"] < d),
            "is_pra_lebaran",
        ] = 1
        df.loc[
            (df["tanggal"] >= d) & (df["tanggal"] <= d + pd.Timedelta(2)), "is_lebaran"
        ] = 1
        df.loc[
            (df["tanggal"] > d + pd.Timedelta(2))
            & (df["tanggal"] <= d + pd.Timedelta(9)),
            "is_pasca_lebaran",
        ] = 1
    df["is_natal_tahunbaru"] = (
        ((df["bulan"] == 12) & (df["tanggal"].dt.day >= 20))
        | ((df["bulan"] == 1) & (df["tanggal"].dt.day <= 7))
    ).astype(int)
    df["is_musim_panen"] = df["bulan"].isin([5, 6, 7, 11, 12, 1]).astype(int)

    # ── C. Lag harga (selektif — hapus lag berkorelasi tinggi) ────────────
    # Simpan: lag_1 (kemarin), lag_7 (minggu lalu), lag_14 (dua minggu),
    #         lag_30 (sebulan lalu). Hapus: lag_2, lag_3, lag_21 (redundan)
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)

    if "harga_cabai_rawit" in df.columns:
        # Rawit sebagai sinyal pasar substitusi jangka pendek
        df["rawit_lag_1"] = df["harga_cabai_rawit"].shift(1)
        df["rawit_lag_7"] = df["harga_cabai_rawit"].shift(7)

    # ── D. Rolling statistics (hanya window 7 & 30, tanpa min/max) ────────
    # min/max di-drop: informasinya sudah tercakup oleh mean + std
    # Window 3 dan 14 di-drop: marginal di antara 7 dan 30
    for w in [7, 30]:
        s = df[TARGET].shift(1)
        df[f"roll_mean_{w}"] = s.rolling(w).mean()
        df[f"roll_std_{w}"] = s.rolling(w).std()

    # ── E. Momentum (hanya jangka pendek) ─────────────────────────────────
    # momentum_30 dan pct_change di-drop: berkorelasi tinggi dengan momentum_7
    df["momentum_7"] = df[TARGET].shift(1) - df[TARGET].shift(7)

    # ── F. Fitur tren & volatilitas (BARU) ────────────────────────────────
    # Rasio tren: apakah harga minggu ini naik lebih cepat dari tren bulanan?
    df["rasio_tren_7_30"] = (
        df["roll_mean_7"] / df["roll_mean_30"].replace(0, np.nan)
    ).fillna(1.0)

    # Koefisien variasi 14 hari: proxy ketidakstabilan / gejolak harga
    roll14_mean = df[TARGET].shift(1).rolling(14).mean().replace(0, np.nan)
    roll14_std = df[TARGET].shift(1).rolling(14).std()
    df["volatilitas_14"] = (roll14_std / roll14_mean).fillna(0)

    # ── G. Cuaca (hanya fitur utama + akumulasi hujan jangka panjang) ─────
    # Dihapus: kelembaban, kec_angin, lama_penyinaran (korelasi tinggi dgn
    #          curah_hujan), roll_suhu_*, roll_hujan_7, roll_hr_hujan_30,
    #          roll_kelembaban_7 (redundan)
    if "curah_hujan" in df.columns:
        df["roll_hujan_30"] = df["curah_hujan"].rolling(30, min_periods=1).sum()
        df["is_hari_hujan"] = (df["curah_hujan"] > 1.0).astype(int)
    # suhu_rata dan curah_hujan dipertahankan langsung dari merge

    # ── H. Diferensiasi harga (PERBAIKAN 4) ──────────────────────────────
    # Menangkap "fluktuasi" secara eksplisit sesuai judul penelitian.
    # diff(n) = harga_hari_ini - harga_n_hari_lalu
    # Ini berbeda dengan momentum_7 yang menggunakan shift(1) sebagai basis
    df["selisih_harga_1"] = df[TARGET].diff(1)  # perubahan harian
    df["selisih_harga_7"] = df[TARGET].diff(7)  # perubahan mingguan

    # Fitur Arah Pergerakan Harga (Klasifikasi)
    df["arah_1"] = np.sign(df["selisih_harga_1"])
    df["arah_7"] = np.sign(df["selisih_harga_7"])
    
    # Fitur Streak Naik: counter berapa hari berturut-turut harga naik
    # Jika arah_1 == 1, tambah 1, jika tidak reset ke 0
    df["streak_naik"] = df["arah_1"].groupby((df["arah_1"] != 1).cumsum()).cumcount()

    # ── I. Fitur konteks historis (PERBAIKAN 4) ───────────────────────────
    # Tahun 2022 = masa pemulihan pasca-COVID, pola harga sangat tidak normal
    # Model perlu "diberitahu" bahwa data 2022 memiliki karakter berbeda
    df["is_pasca_covid"] = (df["tanggal"].dt.year == 2022).astype(int)

    # ── J. Target multi-horizon (PERBAIKAN 3) ─────────────────────────────
    # Membuat kolom target untuk prediksi H+1, H+3, H+7
    # shift(-n) = ambil nilai n hari ke depan sebagai target
    # Kolom ini TIDAK dimasukkan ke feature_cols, hanya dipakai saat training
    df["target_h1"] = df[TARGET].shift(-1)  # prediksi besok
    df["target_h3"] = df[TARGET].shift(-3)  # prediksi 3 hari ke depan
    df["target_h7"] = df[TARGET].shift(-7)  # prediksi 7 hari ke depan

    # Target klasifikasi: 1 jika harga target > harga hari ini, 0 jika tidak (turun/tetap)
    df["arah_target_h1"] = (df["target_h1"] > df[TARGET]).astype(int)
    df["arah_target_h3"] = (df["target_h3"] > df[TARGET]).astype(int)
    df["arah_target_h7"] = (df["target_h7"] > df[TARGET]).astype(int)

    # Karena target numerik di-shift, kita perlu memastikan baris terakhir target klasifikasi
    # menjadi NaN agar sejalan dengan target regresi (mencegah evaluasi pada target palsu)
    df.loc[df["target_h1"].isna(), "arah_target_h1"] = np.nan
    df.loc[df["target_h3"].isna(), "arah_target_h3"] = np.nan
    df.loc[df["target_h7"].isna(), "arah_target_h7"] = np.nan

    log(f"    -> Total fitur dibuat : {df.shape[1] - 2}")
    log(f"    -> Kolom target multi-horizon: target_h1, target_h3, target_h7")
    return df