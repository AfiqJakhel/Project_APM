"""
routes/history.py — Endpoint data historis harga cabai
"""
import sys
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import TARGET
from app.core import predictor

router = APIRouter()


@router.get("/", summary="Data historis harga cabai")
def get_history(
    start: str = Query(None, description="Tanggal mulai YYYY-MM-DD"),
    end  : str = Query(None, description="Tanggal akhir YYYY-MM-DD"),
    limit: int = Query(90,   description="Jumlah hari (default 90)"),
):
    """Ambil data historis harga cabai untuk ditampilkan di grafik."""
    df = predictor.get_dataset().copy()

    if start:
        df = df[df["tanggal"] >= pd.Timestamp(start)]
    if end:
        df = df[df["tanggal"] <= pd.Timestamp(end)]

    cols = ["tanggal", TARGET]
    if "harga_cabai_rawit" in df.columns:
        cols.append("harga_cabai_rawit")

    df = df[cols].tail(limit)
    df["tanggal"] = df["tanggal"].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")
