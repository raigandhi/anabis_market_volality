"""
data_processor.py
-----------------
Menangani pemuatan CSV, filter waktu, dan kalkulasi proksi volatilitas.
Dioptimalkan khusus untuk model runtun waktu (Time-Series) Ekonometrika.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional

ALLOWED_STOCKS = {"ASII", "BBCA", "BBNI", "BBRI", "BMRI",
                  "BYAN", "ICBP", "TLKM", "TPIA", "UNVR"}
DEFAULT_START = pd.Timestamp("2020-01-01")
DEFAULT_END   = pd.Timestamp("2026-02-28")
DATA_FOLDER   = "data"

@st.cache_data(show_spinner=False)
def load_stock(stock: str) -> pd.DataFrame:
    if stock not in ALLOWED_STOCKS:
        raise ValueError(f"'{stock}' tidak ada dalam daftar izin.")

    path = f"{DATA_FOLDER}/{stock}.csv"
    df = pd.read_csv(path, thousands='.', decimal=',')

    df.columns = [c.strip().lstrip("\ufeff").strip('"') for c in df.columns]
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)

    df = df[df["Tanggal"].between(DEFAULT_START, DEFAULT_END)]
    df = df.sort_values("Tanggal").reset_index(drop=True)

    # Return Logaritmik Harian
    df["Return"] = np.log(df["Terakhir"] / df["Terakhir"].shift(1))
    
    # PROKSI VOLATILITAS: Parkinson Volatility (berdasarkan High-Low)
    # Sangat akurat sebagai target pembanding (aktual) untuk volatilitas harian GARCH
    df["Target_Volatility"] = np.sqrt(1 / (4 * np.log(2))) * np.log(df["Tertinggi"] / df["Terendah"]) * 100

    return df.dropna().reset_index(drop=True)


class DataProcessor:
    def __init__(self, stock: str):
        if stock not in ALLOWED_STOCKS:
            raise ValueError(f"Saham '{stock}' tidak valid.")
        self.stock = stock

    def load(self) -> pd.DataFrame:
        return load_stock(self.stock)

    @staticmethod
    def apply_time_filter(
        df: pd.DataFrame,
        mode: str,
        custom_start: Optional[pd.Timestamp] = None,
        custom_end:   Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        latest = df["Tanggal"].max()
        days_map = {
            "7 Hari Terakhir": 7,
            "1 Bulan Terakhir": 30,
            "1 Tahun Terakhir": 365
        }
        if mode in days_map:
            return df[df["Tanggal"] >= latest - pd.Timedelta(days=days_map[mode])].copy()
        if mode == "Custom Range" and custom_start and custom_end:
            return df[df["Tanggal"].between(custom_start, custom_end)].copy()
            
        return df.copy()