"""
eda_analyst.py
--------------
Script EDA mandiri untuk 10 emiten BEI, difokuskan pada analisis volatilitas ekonometrika.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from functools import lru_cache

warnings.filterwarnings("ignore")

DATA_FOLDER, OUTPUT_FOLDER = "data", "eda_output"
ALLOWED_STOCKS = ("ASII", "BBCA", "BBNI", "BBRI", "BMRI", "BYAN", "ICBP", "TLKM", "TPIA", "UNVR")
START, END = pd.Timestamp("2020-01-01"), pd.Timestamp("2026-02-28")

C_BG, C_LIME, C_RED = "#191919", "#BED754", "#750E21"
PALETTE = [C_LIME, "#E3651D", C_RED, "#EEEEEE", "#7ECFE0", "#F4A261", "#E76F51", "#2A9D8F", "#E9C46A", "#264653"]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
plt.rcParams.update({"figure.facecolor": C_BG, "axes.facecolor": "#1E1E1E", "axes.edgecolor": "#333333", 
                     "text.color": "#EEEEEE", "axes.labelcolor": "#EEEEEE", "axes.titlecolor": C_LIME, 
                     "xtick.color": "#EEEEEE", "ytick.color": "#EEEEEE", "grid.color": "#333333", 
                     "grid.alpha": 0.4, "font.family": "monospace", "legend.facecolor": "#111111"})

def save_plot(fig, name):
    path = f"{OUTPUT_FOLDER}/{name}"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"[✓] Tersimpan: {path}")

@lru_cache(maxsize=10)
def load_stock(s: str) -> pd.DataFrame:
    df = pd.read_csv(f"{DATA_FOLDER}/{s}.csv", thousands='.', decimal=',')
    df.columns = [c.strip().lstrip("\ufeff").strip('"') for c in df.columns]
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df = df[df["Tanggal"].between(START, END)].sort_values("Tanggal")
    
    r = np.log(df["Terakhir"] / df["Terakhir"].shift(1))
    parkinson = np.sqrt(1 / (4 * np.log(2))) * np.log(df["Tertinggi"] / df["Terendah"]) * 100
    return df.assign(Return=r, Target_Volatility=parkinson).dropna().reset_index(drop=True)

def plot_return_distributions():
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, (ax, s) in enumerate(zip(axes.flat, ALLOWED_STOCKS)):
        ret = load_stock(s)["Return"] * 100
        ax.hist(ret, bins=60, color=PALETTE[i], alpha=0.8)
        ax.axvline(ret.mean(), color=C_RED, ls="--")
        ax.set(title=s, xlabel="Return (%)"); ax.grid(True, lw=0.3)
    fig.suptitle("Distribusi Return Harian (Uji Normalitas Volatilitas)", fontsize=14, color=C_LIME)
    save_plot(fig, "01_return_distributions.png")

def plot_volatility_clustering():
    """Membuat plot ACF dari Kuadrat Return untuk membuktikan efek Volatility Clustering"""
    fig, axes = plt.subplots(2, 5, figsize=(24, 10))
    for i, (ax, s) in enumerate(zip(axes.flat, ALLOWED_STOCKS)):
        sq_ret = (load_stock(s)["Return"] * 100) ** 2
        plot_acf(sq_ret, ax=ax, lags=40, alpha=0.05, color=PALETTE[i], title=f"{s} (Squared Returns)")
        ax.set_ylim(-0.1, 0.4); ax.grid(True, lw=0.3)
    fig.suptitle("Autocorrelation Function (ACF) - Bukti Volatility Clustering untuk GARCH", fontsize=14, color=C_LIME)
    save_plot(fig, "02_volatility_clustering_acf.png")

def run_adf_tests():
    print(f"\n{'═'*70}\n  UJI STASIONERITAS ADF (Syarat Mutlak GARCH)\n{'═'*70}")
    rows = []
    for s in ALLOWED_STOCKS:
        p_val = adfuller(load_stock(s)["Return"].dropna())[1]
        rows.append({"Saham": s, "Series": "Return Logaritmik", "p-value": round(p_val, 4), "Stasioner?": "✓ Ya" if p_val < 0.05 else "✗ Tidak"})
    print((df := pd.DataFrame(rows)).to_string(index=False))
    df.to_csv(f"{OUTPUT_FOLDER}/adf_results.csv", index=False)

if __name__ == "__main__":
    [func() for func in (plot_return_distributions, plot_volatility_clustering, run_adf_tests)]
    print(f"\n{'═'*60}\n  Semua output telah disimpan ke ./{OUTPUT_FOLDER}/\n{'═'*60}\n")