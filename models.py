"""
models.py
---------
Kumpulan model Ekonometrika (Keluarga ARCH & EWMA) untuk peramalan volatilitas.
Dioptimalkan untuk perbandingan runtun waktu finansial yang apples-to-apples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from abc import ABC, abstractmethod
from arch import arch_model

from data_processor import DataProcessor


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Menghitung metrik evaluasi regresi menggunakan NumPy murni."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}


class VolatilityModel(ABC):
    name: str = "BaseModel"

    def __init__(self):
        self._fitted: bool = False
        self._metrics: dict[str, float] = {}
        self._pred: np.ndarray | None = None

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None: ...

    def predict_test(self) -> np.ndarray:
        if not self._fitted:
            raise ValueError("Model belum dilatih (fit).")
        return self._pred

    @property
    def metrics(self) -> dict[str, float]:
        return self._metrics

    @property
    def test_dates(self) -> pd.Series:
        return self._test_dates

    @property
    def y_test(self) -> np.ndarray:
        return self._y_test

    def _split(self, df: pd.DataFrame) -> None:
        """Membagi data menjadi set pelatihaan (80%) dan pengujian (20%) secara kronologis."""
        y = df["Target_Volatility"].values
        dates = df["Tanggal"]

        split = int(len(y) * 0.8)
        self._split_idx = split
        
        self._y_train, self._y_test = y[:split], y[split:]
        self._train_dates = dates.iloc[:split].reset_index(drop=True)
        self._test_dates  = dates.iloc[split:].reset_index(drop=True)


class ArchFamilyModel(VolatilityModel):
    """
    Kelas Induk (Base Class) untuk seluruh model keluarga ARCH.
    Memudahkan inisialisasi parameter p, o, dan q tanpa menulis ulang logika fitting.
    """
    def __init__(self, name: str, vol_type: str, p: int = 1, o: int = 0, q: int = 1):
        super().__init__()
        self.name = name
        self.vol_type = vol_type
        self.p = p
        self.o = o  # Parameter asimetris (untuk GJR-GARCH dan EGARCH)
        self.q = q

    def fit(self, df: pd.DataFrame) -> None:
        self._split(df)
        returns_pct = df["Return"].values * 100

        # MENCEGAH LEAKAGE: Hitung batas clipping hanya menggunakan data training
        train_returns = returns_pct[:self._split_idx]
        sigma_train = np.std(train_returns)
        returns_clean = np.clip(returns_pct, -4 * sigma_train, 4 * sigma_train)

        # 1. Inisialisasi model sesuai parameter turunan (ARCH/GARCH/EGARCH)
        gm = arch_model(returns_clean, vol=self.vol_type, p=self.p, o=self.o, q=self.q, mean="Zero")
        
        # 2. Latih model HANYA sampai indeks split (Data Training)
        res = gm.fit(last_obs=self._split_idx, disp="off", show_warning=False)

        # 3. Terapkan parameter hasil training ke seluruh time-series
        res_filtered = gm.fix(res.params)

        # 4. Ambil prediksi out-of-sample (Data Testing) di skala harian
        daily_vol_test = np.array(res_filtered.conditional_volatility)[self._split_idx:]
        self._pred = daily_vol_test

        self._metrics = _metrics(self._y_test, self._pred)
        self._fitted  = True


# ── Sub-Class Model Ekonometrika ──────────────────────────────────────────────

class ARCHModel(ArchFamilyModel):
    def __init__(self):
        # ARCH(1) standar: Tidak memiliki parameter GARCH (q=0)
        super().__init__(name="ARCH", vol_type="ARCH", p=1, o=0, q=0)

class GARCHModel(ArchFamilyModel):
    def __init__(self):
        # GARCH(1,1) standar
        super().__init__(name="GARCH", vol_type="GARCH", p=1, o=0, q=1)

class EGARCHModel(ArchFamilyModel):
    def __init__(self):
        # EGARCH(1,1): Mampu menangkap efek asimetris (leverage effect)
        super().__init__(name="EGARCH", vol_type="EGARCH", p=1, o=1, q=1)

class GJRGARCHModel(ArchFamilyModel):
    def __init__(self):
        # GJR-GARCH(1,1): Menggunakan vol='GARCH' tapi dengan ambang batas asimetris (o=1)
        super().__init__(name="GJR-GARCH", vol_type="GARCH", p=1, o=1, q=1)


class EWMAModel(VolatilityModel):
    """
    Model Exponentially Weighted Moving Average (RiskMetrics).
    Tidak menggunakan library 'arch', melainkan kalkulasi Pandas murni.
    """
    name = "EWMA"

    def fit(self, df: pd.DataFrame) -> None:
        self._split(df)
        returns_pct = df["Return"] * 100
        sq_returns = returns_pct ** 2

        # Standar RiskMetrics menggunakan lambda = 0.94 (Alpha = 1 - 0.94 = 0.06)
        # ewm(mean) pada kuadrat return menghasilkan estimasi EWMA Variance
        ewma_var = sq_returns.ewm(alpha=0.06, adjust=False).mean()
        ewma_vol = np.sqrt(ewma_var)

        self._pred = ewma_vol.values[self._split_idx:]
        
        # Isi NaN jika ada dengan nilai rata-rata array untuk mencegah error metrik
        self._pred = np.nan_to_num(self._pred, nan=np.nanmean(self._pred))

        self._metrics = _metrics(self._y_test, self._pred)
        self._fitted  = True


# Registry untuk mempermudah pemanggilan model di Streamlit
MODEL_REGISTRY: dict[str, type[VolatilityModel]] = {
    "ARCH":      ARCHModel,
    "GARCH":     GARCHModel,
    "EGARCH":    EGARCHModel,
    "GJR-GARCH": GJRGARCHModel,
    "EWMA":      EWMAModel,
}

class ModelRunner:
    def __init__(self, df: pd.DataFrame, model_names: list[str]):
        self.df = df
        self.model_names = model_names

    def run(self) -> dict[str, VolatilityModel]:
        trained_models = {}
        for name in self.model_names:
            if cls := MODEL_REGISTRY.get(name):
                model = cls()
                model.fit(self.df)
                trained_models[name] = model
        return trained_models

@st.cache_resource(show_spinner=False)
def fit_models(df: pd.DataFrame, model_names: tuple[str, ...]) -> dict[str, VolatilityModel]:
    """Caching model agar tidak melatih ulang saat berinteraksi di Dashboard UI."""
    runner = ModelRunner(df, list(model_names))
    return runner.run()