"""
ui.py
---------
Antarmuka pengguna (UI) Streamlit untuk Volatility Dashboard Ekonometrika.
Menampilkan Candlestick Chart interaktif dan Komparasi Lintas Emiten.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from data_processor import DataProcessor, ALLOWED_STOCKS
from models import VolatilityModel, fit_models 

C_BG       = "#0E1117"
C_SURFACE  = "#1A1C23"
C_RED      = "#FF4B4B"
C_ORANGE   = "#FFAA00"
C_LIME     = "#00FF87"
C_LIGHT    = "#F0F2F6"
C_PURPLE   = "#9b59b6"
C_BLUE     = "#3498db"
C_BORDER   = "#2D303E"

MODEL_COLORS = {
    "Actual Volatility": C_LIGHT,
    "ARCH":              C_PURPLE, 
    "GARCH":             C_RED,     
    "EGARCH":            C_LIME,    
    "GJR-GARCH":         C_ORANGE,  
    "EWMA":              C_BLUE, 
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C_LIGHT, family="'IBM Plex Mono', monospace", size=12),
    xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, showgrid=True),
    yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, showgrid=True),
    legend=dict(bgcolor="rgba(26,28,35,0.8)", bordercolor=C_BORDER, borderwidth=1, orientation="v", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
)

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@300;500;700&display=swap');
html, body, [class*="css"] {{ background-color: {C_BG}; color: {C_LIGHT}; font-family: 'Space Grotesk', sans-serif; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; }}
h1 {{ color: {C_LIME} !important; font-weight: 700; text-shadow: 0 0 20px rgba(0, 255, 135, 0.2); }}
h2 {{ color: {C_ORANGE} !important; font-size: 1.2rem !important; text-transform: uppercase; letter-spacing: 0.1em; }}
section[data-testid="stSidebar"] {{ background-color: {C_SURFACE} !important; border-right: 1px solid {C_BORDER}; }}
div[data-testid="metric-container"] {{ background: linear-gradient(145deg, #1e2028, #16181e); border: 1px solid {C_BORDER}; border-radius: 12px; padding: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: transform 0.2s ease, box-shadow 0.2s ease; }}
div[data-testid="metric-container"]:hover {{ transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 255, 135, 0.1); border-color: rgba(0, 255, 135, 0.3); }}
div[data-testid="metric-container"] label {{ color: #A0AEC0 !important; font-family: 'IBM Plex Mono', monospace; font-size: 12px !important; letter-spacing: 0.05em; }}
div[data-testid="metric-container"] [data-testid="metric-value"] {{ color: {C_LIGHT} !important; font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem !important; font-weight: 600; }}
[data-testid="stDataFrame"] {{ border: 1px solid {C_BORDER}; border-radius: 8px; overflow: hidden; }}
input[type="number"], .stSelectbox > div > div {{ background: #13151A !important; color: {C_LIGHT} !important; border: 1px solid {C_BORDER} !important; border-radius: 6px; }}
.stMultiSelect span[data-baseweb="tag"] {{ background-color: {C_RED} !important; border-radius: 4px; }}
hr {{ border-color: {C_BORDER} !important; margin: 2rem 0; }}
</style>
"""

class DashboardUI:
    def __init__(self):
        st.set_page_config(page_title="Volatility Intelligence", layout="wide", initial_sidebar_state="expanded")
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        self._setup_sidebar()

    def _setup_sidebar(self):
        sb = st.sidebar
        sb.markdown(f"<h3 style='color:{C_LIME}; text-align:center;'>IDX VOLATILITY</h3><hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
        
        self.menu = sb.radio("Navigasi", ["Analisis Tunggal", "Bandingkan Saham"], label_visibility="collapsed")
        sb.markdown("---")
        
        self.selected_stock = sb.selectbox("Pilih Saham (Emiten)", ALLOWED_STOCKS)
        self.selected_models = sb.multiselect("Pilih Model Peramalan", ["ARCH", "GARCH", "EGARCH", "GJR-GARCH", "EWMA"], default=["GARCH", "EGARCH", "EWMA"])
        self.show_actual = sb.checkbox("Tampilkan Volatilitas Aktual", value=True)
        
        sb.markdown("---")
        self.time_filter = sb.selectbox("Rentang Waktu", ["Semua Waktu", "1 Bulan Terakhir", "1 Tahun Terakhir", "Custom Range"], index=2)
        
        self.custom_start, self.custom_end = None, None
        if self.time_filter == "Custom Range":
            self.custom_start = pd.Timestamp(sb.date_input("Mulai", pd.Timestamp("2020-01-01")))
            self.custom_end   = pd.Timestamp(sb.date_input("Akhir", pd.Timestamp("2026-02-28")))

    def _get_filtered_data(self, stock: str) -> pd.DataFrame:
        df = DataProcessor(stock).load()
        return DataProcessor.apply_time_filter(df, self.time_filter, self.custom_start, self.custom_end)

    def _get_trained_models(self, stock: str, df: pd.DataFrame) -> dict[str, VolatilityModel]:
        return fit_models(df, tuple(self.selected_models))

    @staticmethod
    def _create_fig() -> go.Figure:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT)
        return fig

    def render(self):
        st.title("Monitor Risiko & Ekuitas IDX")
        st.markdown("<p style='color:#A0AEC0; margin-top:-15px;'>Sistem Peramalan Volatilitas Ekonometrika (Keluarga ARCH)</p>", unsafe_allow_html=True)
        
        if self.menu == "Analisis Tunggal": self._page_single()
        elif self.menu == "Bandingkan Saham": self._page_compare()

    def _page_single(self):
        stock = self.selected_stock
        df    = self._get_filtered_data(stock)
        
        st.markdown(f"## Kinerja Saham: {stock}")
        cols = st.columns(4)
        last_price = df["Terakhir"]
        metrics = [
            ("Harga Terakhir", last_price.iloc[-1]),
            ("Harga Rata-rata", last_price.mean()),
            ("Tertinggi (Periode)", last_price.max()),
            ("Terendah (Periode)", last_price.min())
        ]
        for col, (label, val) in zip(cols, metrics):
            col.metric(label, f"Rp {val:,.0f}")

        st.markdown("---")
        
        fig1 = self._create_fig()
        fig1.add_trace(go.Candlestick(
            x=df['Tanggal'], open=df['Pembukaan'], high=df['Tertinggi'], low=df['Terendah'], close=df['Terakhir'],
            increasing_line_color=C_LIME, decreasing_line_color=C_RED, name="OHLC"
        ))
        fig1.update_layout(title="Pergerakan Harga (Candlestick)", yaxis_title="IDR", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = self._create_fig()
        fig2.add_trace(go.Scatter(x=df["Tanggal"], y=df["Target_Volatility"], name="Parkinson Volatility", mode="lines", line=dict(color=C_LIGHT, width=1.5)))
        fig2.update_layout(title="Volatilitas Aktual (Estimasi Parkinson)", yaxis_title="%")
        st.plotly_chart(fig2, use_container_width=True)

        if not self.selected_models: 
            return st.warning("Silakan pilih minimal satu model di menu samping.")

        st.markdown("### Prediksi Volatilitas (Out-of-Sample Test Set)")
        models_dict = self._get_trained_models(stock, df)
        
        fig_pred = self._create_fig()
        first_model = next(iter(models_dict.values()), None)
        
        if first_model and self.show_actual:
            fig_pred.add_trace(go.Scatter(x=first_model.test_dates, y=first_model.y_test, name="Aktual (Parkinson)", line=dict(color=C_LIGHT, dash="dot", width=1), opacity=0.6))

        for mname, model in models_dict.items():
            color = MODEL_COLORS.get(mname, C_LIGHT)
            fig_pred.add_trace(go.Scatter(x=model.test_dates, y=model.predict_test(), name=mname, line=dict(color=color, width=2)))
            
        fig_pred.update_layout(height=450, yaxis_title="Volatilitas Harian (%)")
        st.plotly_chart(fig_pred, use_container_width=True)

        self._render_metrics_table(models_dict)

    def _page_compare(self):
        st.markdown("## Perbandingan Lintas Emiten")
        compare_stocks = st.multiselect("Pilih Emiten untuk Dibandingkan", ALLOWED_STOCKS, default=["BBCA", "BBRI", "TLKM"])
        
        if not compare_stocks: return st.info("Pilih minimal satu saham.")
        if not self.selected_models: return st.warning("Pilih minimal satu model.")

        all_results = {}
        with st.spinner("Melatih model komparasi..."):
            for stock in compare_stocks:
                df = self._get_filtered_data(stock)
                all_results[stock] = self._get_trained_models(stock, df)

        # ── 1. Overlay Volatilitas Antar Saham ──
        st.markdown("### Overlay Volatilitas Antar Saham")
        st.caption("Membandingkan pola prediksi lintas emiten berdasarkan model yang sama.")
        
        # Palet warna khusus untuk membedakan emiten di dalam satu grafik
        stock_colors = [C_LIME, C_ORANGE, "#7ECFE0", "#F4A261", "#E76F51", "#2A9D8F", "#E9C46A", "#264653", C_RED, C_LIGHT]
        
        model_tabs = st.tabs(self.selected_models)
        for tab, mname in zip(model_tabs, self.selected_models):
            with tab:
                fig_over = self._create_fig()
                for j, stock in enumerate(compare_stocks):
                    model_obj = all_results[stock].get(mname)
                    if model_obj is None: continue
                    
                    s_color = stock_colors[j % len(stock_colors)]
                    
                    if self.show_actual:
                        fig_over.add_trace(go.Scatter(
                            x=model_obj.test_dates, y=model_obj.y_test, 
                            name=f"{stock} Aktual", line=dict(color=s_color, dash="dot", width=1), opacity=0.4
                        ))
                        
                    fig_over.add_trace(go.Scatter(
                        x=model_obj.test_dates, y=model_obj.predict_test(), 
                        name=f"{stock} Prediksi", line=dict(color=s_color, width=2)
                    ))
                    
                fig_over.update_layout(height=450, yaxis_title="Volatilitas (%)", title=f"Prediksi {mname} Lintas Emiten")
                st.plotly_chart(fig_over, use_container_width=True)

        st.markdown("---")

        # ── 2. Komparasi Bar Chart & Tabel Evaluasi ──
        st.markdown("### Komparasi Metrik Evaluasi")
        rows = [{"Saham": s, "Model": m, "RMSE": res[m].metrics["RMSE"], "MAE": res[m].metrics["MAE"]} 
                for s, res in all_results.items() for m in res]
        
        if rows:
            eval_df = pd.DataFrame(rows)
            
            # Membagi Bar Chart menjadi 2 kolom (RMSE dan MAE)
            c1, c2 = st.columns(2)
            with c1:
                fig_rmse = px.bar(eval_df, x="Saham", y="RMSE", color="Model", barmode="group", text_auto=".3f", color_discrete_map=MODEL_COLORS)
                fig_rmse.update_layout(**PLOTLY_LAYOUT, bargap=0.2, title="RMSE (Lebih Rendah Lebih Baik)")
                fig_rmse.update_traces(textposition="outside")
                st.plotly_chart(fig_rmse, use_container_width=True)
            with c2:
                fig_mae = px.bar(eval_df, x="Saham", y="MAE", color="Model", barmode="group", text_auto=".3f", color_discrete_map=MODEL_COLORS)
                fig_mae.update_layout(**PLOTLY_LAYOUT, bargap=0.2, title="MAE (Lebih Rendah Lebih Baik)")
                fig_mae.update_traces(textposition="outside")
                st.plotly_chart(fig_mae, use_container_width=True)

            st.markdown("#### Tabel Evaluasi Lengkap")
            # Membuat pivot tabel agar mudah dibaca silang
            pivot_df = eval_df.pivot(index="Saham", columns="Model", values=["RMSE", "MAE"])
            pivot_df.columns = [f"{m} {metric}" for metric, m in pivot_df.columns]
            
            # ── PERUBAHAN ──
            # 1. Kelompokkan nama kolom berdasarkan metriknya
            rmse_cols = [c for c in pivot_df.columns if "RMSE" in c]
            mae_cols  = [c for c in pivot_df.columns if "MAE" in c]

            # 2. Terapkan highlight_min dengan axis=1 (per baris/horizontal) untuk masing-masing kelompok metrik
            styled_df = (
                pivot_df.style
                .highlight_min(subset=rmse_cols, axis=1, color="#1a4731")
                .highlight_min(subset=mae_cols, axis=1, color="#1a4731")
                .format("{:.4f}")
            )
            
            st.dataframe(styled_df, use_container_width=True)


    @staticmethod
    def _render_metrics_table(models_dict: dict):
        rows = [{"Model": mname, "RMSE": m.metrics["RMSE"], "MAE": m.metrics["MAE"]} for mname, m in models_dict.items()]
        if rows:
            st.markdown("### Evaluasi Kinerja (Error Metrics)")
            df_eval = pd.DataFrame(rows).set_index("Model")
            st.dataframe(df_eval.style.highlight_min(color="#1a4731").format("{:.4f}"), use_container_width=True)

if __name__ == "__main__":
    app = DashboardUI()
    app.render()