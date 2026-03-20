import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
from PIL import Image
from pathlib import Path
import plotly.express as px

# --- CONFIG ---
st.set_page_config(page_title="Familias-TCROC MLOps Dashboard", layout="wide", page_icon="📈")
ROOT_DIR = Path("D:/2026/Tesis2026/Familias-TCROC")
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"
TABLES_DIR = ROOT_DIR / "outputs" / "tables"

# --- CSS ---
st.markdown("""
<style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .header { font-size: 32px; font-weight: bold; color: #1c3d5a; margin-bottom: 5px; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
    .subheader { font-size: 20px; font-weight: bold; color: #34495e; margin-top: 20px; margin-bottom: 10px; }
    .sidebar-title { color: #2e86c1; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.markdown("<h1 class='sidebar-title'>🚀 Familias-TCROC</h1>", unsafe_allow_html=True)
page = st.sidebar.radio("Navegación", ["📡 Monitor de Pipeline", "🔬 Optimización & Métricas", "📉 Análisis Individual"])

# Helper functions
def load_img(name):
    path = FIGURES_DIR / name
    if path.exists():
        return Image.open(str(path))
    return None

# --- PAGE 1: MONITOR ---
if page == "📡 Monitor de Pipeline":
    st.markdown("<div class='header'>Monitor de Ejecución In-Real-Time</div>", unsafe_allow_html=True)
    st.markdown("Control centralizado de las fases de ingestión y optimización")
    
    col_run, col_status = st.columns([1, 2])
    
    with col_run:
        st.markdown("### ⚙️ Control Central")
        if st.button("Ejecutar Pipeline Completo", use_container_width=True):
            with st.spinner("Procesando Familias-TCROC (esto tardará unos segundos)..."):
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT_DIR)
                result = subprocess.run(["python", str(ROOT_DIR / "run_all.py")], env=env, capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("¡Pipeline completado con éxito!")
                    st.balloons()
                else:
                    st.error(f"Error: {result.stderr}")
        
        st.info("Fases: Bronze → Silver → Grid Search → Gold → Visualization")

    with col_status:
        st.markdown("### 📊 Estado de Datos")
        for ds in ["combustibles", "pib"]:
            p = TABLES_DIR / f"grid_search_best_{ds}.csv"
            status = "✅ Completado" if p.exists() else "⏳ Pendiente"
            st.write(f"**Dataset {ds.upper()}:** {status}")

    st.markdown("---")
    st.markdown("### 📑 Vista Previa de Datos Crudos (Bronze/Silver)")
    ds_sel = st.selectbox("Seleccionar Dataset:", ["pib", "combustibles"])
    df_prev = pd.read_csv(ROOT_DIR / "data" / "silver" / f"{ds_sel}_clean.csv").head(10)
    st.table(df_prev)

# --- PAGE 2: OPTIMIZACIÓN ---
elif page == "🔬 Optimización & Métricas":
    st.markdown("<div class='header'>Evaluación de Modelos & Optimización</div>", unsafe_allow_html=True)
    st.markdown("Análisis de las superficies de error y selección de parámetros óptimos")
    
    st.markdown("### 🏆 Mejores Parámetros (Resumen)")
    tabs_best = st.tabs(["🌎 PIB Honduras", "⛽ Combustibles"])
    
    for i, ds in enumerate(["pib", "combustibles"]):
        with tabs_best[i]:
            path = TABLES_DIR / f"grid_search_best_{ds}.csv"
            if path.exists():
                df = pd.read_csv(path)
                st.dataframe(df.style.highlight_min(subset=['RMSE'], color='#d5f5e3'), use_container_width=True)
            else:
                st.warning("Ejecuta el Grid Search para ver los mejores resultados.")

    st.markdown("---")
    st.markdown("### 🌡️ Heatmaps de Superficie de Error (W vs Lambda)")
    st.markdown("Este mapa permite verificar la estabilidad de los parámetros y evitar el sobreajuste (overfitting).")
    
    ds_heat = st.selectbox("Seleccionar Dataset para Heatmaps:", ["combustibles", "pib"])
    dataset_cols = ["Super", "Regular", "Diesel", "Kerosene"] if ds_heat == "combustibles" else [
        "PIB_en_Dólares_Corrientes_Millones_de_USD", 
        "Tasa_de_Crecimiento_Anual_del_PIB_",
        "PIB_per_cápita_en_Dólares_Corrientes_USD",
        "PIB_per_cápita_en_Lempiras_a_Precios_Constantes"
    ]
    
    sel_series = st.selectbox("Seleccionar Serie:", dataset_cols)
    sel_metric = st.selectbox("Métrica de Error:", ["RMSE", "MAE", "MAPE", "R2"])
    
    img_heat = load_img(f"heatmap_{ds_heat}_{sel_series}_{sel_metric}.png")
    if img_heat:
        st.image(img_heat, use_container_width=True, caption=f"Superficie de Error {sel_metric} para {sel_series}")
    else:
        st.info("Imagen de heatmap no disponible. Asegúrate de ejecutar la fase de visualización.")

# --- PAGE 3: ANÁLISIS INDIVIDUAL ---
elif page == "📉 Análisis Individual":
    st.markdown("<div class='header'>Gráficos Finales de Ajuste</div>", unsafe_allow_html=True)
    st.markdown("Comparación detallada entre la tendencia original y el ajuste optimizado")
    
    tabs_final = st.tabs(["🌎 Grid PIB", "⛽ Grid Combustibles", "🔍 Inspección Detallada"])
    
    with tabs_final[0]:
        img = load_img("grid_pib_ajustado.png")
        if img: st.image(img, use_container_width=True, caption="Grid Completo PIB")
    
    with tabs_final[1]:
        img = load_img("grid_combustibles_ajustado.png")
        if img: st.image(img, use_container_width=True, caption="Grid Completo Combustibles")

    with tabs_final[2]:
        all_best = list(FIGURES_DIR.glob("final_best_*.png"))
        if all_best:
            sel_img = st.selectbox("Seleccionar Gráfico de Serie:", [f.name for f in all_best])
            st.image(load_img(sel_img), use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(f"**Métricas utilizadas:**\n- **RMSE**: Root Mean Square Error\n- **MAE**: Mean Absolute Error\n- **MedAE**: Median Absolute Error\n- **MSLE**: Mean Sq Log Error\n- **MAPE**: Mean Abs Percentage Error\n- **R2**: Coefficient of Determination")
st.sidebar.markdown("Hecho por **Antigravity AI**")
