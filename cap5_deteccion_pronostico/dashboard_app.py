import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import subprocess
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(
    page_title="TCROC-Markov Visualization Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Root directory logic
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(ROOT_DIR, "outputs", "figures")
TABLES_DIR = os.path.join(ROOT_DIR, "outputs", "tables")
PIPELINES_DIR = os.path.join(ROOT_DIR, "pipelines")

# Brand Colors (from src/config.py)
FUEL_COLORS = {
    'Super': '#374649',
    'Regular': '#982C33',
    'Diesel': '#3F6C3E',
    'Kerosene': '#134966'
}

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .status-card {
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ddd;
        background: white;
        margin-bottom: 10px;
    }
    .status-done { border-left-color: #28a745; }
    .status-running { border-left-color: #ffc107; }
    .status-pending { border-left-color: #6c757d; }
    
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #1e3d59;
        margin-top: 30px;
        margin-bottom: 20px;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def load_img(name):
    path = os.path.join(FIGURES_DIR, name)
    if os.path.exists(path):
        return Image.open(path)
    return None

def load_data_file(name):
    # Try name as is, then try replacing extensions
    base_name = name.split(".")[0]
    csv_path = os.path.join(TABLES_DIR, f"{base_name}.csv")
    xlsx_path = os.path.join(TABLES_DIR, f"{base_name}.xlsx")
    
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    elif os.path.exists(xlsx_path):
        return pd.read_excel(xlsx_path)
    return None

# --- SIDEBAR ---
st.sidebar.title("🚀 Control Panel")
st.sidebar.markdown("---")
run_pipeline = st.sidebar.button("Run Full Pipeline", use_container_width=True)

st.sidebar.markdown("### 🔍 External Figures")
show_extra = st.sidebar.checkbox("Show Hierarchical & Surface Plots", value=True)

st.sidebar.markdown("---")

# --- MAIN PAGE ---
st.title("📈 TCROC-Markov Analysis Dashboard")
st.markdown("Automated Markov Chain Modeling for Fuel Price Dynamics")

# 1. EXECUTION GRID
st.markdown("<div class='section-header'>⚡ Execution Status & Progress</div>", unsafe_allow_html=True)

pipeline_steps = [
    {"id": "01", "name": "Ingestion", "script": "01_ingestion.py", "desc": "ETL Bronze to Silver"},
    {"id": "02", "name": "Processing", "script": "02_processing.py", "desc": "Alpha Calculations"},
    {"id": "grid", "name": "Grid Search", "script": "grid_search.py", "desc": "Hyperparameter Opt."},
    {"id": "03", "name": "Modeling", "script": "03_modeling.py", "desc": "Matrix Estimation"},
    {"id": "04", "name": "Visualization", "script": "04_visualization.py", "desc": "Figure Generation"}
]

# Inicializar estado de sesion
if 'step_status' not in st.session_state:
    st.session_state.step_status = {step['name']: "pending" for step in pipeline_steps}
    st.session_state.progress = 0

cols = st.columns(len(pipeline_steps))

for i, step in enumerate(pipeline_steps):
    status = st.session_state.step_status[step['name']]
    with cols[i]:
        color = "#28a745" if status == "done" else "#ffc107" if status == "running" else "#6c757d"
        st.markdown(f"""
            <div style="padding:15px; border-radius:10px; background:white; border-top: 4px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height:120px;">
                <b style="color:#555">{step['id']}</b><br/>
                <b style="font-size:16px;">{step['name']}</b><br/>
                <small style="color:gray;">{step['desc']}</small>
            </div>
        """, unsafe_allow_html=True)

# RUN LOGIC
if run_pipeline:
    st.session_state.progress = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Configurar PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(ROOT_DIR, "src") + os.pathsep + env.get("PYTHONPATH", "")

    for i, step in enumerate(pipeline_steps):
        st.session_state.step_status[step['name']] = "running"
        st.rerun() if 'rerun' in dir(st) else None
        
        status_text.text(f"Running Step {i+1}: {step['name']}...")
        
        script_path = os.path.join(PIPELINES_DIR, step['script'])
        try:
            # Run in shell to handle potential cross-core imports
            result = subprocess.run(["python", script_path], env=env, capture_output=True, text=True)
            if result.returncode == 0:
                st.session_state.step_status[step['name']] = "done"
            else:
                st.session_state.step_status[step['name']] = "error"
                st.error(f"Error in {step['name']}: {result.stderr}")
                break
        except Exception as e:
            st.session_state.step_status[step['name']] = "error"
            st.error(f"Failed to execute {step['name']}: {e}")
            break
            
        st.session_state.progress = (i + 1) / len(pipeline_steps)
        progress_bar.progress(st.session_state.progress)
        
    status_text.text("✅ Full Pipeline Completed!")
    st.balloons()

# 2. OPTIMIZATION METRICS
st.markdown("<div class='section-header'>🔬 Optimization & Grid Search Results</div>", unsafe_allow_html=True)

opt_col1, opt_col2 = st.columns([1, 2])

with opt_col1:
    best_params = load_data_file("22_best_hyperparameters.xlsx")
    if best_params is not None:
        st.markdown("### 🏆 Optimal Parameters")
        st.dataframe(best_params, use_container_width=True)
    else:
        st.warning("Hyperparameter table not found.")

with opt_col2:
    tab_aic, tab_acc, tab_rmse, tab_bubble, tab_surface = st.tabs(["AIC", "Accuracy", "RMSE", "Bubble Chart", "Surface Opt."])
    with tab_aic:
        img = load_img("1_aic_plot.png")
        if img: st.image(img, use_container_width=True)
    with tab_acc:
        img = load_img("2_accuracy_plot.png")
        if img: st.image(img, use_container_width=True)
    with tab_rmse:
        img = load_img("3_rmse_plot.png")
        if img: st.image(img, use_container_width=True)
    with tab_bubble:
        img = load_img("4_bubble_chart.png")
        if img: st.image(img, use_container_width=True)
    with tab_surface:
        # Cargar desde raiz (figura principal)
        surf_path = os.path.join(ROOT_DIR, "Fig_Optimization_Surface_ENG.png")
        if os.path.exists(surf_path):
            st.image(Image.open(surf_path), use_container_width=True, caption="Hyperparameter Surface Analysis")

# 3. MARKOV TRANSITION ANALYSIS
st.markdown("<div class='section-header'>🕸️ Markov Transition Logic</div>", unsafe_allow_html=True)

tab_graphs, tab_heatmaps, tab_alphas, tab_hierarchical = st.tabs(["Transition Graphs", "Probability Heatmaps", "Alpha Distributions", "Hierarchical Matrix"])

with tab_graphs:
    st.markdown("### Transition Graphs (K-Means)")
    img_panel = load_img("6_final_transition_panel.png")
    if img_panel:
        st.image(img_panel, use_container_width=True)
    else:
        g_cols = st.columns(4)
        for i, fuel in enumerate(FUEL_COLORS.keys()):
            img = load_img(f"graph_kmeans_{fuel.lower()}.png")
            if img: g_cols[i].image(img, caption=fuel)

with tab_heatmaps:
    img_heat = load_img("19_transition_probability_heatmaps.png")
    if img_heat: st.image(img_heat, use_container_width=True)

with tab_alphas:
    img_alpha = load_img("25_alpha_distributions_optimal.png")
    if img_alpha: st.image(img_alpha, use_container_width=True)

with tab_hierarchical:
    hier_path = os.path.join(ROOT_DIR, "Final_P_Matrix_Hierarchical_Full.png")
    if os.path.exists(hier_path):
        st.image(Image.open(hier_path), use_container_width=True, caption="Hierarchical Transition Matrix Heatmap")

# 4. REGIME DYNAMICS & TRENDS
st.markdown("<div class='section-header'>📉 Price Trends & Regime Shading</div>", unsafe_allow_html=True)

fuel_selector = st.selectbox("Select Fuel to Inspect Regimes:", list(FUEL_COLORS.keys()))

col_reg1, col_reg2 = st.columns(2)

with col_reg1:
    st.markdown(f"**Regime Scatters: {fuel_selector}**")
    # Note: Using the k-means plot by default
    img = load_img("8_kmeans_final_regimes_plot.png")
    if img: st.image(img, use_container_width=True)

with col_reg2:
    st.markdown(f"**Price Series Clustering: {fuel_selector}**")
    img = load_img("15_kmeans_final_price_regimes.png")
    if img: st.image(img, use_container_width=True)

# 5. FORECAST & STATISTICS
st.markdown("<div class='section-header'>📑 Statistical Validation & Forecast</div>", unsafe_allow_html=True)

col_stat1, col_stat2 = st.columns(2)

with col_stat1:
    st.markdown("### 📊 Next Week Forecast")
    forecast = load_data_file("16_next_week_forecast.csv") # Assuming CSV/XLSX conversion happened
    if forecast is not None:
        st.dataframe(forecast, use_container_width=True)
    else:
        st.info("Run Step 04 to generate forecast table.")

with col_stat2:
    st.markdown("### 🧬 Sensitivity Analysis (W)")
    img_w = load_img("W_sensitivity_analysis_final_v6.png")
    if img_w: st.image(img_w, use_container_width=True)

# 6. DETAILED AUDIT & LOGS
st.markdown("<div class='section-header'>🔬 Detailed Model Audit</div>", unsafe_allow_html=True)

audit_tab1, audit_tab2, audit_tab3 = st.tabs(["K-Means Metrics", "Quantile Metrics", "Discretization Audit"])

with audit_tab1:
    metrics_km = load_data_file("10_predictive_summary_kmeans_consistent.xlsx")
    if metrics_km is not None:
        st.dataframe(metrics_km, use_container_width=True)

with audit_tab2:
    metrics_q = load_data_file("13_predictive_summary_quantiles.xlsx")
    if metrics_q is not None:
        st.dataframe(metrics_q, use_container_width=True)

with audit_tab3:
    fuel_audit = st.selectbox("Select Fuel Audit Log:", list(FUEL_COLORS.keys()))
    df_audit = load_data_file(f"{fuel_audit}_discretization_audit.csv")
    if df_audit is not None:
        st.dataframe(df_audit.head(100), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built by **Antigravity AI** for Norman Sabillon | © 2026")
