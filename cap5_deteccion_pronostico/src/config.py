import numpy as np
import os

# --- Configuración de Semillas y Reproducebilidad ---
# --- Seed Configuration and Reproducibility ---
SEED = 42
np.random.seed(SEED)

# --- Hiperparámetros de Modelo ---
# --- Model Hyperparameters ---
DEFAULT_W = 2                # Tamaño de ventana para alphas / Window size for alphas
DEFAULT_LAMBDA = 0.95        # Factor de decaimiento / Decay factor
DEFAULT_K = 4                # Número de regímenes por defecto / Default number of regimes
K_RANGE_SEARCH = range(2, 10) # Rango para búsqueda de K / Range for K search

# --- Configuración de Datos ---
# --- Data Configuration ---
CSV_PATH = "data/bronze/Combustibles.csv"
REQUIRED_COLUMNS = ['Super', 'Regular', 'Diesel', 'Kerosene']

# --- Configuración de Salida ---
# --- Output Configuration ---
OUTPUT_DIR = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# --- Estilos de Visualización ---
# --- Visualization Styles ---
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
COLOR_PALETTE = "colorblind"
DPI_QUALITY = 600

# Paleta Nord para tesis
# Nord Palette for thesis
FUEL_COLORS = {
    'Super':    '#81A1C1',   # Frost 3
    'Regular':  '#5E81AC',   # Frost Blue
    'Diesel':   '#EBCB8B',   # Aurora Yellow
    'Kerosene': '#D08770'    # Aurora Orange
}

# Colores de régimen Nord (K≤5)
# Nord regime colors (K≤5)
REGIME_COLORS_NORD = [
    '#BF616A',   # Aurora Red — caída
    '#A3BE8C',   # Aurora Green — estable
    '#EBCB8B',   # Aurora Yellow — moderada
    '#5E81AC',   # Frost Blue — fuerte
    '#B48EAD',   # Aurora Purple — extra
]

# --- Umbrales de Evaluación ---
# --- Evaluation Thresholds ---
ACCURACY_THRESHOLD = 0.95
