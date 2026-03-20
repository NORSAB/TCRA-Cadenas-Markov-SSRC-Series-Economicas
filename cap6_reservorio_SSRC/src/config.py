"""
===================================================================
TCROC-SSRC: Configuración Central
===================================================================
Configuración de hiperparámetros, rutas y constantes para el 
Computador de Reservorio Estructurado Estocástico (SSRC).

Hereda la estructura de datos de cap5_deteccion_pronostico.
===================================================================
"""
import numpy as np
import os

# --- Reproducibilidad ---
SEED = 42
np.random.seed(SEED)

# --- Rutas de datos (compartidas con TCROC-Markov) ---
MARKOV_PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'cap5_deteccion_pronostico')
)
CSV_PATH = os.path.join(MARKOV_PROJECT_DIR, 'data', 'silver', 'fuel_data_standardized.csv')
BEST_HYPERPARAMS_PATH = os.path.join(
    MARKOV_PROJECT_DIR, 'outputs', 'tables', '22_best_hyperparameters.csv'
)

# --- Combustibles ---
FUEL_COLUMNS = ['Regular', 'Super', 'Diesel', 'Kerosene']

# --- Hiperparámetros óptimos TCROC (del Capítulo anterior) ---
TCROC_OPTIMAL = {
    'Regular':  {'W': 50, 'lam': 0.99, 'K': 3},
    'Super':    {'W': 52, 'lam': 0.99, 'K': 4},
    'Diesel':   {'W': 50, 'lam': 0.97, 'K': 3},
    'Kerosene': {'W': 52, 'lam': 0.98, 'K': 5},
}

# RMSE del modelo TCROC-Markov (Tabla 2 del capítulo anterior)
MARKOV_RMSE = {
    'Regular':  0.8394,
    'Super':    0.9769,
    'Diesel':   1.0816,
    'Kerosene': 1.1714,
}

# --- Hiperparámetros del Reservorio (grilla FINA de búsqueda) ---
RESERVOIR_D_VALUES = [20, 30, 40, 50, 60, 75, 100, 150]  # Dimensiones D (8 niveles, granular)
RESERVOIR_RHO_VALUES = [0.70, 0.80, 0.85, 0.90, 0.95, 0.99]  # Radios espectrales (6 niveles)
RESERVOIR_LEAK_RATES = [0.1, 0.3, 0.5, 0.7, 1.0]     # Factor de fuga 'a' (1.0 = ESN clásico)
RESERVOIR_INPUT_SCALE = 1.0                  # omega para W_in
RESERVOIR_SPARSITY = 0.9                     # 90% dispersa
RESERVOIR_WASHOUT = 50                       # Pasos de washout
N_GRID_REALIZATIONS = 5                      # Realizaciones RÁPIDAS para ranking en grilla
N_REALIZATIONS = 30                          # Realizaciones para ensemble final (DM-Test)

# --- Validación ---
TRAIN_SIZE = 312  # Primeras 312 semanas (ene 2017 - dic 2022)

# --- Rutas de salida ---
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

# --- Visualización (Paleta Nord) ---
FUEL_COLORS = {
    'Super':     '#81A1C1',   # Frost 3
    'Regular':   '#5E81AC',   # Frost Blue
    'Diesel':    '#EBCB8B',   # Aurora Yellow
    'Kerosene':  '#D08770',   # Aurora Orange
}

# Colores de régimen Nord (K≤5)
REGIME_COLORS_NORD = [
    '#BF616A',   # Aurora Red — caída
    '#A3BE8C',   # Aurora Green — estable
    '#EBCB8B',   # Aurora Yellow — moderada
    '#5E81AC',   # Frost Blue — fuerte
    '#B48EAD',   # Aurora Purple — extra
]

DPI_QUALITY = 600
