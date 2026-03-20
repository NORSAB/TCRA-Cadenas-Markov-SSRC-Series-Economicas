"""
Configuración del pipeline TCROC-NNLS (Capítulo 4).
Umbrales fijos, K=4, W=2, lambda=1.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "TCROC-Markov_Original", "Combustibles.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Hiperparámetros fijos (Cap 4)
W = 2
LAMBDA = 1.0
K = 4
SEED = 42

# Umbrales fijos para combustibles (Sec 4.X, Ec. de asignación)
# s1: caída (α < -0.01)
# s2: estable (-0.01 ≤ α ≤ 0.02)
# s3: subida (0.02 < α ≤ 0.04)
# s4: alza fuerte (α > 0.04)
THRESHOLDS = [-0.01, 0.02, 0.04]
STATE_NAMES = ["Caída", "Estable", "Subida", "Alza fuerte"]

FUEL_ORDER = ["Super", "Regular", "Diesel", "Kerosene"]

# Particiones para validación predictiva
TRAIN_RATIOS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
