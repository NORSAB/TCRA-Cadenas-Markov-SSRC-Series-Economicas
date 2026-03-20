import os
from pathlib import Path

# Base directories
BASE_DIR = Path("D:/2026/Tesis2026/Familias-TCROC")
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Medallion Architecture
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Outputs
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Crear directorios si no existen
for d in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset Config
DATASETS = {
    "combustibles": {
        "raw_path": BRONZE_DIR / "Combustibles.csv",
        "silver_path": SILVER_DIR / "combustibles_clean.csv",
        "gold_path": GOLD_DIR / "combustibles_results.csv",
        "sep": ",",
        "date_col": "Fecha",
        "series_cols": ["Super", "Regular", "Diesel", "Kerosene"]
    },
    "pib": {
        "raw_path": BRONZE_DIR / "PIB.csv",
        "silver_path": SILVER_DIR / "pib_clean.csv",
        "gold_path": GOLD_DIR / "pib_results.csv",
        "sep": ";",
        "date_col": "Fecha",
        "series_cols": [
            "PIB en Dólares Corrientes (Millones de USD)", 
            "Tasa de Crecimiento Anual del PIB (%)",
            "PIB per cápita en Dólares Corrientes (USD)",
            "PIB per cápita en Lempiras a Precios Constantes"
        ]
    }
}

# TCROC Default Hyperparameters
DEFAULT_W = {
    "combustibles": 52,
    "pib": 5  # Smaller W for annual data
}
DEFAULT_LAMBDA = 0.99

# Visual Config
FUEL_COLORS = {
    'Super': '#2C3E50',
    'Regular': '#E74C3C',
    'Diesel': '#27AE60',
    'Kerosene': '#2980B9'
}

PIB_COLORS = {
    'PIB en Dólares Corrientes (Millones de USD)': '#1A5276',
    'Tasa de Crecimiento Anual del PIB (%)': '#943126',
    'PIB per cápita en Dólares Corrientes (USD)': '#0E6251',
    'PIB per cápita en Lempiras a Precios Constantes': '#7D6608'
}

SERIES_COLORS = {**FUEL_COLORS, **PIB_COLORS}
ORIGINAL_COLOR = '#424949' # Deep Slate Gray
