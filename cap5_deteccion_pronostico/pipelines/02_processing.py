import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.functions.calculate_alphas import calculate_alphas
from src.config import DEFAULT_W, DEFAULT_LAMBDA, REQUIRED_COLUMNS

def run_processing():
    """
    Pipeline de Procesamiento: Mueve datos de Silver a Gold.
    Processing Pipeline: Moves data from Silver to Gold.
    """
    print("\n--- Pipeline 02: Procesamiento (Silver -> Gold) ---")
    
    silver_path = "data/silver/fuel_data_standardized.csv"
    if not os.path.exists(silver_path):
        print("Error: No se encontró el archivo en Silver.")
        return None
        
    data = pd.read_csv(silver_path)
    gold_data = {}
    
    for col in REQUIRED_COLUMNS:
        series = data[col].values
        alphas = calculate_alphas(series, W=DEFAULT_W, lambda_decay=DEFAULT_LAMBDA)
        gold_data[f"{col}_alpha"] = alphas
    
    # Nota: Las series de alphas son más cortas que el original por la ventana W
    # Guardar cada serie en Gold
    os.makedirs("data/gold", exist_ok=True)
    for name, series in gold_data.items():
        np.save(f"data/gold/{name}.npy", series)
        
    print(f"Características (Alphas) guardadas en Gold.")
    return gold_data

if __name__ == "__main__":
    run_processing()
