import os
import sys
import pandas as pd

# Añadir ruta raíz para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.load_fuel_data import load_fuel_data
from src.config import CSV_PATH

def run_ingestion():
    """
    Pipeline de Ingestión: Mueve datos de Bronze a Silver.
    Ingestion Pipeline: Moves data from Bronze to Silver.
    """
    print("\n--- Pipeline 01: Ingestión (Bronze -> Silver) ---")
    
    # Cargar datos desde Bronze
    data, _ = load_fuel_data(CSV_PATH)
    
    if data is not None:
        # Guardar en Silver como CSV estandarizado
        silver_path = "data/silver/fuel_data_standardized.csv"
        os.makedirs("data/silver", exist_ok=True)
        data.to_csv(silver_path, index=False)
        print(f"Datos estandarizados guardados en Silver: {silver_path}")
        return silver_path
    
    return None

if __name__ == "__main__":
    run_ingestion()
