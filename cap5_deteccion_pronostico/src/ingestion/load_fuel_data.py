import pandas as pd
import numpy as np
import os
from src.config import REQUIRED_COLUMNS

def load_fuel_data(file_path):
    """
    Carga los datos de precios de combustibles desde un archivo CSV.
    Loads fuel price data from a specified CSV file.

    Args:
        file_path (str): Ruta al archivo CSV. / Path to the CSV file.

    Returns:
        tuple: (pd.DataFrame, dict) - Datos cargados y diccionario de series. / Loaded data and series dictionary.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Error: El archivo '{file_path}' no existe. / File '{file_path}' does not exist.")
            return None, None
            
        data = pd.read_csv(file_path)
        
        # Estandarizar nombre de columna de fecha
        # Standardize date column name
        if 'Fecha' in data.columns:
            data.rename(columns={'Fecha': 'Date'}, inplace=True)

        if not all(col in data.columns for col in REQUIRED_COLUMNS):
            print(f"Error: El CSV debe contener las columnas: {REQUIRED_COLUMNS}")
            return None, None

        series = {
            col: np.asarray(data[col].values, dtype=float)
            for col in REQUIRED_COLUMNS
        }
        
        return data, series

    except Exception as e:
        print(f"Ocurrió un error inesperado al cargar datos: {e}")
        return None, None
