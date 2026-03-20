import pandas as pd
from src.config import DATASETS

def load_bronze(dataset_name):
    config = DATASETS[dataset_name]
    df = pd.read_csv(config['raw_path'], sep=config['sep'])
    return df

def clean_data(df, dataset_name):
    config = DATASETS[dataset_name]
    
    # Procesar fechas
    if dataset_name == 'pib':
        # PIB column names might have spaces or weird characters from CSV
        # date_col is 'Fecha'
        df[config['date_col']] = pd.to_datetime(df[config['date_col']], format='%Y')
    else:
        df[config['date_col']] = pd.to_datetime(df[config['date_col']])
    
    # Sort and set index
    df = df.sort_values(config['date_col'])
    df = df.set_index(config['date_col'])
    
    # Asegurar tipo numerico
    for col in config['series_cols']:
        # PIB data might have large numbers, ensure they are floats
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    return df

def save_silver(df, dataset_name):
    config = DATASETS[dataset_name]
    df.to_csv(config['silver_path'])
    print(f"Saved silver data for {dataset_name} at {config['silver_path']}")
