import os
import sys
import pandas as pd
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processing.functions.calculate_alphas import calculate_alphas
from src.processing.functions.discretize_series import discretize_series
from src.processing.functions.discretize_series_quantiles import discretize_series_quantiles
from src.models.functions.estimate_transition_matrix import estimate_transition_matrix
from src.evaluation.calculate_aic import calculate_aic
from src.models.functions.assign_regime_names import assign_regime_names
from src.config import REQUIRED_COLUMNS, MODELS_DIR, TABLES_DIR, CSV_PATH

from src.ingestion.load_fuel_data import load_fuel_data

def run_modeling():
    """
    Modeling Pipeline: Trains Markov models (K-Means with OPTIMAL k and Quantiles with fixed k=4 benchmark).
    """
    print("\n--- Pipeline 03: Modelado (Optimizados y Benchmark) ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    # Cargar datos crudos
    _, fuel_series = load_fuel_data(CSV_PATH)
    
    # Cargar hiperparametros optimos
    best_h_path = os.path.join(TABLES_DIR, "22_best_hyperparameters.csv")
    if not os.path.exists(best_h_path):
        print("Error: 22_best_hyperparameters.csv not found. Run grid_search first.")
        return

    best_df = pd.read_csv(best_h_path)
    summary_results = []
    
    for _, row in best_df.iterrows():
        fuel_name = row['Combustible']
        W_opt = int(row['W'])
        L_opt = float(row['Lambda'])
        k_opt = int(row['k'])
        
        series = fuel_series[fuel_name]
        
        # 1. Recalculate Alphas
        alphas = calculate_alphas(series, W=W_opt, lambda_decay=L_opt)
        np.save(f"data/gold/{fuel_name}_alpha.npy", alphas)
        
        # 2. K-Means (Optimal k)
        states_km, centroids = discretize_series(alphas, k=k_opt)
        regime_names_km = assign_regime_names(centroids)
        P_km, _ = estimate_transition_matrix(states_km, k=k_opt)
        aic_km = calculate_aic(states_km, k_opt, P_km)
        
        # 3. Quantiles (Fixed k=4 Benchmark as per Articulol.py FIG-07)
        k_q = 4 
        states_q, boundaries = discretize_series_quantiles(alphas, k=k_q)
        P_q, _ = estimate_transition_matrix(states_q, k=k_q)
        aic_q = calculate_aic(states_q, k_q, P_q)
        
        # Guardar modelos
        np.save(os.path.join(MODELS_DIR, f"{fuel_name}_P_kmeans.npy"), P_km)
        np.save(os.path.join(MODELS_DIR, f"{fuel_name}_centroids.npy"), centroids)
        np.save(os.path.join(MODELS_DIR, f"{fuel_name}_regime_names.npy"), np.array(regime_names_km))
        np.save(os.path.join(MODELS_DIR, f"{fuel_name}_P_quantiles.npy"), P_q)
        np.save(os.path.join(MODELS_DIR, f"{fuel_name}_boundaries.npy"), boundaries)
        
        # Guardar tabla de auditoria
        data_full, _ = load_fuel_data(CSV_PATH)
        audit_subset = data_full.iloc[W_opt-1:].copy()
        if len(audit_subset) > len(alphas):
            audit_subset = audit_subset.iloc[:len(alphas)]
            
        audit_subset['Alpha'] = alphas
        audit_subset['State_KMeans'] = states_km
        audit_subset['State_Quantiles'] = states_q
        
        table_path = os.path.join(TABLES_DIR, f"{fuel_name}_discretization_audit.csv")
        audit_subset[['Date', fuel_name, 'Alpha', 'State_KMeans', 'State_Quantiles']].to_csv(table_path, index=False)
        
        summary_results.append({
            'Combustible': fuel_name, 'W': W_opt, 'Lambda': L_opt, 'k_optimo': k_opt,
            'AIC_KMeans': aic_km, 'AIC_Cuantiles': aic_q
        })
        print(f"Modelos para {fuel_name} entrenados (K-Means k={k_opt} | Quantiles k=4).")
        
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(os.path.join(TABLES_DIR, "modeling_summary.csv"), index=False)
    
    return summary_results

if __name__ == "__main__":
    run_modeling()
