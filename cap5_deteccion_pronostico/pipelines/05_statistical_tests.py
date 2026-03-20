import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import REQUIRED_COLUMNS, TABLES_DIR, DEFAULT_W, DEFAULT_LAMBDA, DEFAULT_K
from src.evaluation.compute_predictive_metrics import compute_predictive_metrics
from src.processing.functions.calculate_alphas import calculate_alphas

def run_statistical_tests():
    print("\n--- STEP 17: Performing Statistical Significance Tests ---")
    
    silver_path = "data/silver/fuel_data_standardized.csv"
    if not os.path.exists(silver_path):
        print("Data not found.")
        return
        
    data = pd.read_csv(silver_path)
    significance_results = []
    
    for fuel in REQUIRED_COLUMNS:
        print(f"Analyzing: {fuel}")
        series = data[fuel].values
        
        # Calcular alphas optimos
        # In a real scenario, we'd load best W/Lambda from grid search
        # For now we use defaults or assume they are optimized
        alphas = calculate_alphas(series, W=DEFAULT_W, lambda_decay=DEFAULT_LAMBDA)
        
        # K-Means Metrics
        acc_km, rmse_km, splits_km = compute_predictive_metrics(
            alphas, series, DEFAULT_K, DEFAULT_W, method='kmeans', return_splits=True
        )
        
        # Quantiles Metrics
        acc_q, rmse_q, splits_q = compute_predictive_metrics(
            alphas, series, 4, DEFAULT_W, method='quantiles', return_splits=True
        )
        
        # Paired T-Test on RMSEs from splits
        rmses_km = [s['RMSE'] for s in splits_km]
        rmses_q = [s['RMSE'] for s in splits_q]
        
        t_stat, p_val = stats.ttest_rel(rmses_km, rmses_q)
        
        significance_results.append({
            'Fuel': fuel,
            'RMSE_KMeans': rmse_km,
            'RMSE_Quantiles': rmse_q,
            'P-Value': p_val,
            'Significant_0.05': p_val < 0.05
        })

    df_sig = pd.DataFrame(significance_results)
    df_sig.to_csv(os.path.join(TABLES_DIR, "statistical_significance.csv"), index=False)
    print("\nStatistical Significance Table:")
    print(df_sig)

if __name__ == "__main__":
    run_statistical_tests()
