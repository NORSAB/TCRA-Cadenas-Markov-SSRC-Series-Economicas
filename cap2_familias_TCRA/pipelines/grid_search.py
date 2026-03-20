import pandas as pd
import numpy as np
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import time

from src.config import DATASETS, TABLES_DIR, GOLD_DIR
from src.models.tcroc_variants import calculate_alpha_family
from src.evaluation.metrics import calculate_metrics

def evaluate_params(args):
    """Worker function for parallel processing."""
    series_name, series_data, W, lam, variant_name, max_W = args
    
    alphas = calculate_alpha_family(series_data, W, lam)
    
    # x_hat = x_{t-1} * (1 + alpha_t)
    y_true = np.asarray(series_data)
    y_pred = np.roll(y_true, 1) * (1 + alphas)
    y_pred[0] = np.nan # First element has no prediction
    
    # FAIR COMPUTATION & CROSS-VALIDATION
    # To fairly compare all models, we evaluate exclusively out-of-sample (starting strictly after max_W).
    valid_start = max_W + 1
    
    if valid_start < len(y_true):
        y_t_eval = y_true[valid_start:]
        y_p_eval = y_pred[valid_start:]
        
        # Validacion cruzada temporal (Backtesting con 4 ventanas expansivas)
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=4)
        
        fold_metrics = []
        for train_index, test_index in tscv.split(y_t_eval):
            # En modelos determinísticos evaluamos el Forward-Test (Test Index)
            y_t_test = y_t_eval[test_index]
            y_p_test = y_p_eval[test_index]
            fold_metrics.append(calculate_metrics(y_t_test, y_p_test))
            
        # Promediar las métricas a través de los 4 Folds de Validación
        metrics = {}
        # Tomar las llaves de la primera evaluación y sacar el np.nanmean
        for key in fold_metrics[0].keys():
            metrics[key] = np.nanmean([fm[key] for fm in fold_metrics])
            
    else:
        # Too little data to compare strictly
        metrics = calculate_metrics(y_true, y_pred)
    
    return {
        "Series": series_name,
        "Variant": variant_name,
        "W": W,
        "Lambda": lam,
        **metrics
    }

def run_grid_search(dataset_key):
    config = DATASETS[dataset_key]
    df = pd.read_csv(config['silver_path'], index_col=0, parse_dates=True)
    
    # Ranges
    if dataset_key == 'combustibles':
        w_range = range(2, 56) # 2 to 55
        max_W = 55
    else:
        w_range = range(2, 11) # 2 to 10
        max_W = 10
        
    lambda_range = np.linspace(0.70, 1.00, 31) # 0.70 to 1.00 explicitly
    variants = ['TCROC', 'TCROCM', 'ETCROC', 'ETCROCM'] # Evaluamos a toda la familia
    
    tasks = []
    for col in config['series_cols']:
        series_data = df[col].values
        for variant in variants:
            if variant == 'TCROC':
                # Base model. Mínimo permitido W=2, Lambda=1.0. 
                tasks.append((col, series_data, 2, 1.0, variant, max_W))
                
            elif variant == 'ETCROC':
                # Decay model, W=2 (min to see decay). 
                for lam in lambda_range:
                    tasks.append((col, series_data, 2, round(lam, 2), variant, max_W))
                    
            elif variant == 'TCROCM':
                for w in w_range:
                    tasks.append((col, series_data, w, 1.0, variant, max_W))
                    
            elif variant == 'ETCROCM':
                for w in w_range:
                    for lam in lambda_range:
                        tasks.append((col, series_data, w, round(lam, 2), variant, max_W))

    print(f"Starting Grid Search for {dataset_key} ({len(tasks)} tasks)...")
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        # Using tqdm to show progress
        list_results = list(tqdm(executor.map(evaluate_params, tasks), total=len(tasks)))
        results.extend(list_results)
        
    df_results = pd.DataFrame(results)
    
    # Guardar resultados detallados
    output_path = TABLES_DIR / f"grid_search_detailed_{dataset_key}.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Detailed results saved at {output_path}")
    
    # Find best of the best per series using a hierarchy metric rule
    # Rule hierarchy: 1. RMSE (lowest), 2. MAE (lowest), 3. MAPE (lowest)
    best_results = []
    for col in config['series_cols']:
        df_series = df_results[df_results['Series'] == col].copy()
        if not df_series.empty:
            # Drop NaN to be safe
            df_series = df_series.dropna(subset=['RMSE', 'MAE', 'MAPE'])
            df_sorted = df_series.sort_values(by=['RMSE', 'MAE', 'MAPE'], ascending=[True, True, True])
            best = df_sorted.iloc[0]
            best_results.append(best)
            
    df_best = pd.DataFrame(best_results)
    best_output_path = TABLES_DIR / f"grid_search_best_{dataset_key}.csv"
    df_best.to_csv(best_output_path, index=False)
    print(f"Best parameters for {dataset_key}:\n", df_best[["Series", "Variant", "W", "Lambda", "RMSE"]])
    
    return df_best

if __name__ == "__main__":
    for ds in ["combustibles", "pib"]:
        run_grid_search(ds)
