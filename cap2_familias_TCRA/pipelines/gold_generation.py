import pandas as pd
import numpy as np
from src.config import DATASETS, DEFAULT_W, DEFAULT_LAMBDA, TABLES_DIR
from src.models.tcroc_variants import calculate_alpha_family
from src.utils.helpers import save_csv

def run_gold():
    for ds in ["combustibles", "pib"]:
        config = DATASETS[ds]
        df_silver = pd.read_csv(config['silver_path'], index_col=0, parse_dates=True)
        
        # Try to load best parameters if they exist
        best_params_path = TABLES_DIR / f"grid_search_best_{ds}.csv"
        best_params = {}
        if best_params_path.exists():
            print(f"Loading best parameters from {best_params_path}")
            df_best = pd.read_csv(best_params_path)
            for _, row in df_best.iterrows():
                best_params[row['Series']] = {
                    'W': int(row['W']),
                    'Lambda': float(row['Lambda'])
                }
        
        all_results = []
        for col in config['series_cols']:
            # Use found best params, or fallback to defaults
            params = best_params.get(col, {'W': DEFAULT_W[ds], 'Lambda': DEFAULT_LAMBDA})
            w_val = params['W']
            l_val = params['Lambda']
            
            print(f"Applying best params for {ds} - {col}: W={w_val}, L={l_val}")
            
            # Generar 4 variantes usando "Mejor W/L" para las variables
            v = df_silver[col].values
            df_v = pd.DataFrame(index=df_silver.index)
            df_v[f"{col}_TCROC"] = calculate_alpha_family(v, 2, 1.0)
            df_v[f"{col}_TCROCM"] = calculate_alpha_family(v, w_val, 1.0)
            df_v[f"{col}_ETCROC"] = calculate_alpha_family(v, 2, l_val)
            df_v[f"{col}_ETCROCM"] = calculate_alpha_family(v, w_val, l_val)
            
            all_results.append(df_v)
            
        df_gold = pd.concat(all_results, axis=1)
        
        # Guardar resultados gold
        output_path = config['gold_path']
        df_gold.to_csv(output_path)
        print(f"Saved gold results for {ds} at {output_path}")
        
        # Also save as individual table in outputs
        save_csv(df_gold, f"{ds}_final_alphas")

if __name__ == "__main__":
    run_gold()
