import pandas as pd
from src.config import TABLES_DIR

def run_tables():
    print("--- Extracting Best Metrics Combinations ---")
    metrics = ["RMSE", "MAE", "MedAE", "MSLE", "MAPE", "R2"]
    
    for ds in ["combustibles", "pib"]:
        path = TABLES_DIR / f"grid_search_detailed_{ds}.csv"
        if not path.exists():
            continue
            
        df = pd.read_csv(path)
        
        # Dejamos todas las variantes (TCROCM, ETCROCM) para ver cuál es realmente la mejor
        df_opt = df
        if df_opt.empty:
            continue
            
        series = df_opt['Series'].unique()
        best_rows = []
        
        for ser in series:
            df_s = df_opt[df_opt['Series'] == ser]
            
            for m in metrics:
                if m not in df_s.columns:
                    continue
                
                df_clean = df_s.dropna(subset=[m])
                if df_clean.empty:
                    continue
                    
                if m == 'R2':
                    top = df_clean.loc[df_clean[m].idxmax()].copy()
                else:
                    top = df_clean.loc[df_clean[m].idxmin()].copy()
                
                top['Optimized_For'] = m
                best_rows.append(top)
        
        if best_rows:
            df_best_all = pd.DataFrame(best_rows)
            # Organizar la vista
            cols_order = ['Series', 'Optimized_For', 'Variant', 'W', 'Lambda'] + metrics
            cols_order = [c for c in cols_order if c in df_best_all.columns]
            df_best_all = df_best_all[cols_order]
            
            out_file = TABLES_DIR / f"best_combinations_per_metric_{ds}.csv"
            df_best_all.to_csv(out_file, index=False)
            print(f"Saved: {out_file.name} (Tus 48 combinaciones para análisis)")

if __name__ == "__main__":
    run_tables()
