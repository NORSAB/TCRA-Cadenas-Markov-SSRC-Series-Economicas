import os
# OPTIMIZATION: Prevent NumPy/Scikit-learn from spawning threads in parallel processes
# Reducir overhead de cambio de contexto en ProcessPoolExecutor
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import pandas as pd
import numpy as np
import time
import tkinter as tk
from tkinter import ttk
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error

# Add root to path
from src.ingestion.load_fuel_data import load_fuel_data
from src.processing.functions.calculate_alphas import calculate_alphas
from src.evaluation.compute_predictive_metrics import compute_predictive_metrics
from src.config import (
    CSV_PATH, TABLES_DIR, SEED, REQUIRED_COLUMNS
)
from src.processing.functions.discretize_series import discretize_series
from src.models.functions.estimate_transition_matrix_mle import estimate_transition_matrix_mle

def evaluate_fuel_chunk(fuel_name, series, w_values, lambda_range, k_range_search):
    """
    Evaluates a chunk of the grid (specific W values) for a fuel.
    OPTIMIZED: Computes alphas once per (W, L) pair and iterates K.
    """
    np.random.seed(SEED) # Ensure reproducibility within each process
    local_results = []
    
    # Outer loops: Hyperparameters that affect Alphas
    for w_val in w_values:
        for l_val in lambda_range:
            # CACHE OPTIMIZATION: Calculate Alphas ONCE for this (W, L) pair
            try:
                alphas = calculate_alphas(series, W=w_val, lambda_decay=l_val)
            except Exception:
                continue

            # Inner loop: K values (using the SAME alphas)
            for k_val in k_range_search:
                # Basic check: Need enough data points
                if len(alphas) < k_val * 2: continue

                try:
                    # 1. AIC calculation on full series (matching notebook)
                    states_full, _ = discretize_series(alphas, k_val)
                    P_full, C_full = estimate_transition_matrix_mle(states_full, k_val)
                    
                    # Log-likelihood + AIC
                    log_likelihood = np.sum(C_full * np.log(P_full + 1e-9))
                    aic = 2 * k_val * (k_val - 1) - 2 * log_likelihood

                    # 2. Cross-validation (matching notebook's compute_predictive_metrics_detailed)
                    # Note: compute_predictive_metrics handles its own internal checks
                    partition_results = compute_predictive_metrics(alphas, series, k_val, w_val)
                    
                    for res in partition_results:
                        local_results.append({
                            'Fuel Series': fuel_name, 'W': w_val, 'Lambda': l_val,
                            'k': k_val, 'Partition': res['Partition'],
                            'Accuracy': res['Accuracy'], 'RMSE': res['RMSE'], 'AIC': aic
                        })
                except Exception as e:
                     # Log specific error but continue with next K
                    # print(f"Error in evaluate_fuel_chunk (W={w_val}, L={l_val}, K={k_val}): {e}", flush=True)
                    continue
                    
    return local_results

def execute_grid_search():
    print("\n--- PASO FINAL: Iniciando Búsqueda Grid 3D Paralelizada (GUI Activa) ---")
    print("⚠️  Utilizando Tkinter para monitoreo en tiempo real.")

    # Cargar datos
    _, fuel_series = load_fuel_data(CSV_PATH)
    if fuel_series is None:
        print("Error: No se pudieron cargar los datos.")
        return

    # Exact Search Space from Notebook
    w_range = list(range(2, 53)) # 51 values
    lambda_range = np.round(np.arange(0.8, 1.01, 0.01), 2) # 21 values
    k_range_search = list(range(2, 10)) # 8 values
    
    # Total combinations per fuel (before partitioning)
    # Bucle interno: w (chunk) * lambda(21) * k(8)
    # Total iter per W = 21 * 8 = 168
    total_iters_per_fuel = len(w_range) * len(lambda_range) * len(k_range_search)

    # Core Management
    num_physical_cores = os.cpu_count() or 4
    # User requested to use 16 cores max if 20 are available
    max_workers = min(16, num_physical_cores)
    print(f"Cores detectados: {num_physical_cores}. Usando límite de: {max_workers} núcleos.")

    # GUI Setup
    root = tk.Tk()
    root.title("Grid Search Progress - TCROC Markov Model")
    root.geometry("700x500")
    root.configure(bg="#f0f0f0")

    header = tk.Label(root, text="Optimizando Hiperparámetros (Grid Search)", font=("Arial", 16, "bold"), bg="#f0f0f0")
    header.pack(pady=10)

    sub_header = tk.Label(root, text=f"Ejecutando en paralelo con {max_workers} núcleos...", font=("Arial", 10), bg="#f0f0f0", fg="gray")
    sub_header.pack(pady=5)

    progress_frame = tk.Frame(root, bg="#f0f0f0")
    progress_frame.pack(fill="both", expand=True, padx=20, pady=10)

    # State containers
    fuel_progress_widgets = {}
    fuel_stats = {} # {fuel: {'completed': 0, 'total': N, 'start_time': t, 'end_time': None}}

    for fuel in REQUIRED_COLUMNS:
        f_frame = tk.Frame(progress_frame, bg="white", bd=1, relief="solid")
        f_frame.pack(fill="x", pady=5, padx=5, ipady=5)
        
        lbl_name = tk.Label(f_frame, text=f"{fuel}", font=("Arial", 12, "bold"), bg="white", width=15, anchor="w")
        lbl_name.pack(side="left", padx=10)
        
        pbar = ttk.Progressbar(f_frame, orient="horizontal", length=300, mode="determinate")
        pbar.pack(side="left", padx=10, fill="x", expand=True)
        
        lbl_pct = tk.Label(f_frame, text="0.0%", font=("Arial", 10), bg="white", width=8)
        lbl_pct.pack(side="left", padx=5)
        
        lbl_time = tk.Label(f_frame, text="00:00", font=("Arial", 10), bg="white", width=10)
        lbl_time.pack(side="left", padx=5)
        
        fuel_progress_widgets[fuel] = {'pbar': pbar, 'lbl_pct': lbl_pct, 'lbl_time': lbl_time}
        fuel_stats[fuel] = {'completed': 0, 'total': total_iters_per_fuel, 'start_time': None, 'end_time': None}

    total_time_lbl = tk.Label(root, text="Tiempo Total: 00:00", font=("Arial", 12, "bold"), bg="#f0f0f0")
    total_time_lbl.pack(pady=10)

    # Prepare Execution
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures_map = {} # future -> (fuel, expected_iterations)
    
    # SUBMIT ALL TASKS (INTERLEAVED)
    # Collect all tasks first to shuffle/interleave them, prioritizing simultaneous progress across all fuels.
    # Recolectar todas las tareas primero para barajarlas/intercalarlas, priorizando el progreso simultáneo en todos los combustibles.
    
    global_start_time = time.time()
    
    all_tasks = []
    import random
    
    lambda_chunks = np.array_split(lambda_range, 2)
    
    for fuel_name in REQUIRED_COLUMNS:
        if fuel_name not in fuel_series: continue
        series = fuel_series[fuel_name]
        fuel_stats[fuel_name]['start_time'] = time.time()
        
        # INCREASED GRANULARITY: ~100 tasks per fuel
        for w_val in w_range:
            for l_chunk in lambda_chunks:
                if len(l_chunk) == 0: continue
                
                # Task Definition
                task = {
                    'fuel_name': fuel_name,
                    'series': series,
                    'w_subset': [w_val],
                    'l_subset': l_chunk.tolist(),
                    'k_range': k_range_search,
                    'expected_iters': 1 * len(l_chunk) * len(k_range_search)
                }
                all_tasks.append(task)

    # Shuffle tasks to ensure parallel execution across different fuels
    # Barajar tareas para asegurar ejecución paralela entre diferentes combustibles
    random.shuffle(all_tasks)

    # Submit tasks
    for task in all_tasks:
        future = executor.submit(
            evaluate_fuel_chunk, 
            task['fuel_name'], 
            task['series'], 
            task['w_subset'], 
            task['l_subset'], 
            task['k_range']
        )
        futures_map[future] = (task['fuel_name'], task['expected_iters'])

    all_results = []
    
    def update_ui():
        # Verificar futuros activos
        done_futures = [f for f in futures_map.keys() if f.done()]
        
        for f in done_futures:
            fuel_name, expected_iters = futures_map.pop(f) # Remove from tracking
            try:
                res = f.result()
                all_results.extend(res)
                
                # Actualizar estadisticas
                fuel_stats[fuel_name]['completed'] += expected_iters
                
                # Verificar si fuel finished
                if fuel_stats[fuel_name]['completed'] >= fuel_stats[fuel_name]['total']:
                    if fuel_stats[fuel_name]['end_time'] is None:
                        fuel_stats[fuel_name]['end_time'] = time.time() # Mark finish time
                        
            except Exception as e:
                print(f"Error in future for {fuel_name}: {e}")
        
        # Actualizar widgets
        all_finished = True
        current_time = time.time()
        
        for fuel in REQUIRED_COLUMNS:
            stats = fuel_stats[fuel]
            widgets = fuel_progress_widgets[fuel]
            
            progress = min(1.0, stats['completed'] / stats['total'])
            widgets['pbar']['value'] = progress * 100
            widgets['lbl_pct'].config(text=f"{progress*100:.1f}%")
            
            # Time tracking
            if stats['end_time']:
                elapsed = stats['end_time'] - stats['start_time']
                widgets['lbl_time'].config(text=f"{elapsed:.1f}s", fg="green")
            else:
                elapsed = current_time - stats['start_time']
                widgets['lbl_time'].config(text=f"{elapsed:.1f}s", fg="black")
                all_finished = False

        total_elapsed = current_time - global_start_time
        total_time_lbl.config(text=f"Tiempo Total: {total_elapsed/60:.2f} min")

        if all_finished and not futures_map:
            root.after(1000, finish_process) # Wait 1s then close
        else:
            root.after(100, update_ui) # Poll every 100ms

    def finish_process():
        executor.shutdown(wait=False)
        root.destroy()

    root.after(100, update_ui)
    root.mainloop()

    # --- POST-PROCESSING (After Window Closes) ---
    print("\n✅ Interfaz Gráfica cerrada. Procesando resultados finales...")
    
    if all_results:
        os.makedirs(TABLES_DIR, exist_ok=True)
        df_detailed = pd.DataFrame(all_results).dropna()
        
        # Translate columns to Spanish for output
        df_detailed.rename(columns={
            'Fuel Series': 'Combustible', 
            'Partition': 'Partición',
            'Accuracy': 'Exactitud'
        }, inplace=True)
        
        df_detailed.to_csv(os.path.join(TABLES_DIR, "20_final_grid_search_detailed.csv"), index=False, float_format='%.10f')
        
        df_summary = df_detailed.groupby(['Combustible', 'W', 'Lambda', 'k']).agg(
            Exactitud=('Exactitud', 'mean'),
            RMSE=('RMSE', 'mean'),
            AIC=('AIC', 'first')
        ).reset_index()
        df_summary.to_csv(os.path.join(TABLES_DIR, "21_grid_search_summary.csv"), index=False, float_format='%.10f')
        
        # Selection
        print("Aplicando lógica de selección y guardando tablas...")
        df_realistic = df_summary[df_summary['Exactitud'] < 0.95].copy()
        
        if not df_realistic.empty:
            fuel_order = ['Super', 'Regular', 'Diesel', 'Kerosene']
            df_realistic['Combustible'] = pd.Categorical(df_realistic['Combustible'], categories=fuel_order, ordered=True)
            df_best = df_realistic.sort_values(
                by=['Combustible', 'RMSE', 'Exactitud', 'AIC'], 
                ascending=[True, True, False, True]
            ).drop_duplicates(subset=['Combustible'], keep='first')
            
            df_best.to_csv(os.path.join(TABLES_DIR, "22_best_hyperparameters.csv"), index=False, float_format='%.10f')
            df_best.to_csv(os.path.join(TABLES_DIR, "best_hyperparameters.csv"), index=False, float_format='%.10f')

            print("\n" + "="*60)
            print("RESULTADOS FINALES DE OPTIMIZACIÓN (Guardados en tablas)")
            print("="*60)
            print(df_best.to_string(index=False))
            
            # Print timing summary for user confirmation
            print("\nTIEMPOS DE EJECUCIÓN PARALELA:")
            for fuel in REQUIRED_COLUMNS:
                s = fuel_stats[fuel]
                t = s['end_time'] - s['start_time'] if s['end_time'] else 0
                print(f"  - {fuel:10s}: {t:.2f} segundos")
            
    else:
        print("Error: El Grid Search no produjo resultados.")

if __name__ == "__main__":
    execute_grid_search()
