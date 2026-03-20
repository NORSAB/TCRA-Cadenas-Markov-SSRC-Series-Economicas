"""
===================================================================
Pipeline 03b: Comparison Markov vs SSRC (TURBO v3 — Reproducible)
===================================================================
Usa grids pre-computados (paso 02b).
Ensemble con NNLS + create_reservoir() oficial + H_full pre-cómputo.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import numpy as np
import tkinter as tk
from tkinter import ttk
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import (FUEL_COLUMNS, TCROC_OPTIMAL, MARKOV_RMSE, MODELS_DIR,
                    OUTPUT_DIR, N_REALIZATIONS, N_GRID_REALIZATIONS,
                    RESERVOIR_WASHOUT, TRAIN_SIZE, SEED)

from evaluation.functions.compare_models import compare_markov_vs_ssrc, save_comparison_csv
from evaluation.functions.markov_benchmark import predict_markov_rolling_window
from evaluation.functions.stats_tests import diebold_mariano_test

# Numba JIT (misma función que 02b)
try:
    from numba import njit

    @njit(cache=True)
    def _propagate_jit(alphas, W_in_col, W_res, leak_rate, D, T):
        H = np.zeros((D, T))
        h = np.zeros(D)
        a = leak_rate
        for t in range(T):
            u = alphas[t]
            pre = np.zeros(D)
            for j in range(D):
                pre[j] = W_in_col[j] * u
                for k in range(D):
                    pre[j] += W_res[j, k] * h[k]
            for j in range(D):
                h[j] = (1.0 - a) * h[j] + a * np.tanh(pre[j])
            H[:, t] = h
        return H

    NUMBA_OK = True
except ImportError:
    NUMBA_OK = False


def _ensemble_realization(fuel, alphas, prices, D, rho, leak_rate, seed,
                           train_size, washout):
    """
    Una realización del ensemble final.
    Usa create_reservoir() + NNLS + H_full pre-cómputo.
    """
    from scipy.optimize import nnls
    from reservoir.functions.create_reservoir import create_reservoir
    
    # Crear reservorio con función oficial
    W_in, W_res = create_reservoir(
        input_dim=1, reservoir_dim=D, spectral_radius=rho,
        sparsity=0.9, seed=seed
    )
    
    T = len(alphas)
    n_test = T - train_size - 1
    
    # Pre-computar propagación completa
    W_in_col = W_in.ravel()
    if NUMBA_OK:
        H_full = _propagate_jit(alphas, W_in_col, W_res, leak_rate, D, T)
    else:
        H_full = np.zeros((D, T))
        h = np.zeros(D)
        a = leak_rate
        for t in range(T):
            pre = W_in_col * alphas[t] + W_res @ h
            h = (1.0 - a) * h + a * np.tanh(pre)
            H_full[:, t] = h
    
    # Rolling window con NNLS
    predictions = np.zeros(n_test)
    actuals = np.zeros(n_test)
    
    for i in range(n_test):
        train_end = train_size + i
        actuals[i] = prices[train_end + 1]
        
        try:
            H_train = H_full[:, washout : train_end]
            targets = alphas[washout + 1 : train_end]
            H_aligned = H_train[:, :-1]
            
            min_len = min(H_aligned.shape[1], len(targets))
            if min_len == 0:
                predictions[i] = prices[train_end]
                continue
            
            H_aligned = H_aligned[:, :min_len]
            targets_a = targets[:min_len]
            
            # NNLS (Def. 6.3)
            try:
                W_out, _ = nnls(H_aligned.T, targets_a,
                                maxiter=D * min_len * 5)
            except RuntimeError:
                reg = 1e-6 * np.eye(D)
                W_out = np.linalg.solve(
                    H_aligned @ H_aligned.T + reg,
                    H_aligned @ targets_a
                )
                W_out = np.maximum(W_out, 0)
            
            h_predict = H_full[:, train_end]
            alpha_pred = float(W_out @ h_predict)
            predictions[i] = prices[train_end] * (1.0 + alpha_pred)
        except Exception:
            predictions[i] = prices[train_end]
    
    return {'fuel': fuel, 'predictions': predictions, 'actuals': actuals}


def main():
    print("=" * 60)
    print("Comparación Markov vs SSRC (TURBO v3, Reproducible)")
    print("=" * 60)
    
    data_path = os.path.join(MODELS_DIR, 'prepared_data.npz')
    if not os.path.exists(data_path):
        print("Error: No se encontró prepared_data.npz.")
        return
    data = np.load(data_path)
    
    # Warmup Numba
    if NUMBA_OK:
        _propagate_jit(np.zeros(10), np.zeros(5), np.zeros((5, 5)), 1.0, 5, 10)
        print("  ⚡ Numba JIT listo")
    
    # Cargar grids pre-computados
    best_configs = {}
    for fuel in FUEL_COLUMNS:
        grid_path = os.path.join(MODELS_DIR, f'{fuel.lower()}_ssrc_grid.npz')
        if os.path.exists(grid_path):
            grid_data = np.load(grid_path, allow_pickle=True)
            grid = grid_data['grid']
            best = min(grid, key=lambda x: x['rmse_mean'])
            best_configs[fuel] = best
            leak_s = f", a={best.get('leak_rate', 1.0):.1f}" if best.get('leak_rate', 1.0) != 1.0 else ""
            print(f"  {fuel}: D={best['D']}, ρ={best['rho']:.2f}{leak_s}, "
                  f"RMSE={best['rmse_mean']:.4f}")
        else:
            best_configs[fuel] = {'D': 100, 'rho': 0.80, 'leak_rate': 1.0,
                                  'rmse_mean': 0, 'rmse_std': 0}
    
    # GUI
    root = tk.Tk()
    root.title("Modelo Competitivo: Markov vs SSRC")
    root.geometry("850x550")
    root.configure(bg="#ffffff")

    tk.Label(root, text="⚔ Duelo de Modelos: Markov vs SSRC", 
             font=("Segoe UI", 18, "bold"), bg="#ffffff", fg="#1e3d59").pack(pady=15)
    
    table_frame = tk.Frame(root, bg="#ffffff")
    table_frame.pack(fill="both", expand=True, padx=20)

    h_bg = "#f0f4f8"
    headers = ["Combustible", "Markov", "SSRC", "Δ%", "Config", "DM", "P-Value"]
    for j, h in enumerate(headers):
        tk.Label(table_frame, text=h, font=("Segoe UI", 10, "bold"), bg=h_bg,
                 width=12, relief="flat", padx=4, pady=6).grid(
                     row=0, column=j, sticky="nsew")

    rows_widgets = {}
    for i, fuel in enumerate(FUEL_COLUMNS):
        tk.Label(table_frame, text=fuel, font=("Segoe UI", 10, "bold"),
                 bg="white").grid(row=i+1, column=0, pady=8)
        tk.Label(table_frame, text=f"{MARKOV_RMSE[fuel]:.4f}",
                 font=("Segoe UI", 10), bg="white").grid(row=i+1, column=1)
        
        ssrc_lbl = tk.Label(table_frame, text="⏳", font=("Segoe UI", 10, "italic"),
                           bg="white", fg="gray")
        ssrc_lbl.grid(row=i+1, column=2)
        delta_lbl = tk.Label(table_frame, text="—", font=("Segoe UI", 10, "bold"),
                            bg="white")
        delta_lbl.grid(row=i+1, column=3)
        cfg_lbl = tk.Label(table_frame, text="—", font=("Segoe UI", 9),
                          bg="white", fg="#555")
        cfg_lbl.grid(row=i+1, column=4)
        dm_lbl = tk.Label(table_frame, text="—", font=("Segoe UI", 10), bg="white")
        dm_lbl.grid(row=i+1, column=5)
        pv_lbl = tk.Label(table_frame, text="—", font=("Segoe UI", 10), bg="white")
        pv_lbl.grid(row=i+1, column=6)
        
        rows_widgets[fuel] = {'ssrc': ssrc_lbl, 'delta': delta_lbl,
                              'cfg': cfg_lbl, 'dm': dm_lbl, 'pv': pv_lbl}

    status_var = tk.StringVar(value="Calculando ensembles (NNLS + Numba)...")
    tk.Label(root, textvariable=status_var, font=("Segoe UI", 10),
            bg="#ffffff").pack(pady=15)

    comparisons = []
    max_workers = min(16, os.cpu_count() or 4)
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures_map = {}
    fuel_preds = {f: [] for f in FUEL_COLUMNS}
    fuel_actuals = {f: None for f in FUEL_COLUMNS}
    
    global_start = time.time()
    
    # Semillas deterministas para ensemble
    for fuel in FUEL_COLUMNS:
        alphas = data[f'{fuel}_alpha']
        prices = data[f'{fuel}_prices']
        best = best_configs[fuel]
        D = int(best['D'])
        rho = float(best['rho'])
        leak = float(best.get('leak_rate', 1.0))
        
        for r in range(N_REALIZATIONS):
            seed = SEED + r  # Semilla simple y determinista
            f = executor.submit(_ensemble_realization, fuel, alphas, prices,
                               D, rho, leak, seed, TRAIN_SIZE, RESERVOIR_WASHOUT)
            futures_map[f] = fuel
    
    ensemble_done = {f: 0 for f in FUEL_COLUMNS}
    
    def update_ui():
        done_futures = [f for f in futures_map if f.done()]
        
        for f in done_futures:
            fuel = futures_map.pop(f)
            try:
                res = f.result()
                fuel_preds[fuel].append(res['predictions'])
                if fuel_actuals[fuel] is None:
                    fuel_actuals[fuel] = res['actuals']
            except Exception as e:
                print(f"  Error ensemble {fuel}: {e}")
            ensemble_done[fuel] += 1
        
        # Verificar si any fuel just completed all realizations
        for fuel in FUEL_COLUMNS:
            if (ensemble_done[fuel] == N_REALIZATIONS and 
                    fuel not in [c.get('fuel', c.get('serie')) for c in comparisons]):
                best = best_configs[fuel]
                s_preds = np.mean(fuel_preds[fuel], axis=0)
                actuals = fuel_actuals[fuel]
                
                # Markov benchmark
                k_val = TCROC_OPTIMAL[fuel]['K']
                alphas = data[f'{fuel}_alpha']
                prices = data[f'{fuel}_prices']
                m_preds, _ = predict_markov_rolling_window(
                    alphas, prices, TRAIN_SIZE, k_val)
                
                # DM Test
                min_len = min(len(m_preds), len(s_preds), len(actuals))
                dm_stat, p_val = diebold_mariano_test(
                    actuals[:min_len], m_preds[:min_len], s_preds[:min_len])
                
                comp = compare_markov_vs_ssrc(MARKOV_RMSE[fuel], best, fuel)
                comp['dm_stat'] = dm_stat
                comp['p_value'] = p_val
                comparisons.append(comp)
                
                # Actualizar interfaz
                w = rows_widgets[fuel]
                w['ssrc'].config(text=f"{comp['ssrc_rmse_mean']:.4f}",
                                fg="black", font=("Segoe UI", 10))
                d_val = comp['delta_rmse_pct']
                d_color = "#2e7d32" if d_val < 0 else "#c62828"
                w['delta'].config(text=f"{d_val:+.1f}%", fg=d_color)
                
                D = int(best['D'])
                rho = float(best['rho'])
                leak = float(best.get('leak_rate', 1.0))
                leak_s = f",{leak:.1f}" if leak != 1.0 else ""
                w['cfg'].config(text=f"{D},{rho:.2f}{leak_s}")
                w['dm'].config(text=f"{dm_stat:.2f}")
                if p_val < 0.05:
                    w['pv'].config(text=f"{p_val:.4f} ★", fg="#1565c0",
                                  font=("Segoe UI", 10, "bold"))
                else:
                    w['pv'].config(text=f"{p_val:.3f}", fg="black")
        
        elapsed = time.time() - global_start
        total_done = sum(ensemble_done.values())
        total_tasks = N_REALIZATIONS * len(FUEL_COLUMNS)
        status_var.set(
            f"Ensemble: {total_done}/{total_tasks} | "
            f"Comparados: {len(comparisons)}/4 | {elapsed:.0f}s"
        )
        
        if len(comparisons) == len(FUEL_COLUMNS):
            status_var.set(f"✅ Completado en {elapsed:.0f}s")
            root.after(1500, finalize)
        else:
            root.after(200, update_ui)

    def finalize():
        csv_path = os.path.join(OUTPUT_DIR, 'tables', 'ssrc_comparison.csv')
        save_comparison_csv(comparisons, csv_path)
        
        print("\n" + "=" * 65)
        print("RESUMEN COMPARATIVO (NNLS, Reproducible)")
        print("=" * 65)
        for c in comparisons:
            sig = "★" if c['p_value'] < 0.05 else " "
            fuel = c.get('fuel', c.get('serie', ''))
            print(f"  {fuel:10s}: Markov={MARKOV_RMSE.get(fuel, 0):.4f} → "
                  f"SSRC={c['ssrc_rmse_mean']:.4f} ({c['delta_rmse_pct']:+.1f}%) "
                  f"DM={c['dm_stat']:.2f} p={c['p_value']:.4f} {sig}")
        
        executor.shutdown(wait=False)
        root.destroy()

    root.after(300, update_ui)
    root.mainloop()

if __name__ == "__main__":
    main()
