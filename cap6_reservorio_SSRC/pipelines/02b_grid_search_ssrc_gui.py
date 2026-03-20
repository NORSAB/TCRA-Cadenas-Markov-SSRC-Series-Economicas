"""
===================================================================
Pipeline 02b: Grid Search SSRC TURBO v3 (Reproducible + Correcto)
===================================================================
Optimizaciones que NO cambian resultados:
  1. Numba JIT para propagación del reservorio (~50x)
  2. Pre-computa H_full UNA sola vez (equivalencia causal) (~75x)
  3. Carga de grids pre-computados en paso 04

Garantías de reproducibilidad:
  - Misma semilla → mismos reservorios
  - Solver NNLS (Definición 6.3: W_out ≥ 0) como en la teoría
  - create_reservoir() del módulo oficial
  - random.seed(SEED) para shuffle determinista
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import numpy as np
import random
import tkinter as tk
from tkinter import ttk
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import (FUEL_COLUMNS, MODELS_DIR,
                    RESERVOIR_D_VALUES, RESERVOIR_RHO_VALUES,
                    RESERVOIR_LEAK_RATES,
                    N_GRID_REALIZATIONS, RESERVOIR_WASHOUT, TRAIN_SIZE, SEED)


# =====================================================================
# Numba JIT — solo acelera cómputo, NO cambia resultados
# =====================================================================
try:
    from numba import njit

    @njit(cache=True)
    def _propagate_jit(alphas, W_in_col, W_res, leak_rate, D, T):
        """Propagación Leaky-ESN compilada a nativo. Idéntica a la ecuación 6.3."""
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


def _propagate_numpy(alphas, W_in_col, W_res, leak_rate, D, T):
    """Fallback NumPy — mismo resultado que Numba, más lento."""
    H = np.zeros((D, T))
    h = np.zeros(D)
    a = leak_rate
    for t in range(T):
        pre = W_in_col * alphas[t] + W_res @ h
        h = (1.0 - a) * h + a * np.tanh(pre)
        H[:, t] = h
    return H


def _single_realization(fuel, alphas, prices, D, rho, leak_rate, seed,
                         train_size, washout):
    """
    UNA realización con pre-cómputo de H_full + NNLS (correcto teóricamente).
    
    Optimización clave: la propagación es CAUSAL — h_t solo depende de
    alphas[0:t+1]. Por tanto, pre-computar H_full para toda la serie 
    y luego rebanar es MATEMÁTICAMENTE IDÉNTICO a re-propagar cada ventana.
    """
    from scipy.optimize import nnls
    from reservoir.functions.create_reservoir import create_reservoir
    
    # 1. Crear reservorio con la función oficial (reproducible y consistente)
    W_in, W_res = create_reservoir(
        input_dim=1, reservoir_dim=D, spectral_radius=rho,
        sparsity=0.9, seed=seed
    )
    
    T = len(alphas)
    n_test = T - train_size - 1
    
    # 2. PRE-COMPUTAR H_full UNA sola vez
    #    Teorema: h_t = f(alphas[0:t+1], W_in, W_res, a, h_0)
    #    => H_full[:, t] es idéntico sin importar si propagamos T o T+k pasos.
    W_in_col = W_in.ravel()
    if NUMBA_OK:
        H_full = _propagate_jit(alphas, W_in_col, W_res, leak_rate, D, T)
    else:
        H_full = _propagate_numpy(alphas, W_in_col, W_res, leak_rate, D, T)
    
    # 3. Rolling window con NNLS (Def. 6.3: W_out >= 0)
    predictions = np.zeros(n_test)
    actuals = np.zeros(n_test)
    
    for i in range(n_test):
        train_end = train_size + i
        actuals[i] = prices[train_end + 1]
        
        try:
            # Rebanar estados pre-computados (equivalente causal)
            H_train = H_full[:, washout : train_end]
            targets = alphas[washout + 1 : train_end]
            H_aligned = H_train[:, :-1]
            
            min_len = min(H_aligned.shape[1], len(targets))
            if min_len == 0:
                predictions[i] = prices[train_end]
                continue
            
            H_aligned = H_aligned[:, :min_len]
            targets_a = targets[:min_len]
            
            # NNLS readout (Definición 6.3: W_out >= 0)
            try:
                W_out, _ = nnls(H_aligned.T, targets_a,
                                maxiter=D * min_len * 5)
            except RuntimeError:
                # Fallback: Ridge + clipping no-negativo
                reg = 1e-6 * np.eye(D)
                W_out = np.linalg.solve(
                    H_aligned @ H_aligned.T + reg,
                    H_aligned @ targets_a
                )
                W_out = np.maximum(W_out, 0)
            
            # Predecir: usar estado h_{train_end} = H_full[:, train_end]
            # (estado DESPUÉS de procesar alphas[train_end])
            h_predict = H_full[:, train_end]
            alpha_pred = float(W_out @ h_predict)
            predictions[i] = prices[train_end] * (1.0 + alpha_pred)
        except Exception:
            predictions[i] = prices[train_end]
    
    errors = actuals - predictions
    rmse = float(np.sqrt(np.mean(errors**2)))
    return {'fuel': fuel, 'D': D, 'rho': rho, 'leak_rate': leak_rate, 'rmse': rmse}


def main():
    print("=" * 60)
    print("Grid Search SSRC TURBO v3 (Numba + PreH + NNLS)")
    print("=" * 60)
    
    data_path = os.path.join(MODELS_DIR, 'prepared_data.npz')
    if not os.path.exists(data_path):
        print("Error: No se encontró prepared_data.npz.")
        return
    data = np.load(data_path)
    
    # Warmup Numba JIT
    if NUMBA_OK:
        print("  Compilando Numba JIT...")
        _propagate_jit(np.zeros(10), np.zeros(5), np.zeros((5, 5)), 1.0, 5, 10)
        print("  ⚡ Numba JIT compilado y cacheado")
    else:
        print("  ⚠ Numba no disponible, usando NumPy")
    
    configs_per_fuel = (len(RESERVOIR_D_VALUES) * len(RESERVOIR_RHO_VALUES) 
                        * len(RESERVOIR_LEAK_RATES))
    total_tasks = configs_per_fuel * len(FUEL_COLUMNS) * N_GRID_REALIZATIONS
    
    num_cores = os.cpu_count() or 4
    max_workers = min(16, num_cores)
    
    print(f"  Semilla global: {SEED}")
    print(f"  Configs/fuel: {configs_per_fuel}")
    print(f"  Realizaciones/config: {N_GRID_REALIZATIONS}")
    print(f"  Total tareas atómicas: {total_tasks}")
    print(f"  Workers paralelos: {max_workers}")
    print(f"  Solver: NNLS (W_out ≥ 0, Definición 6.3)")
    
    # GUI
    root = tk.Tk()
    root.title("Grid Search SSRC TURBO v3")
    root.geometry("850x500")
    root.configure(bg="#1a1a2e")

    ttk.Label(root, text=f"⚡ SSRC TURBO v3 — {total_tasks} tareas (Numba+PreH+NNLS)", 
              font=("Segoe UI", 13, "bold"), background="#1a1a2e", 
              foreground="white").pack(pady=10)
    
    subtitle = (f"Seed={SEED} | {len(RESERVOIR_D_VALUES)}D × "
                f"{len(RESERVOIR_RHO_VALUES)}ρ × {len(RESERVOIR_LEAK_RATES)}a × "
                f"{N_GRID_REALIZATIONS}R | Workers={max_workers}")
    tk.Label(root, text=subtitle, font=("Segoe UI", 9), 
             bg="#1a1a2e", fg="#888").pack()
    
    main_frame = tk.Frame(root, bg="#1a1a2e")
    main_frame.pack(fill="both", expand=True, padx=20, pady=10)

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("turbo.Horizontal.TProgressbar", troughcolor='#0f3460',
                    background='#00b894', darkcolor='#00b894',
                    lightcolor='#55efc4')

    widgets = {}
    stats = {}
    tasks_per_fuel = configs_per_fuel * N_GRID_REALIZATIONS

    for fuel in FUEL_COLUMNS:
        f = tk.Frame(main_frame, bg="#16213e", bd=0)
        f.pack(fill="x", pady=4, ipady=4)
        
        tk.Label(f, text=fuel, font=("Segoe UI", 11, "bold"),
                 bg="#16213e", fg="white", width=12).pack(side="left", padx=10)
        pbar = ttk.Progressbar(f, length=350, mode='determinate',
                               style="turbo.Horizontal.TProgressbar")
        pbar.pack(side="left", padx=10, expand=True, fill="x")
        lbl = tk.Label(f, text=f"0/{tasks_per_fuel}",
                      font=("Segoe UI", 9), bg="#16213e", fg="#ddd", width=14)
        lbl.pack(side="left", padx=5)
        lbl_best = tk.Label(f, text="☆ —", font=("Segoe UI", 9, "italic"),
                           bg="#16213e", fg="#55efc4", width=30)
        lbl_best.pack(side="left", padx=5)
        
        widgets[fuel] = {'pbar': pbar, 'lbl': lbl, 'best': lbl_best}
        stats[fuel] = {'done': 0, 'total': tasks_per_fuel, 'results': []}

    status_var = tk.StringVar(value="Preparando tareas...")
    tk.Label(root, textvariable=status_var, font=("Segoe UI", 10, "italic"),
            bg="#1a1a2e", fg="#ffd32a").pack(pady=10)

    # === Construir tareas con semillas deterministas ===
    all_tasks = []
    for fuel in FUEL_COLUMNS:
        alphas = data[f'{fuel}_alpha']
        prices = data[f'{fuel}_prices']
        for D in RESERVOIR_D_VALUES:
            for rho in RESERVOIR_RHO_VALUES:
                for leak in RESERVOIR_LEAK_RATES:
                    for r in range(N_GRID_REALIZATIONS):
                        # Semilla DETERMINISTA por (D, rho, leak, r)
                        seed = SEED + r * 1000 + D * 7 + int(rho * 100) + int(leak * 10)
                        all_tasks.append((fuel, alphas, prices, D, rho, leak, seed))
    
    # Shuffle DETERMINISTA (misma semilla → misma orden siempre)
    random.seed(SEED)
    random.shuffle(all_tasks)
    
    # Submit todas al pool
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures_map = {}
    for task in all_tasks:
        future = executor.submit(_single_realization, *task, TRAIN_SIZE, RESERVOIR_WASHOUT)
        futures_map[future] = task[0]
    
    global_start = time.time()
    
    def update_ui():
        done_futures = [f for f in futures_map if f.done()]
        
        for f in done_futures:
            fuel = futures_map.pop(f)
            try:
                res = f.result()
                stats[fuel]['results'].append(res)
            except Exception as e:
                print(f"  ERROR {fuel}: {e}")
            stats[fuel]['done'] += 1
        
        for fuel in FUEL_COLUMNS:
            s = stats[fuel]
            w = widgets[fuel]
            prog = (s['done'] / s['total']) * 100
            w['pbar']['value'] = prog
            w['lbl'].config(text=f"{s['done']}/{s['total']} ({prog:.0f}%)")
            
            if s['results']:
                from itertools import groupby
                sorted_r = sorted(s['results'],
                                  key=lambda x: (x['D'], x['rho'], x['leak_rate']))
                best_rmse = float('inf')
                best_cfg = ''
                for key, group in groupby(
                    sorted_r, key=lambda x: (x['D'], x['rho'], x['leak_rate'])
                ):
                    grp = list(group)
                    mean_rmse = np.mean([g['rmse'] for g in grp])
                    if mean_rmse < best_rmse:
                        best_rmse = mean_rmse
                        D_k, rho_k, leak_k = key
                        leak_s = f",a={leak_k:.1f}" if leak_k != 1.0 else ""
                        best_cfg = f"D={D_k},ρ={rho_k:.2f}{leak_s}"
                w['best'].config(text=f"☆ {best_rmse:.4f} ({best_cfg})")
        
        elapsed = time.time() - global_start
        total_done = sum(s['done'] for s in stats.values())
        rate = total_done / elapsed if elapsed > 0 else 0
        remaining = total_tasks - total_done
        eta = remaining / rate if rate > 0 else 0
        
        status_var.set(
            f"⚡ {total_done}/{total_tasks} | {elapsed:.0f}s | "
            f"{rate:.1f}/s | ETA: {eta:.0f}s"
        )

        if not futures_map:
            elapsed = time.time() - global_start
            status_var.set(f"✅ Completado en {elapsed:.1f}s ({rate:.1f} tareas/s)")
            root.after(1000, finalize)
        else:
            root.after(150, update_ui)

    def finalize():
        from itertools import groupby
        
        print("\n" + "=" * 60)
        print("RESULTADOS GRID SEARCH (NNLS, reproducible)")
        print("=" * 60)
        
        for fuel in FUEL_COLUMNS:
            results = stats[fuel]['results']
            sorted_r = sorted(results,
                              key=lambda x: (x['D'], x['rho'], x['leak_rate']))
            grid = []
            for key, group in groupby(
                sorted_r, key=lambda x: (x['D'], x['rho'], x['leak_rate'])
            ):
                grp = list(group)
                rmse_all = [g['rmse'] for g in grp]
                D_k, rho_k, leak_k = key
                grid.append({
                    'D': D_k, 'rho': rho_k, 'leak_rate': leak_k,
                    'rmse_mean': np.mean(rmse_all),
                    'rmse_std': np.std(rmse_all),
                    'rmse_all': rmse_all
                })
            
            grid_path = os.path.join(MODELS_DIR, f'{fuel.lower()}_ssrc_grid.npz')
            np.savez(grid_path, grid=grid)
            
            best = min(grid, key=lambda x: x['rmse_mean'])
            leak_s = f", a={best['leak_rate']:.1f}" if best['leak_rate'] != 1.0 else ""
            print(f"  {fuel:10s}: RMSE={best['rmse_mean']:.4f}±{best['rmse_std']:.4f} "
                  f"(D={best['D']}, ρ={best['rho']:.2f}{leak_s})")
        
        executor.shutdown(wait=False)
        root.destroy()

    root.after(200, update_ui)
    root.mainloop()
    
    elapsed_total = time.time() - global_start
    print(f"\n✅ Grid Search v3 completado en {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

if __name__ == "__main__":
    main()
