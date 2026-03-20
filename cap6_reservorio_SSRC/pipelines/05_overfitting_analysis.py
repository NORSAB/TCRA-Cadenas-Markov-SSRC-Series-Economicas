"""
===================================================================
Análisis de Overfitting por Dimensión D + Comparación NNLS vs Ridge
===================================================================
Genera tablas CSV detalladas para validar si D=150 es genuinamente 
óptimo o es un falso positivo (overfitting).

Métricas de diagnóstico:
  1. RMSE por D (promedio sobre ρ y a)
  2. Varianza por D (alta varianza = overfitting)
  3. Ratio D/T_eff (si > 0.5 → riesgo de sobreajuste)
  4. Comparación NNLS vs Ridge (mismos seeds)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import time
import numpy as np
import csv
from concurrent.futures import ProcessPoolExecutor
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import (FUEL_COLUMNS, MODELS_DIR, OUTPUT_DIR,
                    RESERVOIR_D_VALUES, RESERVOIR_RHO_VALUES,
                    RESERVOIR_LEAK_RATES,
                    N_GRID_REALIZATIONS, RESERVOIR_WASHOUT, TRAIN_SIZE, SEED)

# Numba 
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


def _run_realization(fuel, alphas, prices, D, rho, leak_rate, seed,
                      train_size, washout, solver='nnls'):
    """
    Una realización con solver seleccionable (NNLS o Ridge).
    Retorna RMSE + diagnósticos de overfitting.
    """
    from scipy.optimize import nnls
    from reservoir.functions.create_reservoir import create_reservoir
    
    W_in, W_res = create_reservoir(
        input_dim=1, reservoir_dim=D, spectral_radius=rho,
        sparsity=0.9, seed=seed
    )
    
    T = len(alphas)
    n_test = T - train_size - 1
    T_eff = train_size - washout - 1  # T efectivo del primer window
    ratio_D_Teff = D / T_eff
    
    # Pre-compute H_full
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
    
    predictions = np.zeros(n_test)
    actuals = np.zeros(n_test)
    n_nonzero_weights = []
    train_rmse_list = []
    
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
            
            if solver == 'nnls':
                try:
                    W_out, residual = nnls(H_aligned.T, targets_a,
                                           maxiter=D * min_len * 5)
                except RuntimeError:
                    reg = 1e-6 * np.eye(D)
                    W_out = np.linalg.solve(
                        H_aligned @ H_aligned.T + reg, H_aligned @ targets_a)
                    W_out = np.maximum(W_out, 0)
            else:  # Ridge
                reg = 1e-6 * np.eye(D)
                W_out = np.linalg.solve(
                    H_aligned @ H_aligned.T + reg, H_aligned @ targets_a)
            
            # Diagnósticos
            n_nonzero_weights.append(np.sum(np.abs(W_out) > 1e-10))
            train_pred = W_out @ H_aligned
            train_rmse = np.sqrt(np.mean((train_pred - targets_a)**2))
            train_rmse_list.append(train_rmse)
            
            # Predict
            h_predict = H_full[:, train_end]
            alpha_pred = float(W_out @ h_predict)
            predictions[i] = prices[train_end] * (1.0 + alpha_pred)
        except Exception:
            predictions[i] = prices[train_end]
    
    errors = actuals - predictions
    test_rmse = float(np.sqrt(np.mean(errors**2)))
    avg_train_rmse = float(np.mean(train_rmse_list)) if train_rmse_list else float('inf')
    avg_nonzero = float(np.mean(n_nonzero_weights)) if n_nonzero_weights else 0
    
    # Gap = train_rmse vs test_rmse (overfitting indicator)
    gap = test_rmse - avg_train_rmse if avg_train_rmse < float('inf') else 0
    
    return {
        'fuel': fuel, 'D': D, 'rho': rho, 'leak_rate': leak_rate,
        'solver': solver,
        'test_rmse': test_rmse,
        'train_rmse': avg_train_rmse,
        'gap': gap,
        'ratio_D_Teff': ratio_D_Teff,
        'avg_nonzero_weights': avg_nonzero,
        'seed': seed
    }


def main():
    print("=" * 70)
    print("ANÁLISIS DE OVERFITTING y COMPARACIÓN NNLS vs RIDGE")
    print("=" * 70)
    
    data_path = os.path.join(MODELS_DIR, 'prepared_data.npz')
    data = np.load(data_path)
    
    # Warmup Numba
    if NUMBA_OK:
        _propagate_jit(np.zeros(10), np.zeros(5), np.zeros((5,5)), 1.0, 5, 10)
        print("  ⚡ Numba JIT listo")
    
    tables_dir = os.path.join(OUTPUT_DIR, 'tables')
    os.makedirs(tables_dir, exist_ok=True)
    
    max_workers = min(16, os.cpu_count() or 4)
    executor = ProcessPoolExecutor(max_workers=max_workers)
    
    # =====================================================
    # FASE 1: Correr NNLS y Ridge en paralelo (mismos seeds)
    # =====================================================
    print("\n[FASE 1] Ejecutando NNLS y Ridge con mismos seeds...")
    
    all_futures = []
    for fuel in FUEL_COLUMNS:
        alphas = data[f'{fuel}_alpha']
        prices = data[f'{fuel}_prices']
        for D in RESERVOIR_D_VALUES:
            for rho in RESERVOIR_RHO_VALUES:
                for leak in RESERVOIR_LEAK_RATES:
                    for r in range(N_GRID_REALIZATIONS):
                        seed = SEED + r * 1000 + D * 7 + int(rho * 100) + int(leak * 10)
                        # NNLS
                        f1 = executor.submit(_run_realization, fuel, alphas, prices,
                                            D, rho, leak, seed, TRAIN_SIZE, 
                                            RESERVOIR_WASHOUT, 'nnls')
                        all_futures.append(f1)
                        # Ridge (mismos seeds!)
                        f2 = executor.submit(_run_realization, fuel, alphas, prices,
                                            D, rho, leak, seed, TRAIN_SIZE,
                                            RESERVOIR_WASHOUT, 'ridge')
                        all_futures.append(f2)
    
    total = len(all_futures)
    print(f"  Total tareas: {total} (NNLS + Ridge)")
    
    t0 = time.time()
    results_all = []
    done = 0
    for f in all_futures:
        try:
            res = f.result()
            results_all.append(res)
        except Exception as e:
            print(f"  Error: {e}")
        done += 1
        if done % 500 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total - done) / rate
            print(f"  {done}/{total} ({elapsed:.0f}s, ETA {eta:.0f}s)")
    
    elapsed = time.time() - t0
    print(f"  ✅ Completado en {elapsed:.0f}s")
    
    executor.shutdown()
    
    # =====================================================
    # FASE 2: Generar tablas CSV
    # =====================================================
    from itertools import groupby
    
    # ----- TABLA 1: RMSE por D (promedio sobre ρ,a,realizaciones) -----
    print("\n[TABLA 1] RMSE por D — Diagnóstico de overfitting...")
    
    csv_path = os.path.join(tables_dir, 'T_overfitting_by_D.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Fuel', 'D', 'Solver', 'TestRMSE_mean', 'TestRMSE_std',
                     'TrainRMSE_mean', 'Gap_mean', 'Ratio_D_Teff',
                     'AvgNonzeroW', 'N_configs'])
        
        for fuel in FUEL_COLUMNS:
            for solver in ['nnls', 'ridge']:
                fuel_solver = [r for r in results_all 
                               if r['fuel'] == fuel and r['solver'] == solver]
                
                for D in RESERVOIR_D_VALUES:
                    d_results = [r for r in fuel_solver if r['D'] == D]
                    if not d_results:
                        continue
                    
                    test_rmses = [r['test_rmse'] for r in d_results]
                    train_rmses = [r['train_rmse'] for r in d_results]
                    gaps = [r['gap'] for r in d_results]
                    nonzeros = [r['avg_nonzero_weights'] for r in d_results]
                    ratio = d_results[0]['ratio_D_Teff']
                    
                    w.writerow([
                        fuel, D, solver.upper(),
                        f"{np.mean(test_rmses):.6f}",
                        f"{np.std(test_rmses):.6f}",
                        f"{np.mean(train_rmses):.6f}",
                        f"{np.mean(gaps):.6f}",
                        f"{ratio:.4f}",
                        f"{np.mean(nonzeros):.1f}",
                        len(d_results)
                    ])
    
    print(f"  → {csv_path}")
    
    # ----- TABLA 2: Top 10 configs por fuel (NNLS vs Ridge) -----
    print("\n[TABLA 2] Top 10 configs NNLS vs Ridge por fuel...")
    
    csv_path = os.path.join(tables_dir, 'T_top10_nnls_vs_ridge.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Fuel', 'Rank', 'Solver', 'D', 'rho', 'leak_rate',
                     'TestRMSE_mean', 'TestRMSE_std', 'TrainRMSE_mean',
                     'Gap', 'AvgNonzeroW'])
        
        for fuel in FUEL_COLUMNS:
            for solver in ['nnls', 'ridge']:
                fuel_solver = [r for r in results_all 
                               if r['fuel'] == fuel and r['solver'] == solver]
                
                sorted_r = sorted(fuel_solver,
                                  key=lambda x: (x['D'], x['rho'], x['leak_rate']))
                configs = []
                for key, group in groupby(
                    sorted_r, key=lambda x: (x['D'], x['rho'], x['leak_rate'])
                ):
                    grp = list(group)
                    D_k, rho_k, leak_k = key
                    configs.append({
                        'D': D_k, 'rho': rho_k, 'leak_rate': leak_k,
                        'test_rmse_mean': np.mean([g['test_rmse'] for g in grp]),
                        'test_rmse_std': np.std([g['test_rmse'] for g in grp]),
                        'train_rmse_mean': np.mean([g['train_rmse'] for g in grp]),
                        'gap': np.mean([g['gap'] for g in grp]),
                        'avg_nonzero': np.mean([g['avg_nonzero_weights'] for g in grp])
                    })
                
                configs.sort(key=lambda x: x['test_rmse_mean'])
                
                for rank, c in enumerate(configs[:10], 1):
                    leak_s = f"{c['leak_rate']:.1f}"
                    w.writerow([
                        fuel, rank, solver.upper(),
                        c['D'], f"{c['rho']:.2f}", leak_s,
                        f"{c['test_rmse_mean']:.6f}",
                        f"{c['test_rmse_std']:.6f}",
                        f"{c['train_rmse_mean']:.6f}",
                        f"{c['gap']:.6f}",
                        f"{c['avg_nonzero']:.1f}"
                    ])
    
    print(f"  → {csv_path}")
    
    # ----- TABLA 3: Comparación directa NNLS vs Ridge (mejor config) -----
    print("\n[TABLA 3] Comparación directa NNLS vs Ridge...")
    
    csv_path = os.path.join(tables_dir, 'T_nnls_vs_ridge_comparison.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Fuel', 'Best_NNLS_D', 'Best_NNLS_rho', 'Best_NNLS_leak',
                     'NNLS_TestRMSE', 'NNLS_TrainRMSE', 'NNLS_Gap', 'NNLS_NonzeroW',
                     'Best_Ridge_D', 'Best_Ridge_rho', 'Best_Ridge_leak',
                     'Ridge_TestRMSE', 'Ridge_TrainRMSE', 'Ridge_Gap',
                     'Delta_NNLS_vs_Ridge_pct', 'Ganador'])
        
        for fuel in FUEL_COLUMNS:
            for solver in ['nnls', 'ridge']:
                fuel_solver = [r for r in results_all 
                               if r['fuel'] == fuel and r['solver'] == solver]
                sorted_r = sorted(fuel_solver,
                                  key=lambda x: (x['D'], x['rho'], x['leak_rate']))
                configs = []
                for key, group in groupby(
                    sorted_r, key=lambda x: (x['D'], x['rho'], x['leak_rate'])
                ):
                    grp = list(group)
                    D_k, rho_k, leak_k = key
                    configs.append({
                        'D': D_k, 'rho': rho_k, 'leak_rate': leak_k,
                        'test_rmse_mean': np.mean([g['test_rmse'] for g in grp]),
                        'train_rmse_mean': np.mean([g['train_rmse'] for g in grp]),
                        'gap': np.mean([g['gap'] for g in grp]),
                        'avg_nonzero': np.mean([g['avg_nonzero_weights'] for g in grp])
                    })
                
                if solver == 'nnls':
                    best_nnls = min(configs, key=lambda x: x['test_rmse_mean'])
                else:
                    best_ridge = min(configs, key=lambda x: x['test_rmse_mean'])
            
            delta = (best_nnls['test_rmse_mean'] - best_ridge['test_rmse_mean']) / best_ridge['test_rmse_mean'] * 100
            ganador = 'NNLS' if best_nnls['test_rmse_mean'] < best_ridge['test_rmse_mean'] else 'Ridge'
            
            w.writerow([
                fuel,
                best_nnls['D'], f"{best_nnls['rho']:.2f}", f"{best_nnls['leak_rate']:.1f}",
                f"{best_nnls['test_rmse_mean']:.6f}",
                f"{best_nnls['train_rmse_mean']:.6f}",
                f"{best_nnls['gap']:.6f}",
                f"{best_nnls['avg_nonzero']:.1f}",
                best_ridge['D'], f"{best_ridge['rho']:.2f}", f"{best_ridge['leak_rate']:.1f}",
                f"{best_ridge['test_rmse_mean']:.6f}",
                f"{best_ridge['train_rmse_mean']:.6f}",
                f"{best_ridge['gap']:.6f}",
                f"{delta:+.2f}%",
                ganador
            ])
    
    print(f"  → {csv_path}")
    
    # ----- TABLA 4: Grid completa NNLS para audit trail -----
    print("\n[TABLA 4] Grid completa NNLS (audit trail)...")
    
    csv_path = os.path.join(tables_dir, 'T_full_grid_nnls.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Fuel', 'D', 'rho', 'leak_rate', 'TestRMSE_mean', 'TestRMSE_std',
                     'TrainRMSE_mean', 'Gap', 'AvgNonzeroW', 'Ratio_D_Teff', 'N_realizations'])
        
        for fuel in FUEL_COLUMNS:
            fuel_nnls = [r for r in results_all 
                         if r['fuel'] == fuel and r['solver'] == 'nnls']
            sorted_r = sorted(fuel_nnls,
                              key=lambda x: (x['D'], x['rho'], x['leak_rate']))
            
            for key, group in groupby(
                sorted_r, key=lambda x: (x['D'], x['rho'], x['leak_rate'])
            ):
                grp = list(group)
                D_k, rho_k, leak_k = key
                w.writerow([
                    fuel, D_k, f"{rho_k:.2f}", f"{leak_k:.1f}",
                    f"{np.mean([g['test_rmse'] for g in grp]):.6f}",
                    f"{np.std([g['test_rmse'] for g in grp]):.6f}",
                    f"{np.mean([g['train_rmse'] for g in grp]):.6f}",
                    f"{np.mean([g['gap'] for g in grp]):.6f}",
                    f"{np.mean([g['avg_nonzero_weights'] for g in grp]):.1f}",
                    f"{grp[0]['ratio_D_Teff']:.4f}",
                    len(grp)
                ])
    
    print(f"  → {csv_path}")
    
    # =====================================================
    # RESUMEN EN CONSOLA
    # =====================================================
    print("\n" + "=" * 70)
    print("DIAGNÓSTICO DE OVERFITTING POR D")
    print("=" * 70)
    
    for fuel in FUEL_COLUMNS:
        print(f"\n  {fuel}:")
        print(f"  {'D':>5} {'TestRMSE':>10} {'TrainRMSE':>10} {'Gap':>8} "
              f"{'Std':>8} {'D/Teff':>7} {'NonZeroW':>9}")
        print(f"  {'—'*5} {'—'*10} {'—'*10} {'—'*8} {'—'*8} {'—'*7} {'—'*9}")
        
        fuel_nnls = [r for r in results_all 
                     if r['fuel'] == fuel and r['solver'] == 'nnls']
        
        for D in RESERVOIR_D_VALUES:
            d_results = [r for r in fuel_nnls if r['D'] == D]
            if not d_results:
                continue
            
            test_mean = np.mean([r['test_rmse'] for r in d_results])
            train_mean = np.mean([r['train_rmse'] for r in d_results])
            gap_mean = np.mean([r['gap'] for r in d_results])
            test_std = np.std([r['test_rmse'] for r in d_results])
            ratio = d_results[0]['ratio_D_Teff']
            nonzero = np.mean([r['avg_nonzero_weights'] for r in d_results])
            
            # Flag overfitting
            flag = ""
            if ratio > 0.5:
                flag += " ⚠D/Teff"
            if gap_mean > 0.5:
                flag += " ⚠GAP"
            if test_std > 0.05:
                flag += " ⚠VAR"
            
            print(f"  {D:5d} {test_mean:10.4f} {train_mean:10.4f} {gap_mean:8.4f} "
                  f"{test_std:8.4f} {ratio:7.3f} {nonzero:9.1f}{flag}")
    
    print("\n" + "=" * 70)
    print("COMPARACIÓN NNLS vs RIDGE (mejor config por fuel)")
    print("=" * 70)
    
    for fuel in FUEL_COLUMNS:
        for solver in ['nnls', 'ridge']:
            fuel_solver = [r for r in results_all 
                           if r['fuel'] == fuel and r['solver'] == solver]
            sorted_r = sorted(fuel_solver,
                              key=lambda x: (x['D'], x['rho'], x['leak_rate']))
            configs = []
            for key, group in groupby(
                sorted_r, key=lambda x: (x['D'], x['rho'], x['leak_rate'])
            ):
                grp = list(group)
                configs.append({
                    'D': key[0], 'rho': key[1], 'leak': key[2],
                    'rmse': np.mean([g['test_rmse'] for g in grp]),
                    'nonzero': np.mean([g['avg_nonzero_weights'] for g in grp])
                })
            best = min(configs, key=lambda x: x['rmse'])
            leak_s = f", a={best['leak']:.1f}" if best['leak'] != 1.0 else ""
            print(f"  {fuel:10s} {solver.upper():5s}: RMSE={best['rmse']:.4f} "
                  f"(D={best['D']}, ρ={best['rho']:.2f}{leak_s}) "
                  f"NonZeroW={best['nonzero']:.0f}/{best['D']}")
    
    print(f"\n✅ Análisis completado. Tablas en: {tables_dir}")


if __name__ == "__main__":
    main()
