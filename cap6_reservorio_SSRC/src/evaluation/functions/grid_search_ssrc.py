"""
grid_search_ssrc.py — Búsqueda en grilla PARALELA de hiperparámetros.

Para cada (D, rho, leak_rate), ejecuta N realizaciones con diferentes 
semillas y reporta media ± std del RMSE.

Usa ProcessPoolExecutor para paralelizar entre realizaciones.
"""
import numpy as np
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from reservoir.functions.create_reservoir import create_reservoir
from evaluation.functions.rolling_window_ssrc import run_ssrc_rolling_window


def _evaluate_single_realization(args):
    """Función atómica para ProcessPoolExecutor."""
    alpha_full, prices_full, train_size, D, rho, leak_rate, washout, \
        augment_input, use_nnls, seed = args

    input_dim = 2 if augment_input else 1
    W_in, W_res = create_reservoir(
        input_dim=input_dim,
        reservoir_dim=D,
        spectral_radius=rho,
        input_scale=1.0,
        sparsity=0.9,
        seed=seed
    )

    res = run_ssrc_rolling_window(
        alpha_full=alpha_full,
        prices_full=prices_full,
        train_size=train_size,
        W_in=W_in,
        W_res=W_res,
        washout=washout,
        augment_input=augment_input,
        use_nnls=use_nnls,
        leak_rate=leak_rate
    )

    return res['rmse']


def grid_search_ssrc(
    alpha_full: np.ndarray,
    prices_full: np.ndarray,
    train_size: int,
    D_values: List[int] = [20, 50, 75, 100, 150],
    rho_values: List[float] = [0.70, 0.80, 0.85, 0.90, 0.95, 0.99],
    leak_rates: List[float] = [1.0],
    n_realizations: int = 30,
    washout: int = 50,
    augment_input: bool = False,
    use_nnls: bool = True,
    parallel: bool = True,
    verbose: bool = True
) -> Dict:
    """
    Búsqueda en grilla sobre hiperparámetros del reservorio.
    
    Ahora incluye leak_rate y paralelización por realizaciones.

    Retorna
    -------
    results : dict con 'grid', 'best'
    """
    grid_results = []
    total_configs = len(D_values) * len(rho_values) * len(leak_rates)
    config_idx = 0

    max_workers = max(1, multiprocessing.cpu_count() - 1) if parallel else 1

    for D in D_values:
        for rho in rho_values:
            for leak in leak_rates:
                config_idx += 1

                # Preparar argumentos para cada realización
                args_list = []
                for r in range(n_realizations):
                    seed = r * 1000 + int(D * 100 + rho * 1000 + leak * 100)
                    args_list.append((
                        alpha_full, prices_full, train_size,
                        D, rho, leak, washout,
                        augment_input, use_nnls, seed
                    ))

                # Ejecutar realizaciones en paralelo
                if parallel and n_realizations > 1:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(_evaluate_single_realization, a) for a in args_list]
                        rmse_list = [f.result() for f in futures]
                else:
                    rmse_list = [_evaluate_single_realization(a) for a in args_list]

                rmse_mean = np.mean(rmse_list)
                rmse_std = np.std(rmse_list)

                entry = {
                    'D': D,
                    'rho': rho,
                    'leak_rate': leak,
                    'rmse_mean': rmse_mean,
                    'rmse_std': rmse_std,
                    'rmse_all': rmse_list
                }
                grid_results.append(entry)

                if verbose:
                    leak_str = f", a={leak:.1f}" if leak != 1.0 else ""
                    print("  [{}/{}] D={:3d}, rho={:.2f}{} -> RMSE = {:.4f} +/- {:.4f}".format(
                        config_idx, total_configs,
                        D, rho, leak_str, rmse_mean, rmse_std))

    best_idx = np.argmin([r['rmse_mean'] for r in grid_results])
    best = grid_results[best_idx]

    return {
        'grid': grid_results,
        'best': best,
    }
