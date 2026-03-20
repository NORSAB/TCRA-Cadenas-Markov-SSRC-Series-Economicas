"""
rolling_window_ssrc.py — Validación por ventanas deslizantes para SSRC.

Mismo protocolo que TCROC-Markov (Capítulo anterior).
Optimizado con Leaky-ESN y Ridge adaptativo.
"""
import numpy as np
import warnings
from typing import Dict

from reservoir.functions.propagate_reservoir import propagate_reservoir
from reservoir.functions.estimate_readout import estimate_readout_nnls, estimate_readout_ridge


def run_ssrc_single_window(
    alpha_train: np.ndarray,
    alpha_test_point: float,
    price_current: float,
    W_in: np.ndarray,
    W_res: np.ndarray,
    washout: int,
    augment_input: bool = False,
    use_nnls: bool = True,
    ridge_alpha: float = 1e-6,
    leak_rate: float = 1.0
) -> float:
    """
    Ejecuta SSRC para una ventana de entrenamiento y predice
    el siguiente precio.

    Retorna
    -------
    price_predicted : float
    """
    # 1. Propagar reservorio con datos de entrenamiento
    H_train = propagate_reservoir(
        alpha_train, W_in, W_res, washout, augment_input,
        leak_rate=leak_rate
    )

    # 2. Targets: alpha_{t+1}
    targets = alpha_train[washout + 1:]
    H_aligned = H_train[:, :-1]

    if H_aligned.shape[1] < H_aligned.shape[0]:
        warnings.warn("T_eff < D: sistema subdeterminado.")

    if H_aligned.shape[1] == 0 or len(targets) == 0:
        return price_current  # Fallback

    # Alinear dimensiones
    min_len = min(H_aligned.shape[1], len(targets))
    H_aligned = H_aligned[:, :min_len]
    targets = targets[:min_len]

    # 3. Estimar W_out (Ridge adaptativo basado en número de condición)
    if use_nnls:
        W_out = estimate_readout_nnls(H_aligned, targets)
    else:
        # Ridge adaptativo: alpha proporcional a kappa(H)
        try:
            kappa = np.linalg.cond(H_aligned.T)
            adaptive_alpha = ridge_alpha * min(kappa / 1e6, 1.0)
        except:
            adaptive_alpha = ridge_alpha
        W_out = estimate_readout_ridge(H_aligned, targets, adaptive_alpha)

    # 4. Propagar un paso más con leaky integrator
    h_last = H_train[:, -1]
    if augment_input:
        u_new = np.array([alpha_test_point, alpha_test_point**2])
    else:
        u_new = np.array([alpha_test_point])

    a = leak_rate
    h_new = (1.0 - a) * h_last + a * np.tanh(W_in @ u_new + W_res @ h_last)

    # 5. Predecir alpha_{t+1} y reconstruir precio
    alpha_predicted = W_out @ h_new
    price_predicted = price_current * (1.0 + alpha_predicted)

    return price_predicted


def run_ssrc_rolling_window(
    alpha_full: np.ndarray,
    prices_full: np.ndarray,
    train_size: int,
    W_in: np.ndarray,
    W_res: np.ndarray,
    washout: int = 50,
    augment_input: bool = False,
    use_nnls: bool = True,
    ridge_alpha: float = 1e-6,
    leak_rate: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Ejecuta validación por ventanas deslizantes.
    Protocolo idéntico al del capítulo de combustibles.

    Retorna
    -------
    results : dict con 'predictions', 'actuals', 'rmse', 'errors'
    """
    T = len(alpha_full)
    n_test = T - train_size - 1

    if n_test <= 0:
        return {'predictions': np.array([]), 'actuals': np.array([]),
                'rmse': np.inf, 'errors': np.array([])}

    predictions = np.zeros(n_test)
    actuals = np.zeros(n_test)

    for i in range(n_test):
        train_end = train_size + i
        alpha_train = alpha_full[:train_end]
        alpha_test = alpha_full[train_end]

        price_current = prices_full[train_end]
        price_actual = prices_full[train_end + 1]

        try:
            pred = run_ssrc_single_window(
                alpha_train=alpha_train,
                alpha_test_point=alpha_test,
                price_current=price_current,
                W_in=W_in,
                W_res=W_res,
                washout=washout,
                augment_input=augment_input,
                use_nnls=use_nnls,
                ridge_alpha=ridge_alpha,
                leak_rate=leak_rate
            )
        except Exception as e:
            warnings.warn("Error en ventana {}: {}. Usando naive.".format(i, e))
            pred = price_current

        predictions[i] = pred
        actuals[i] = price_actual

    errors = actuals - predictions
    rmse = np.sqrt(np.mean(errors**2))

    return {
        'predictions': predictions,
        'actuals': actuals,
        'rmse': rmse,
        'errors': errors
    }
