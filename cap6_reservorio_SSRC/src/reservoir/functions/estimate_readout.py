"""
estimate_readout.py — Estimación de la capa de lectura W_out.

Implementa el Teorema 6.3 del capítulo: NNLS para W_out.
También incluye Ridge como alternativa de comparación.
"""
import numpy as np
from scipy.optimize import nnls


def estimate_readout_nnls(
    H: np.ndarray,
    targets: np.ndarray
) -> np.ndarray:
    """
    Estima W_out vía NNLS (Non-Negative Least Squares).

    Resuelve: w_out = argmin_{w >= 0} ||targets - H^T w||^2

    Parámetros
    ----------
    H : ndarray (D, T_eff)
        Matriz de estados del reservorio.
    targets : ndarray (T_eff,)
        Valores objetivo.

    Retorna
    -------
    W_out : ndarray (D,)
        Vector de pesos de lectura no negativos.
    """
    try:
        W_out, _ = nnls(H.T, targets, maxiter=H.shape[0] * H.shape[1] * 5)
        return W_out
    except RuntimeError:
        # Fallback a Ridge si NNLS no converge
        D = H.shape[0]
        W_out = np.linalg.solve(
            H @ H.T + 1e-6 * np.eye(D),
            H @ targets
        )
        return np.maximum(W_out, 0)  # Proyectar a no-negativo


def estimate_readout_ridge(
    H: np.ndarray,
    targets: np.ndarray,
    ridge_alpha: float = 1e-6
) -> np.ndarray:
    """
    Alternativa: W_out vía Ridge Regression (sin restricción >=0).

    W_out = (H H^T + alpha*I)^{-1} H targets

    Parámetros
    ----------
    H : ndarray (D, T_eff)
    targets : ndarray (T_eff,)
    ridge_alpha : float — Regularización. Default: 1e-6.

    Retorna
    -------
    W_out : ndarray (D,)
    """
    D = H.shape[0]
    W_out = np.linalg.solve(
        H @ H.T + ridge_alpha * np.eye(D),
        H @ targets
    )
    return W_out
