"""
predict_ssrc.py — Genera pronósticos: y_hat = W_out^T h_t
"""
import numpy as np


def predict_ssrc(
    W_out: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    """
    Genera pronósticos del SSRC.

    Parámetros
    ----------
    W_out : ndarray (D,)
    H : ndarray (D, T_eff)

    Retorna
    -------
    predictions : ndarray (T_eff,)
    """
    return W_out @ H
