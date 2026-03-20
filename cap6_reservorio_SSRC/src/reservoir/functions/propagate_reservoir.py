"""
propagate_reservoir.py — Propaga la señal alpha_t a través del reservorio.

Implementa la ecuación de actualización Leaky-ESN:
    h_t = (1 - a) * h_{t-1} + a * tanh(W_in * u_t + W_res * h_{t-1})

Cuando a = 1.0, se reduce al ESN clásico (Definición 6.3).
"""
import numpy as np


def propagate_reservoir(
    alpha_series: np.ndarray,
    W_in: np.ndarray,
    W_res: np.ndarray,
    washout: int = 50,
    augment_input: bool = False,
    leak_rate: float = 1.0
) -> np.ndarray:
    """
    Propaga la señal TCROC alpha_t a través del reservorio.

    Parámetros
    ----------
    alpha_series : ndarray (T,)
        Serie TCROC alpha_t.
    W_in : ndarray (D, d)
        Matriz de entrada del reservorio.
    W_res : ndarray (D, D)
        Matriz de reservorio.
    washout : int
        Pasos iniciales a descartar (transitorios). Default: 50.
    augment_input : bool
        Si True, u_t = [alpha_t, alpha_t^2]. Default: False.
    leak_rate : float
        Factor de fuga 'a' del Leaky Integrator (0 < a <= 1).
        a = 1.0 → ESN clásico (sin fuga).
        a < 1.0 → Memoria más larga (suavizado exponencial).

    Retorna
    -------
    H : ndarray (D, T_effective)
        Matriz de estados post-washout. T_effective = T - washout.
    """
    T = len(alpha_series)
    D = W_res.shape[0]

    # Construir secuencia de entradas
    if augment_input:
        U = np.column_stack([alpha_series, alpha_series**2])
    else:
        U = alpha_series.reshape(-1, 1)

    # Propagar: h_0 = 0
    H_all = np.zeros((D, T))
    h = np.zeros(D)
    a = leak_rate

    for t in range(T):
        # Leaky Integrator ESN (Jaeger, 2007):
        # h_t = (1-a)*h_{t-1} + a*tanh(W_in*u_t + W_res*h_{t-1})
        h = (1.0 - a) * h + a * np.tanh(W_in @ U[t] + W_res @ h)
        H_all[:, t] = h

    # Descartar washout
    H = H_all[:, washout:]
    return H
