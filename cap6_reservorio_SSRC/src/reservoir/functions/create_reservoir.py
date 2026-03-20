"""
create_reservoir.py — Genera matrices fijas W_in y W_res.

Corresponde a Definición 6.3 (SSRC) del capítulo de tesis.
"""
import numpy as np
from typing import Tuple, Optional


def create_reservoir(
    input_dim: int,
    reservoir_dim: int,
    spectral_radius: float,
    input_scale: float = 1.0,
    sparsity: float = 0.9,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera las matrices fijas del reservorio (W_in, W_res).

    Parámetros
    ----------
    input_dim : int
        Dimensión de la entrada (d). Para TCROC escalar, d=1.
    reservoir_dim : int
        Dimensión del reservorio (D). Típico: 20, 50, 100.
    spectral_radius : float
        Radio espectral objetivo (rho_res). Debe ser < 1 para ESP.
    input_scale : float
        Escala omega de W_in. Default: 1.0.
    sparsity : float
        Fracción de ceros en W_res. Default: 0.9.
    seed : int, optional
        Semilla para reproducibilidad.

    Retorna
    -------
    W_in : ndarray (D, d) — Matriz de entrada fija.
    W_res : ndarray (D, D) — Matriz de reservorio reescalada.
    """
    rng = np.random.RandomState(seed)

    # W_in: Uniforme en [-omega, omega]
    W_in = rng.uniform(-input_scale, input_scale,
                       size=(reservoir_dim, input_dim))

    # W_res: Dispersa aleatoria, reescalada espectralmente
    W_res_raw = rng.randn(reservoir_dim, reservoir_dim)
    mask = rng.rand(reservoir_dim, reservoir_dim) > sparsity
    W_res_raw *= mask

    # Reescalar: W_res ← rho_res * W_res^(0) / rho(W_res^(0))
    eigenvalues = np.linalg.eigvals(W_res_raw)
    rho_current = np.max(np.abs(eigenvalues))

    if rho_current > 0:
        W_res = spectral_radius * W_res_raw / rho_current
    else:
        W_res = W_res_raw

    return W_in, W_res
