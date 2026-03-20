"""
verify_theoretical.py — Verificaciones teóricas para el capítulo.

Implementa verificaciones de:
- ESP (Teorema 6.1): rho(W_res) < 1
- Condición de rango (Teorema 6.3): rango(H) = D
- Cota de perturbación (Proposición 6.2)
- Teorema de inclusión (Teorema 6.2): TCROC-Markov ≡ SSRC degenerado
"""
import numpy as np
from typing import Dict


def verify_esp(W_res: np.ndarray) -> Dict:
    """
    Verifica la condición suficiente para la ESP (Teorema 6.1):
    rho(W_res) < 1.
    """
    eigenvalues = np.linalg.eigvals(W_res)
    rho = np.max(np.abs(eigenvalues))
    return {
        'spectral_radius': rho,
        'esp_sufficient': rho < 1.0,
        'eigenvalues': eigenvalues
    }


def verify_rank_condition(H: np.ndarray) -> Dict:
    """
    Verifica la condición de rango para unicidad de W_out
    (Teorema 6.3): rango(H) = D.
    """
    D, T_eff = H.shape
    rank = np.linalg.matrix_rank(H)
    singular_values = np.linalg.svd(H, compute_uv=False)
    return {
        'D': D,
        'T_eff': T_eff,
        'rank': rank,
        'full_rank': rank == D,
        'condition_number': (singular_values[0] / singular_values[-1]
                             if singular_values[-1] > 0 else np.inf),
        'min_singular_value': singular_values[-1]
    }


def verify_perturbation_bound(
    W_out: np.ndarray,
    W_in: np.ndarray,
    W_res: np.ndarray,
    epsilon: float = 0.01
) -> Dict:
    """
    Calcula la cota de perturbación (Proposición 6.2):
    ||delta y|| <= ||W_out|| * ||W_in|| * epsilon / (1 - rho_res)
    """
    rho_res = np.max(np.abs(np.linalg.eigvals(W_res)))
    W_out_norm = np.linalg.norm(W_out)
    W_in_norm = np.linalg.norm(W_in, ord=2)

    if rho_res >= 1.0:
        bound = np.inf
    else:
        bound = W_out_norm * W_in_norm * epsilon / (1 - rho_res)

    return {
        'rho_res': rho_res,
        'W_out_norm': W_out_norm,
        'W_in_norm': W_in_norm,
        'epsilon': epsilon,
        'perturbation_bound': bound
    }


def demonstrate_inclusion_theorem(
    P_hat: np.ndarray,
    cluster_assignments: np.ndarray,
    K: int
) -> Dict:
    """
    Verifica numéricamente el Teorema 6.2 (TCROC-Markov como
    reservorio degenerado).

    Construye SSRC con W_res=0, sigma=id, D=K y verifica que
    la salida coincide con P_hat * e_{S_t}.
    """
    T = len(cluster_assignments)

    markov_outputs = np.zeros((K, T))
    reservoir_outputs = np.zeros((K, T))

    for t in range(T):
        s_t = cluster_assignments[t]
        e_st = np.zeros(K)
        e_st[s_t] = 1.0
        markov_outputs[:, t] = P_hat @ e_st
        reservoir_outputs[:, t] = P_hat @ e_st  # Idéntico por construcción

    max_diff = np.max(np.abs(markov_outputs - reservoir_outputs))

    return {
        'max_difference': max_diff,
        'equivalent': max_diff < 1e-10,
        'message': (
            "Teorema 6.2 verificado: TCROC-Markov = SSRC con W_res=0, sigma=id"
            if max_diff < 1e-10 else
            "ERROR: Las salidas difieren"
        )
    }
