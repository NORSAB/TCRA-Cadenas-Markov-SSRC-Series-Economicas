import numpy as np

def calculate_aic(states, k, P_matrix, C_matrix=None):
    """
    Calcula el Criterio de Información de Akaike (AIC).
    Calculates the Akaike Information Criterion (AIC).
    
    Args:
        states (array): Estados observados.
        k (int): Número de estados.
        P_matrix (matrix): Matriz de transición Pr[j,i].
        C_matrix (matrix): Matriz de conteos C[j,i]. Opional.
        
    Returns:
        float: Valor AIC.
    """
    if C_matrix is None:
        C_matrix = np.zeros((k, k), dtype=int)
        from_states = states[:-1]
        to_states = states[1:]
        np.add.at(C_matrix, (to_states, from_states), 1)

    # Use the logic from Articulol.py
    # log_likelihood = sum(C * log(P))
    # We add a small epsilon to P to avoid log(0)
    log_likelihood = np.sum(C_matrix[C_matrix > 0] * np.log(P_matrix[C_matrix > 0] + 1e-12))
    
    if not np.isfinite(log_likelihood):
        return np.inf

    num_params = float(k * (k - 1))
    aic = 2 * num_params - 2 * log_likelihood
    
    return aic
