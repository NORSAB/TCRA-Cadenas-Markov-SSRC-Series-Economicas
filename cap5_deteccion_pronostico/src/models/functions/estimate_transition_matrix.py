import numpy as np
from .srep_estimator import srep_estimator

def estimate_transition_matrix(states, k):
    """
    Estimates the transition matrix using SRep (Robust NNLS).
    Estima la matriz de transición usando SRep (NNLS Robusto).
    
    Args:
        states (np.array): Sequence of discrete states.
        k (int): Number of possible states.
        
    Returns:
        tuple: (P_matrix, count_matrix)
    """
    # 1. One-hot encoding
    num_samples = len(states)
    one_hot = np.zeros((k, num_samples))
    one_hot[states, np.arange(num_samples)] = 1
    
    # 2. SRep Estimation
    if num_samples > 1:
        P_matrix = srep_estimator(one_hot, num_samples - 1)
    else:
        P_matrix = np.full((k, k), 1.0/k)
        
    # 3. Calculation of Count Matrix for AIC
    C = np.zeros((k, k), dtype=int)
    from_states = states[:-1]
    to_states = states[1:]
    np.add.at(C, (to_states, from_states), 1)
    
    return P_matrix, C
