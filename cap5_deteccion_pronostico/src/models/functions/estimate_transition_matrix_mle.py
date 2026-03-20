import numpy as np

def estimate_transition_matrix_mle(states, k):
    """
    Estimates the transition matrix using Maximum Likelihood Estimation (Simple Counting).
    Used primarily for AIC calculation in Grid Search where robust SRep may be overkill or lack direct interpretation.
    
    Args:
        states (np.array): Sequence of discrete states.
        k (int): Number of possible states.
        
    Returns:
        tuple: (P_matrix, count_matrix)
    """
    P = np.zeros((k, k))
    C = np.zeros((k, k))
    # Optimized Vectorization (eliminates slow Python loop)
    # P[next_state, current_state]
    if len(states) > 1:
        # states[:-1] = current states (from)
        # states[1:]  = next states (to)
        np.add.at(C, (states[1:], states[:-1]), 1)
    
    # Normalize columns to get probabilities
    col_sums = C.sum(axis=0)
    # Avoid division by zero
    P = np.divide(C, col_sums, out=np.zeros_like(C), where=col_sums!=0)
    
    return P, C
