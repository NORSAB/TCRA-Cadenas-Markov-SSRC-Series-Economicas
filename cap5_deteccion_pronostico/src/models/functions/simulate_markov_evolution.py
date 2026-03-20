import numpy as np

def simulate_markov_evolution(P, initial_state_idx=0, steps=100):
    """
    Simula la evolución de las probabilidades de estado en una cadena de Markov.
    Simulates state probability evolution in a Markov chain.
    """
    k = P.shape[0]
    trajectory = np.zeros((k, steps))
    
    # Vector de probabilidad inicial
    p_t = np.zeros(k)
    p_t[initial_state_idx] = 1.0
    trajectory[:, 0] = p_t
    
    for t in range(1, steps):
        p_t = P @ p_t
        trajectory[:, t] = p_t
        
    return trajectory
