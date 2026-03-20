import numpy as np

def discretize_series_quantiles(alphas, k=4):
    """
    Discretiza la serie de alphas usando cuantiles.
    Discretizes the alpha series using quantiles.
    
    Args:
        alphas (np.array): Serie continua. / Continuous series.
        k (int): Número de regímenes (cuantiles). / Number of regimes.
        
    Returns:
        tuple: (estados, fronteras) / (states, boundaries).
    """
    quantiles = np.linspace(0, 1, k + 1)[1:-1]
    boundaries = np.quantile(alphas, quantiles)
    
    states = np.zeros_like(alphas, dtype=int)
    for i, b in enumerate(boundaries):
        states[alphas >= b] = i + 1
        
    return states, boundaries
