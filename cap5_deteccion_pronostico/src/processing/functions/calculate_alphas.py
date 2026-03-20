import numpy as np

def calculate_alphas(series, W, lambda_decay):
    """
    Computes the weighted relative change of a time series.
    EXACT replication of Articulol.py logic, including the k-1 index wrapping.
    """
    v = np.asarray(series)
    if len(v) < W:
        return np.array([])
        
    alphas = []
    # Matching the exact loop from notebook
    for t in range(W - 1, len(v)):
        # range(t - W + 1, t + 1)
        k_range = range(t - W + 1, t + 1)
        
        # Literal terms calculation from notebook
        numerator_terms = [lambda_decay**(t - k) * v[k] * v[k-1] for k in k_range]
        denominator_terms = [lambda_decay**(t - k) * v[k-1]**2 for k in k_range]
        
        numerator = np.sum(numerator_terms)
        denominator = np.sum(denominator_terms)
        
        if denominator == 0:
            beta_t = 1.0
        else:
            beta_t = numerator / denominator
            
        alphas.append(beta_t - 1)
        
    return np.array(alphas)
