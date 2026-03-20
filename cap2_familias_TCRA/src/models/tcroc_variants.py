import numpy as np
import pandas as pd

def calculate_alpha_family(series, W, lambda_decay):
    """
    Calculates the alpha series for a given W and lambda.
    Optimized with vectorization inside the loop.
    """
    v = np.asarray(series, dtype=float)
    n = len(v)
    if n < W + 1:
        return np.full(n, np.nan)
    
    # Pre-calculate weights
    weights = lambda_decay ** np.arange(W - 1, -1, -1)
    
    alphas = np.full(n, np.nan)
    
    # We start at index W+1 because we need history strictly before `t` to prevent data leakage.
    # To predict v[t], we only use data up to t-1. 
    # For k from t-W to t-1:
    #   Numerator: \sum w_k * v_k * v_{k-1}
    #   Denominator: \sum w_k * v_{k-1}^2
    
    for t in range(W + 1, n):
        # Data strictly up to t-1 (length W)
        vk = v[t-W : t]
        vkm1 = v[t-W-1 : t-1]
        
        num = np.sum(weights * vk * vkm1)
        den = np.sum(weights * vkm1**2)
        
        if den == 0:
            beta_t = 1.0
        else:
            beta_t = num / den
        
        alphas[t] = beta_t - 1
        
    return alphas

def get_variant_params(variant_name, W_m, lambda_e):
    """Returns the (W, lambda) pair for a specific variant name."""
    if variant_name == 'TCROC':
        return 2, 1.0
    elif variant_name == 'TCROCM':
        return W_m, 1.0
    elif variant_name == 'ETCROC':
        return 2, lambda_e # Minimum W for lambda to have impact
    elif variant_name == 'ETCROCM':
        return W_m, lambda_e
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

def get_tcroc_variants(series, W_m=52, lambda_e=0.99):
    """
    Computes the four variants for a series as a DataFrame.
    """
    df_variants = pd.DataFrame(index=series.index if hasattr(series, 'index') else None)
    df_variants['TCROC'] = calculate_alpha_family(series, 2, 1.0)
    df_variants['TCROCM'] = calculate_alpha_family(series, W_m, 1.0)
    df_variants['ETCROC'] = calculate_alpha_family(series, 2, lambda_e)
    df_variants['ETCROCM'] = calculate_alpha_family(series, W_m, lambda_e)
    
    return df_variants
