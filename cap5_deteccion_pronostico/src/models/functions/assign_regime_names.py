import numpy as np

def assign_regime_names(centroids):
    """
    Dynamically assigns descriptive names to regimes based on their centroid values.
    Asigna nombres descriptivos a los regímenes según sus centroides.
    
    Args:
        centroids (np.ndarray): A sorted array of regime centroids.

    Returns:
        list: Names corresponding to each centroid.
    """
    k = len(centroids)
    names = [''] * k

    # Most stable regime (closest to zero)
    stable_idx = np.argmin(np.abs(centroids))
    names[stable_idx] = 'Stable'
    
    # Extremes
    if stable_idx != 0:
        names[0] = 'Strong Decrease'
    if stable_idx != k - 1:
        names[k - 1] = 'Strong Increase'

    # Intermediates
    for i in range(k):
        if names[i] == '':
            if i < stable_idx:
                names[i] = 'Moderate Decrease'
            elif i > stable_idx:
                names[i] = 'Moderate Increase'
                
    if k == 2:
        names[0] = 'Decrease'
        names[1] = 'Increase'
        
    return names
