import numpy as np
from sklearn.cluster import KMeans

def discretize_series(alphas, k):
    """
    Discretizes a continuous series (alphas) using K-Means clustering.
    Discretiza una serie continua (alphas) usando agrupamiento K-Means.
    Matching Articulol.py exactly.
    """
    alphas_reshaped = alphas.reshape(-1, 1)
    # Reverting to original settings for exact result replication
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(alphas_reshaped)
    states = kmeans.predict(alphas_reshaped)
    centroids = kmeans.cluster_centers_.flatten()
    
    # Sort centroids and map states to match the order (standard in Articulol.py)
    # Ordenar centroides y mapear estados para coincidir el orden (estándar en Articulol.py)
    sorted_idx = np.argsort(centroids)
    sorted_centroids = centroids[sorted_idx]
    
    # Mapeo: etiqueta original -> etiqueta ordenada
    # Crear mapeo: etiqueta_original -> etiqueta_ordenada
    mapping = np.zeros_like(sorted_idx)
    mapping[sorted_idx] = np.arange(k)
    sorted_states = mapping[states]
    
    return sorted_states, sorted_centroids
