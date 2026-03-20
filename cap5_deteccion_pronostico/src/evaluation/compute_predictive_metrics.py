import numpy as np
import warnings
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.cluster import KMeans
from src.models.functions.estimate_transition_matrix_mle import estimate_transition_matrix_mle

def compute_predictive_metrics(alphas, series, k, W, method='kmeans', use_cv=False, return_splits=False):
    """
    Evaluates predictive performance using rolling training windows.
    EXACT replication of Articulol.py logic and indexing.
    """
    price_series_values = np.asarray(series)
    T_total = len(price_series_values)
    results_per_split = []
    
    # Ratios used in Articulol.py
    ratios = np.arange(0.60, 0.96, 0.05)
    splits = [(np.arange(0, int(len(alphas) * r)), np.arange(int(len(alphas) * r), len(alphas))) for r in ratios]

    import warnings

    for idx, (train_idx, test_idx) in enumerate(splits):
        accuracy, rmse = np.nan, np.nan # Initialize for each split

        # ROBUSTEZ DINAMICA: 
        # Verificar datos de entrenamiento suficientes to form K clusters (at least 2 points per expected cluster is a safe minimum)
        if len(train_idx) < 2 * k: 
            continue
        
        # Verificar datos de prueba estadisticamente relevantes
        if len(test_idx) < 5:
            continue

        alphas_train, alphas_test = alphas[train_idx], alphas[test_idx]
        
        # --- Discretization (Fit on Train, Transform on Test) ---
        try:
            if method == 'kmeans':
                alphas_reshaped = alphas_train.reshape(-1, 1)
                # Suppress KMeans convergence warnings for very small datasets
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(alphas_reshaped)
                train_states = kmeans.labels_
                regime_reps = kmeans.cluster_centers_.flatten()
                
                # Map test using nearest centroid
                test_states = np.array([np.argmin(np.abs(alpha - regime_reps)) for alpha in alphas_test])
            else: # quantiles
                boundaries = np.quantile(alphas_train, np.linspace(0, 1, k + 1)[1:-1])
                train_states = np.digitize(alphas_train, boundaries)
                test_states = np.digitize(alphas_test, boundaries)
                # Centroids
                regime_reps = np.array([alphas_train[train_states == i].mean() if np.any(train_states==i) else 0 for i in range(k)])
                regime_reps[np.isnan(regime_reps)] = 0
        except Exception:
            # If clustering fails (e.g. singular matrix, empty bins), skip this split
            continue

        # --- Model Training (Transition Matrix) ---
        # Optimized Logic: Use vectorized MLE estimation
        try:
            P_hat, _ = estimate_transition_matrix_mle(train_states, k)
        except Exception:
            # Manejar estados vacios o casos limite
            continue
        
        if len(train_states) > 0:
            predicted_states = []
            current_state = train_states[-1]
            for _ in range(len(alphas_test)):
                if current_state < P_hat.shape[1]:
                    next_state = np.argmax(P_hat[:, current_state])
                else:
                    next_state = 0
                predicted_states.append(next_state)
                current_state = next_state
            
            # Accuracy - With warning suppression
            if len(test_states) == len(predicted_states):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    accuracy = accuracy_score(test_states, predicted_states)
            
            # RMSE Logic
            predicted_alphas = np.array([regime_reps[s] for s in predicted_states])
            last_prices_idx = test_idx + W - 2
            actual_prices_idx = test_idx + W - 1
            
            if np.max(last_prices_idx) < T_total and np.max(actual_prices_idx) < T_total:
                last_prices = price_series_values[last_prices_idx]
                actual_prices = price_series_values[actual_prices_idx]
                
                check_len = min(len(predicted_alphas), len(last_prices), len(actual_prices))
                if check_len > 0:
                    pred_p = last_prices[:check_len] * (1 + predicted_alphas[:check_len])
                    act_p = actual_prices[:check_len]
                    rmse = np.sqrt(mean_squared_error(act_p, pred_p))
        
        results_per_split.append({
            'Partition': f"{int(ratios[idx]*100)}/{int((1-ratios[idx])*100)}",
            'Accuracy': accuracy,
            'RMSE': rmse
        })
        
    if return_splits:
        # Promedios para compatibilidad; retorna lista
        avg_acc = np.nanmean([r['Accuracy'] for r in results_per_split])
        avg_rmse = np.nanmean([r['RMSE'] for r in results_per_split])
        return avg_acc, avg_rmse, results_per_split
        
    return results_per_split
