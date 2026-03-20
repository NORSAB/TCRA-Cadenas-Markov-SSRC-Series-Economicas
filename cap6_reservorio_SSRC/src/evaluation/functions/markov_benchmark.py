"""
===================================================================
Benchmark: Pronóstico TCROC-Markov
===================================================================
Calcula las predicciones de la cadena de Markov para comparación.
"""
import sys
import os
import numpy as np

# Usar constantes absolutas para evitar errores de cálculo de ruta
MARKOV_SRC = os.path.abspath(r"d:\2026\Tesis2026\cap5_deteccion_pronostico\src")

if MARKOV_SRC not in sys.path:
    # Insertar al inicio para prioridad
    sys.path.insert(0, MARKOV_SRC)

# Importar funciones de los submódulos específicos
try:
    from processing.functions.discretize_series import discretize_series
    from models.functions.estimate_transition_matrix import estimate_transition_matrix
except ImportError:
    # Intento alternativo si las carpetas no tienen __init__.py o se comportan distinto
    print(f"DEBUG: Intentando carga manual de módulos desde {MARKOV_SRC}")
    import importlib.util
    
    def load_module_from_path(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    ds_mod = load_module_from_path("ds", os.path.join(MARKOV_SRC, "processing", "functions", "discretize_series.py"))
    tm_mod = load_module_from_path("tm", os.path.join(MARKOV_SRC, "models", "functions", "estimate_transition_matrix.py"))
    discretize_series = ds_mod.discretize_series
    estimate_transition_matrix = tm_mod.estimate_transition_matrix

def predict_markov_rolling_window(alphas, prices, train_size, K):
    """
    Predicción 1-step-ahead usando TCROC-Markov en ventana deslizante.
    """
    n_test = len(alphas) - train_size - 1
    preds = np.zeros(n_test)
    actuals = prices[train_size+1 : train_size+1+n_test]
    
    for i in range(n_test):
        t = train_size + i
        current_price = prices[t]
        
        # Entrenamiento local
        train_alphas = alphas[:t]
        try:
            train_states, centroids = discretize_series(train_alphas, K)
            P_train, _ = estimate_transition_matrix(train_states, K)
            
            # Identificar estado actual
            current_alpha = alphas[t]
            current_state = np.argmin(np.abs(current_alpha - centroids))
            
            # Predicción
            next_state = np.argmax(P_train[:, current_state])
            pred_alpha = centroids[next_state]
            
            preds[i] = current_price * (1 + pred_alpha)
        except Exception as e:
            preds[i] = current_price
            
    return preds, actuals
