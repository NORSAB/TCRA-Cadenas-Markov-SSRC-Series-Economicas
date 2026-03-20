"""
===================================================================
Pipeline 01: Preparación de Datos (reutiliza datos de TCROC-Markov)
===================================================================
Carga los datos y calcula alpha_t para cada combustible usando las
funciones existentes de TCROC-Markov (NO duplica código).
"""
import sys
import os
import numpy as np
import pandas as pd

# Agregar src al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import (CSV_PATH, FUEL_COLUMNS, TCROC_OPTIMAL,
                     OUTPUT_DIR, MODELS_DIR)

# Agregar src de TCROC-Markov para reutilizar funciones
MARKOV_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'cap5_deteccion_pronostico', 'src'))
sys.path.insert(0, MARKOV_SRC)

from processing.functions.calculate_alphas import calculate_alphas
from processing.functions.discretize_series import discretize_series
from models.functions.estimate_transition_matrix import estimate_transition_matrix


def main():
    print("=" * 60)
    print("PASO 1: Preparación de datos para TCROC-SSRC")
    print("=" * 60)

    # Crear directorios de salida
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)

    # Cargar datos
    print("\nCargando datos desde: {}".format(CSV_PATH))
    df = pd.read_csv(CSV_PATH)
    print("Shape: {}".format(df.shape))

    fuel_data = {}

    for fuel in FUEL_COLUMNS:
        hp = TCROC_OPTIMAL[fuel]
        W, lam, K = hp['W'], hp['lam'], hp['K']
        print("\n--- {} (W={}, lambda={}, K={}) ---".format(fuel, W, lam, K))

        series = df[fuel].values

        # Calcular alpha_t (reutiliza función de TCROC-Markov)
        alphas = calculate_alphas(series, W, lam)
        print("  alphas: {} valores".format(len(alphas)))

        # Discretizar (reutiliza función de TCROC-Markov)
        states, centroids = discretize_series(alphas, K)
        print("  estados: {} valores, centroides: {}".format(
            len(states), np.round(centroids, 4)))

        # Estimar P_hat (reutiliza función de TCROC-Markov)
        P_hat, counts = estimate_transition_matrix(states, K)
        print("  P_hat estimada ({}x{})".format(K, K))

        # Alinear precios con alphas
        # alphas empieza desde index W-1, así que prices_aligned = series[W-1:]
        prices_aligned = series[W - 1:]

        fuel_data[fuel] = {
            'alpha': alphas,
            'prices': prices_aligned,
            'states': states,
            'centroids': centroids,
            'P_hat': P_hat,
            'W': W,
            'lam': lam,
            'K': K,
        }

    # Guardar datos preparados
    save_path = os.path.join(MODELS_DIR, 'prepared_data.npz')
    np.savez(save_path, **{
        '{}_alpha'.format(f): fuel_data[f]['alpha'] for f in FUEL_COLUMNS
    }, **{
        '{}_prices'.format(f): fuel_data[f]['prices'] for f in FUEL_COLUMNS
    }, **{
        '{}_states'.format(f): fuel_data[f]['states'] for f in FUEL_COLUMNS
    }, **{
        '{}_P_hat'.format(f): fuel_data[f]['P_hat'] for f in FUEL_COLUMNS
    })
    print("\nDatos guardados en: {}".format(save_path))
    print("\nPASO 1 COMPLETADO")


if __name__ == '__main__':
    main()
