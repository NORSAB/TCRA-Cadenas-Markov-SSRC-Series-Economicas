"""
===================================================================
Pipeline 03: Comparación TCROC-Markov vs TCROC-SSRC
===================================================================
Búsqueda en grilla de hiperparámetros del reservorio y comparación
con el modelo TCROC-Markov del capítulo anterior.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import (FUEL_COLUMNS, TCROC_OPTIMAL, MARKOV_RMSE, MODELS_DIR,
                     OUTPUT_DIR, RESERVOIR_D_VALUES, RESERVOIR_RHO_VALUES,
                     N_REALIZATIONS, RESERVOIR_WASHOUT, TRAIN_SIZE)

from evaluation.functions.grid_search_ssrc import grid_search_ssrc
from evaluation.functions.compare_models import (
    compare_markov_vs_ssrc, print_latex_table, save_comparison_csv
)


def main():
    print("=" * 60)
    print("PASO 3: Comparacion TCROC-Markov vs TCROC-SSRC")
    print("=" * 60)

    # Cargar datos preparados
    data = np.load(os.path.join(MODELS_DIR, 'prepared_data.npz'))

    comparisons = []

    for fuel in FUEL_COLUMNS:
        hp = TCROC_OPTIMAL[fuel]
        alphas = data['{}_alpha'.format(fuel)]
        prices = data['{}_prices'.format(fuel)]
        markov_rmse = MARKOV_RMSE[fuel]

        print("\n" + "=" * 60)
        print("  {} (W={}, lambda={}, K={})".format(
            fuel, hp['W'], hp['lam'], hp['K']))
        print("  TCROC-Markov RMSE: {:.4f}".format(markov_rmse))
        print("=" * 60)

        # Grid search SSRC
        print("\nGrid search ({} configs x {} realizaciones)...".format(
            len(RESERVOIR_D_VALUES) * len(RESERVOIR_RHO_VALUES),
            N_REALIZATIONS))

        ssrc_results = grid_search_ssrc(
            alpha_full=alphas,
            prices_full=prices,
            train_size=TRAIN_SIZE,
            D_values=RESERVOIR_D_VALUES,
            rho_values=RESERVOIR_RHO_VALUES,
            n_realizations=N_REALIZATIONS,
            washout=RESERVOIR_WASHOUT,
            augment_input=False,
            use_nnls=True,
            verbose=True
        )

        # Comparar
        comp = compare_markov_vs_ssrc(
            markov_rmse=markov_rmse,
            ssrc_results=ssrc_results['best'],
            series_name=fuel
        )
        comparisons.append(comp)

        print("\n  -> Mejor SSRC: D={}, rho={:.2f}".format(
            comp['D_star'], comp['rho_star']))
        print("  -> RMSE: {}".format(comp['ssrc_rmse']))
        print("  -> Delta RMSE: {:+.2f}%".format(comp['delta_rmse_pct']))

        # Guardar resultados de grilla por combustible
        grid_path = os.path.join(MODELS_DIR, '{}_ssrc_grid.npz'.format(fuel.lower()))
        np.savez(grid_path,
                 grid=[str(g) for g in ssrc_results['grid']],
                 best_D=ssrc_results['best']['D'],
                 best_rho=ssrc_results['best']['rho'],
                 best_rmse_mean=ssrc_results['best']['rmse_mean'],
                 best_rmse_std=ssrc_results['best']['rmse_std'])

    # Tabla LaTeX
    print("\n" + "=" * 60)
    print("  TABLA LATEX PARA EL CAPITULO")
    print("=" * 60 + "\n")
    print_latex_table(comparisons)

    # Guardar CSV
    csv_path = os.path.join(OUTPUT_DIR, 'tables', 'ssrc_comparison.csv')
    save_comparison_csv(comparisons, csv_path)

    print("\nPASO 3 COMPLETADO")


if __name__ == '__main__':
    main()
