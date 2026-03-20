"""
===================================================================
Pipeline 02: Verificaciones Teóricas
===================================================================
Verifica los teoremas del capítulo:
- Teorema 6.1 (ESP): rho(W_res) < 1
- Teorema 6.2 (Inclusión): TCROC-Markov = SSRC degenerado
- Teorema 6.3 (Condición de rango): rango(H) = D
- Proposición 6.2 (Cota de perturbación)
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import (FUEL_COLUMNS, TCROC_OPTIMAL, MODELS_DIR, OUTPUT_DIR,
                     RESERVOIR_WASHOUT)

from reservoir.functions.create_reservoir import create_reservoir
from reservoir.functions.propagate_reservoir import propagate_reservoir
from reservoir.functions.estimate_readout import estimate_readout_nnls
from reservoir.functions.verify_theoretical import (
    verify_esp, verify_rank_condition,
    verify_perturbation_bound, demonstrate_inclusion_theorem
)


def main():
    print("=" * 60)
    print("PASO 2: Verificaciones Teóricas")
    print("=" * 60)

    # Cargar datos preparados
    data = np.load(os.path.join(MODELS_DIR, 'prepared_data.npz'))

    results = []

    for fuel in FUEL_COLUMNS:
        hp = TCROC_OPTIMAL[fuel]
        K = hp['K']
        alphas = data['{}_alpha'.format(fuel)]
        states = data['{}_states'.format(fuel)]
        P_hat = data['{}_P_hat'.format(fuel)]

        print("\n" + "=" * 60)
        print("  {}  (K={})".format(fuel, K))
        print("=" * 60)

        # --- Teorema 6.2: TCROC-Markov = SSRC degenerado ---
        print("\n[Teorema 6.2] Inclusion:")
        incl = demonstrate_inclusion_theorem(P_hat, states, K)
        print("  {}".format(incl['message']))

        # --- Teorema 6.1: ESP ---
        D_test = 50
        rho_test = 0.95
        W_in, W_res = create_reservoir(
            input_dim=1, reservoir_dim=D_test,
            spectral_radius=rho_test, seed=42
        )
        esp = verify_esp(W_res)
        print("\n[Teorema 6.1] ESP (D={}, rho_target={:.2f}):".format(
            D_test, rho_test))
        print("  rho(W_res) = {:.6f}".format(esp['spectral_radius']))
        print("  ESP garantizada: {}".format(esp['esp_sufficient']))

        # --- Teorema 6.3: Condición de rango ---
        H = propagate_reservoir(alphas, W_in, W_res, washout=RESERVOIR_WASHOUT)
        rank_info = verify_rank_condition(H)
        print("\n[Teorema 6.3] Condicion de rango:")
        print("  D={}, T_eff={}, rango(H)={}".format(
            rank_info['D'], rank_info['T_eff'], rank_info['rank']))
        print("  Rango completo: {}".format(rank_info['full_rank']))
        print("  Numero de condicion: {:.2f}".format(rank_info['condition_number']))

        # --- Proposición 6.2: Cota de perturbación ---
        targets = alphas[RESERVOIR_WASHOUT + 1:]
        H_aligned = H[:, :-1]
        min_len = min(H_aligned.shape[1], len(targets))
        W_out = estimate_readout_nnls(H_aligned[:, :min_len], targets[:min_len])

        bound = verify_perturbation_bound(W_out, W_in, W_res, epsilon=0.01)
        print("\n[Proposicion 6.2] Cota de perturbacion (eps=0.01):")
        print("  ||delta y|| <= {:.6f}".format(bound['perturbation_bound']))

        results.append({
            'fuel': fuel,
            'inclusion_ok': incl['equivalent'],
            'esp_ok': esp['esp_sufficient'],
            'rank_ok': rank_info['full_rank'],
            'perturbation_bound': bound['perturbation_bound']
        })

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE VERIFICACIONES")
    print("=" * 60)
    print("{:12s} {:10s} {:8s} {:10s} {:12s}".format(
        'Combustible', 'Inclusion', 'ESP', 'Rango', 'Cota pert.'))
    print("-" * 54)
    for r in results:
        print("{:12s} {:10s} {:8s} {:10s} {:.6f}".format(
            r['fuel'],
            'OK' if r['inclusion_ok'] else 'FAIL',
            'OK' if r['esp_ok'] else 'FAIL',
            'OK' if r['rank_ok'] else 'FAIL',
            r['perturbation_bound']
        ))

    print("\nPASO 2 COMPLETADO")


if __name__ == '__main__':
    main()
