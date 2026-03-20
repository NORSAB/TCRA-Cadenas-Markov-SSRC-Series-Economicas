"""
compare_models.py — Comparación TCROC-Markov vs TCROC-SSRC.

Genera filas de la tabla comparativa y salida LaTeX.
"""
from typing import Dict, List


def compare_markov_vs_ssrc(
    markov_rmse: float,
    ssrc_results: Dict,
    series_name: str
) -> Dict:
    """
    Genera una fila de la tabla comparativa del capítulo.
    """
    ssrc_rmse_mean = ssrc_results['rmse_mean']
    ssrc_rmse_std = ssrc_results['rmse_std']
    delta_rmse = (ssrc_rmse_mean - markov_rmse) / markov_rmse * 100

    return {
        'fuel': series_name,
        'serie': series_name,
        'markov_rmse': markov_rmse,
        'ssrc_rmse_mean': ssrc_rmse_mean,
        'ssrc_rmse_std': ssrc_rmse_std,
        'ssrc_rmse': "{:.4f} +/- {:.4f}".format(ssrc_rmse_mean, ssrc_rmse_std),
        'D_star': ssrc_results['D'],
        'rho_star': ssrc_results['rho'],
        'leak_star': ssrc_results.get('leak_rate', 1.0),
        'delta_rmse_pct': delta_rmse,
        'ssrc_wins': delta_rmse < 0,
        'dm_stat': ssrc_results.get('dm_stat', 0.0),
        'p_value': ssrc_results.get('p_value', 1.0)
    }


def print_latex_table(comparisons: List[Dict]):
    """
    Imprime tabla comparativa en formato LaTeX.
    """
    print(r"\begin{tabular}{@{}lcccc@{}}")
    print(r"\toprule")
    print(r"\textbf{Serie} & \textbf{TCROC-Markov} & "
          r"\textbf{TCROC-SSRC} & $(D^*, \rho^*)$ & "
          r"$\Delta$\textbf{RMSE} \\")
    print(r"\midrule")

    for c in comparisons:
        delta_str = "{:+.2f}\\%".format(c['delta_rmse_pct'])
        print("{:10s} & ${:.4f}$ & ${:.4f} \\pm {:.4f}$ & $({}, {:.2f})$ & ${}$ \\\\".format(
            c['serie'], c['markov_rmse'],
            c['ssrc_rmse_mean'], c['ssrc_rmse_std'],
            c['D_star'], c['rho_star'], delta_str))

    print(r"\bottomrule")
    print(r"\end{tabular}")


def save_comparison_csv(comparisons: List[Dict], filepath: str):
    """
    Guarda resultados de comparación en CSV.
    """
    import csv
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'serie', 'markov_rmse', 'ssrc_rmse_mean', 'ssrc_rmse_std',
            'D_star', 'rho_star', 'leak_star', 'delta_rmse_pct', 'ssrc_wins',
            'dm_stat', 'p_value'
        ])
        writer.writeheader()
        for c in comparisons:
            writer.writerow({k: c[k] for k in writer.fieldnames})
    print("Resultados guardados en: {}".format(filepath))
