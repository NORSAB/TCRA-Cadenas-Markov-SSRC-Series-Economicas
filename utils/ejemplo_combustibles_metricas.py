import pandas as pd
import numpy as np

def compute_tcroc(v, W=None, lambd=1.0):
    v = np.asarray(v, dtype=float)
    T = len(v) - 1
    if W is None or W > T: W = T
    if W < 1: W = 1
    start = T - W + 1
    v_target = v[start : T+1]
    v_lag    = v[start-1 : T]
    weights = lambd ** np.arange(W - 1, -1, -1) if lambd != 1.0 else 1.0
    num = np.sum(weights * v_target * v_lag)
    den = np.sum(weights * (v_lag ** 2))
    beta = num / den if den != 0 else 1.0
    return beta

splits = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
split_labels = ["60/40", "65/35", "70/30", "75/25", "80/20", "85/15", "90/10", "95/5"]
ws = list(range(2, 11))
lambdas = np.arange(0.70, 1.01, 0.05)

df = pd.read_csv(r"D:\2026\Tesis2026\Familias-TCROC\data\silver\combustibles_clean.csv")
cols = ['Super', 'Regular', 'Diesel', 'Kerosene']

best_params_per_series = {}

for col in cols:
    v = df[col].values
    N = len(v)
    
    # We already know W=2, Lam=0.7 are best based on previous run.
    # Let's just calculate the detailed metrics for these best params.
    best_W = 2
    best_lam = 0.7
    
    mape_splits_vals = []
    rmse_splits_vals = []
    
    for s in splits:
        train_len = int(N * s)
        test_len = N - train_len
        
        errors_sq = []
        errors_abs_pct = []
        for i in range(test_len):
            curr_train = v[:train_len + i]
            beta = compute_tcroc(curr_train, W=best_W, lambd=best_lam)
            pred = curr_train[-1] * beta
            actual = v[train_len + i]
            errors_sq.append((pred - actual)**2)
            errors_abs_pct.append(abs(pred - actual) / actual)
            
        rmse_splits_vals.append(np.sqrt(np.mean(errors_sq)))
        mape_splits_vals.append(np.mean(errors_abs_pct) * 100) # percentage
        
    avg_rmse = np.mean(rmse_splits_vals)
    avg_mape = np.mean(mape_splits_vals)
    
    best_params_per_series[col] = {
        'W': best_W,
        'lambda': best_lam,
        'MAPE_splits': mape_splits_vals,
        'Avg_MAPE': avg_mape
    }

latex_code = r"""\begin{table}[H]
\centering
\resizebox{\textwidth}{!}{
\begin{tabular}{@{}lccccccccc@{}}
\toprule
& \multicolumn{8}{c}{\textbf{Desempeño MAPE (\%) por Partición Expansiva}} & \\
\cmidrule(lr){2-9}
\textbf{Macro-Serie} & \textbf{60/40} & \textbf{65/35} & \textbf{70/30} & \textbf{75/25} & \textbf{80/20} & \textbf{85/15} & \textbf{90/10} & \textbf{95/5} & \textbf{Promedio Global} \\
\midrule
REPL_ROWS
\bottomrule
\end{tabular}
}
\caption{Evolución del error predictivo porcentual absoluto (MAPE) a lo largo de los anclajes temporales de validación un-paso-adelante para los cuatro macro-indicadores energéticos.}
\label{tab:cv_combustibles_tcroc}
\end{table}"""

rows = ""
for col in cols:
    d = best_params_per_series[col]
    row_vals = [f"{val:.2f}\\%" for val in d['MAPE_splits']]
    row_str = " & ".join(row_vals)
    rows += f"\\textbf{{{col}}} & {row_str} & \\textbf{{{d['Avg_MAPE']:.2f}\\%}} \\\\\n"

latex_code = latex_code.replace("REPL_ROWS", rows.strip())

with open("snippet_latex_combustibles.tex", "w", encoding="utf-8") as f:
    f.write(latex_code)
