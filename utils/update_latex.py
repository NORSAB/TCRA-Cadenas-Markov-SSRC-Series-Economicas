import pandas as pd
import numpy as np
import re

params = {
    'TCROC': {'W': None, 'lambd': 1.0},
    'TCROCM': {'W': 5, 'lambd': 1.0},
    'ETCROC': {'W': None, 'lambd': 0.9},
    'ETCROCM': {'W': 5, 'lambd': 0.9}
}

def compute_tcroc(v, W=None, lambd=1.0):
    v = np.asarray(v, dtype=float)
    T = len(v) - 1
    if W is None or W > T: W = T
    start = T - W + 1
    weights = lambd ** np.arange(W - 1, -1, -1) if lambd != 1.0 else 1.0
    den = np.sum(weights * (v[start-1:T] ** 2))
    beta = np.sum(weights * v[start:T+1] * v[start-1:T]) / den if den != 0 else 1.0
    return beta - 1.0, beta

df = pd.read_csv(r"D:\2026\Tesis2026\Familias-TCROC\data\silver\pib_clean.csv")
v_pib = df['PIB en Dólares Corrientes (Millones de USD)'].values

# Calcular TCROC basico for the first example
a_tcroc, b_tcroc = compute_tcroc(v_pib)

# Calcular variantes for the comparison table 2025/2026
latex_table_rows = []
for var, p in params.items():
    a, b = compute_tcroc(v_pib, W=p['W'], lambd=p['lambd'])
    # formatting specifically for PIB in USD millions, but let's just make it simple formatting.
    p25 = v_pib[-1] * b
    p26 = p25 * b
    latex_table_rows.append(f"{var} & {b:.4f} & {p25:,.0f} & {p26:,.0f} \\\\".replace(",", "\,"))

latex_snippet = r"""% -------------------------------------------------------------------
% Ejemplos Numéricos
% -------------------------------------------------------------------
\section{Ejemplos Numéricos Sectoriales (Perspectiva a 2026)}
\label{sec:tcroc_ejemplos}

Para validar empíricamente la asimilación matemática del operador, se aplica la familia TCROC sobre la macro-dimensión del **PIB en Dólares Corrientes** de la República de Honduras (periodo 2000--2024, con \(T = 24\) intervalos o transiciones observadas). Se estiman las proyecciones resultantes bajo los cuatro esquemas de asimilación paramétrica estudiados para los años fiscales 2025 y 2026.

\begin{examplex}[Proyecciones del PIB Corriente - Variante Base y Extensiones]
\label{ex:pib_dolares_2026}
Sea la secuencia empírica del PIB en dólares corrientes (millones USD) cerrando el año fiscal 2024 con un valor histórico auditado de \(37,093.57\) millones USD. 

Bajo la **TCROC global (\(W=T, \lambda=1\))**, el sumatorio agnóstico completo hereda una inercia constante \(\hat{\beta} \approx 1.0704\), indicando una tasa de asimilación proyectiva estricta de \(\alpha_T \approx 7.04\%\). 

No obstante, las perturbaciones dinámicas recientes se capturan de forma asimétrica permitiendo el ajuste de memoria y el decaimiento. En la Tabla~\ref{tab:proy_tcroc_pib}, se modelizan los escenarios inerciales divergentes según el ensamble paramétrico de pre-condicionamiento que será entregado a los agrupadores de Markov.

\begin{table}[H]
\centering
\begin{tabular}{@{}lccc@{}}
\toprule
\textbf{Variante Operacional} & \textbf{Factor ($\hat{\beta}$)} & \textbf{Proyección 2025} & \textbf{Proyección 2026} \\
\midrule
TCROC ($W=T, \lambda=1.0$)   & REPL_TCROC_BETA & REPL_TCROC_25 & REPL_TCROC_26 \\
TCROCM ($W=5, \lambda=1.0$)   & REPL_TCROCM_BETA & REPL_TCROCM_25 & REPL_TCROCM_26 \\
ETCROC ($W=T, \lambda=0.9$)  & REPL_ETCROC_BETA & REPL_ETCROC_25 & REPL_ETCROC_26 \\
ETCROCM ($W=5, \lambda=0.9$) & REPL_ETCROCM_BETA & REPL_ETCROCM_25 & REPL_ETCROCM_26 \\
\bottomrule
\end{tabular}
\caption{Modelación paramétrica divergente del PIB de Honduras (millones USD) para los años 2025--2026 bajo las cuatro formulaciones de la TCROC.}
\label{tab:proy_tcroc_pib}
\end{table}
\end{examplex}

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{figures/EjemploPIB_TCROC_4Variantes.png}
\caption{Comparativa del espacio vectorial de entrenamiento (2000-2024) y bifurcación predictiva (2025-2026) según la asimilación del filtro TCROC para los cuatro macro-indicadores del PIB Hondureño.}
\label{fig:pib_tcroc_grid}
\end{figure}
"""

for i, var in enumerate(['TCROC', 'TCROCM', 'ETCROC', 'ETCROCM']):
    b = latex_table_rows[i].split('&')[1].strip()
    p25 = latex_table_rows[i].split('&')[2].strip()
    p26 = latex_table_rows[i].split('&')[3].strip().replace('\\\\', '').strip()
    latex_snippet = latex_snippet.replace(f'REPL_{var}_BETA', b)
    latex_snippet = latex_snippet.replace(f'REPL_{var}_25', p25)
    latex_snippet = latex_snippet.replace(f'REPL_{var}_26', p26)

with open(r"D:\2026\Tesis2026\Capitulo 1 IEEE\main.tex", 'r', encoding='utf-8') as f:
    tex = f.read()

# Replace from "% Ejemplos Numéricos" to right before "\section{Markov" or Kronecker
start_idx = tex.find("% Ejemplos Numéricos")
end_idx = tex.find("% -------------------------------------------------------------------", start_idx + 10)
end_idx = tex.find(r"\section{", end_idx)

if start_idx != -1 and end_idx != -1:
    new_tex = tex[:start_idx] + latex_snippet + "\n" + tex[end_idx:]
    with open(r"D:\2026\Tesis2026\Capitulo 1 IEEE\main.tex", 'w', encoding='utf-8') as f:
        f.write(new_tex)
else:
    print("No se encontró el bloque a reemplazar.")
