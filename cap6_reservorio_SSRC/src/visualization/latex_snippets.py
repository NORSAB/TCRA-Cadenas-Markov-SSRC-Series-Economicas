"""
===================================================================
Algoritmo 1: Paso de Tiempo SSRC (Snippet LaTeX)
===================================================================
Genera el código LaTeX para incluir el algoritmo formal en la tesis.
"""
import os

def generate_latex_algorithm(output_path):
    latex_code = r"""
\begin{algorithm}[H]
\caption{Entrenamiento y Predicción SSRC (State Space Reservoir Computing)}
\label{alg:ssrc}
\begin{algorithmic}[1]
\REQUIRE Series de retornos $\{ \alpha_t \}_{t=1}^T$, dimensión $D$, radio espectral $\rho$, tasa de fuga $a$
\ENSURE Pesos de salida $\mathbf{W}_{out}$, Predicciones $\hat{y}_{t+1}$

\STATE \textbf{Inicialización:} 
\STATE Crear matriz de entrada $\mathbf{W}_{in} \in \mathbb{R}^{D \times 1}$ y reservorio $\mathbf{W}_{res} \in \mathbb{R}^{D \times D}$
\STATE Escalar $\mathbf{W}_{res}$ tal que su radio espectral $\lambda_{max}(\mathbf{W}_{res}) = \rho$
\STATE Inicializar estado oculto $\mathbf{h}_0 = \mathbf{0} \in \mathbb{R}^D$

\STATE \textbf{Propagación (Leaky Integrator ESN):}
\FOR{$t = 1$ \TO $T$}
    \STATE $\mathbf{h}_t = (1 - a) \mathbf{h}_{t-1} + a \cdot \tanh(\mathbf{W}_{in} \alpha_t + \mathbf{W}_{res} \mathbf{h}_{t-1})$
\ENDFOR

\STATE \textbf{Washout:} Descartar los primeros $W_{wash}$ estados para eliminar transitorios.

\STATE \textbf{Entrenamiento (Readout):}
\STATE Resolver $\mathbf{W}_{out}$ mediante mínimos cuadrados no negativos (NNLS):
\STATE $\min_{\mathbf{W}_{out}} \| \mathbf{H} \mathbf{W}_{out} - \mathbf{y} \|_2$ s.t. $\mathbf{W}_{out} \geq 0$

\STATE \textbf{Predicción:}
\STATE $\mathbf{h}_{T+1} = (1 - a) \mathbf{h}_T + a \cdot \tanh(\mathbf{W}_{in} \alpha_T + \mathbf{W}_{res} \mathbf{h}_T)$
\STATE $\hat{\alpha}_{T+1} = \mathbf{W}_{out}^\top \mathbf{h}_{T+1}$
\STATE $\hat{P}_{T+1} = P_T (1 + \hat{\alpha}_{T+1})$
\end{algorithmic}
\end{algorithm}
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    print(f"  -> Snippet Algoritmo LaTeX generado en: {output_path}")

def generate_equations_table(output_path):
    latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Mapeo Conceptual: SSRC vs Dinámica Petrolera}
\label{tab:ssrc_mapping}
\begin{tabular}{@{}llp{8cm}@{}}
\toprule
\textbf{Símbolo} & \textbf{Técnica} & \textbf{Interpretación en el Mercado} \\ \midrule
$\alpha_t$ & Input & Retorno logarítmico semanal del combustible. \\
$\mathbf{h}_t$ & State & Activación de la memoria neuronal (estado latente del mercado). \\
$\mathbf{W}_{res}$ & Reservorio & Inercia temporal y correlaciones pasadas de la serie. \\
$\rho$ & Radio Esp. & Persistencia de la memoria (valores cercanos a 1 indican memoria larga). \\
$a$ & Tasa de fuga & Factor del Leaky Integrator ($a=1$: ESN cl\'asico; $a<1$: suavizado exponencial). \\
$\mathbf{W}_{out}$ & Readout & Influencia de los regímenes latentes en el precio futuro. \\ \bottomrule
\end{tabular}
\end{table}
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print(f"  -> Snippet Tabla Ecuaciones LaTeX generado en: {output_path}")
