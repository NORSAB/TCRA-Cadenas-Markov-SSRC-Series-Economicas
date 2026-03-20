"""
Genera datos REALES para reemplazar las figuras ilustrativas del Apéndice A3.
Lee las matrices P y las series α reales de los outputs del modelo TCROC-Markov.

Produce:
  1. Coordenadas TikZ para la trayectoria de S_t con datos reales (Super y Regular)
  2. Probabilidades de transición reales para el diagrama de estados
  3. Nota de texto actualizada (sin "carácter ilustrativo")

Uso:
  python generar_apendice_figuras_reales.py
"""
import numpy as np
import os

# ===== RUTAS =====
MODELS_DIR = r"D:\2026\Tesis2026\TCROC-Markov_Nuevo\outputs\models"
GOLD_DIR   = r"D:\2026\Tesis2026\TCROC-Markov_Nuevo\data\gold"
OUTPUT_DIR = r"D:\2026\Tesis2026\TCROC-Adicionales"

# ===== 1. CARGAR MATRICES P REALES =====
print("=" * 70)
print("CARGANDO DATOS REALES DEL MODELO TCROC-MARKOV")
print("=" * 70)

# Usar Super como ejemplo principal (tiene K=4 o K=5 estados)
fuels = ['Super', 'Regular', 'Diesel', 'Kerosene']
for fuel in fuels:
    p_path = os.path.join(MODELS_DIR, f"{fuel}_P_kmeans.npy")
    c_path = os.path.join(MODELS_DIR, f"{fuel}_centroids.npy")
    r_path = os.path.join(MODELS_DIR, f"{fuel}_regime_names.npy")
    
    if os.path.exists(p_path):
        P = np.load(p_path)
        print(f"\n--- {fuel} ---")
        print(f"  Tamaño P: {P.shape}")
        print(f"  P =\n{np.round(P, 4)}")
        if os.path.exists(c_path):
            centroids = np.load(c_path)
            print(f"  Centroids: {centroids}")
        if os.path.exists(r_path):
            names = np.load(r_path, allow_pickle=True)
            print(f"  Régimen names: {names}")

# ===== 2. GENERAR TRAYECTORIAS REALES DE S_t =====
print("\n" + "=" * 70)
print("GENERANDO TRAYECTORIAS REALES DE S_t")
print("=" * 70)

# Cargar las series alpha reales 
for fuel in ['Super', 'Regular']:
    alpha_path = os.path.join(GOLD_DIR, f"{fuel}_alpha.npy")
    p_path = os.path.join(MODELS_DIR, f"{fuel}_P_kmeans.npy")
    c_path = os.path.join(MODELS_DIR, f"{fuel}_centroids.npy")
    
    if os.path.exists(alpha_path):
        alphas = np.load(alpha_path)
        centroids = np.load(c_path) if os.path.exists(c_path) else None
        P = np.load(p_path)
        K = P.shape[0]
        
        # Asignar estados via K-Means (nearest centroid)
        if centroids is not None:
            # Assign each alpha to nearest centroid
            states = np.array([np.argmin(np.abs(a - centroids)) + 1 for a in alphas])
        else:
            # Fallback: use quantile-based assignment
            thresholds = np.percentile(alphas, [25, 50, 75])
            states = np.digitize(alphas, thresholds) + 1
        
        # Tomar los primeros 10 puntos para la figura TikZ
        n_points = min(10, len(states))
        tikz_coords = " ".join([f"({t+1},{states[t]})" for t in range(n_points)])
        
        print(f"\n--- {fuel} (K={K}) ---")
        print(f"  Total alpha values: {len(alphas)}")
        print(f"  First 10 states: {states[:n_points]}")
        print(f"  TikZ coordinates ({fuel}):")
        print(f"    \\addplot+[...] coordinates {{")
        print(f"      {tikz_coords}")
        print(f"    }};")

# ===== 3. GENERAR DATOS PARA EL DIAGRAMA DE TRANSICIONES =====
print("\n" + "=" * 70)
print("PROBABILIDADES REALES PARA EL DIAGRAMA DE TRANSICIONES")
print("=" * 70)

# Usar Super como combustible de referencia para el diagrama
fuel = "Super"
p_path = os.path.join(MODELS_DIR, f"{fuel}_P_kmeans.npy")
P = np.load(p_path)
K = P.shape[0]

print(f"\nMatriz P real ({fuel}, K={K}):")
print(np.round(P, 2))

# Generar texto TikZ para las transiciones
# P[i,j] = P(ir a estado i | estar en estado j)
# Es decir, columna j -> fila i
print(f"\n--- Transiciones para diagrama TikZ (K={min(K, 4)}) ---")
K_diag = min(K, 4)

# Self-loops
for j in range(K_diag):
    prob = P[j, j]
    if prob > 0.01:
        print(f"  s{j+1} -> s{j+1} (self-loop): {prob:.2f}")

# Cross-transitions
for j in range(K_diag):
    for i in range(K_diag):
        if i != j:
            prob = P[i, j]
            if prob > 0.05:  # Only show significant transitions
                print(f"  s{j+1} -> s{i+1}: {prob:.2f}")

# ===== 4. GENERAR TRAYECTORIAS CON DOS LAMBDA DISTINTOS =====
print("\n" + "=" * 70)
print("TRAYECTORIAS CON DOS VALORES DE λ DIFERENTES")
print("=" * 70)

# Para la figura del apéndice necesitamos dos trayectorias:
# una con λ=0.95 y otra con λ=0.75
# Usamos los datos reales del pipeline para Super

# Cargar datos crudos de combustibles
data_path = r"D:\2026\Tesis2026\TCROC-Markov_Nuevo\data\silver"
bronze_path = r"D:\2026\Tesis2026\TCROC-Markov_Nuevo\data\bronze"

# Buscamos el CSV o archivo de datos
import glob
csv_files = glob.glob(os.path.join(data_path, "*.csv")) + \
            glob.glob(os.path.join(data_path, "*.npy")) + \
            glob.glob(os.path.join(bronze_path, "*.csv")) + \
            glob.glob(os.path.join(bronze_path, "*.xlsx"))
print(f"Archivos de datos encontrados: {csv_files}")

# Función TCROC para calcular alpha con un lambda dado
def calculate_alpha(v, W, lam):
    """Calcula la serie de tasas relativas alpha_t."""
    alphas = []
    T = len(v)
    weights = np.array([lam**(t) for t in range(W)])[::-1]
    for t in range(W, T):
        v_current_window = v[t-W+1 : t+1]
        v_previous_window = v[t-W : t]
        numerator = np.sum(weights * v_current_window * v_previous_window)
        denominator = np.sum(weights * v_previous_window**2)
        if denominator == 0:
            alpha = 0
        else:
            alpha = -1 + (numerator / denominator)
        alphas.append(alpha)
    return np.array(alphas)

# Cargar best hyperparams
best_path = os.path.join(r"D:\2026\Tesis2026\TCROC-Markov_Nuevo\outputs\tables", 
                          "best_hyperparameters.csv")
if os.path.exists(best_path):
    import csv
    with open(best_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            print(f"  {row}")

# ===== 5. GENERAR LATEX COMPLETO PARA EL APÉNDICE =====
print("\n" + "=" * 70)
print("LATEX GENERADO PARA EL APÉNDICE A3")
print("=" * 70)

# Generar coordenadas TikZ usando datos reales de Super
fuel = "Super"
alpha_path = os.path.join(GOLD_DIR, f"{fuel}_alpha.npy")
c_path = os.path.join(MODELS_DIR, f"{fuel}_centroids.npy")
p_path = os.path.join(MODELS_DIR, f"{fuel}_P_kmeans.npy")

alphas = np.load(alpha_path)
centroids = np.load(c_path)
P = np.load(p_path)
K = P.shape[0]

# Assign states
states = np.array([np.argmin(np.abs(a - centroids)) + 1 for a in alphas])

# For 2nd trajectory, simulate from Markov chain with same P
np.random.seed(42)
states_sim = [states[0]]
for t in range(1, min(10, len(states))):
    current = states_sim[-1] - 1
    probs = P[:, current]
    probs = probs / probs.sum()  # normalize
    next_state = np.random.choice(range(K), p=probs) + 1
    states_sim.append(next_state)

n_points = 10
real_coords = " ".join([f"({t+1},{states[t]})" for t in range(n_points)])
sim_coords = " ".join([f"({t+1},{states_sim[t]})" for t in range(n_points)])

# Obtener probabilidades para diagrama (K=4 or less)
K_diag = min(K, 4)

latex_output = f"""
% ============================================================
% APÉNDICE A3 — Figuras con datos REALES (Gasolina Super)
% Generado automáticamente desde los outputs del modelo TCROC-Markov
% ============================================================

\\section{{Ilustraciones del Proceso Estocástico \\texorpdfstring{{$\\{{S_t\\}}$}}{{St}}}}
\\label{{apx:figuras_proceso_st}}

\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
  \\begin{{axis}}[
    width=0.8\\textwidth,
    xlabel={{Tiempo \\( t \\)}}, ylabel={{Estado \\( S_t \\)}},
    ymin=0.5, ymax={K + 0.5},
    ytick={{{','.join([str(i) for i in range(1, K+1)])}}}, 
    yticklabels={{{','.join([f'\\\\textnormal{{s\\\\_{i}}}' for i in range(1, K+1)])}}},
    legend pos=north west,
    grid=major,
    title={{Trayectorias del proceso $\\{{S_t\\}}$ (Gasolina Super)}}
  ]
  \\addplot+[mark=*, thick, nordFrost] coordinates {{
    {real_coords}
  }};
  \\addplot+[mark=triangle*, thick, nordRed] coordinates {{
    {sim_coords}
  }};
  \\legend{{Observada, Simulada ($\\hat{{P}}$)}}
  \\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Trayectorias del proceso $\\{{S_t\\}}$ para la Gasolina~Super: 
la secuencia observada (datos reales del período 2017~--~2025) frente a 
una realización simulada a partir de la matriz $\\hat{{P}}$ estimada 
por NNLS.}}
\\label{{fig:apx_st_trayectorias}}
\\end{{figure}}

\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}[node distance=2cm and 2.5cm, >=Stealth]
"""

# State nodes
state_fills = ['nordRed!15', 'nordGreen!15', 'nordYellow!15', 'nordFrost!15']
node_positions = [
    ('s1', '', state_fills[0]),
    ('s2', '[right=of s1]', state_fills[1]),
    ('s3', '[below=of s2]', state_fills[2]),
    ('s4', '[left=of s3]', state_fills[3]),
]
for name, pos, fill in node_positions[:K_diag]:
    latex_output += f"\\node[state, fill={fill}] ({name}) {pos} {{\\( {name} \\)}};\n"

latex_output += "\n"

# Transitions (only significant ones > 0.05)
for j in range(K_diag):
    for i in range(K_diag):
        prob = P[i, j]
        if prob > 0.05:
            p_str = f"{prob:.2f}"
            if i == j:
                # Self-loop
                loop_pos = "above" if i < 2 else "below"
                latex_output += f"\\draw[->] (s{j+1}) to[loop {loop_pos}] node {{{p_str}}} (s{j+1});\n"
            else:
                # Verificar arista inversa para curva
                reverse_prob = P[j, i]
                if reverse_prob > 0.05:
                    bend = "bend left"
                else:
                    bend = "bend left=15"
                # Determine label position
                if j < i:
                    label_pos = "right" if (j % 2 == 0) else "below"
                else:
                    label_pos = "left" if (j % 2 == 1) else "above"
                latex_output += f"\\draw[->] (s{j+1}) to[{bend}] node[{label_pos}] {{{p_str}}} (s{i+1});\n"

latex_output += f"""\\end{{tikzpicture}}
\\caption{{Diagrama de transiciones entre estados $s_j$ para la 
Gasolina~Super con $K={K_diag}$ regímenes. Las probabilidades 
corresponden a la matriz $\\hat{{P}}$ estimada por NNLS 
(véase Capítulo~\\ref{{cap:modelo_hibrido}}).}}
\\label{{fig:apx_diagrama_transiciones}}
\\end{{figure}}

\\vspace{{1em}}
\\noindent\\textbf{{Nota:}} Las figuras de esta sección están generadas 
a partir de las matrices $\\hat{{P}}$ estimadas con datos reales de 
precios de combustibles de Honduras (2017--2025). La trayectoria 
observada corresponde a la serie de la Gasolina~Super procesada con 
los hiperparámetros óptimos obtenidos en el Capítulo~\\ref{{cap:regimenes_combustibles}}.
"""

print(latex_output)

# Guardar en archivo
output_file = os.path.join(OUTPUT_DIR, "apendice_A3_datos_reales.tex")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(latex_output)
print(f"\n✅ Archivo LaTeX guardado: {output_file}")

# ===== RESUMEN =====
print("\n" + "=" * 70)
print("✅ RESUMEN")
print("=" * 70)
print(f"  - Matrices P reales cargadas para {len(fuels)} combustibles")
print(f"  - Trayectorias generadas con datos reales de {fuel}")
print(f"  - K = {K} estados (centroides: {np.round(centroids, 4)})")
print(f"  - Archivo LaTeX generado: {output_file}")
print(f"  - SIGUIENTE PASO: Copiar el contenido a Apendices.tex")
print(f"    reemplazando las líneas 53-106 (sección A3 actual)")
