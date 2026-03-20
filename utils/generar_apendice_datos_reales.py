# -*- coding: utf-8 -*-
"""
Genera figuras LaTeX/TikZ para el Apendice A3 con datos REALES
de combustibles (Super y Regular) del modelo TCROC-Markov.

Salida: apendice_A3_datos_reales_v2.tex
"""
import numpy as np
import os

# ── Configuracion ──────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
MODELS_DIR = os.path.join(PROJ, "TCROC-Markov_Nuevo", "outputs", "models")
GOLD_DIR = os.path.join(PROJ, "TCROC-Markov_Nuevo", "data", "gold")
OUTPUT_DIR = BASE
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "apendice_A3_datos_reales_v2.tex")

# Colores Nord para estados (K=4 siempre en el modelo)
NORD_COLORS = {
    1: "nordRed",
    2: "nordFrost",
    3: "nordGreen",
    4: "nordYellow",
}
STATE_NAMES = {
    1: "Descenso fuerte",
    2: "Estable/leve baja",
    3: "Alza moderada",
    4: "Alza fuerte",
}
K = 4  # Todos los combustibles usan K=4 estados en el modelo


def load_fuel_data(fuel_name):
    """Carga alpha, P, centroids, boundaries para un combustible."""
    alpha = np.load(os.path.join(GOLD_DIR, f"{fuel_name}_alpha.npy"), allow_pickle=True)
    P = np.load(os.path.join(MODELS_DIR, f"{fuel_name}_transition_matrix.npy"))
    centroids = np.load(os.path.join(MODELS_DIR, f"{fuel_name}_centroids.npy"))
    boundaries = np.load(os.path.join(MODELS_DIR, f"{fuel_name}_boundaries.npy"), allow_pickle=True)
    return alpha, P, centroids, boundaries


def assign_states(alpha, boundaries):
    """Asigna estados 1..K basado en boundaries (digitize con 3 boundaries => 4 bins)."""
    return np.digitize(alpha, boundaries) + 1


def find_varied_window(states, window_size=20):
    """Encuentra la ventana con mayor variedad de estados y transiciones."""
    best_start = 0
    best_score = 0
    for i in range(len(states) - window_size):
        window = states[i:i + window_size]
        unique = len(set(window))
        transitions = sum(1 for j in range(1, len(window)) if window[j] != window[j-1])
        score = unique * 10 + transitions
        if score > best_score:
            best_score = score
            best_start = i
    return best_start


def simulate_trajectory(P, initial_state, n_steps, rng=None):
    """Simula una trayectoria de estados usando la matriz P."""
    if rng is None:
        rng = np.random.default_rng(42)
    trajectory = [initial_state]
    for _ in range(n_steps - 1):
        current = trajectory[-1] - 1  # 0-indexed
        probs = P[current]
        probs = probs / probs.sum()
        next_state = rng.choice(range(1, K + 1), p=probs)
        trajectory.append(next_state)
    return trajectory


def gen_tikz_trajectory(fuel_name, alpha, P, boundaries, window_size=20):
    """Genera TikZ para trayectorias observada y simulada."""
    states = assign_states(alpha, boundaries)

    # Encontrar ventana con maxima variedad
    start = find_varied_window(states, window_size)
    obs_window = states[start:start + window_size]

    # Simular desde mismo estado inicial
    sim_window = simulate_trajectory(P, obs_window[0], window_size)

    # Generar coordenadas TikZ
    obs_coords = " ".join(f"({t+1},{s})" for t, s in enumerate(obs_window))
    sim_coords = " ".join(f"({t+1},{s})" for t, s in enumerate(sim_window))

    ytick_labels = ",".join(
        f"\\textnormal{{{STATE_NAMES[i]}}}" for i in range(1, K + 1)
    )
    yticks = ",".join(str(i) for i in range(1, K + 1))

    tikz = f"""\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
  \\begin{{axis}}[
    width=0.85\\textwidth, height=6cm,
    xlabel={{Periodo $t$ (semanas)}}, ylabel={{Estado $S_t$}},
    ymin=0.5, ymax={K + 0.5},
    xmin=0.5, xmax={window_size + 0.5},
    ytick={{{yticks}}},
    yticklabels={{{ytick_labels}}},
    legend pos=north east,
    legend style={{font=\\footnotesize}},
    grid=major,
    grid style={{dashed, opacity=0.4}},
    title={{Trayectorias del proceso $\\{{S_t\\}}$ --- {fuel_name}}},
    title style={{font=\\bfseries}},
  ]
  \\addplot+[mark=*, thick, nordFrost, mark size=2.5pt] coordinates {{
    {obs_coords}
  }};
  \\addplot+[mark=triangle*, thick, nordRed, mark size=2.5pt, dashed] coordinates {{
    {sim_coords}
  }};
  \\legend{{Observada ($\\hat{{P}}$ real), Simulada ($\\hat{{P}}$ Markov)}}
  \\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Trayectorias del proceso $\\{{S_t\\}}$ para combustible \\textbf{{{fuel_name}}}: 
secuencia observada (asignada por $\\alpha_t$ y centroides $K$-means con $K={K}$) 
vs.\\ trayectoria simulada a partir de la matriz de transici\\'on $\\hat{{P}}$ estimada.
Ventana de {window_size} periodos seleccionada por m\\'axima variabilidad de reg\\'imenes.}}
\\label{{fig:apx_trayectorias_{fuel_name.lower()}}}
\\end{{figure}}"""

    print(f"  [{fuel_name}] Ventana t={start+1}..{start+window_size}")
    print(f"  [{fuel_name}] Observada: {list(obs_window)}")
    print(f"  [{fuel_name}] Simulada:  {sim_window}")

    return tikz


def gen_tikz_transition_diagram(fuel_name, P):
    """Genera TikZ para diagrama de transicion con probabilidades reales."""
    positions = [
        ("s1", "nordRed!15",    "",           ""),
        ("s2", "nordFrost!15",  "right=of s1",""),
        ("s3", "nordGreen!15",  "below=of s2",""),
        ("s4", "nordYellow!15", "left=of s3", ""),
    ]
    loop_positions = ["above", "above", "below", "below"]

    # Nodos
    nodes_tex = []
    for idx, (sid, color, pos, _) in enumerate(positions):
        if pos:
            nodes_tex.append(
                f"\\node[state, fill={color}] ({sid}) [{pos}] {{$s_{idx+1}$}};"
            )
        else:
            nodes_tex.append(
                f"\\node[state, fill={color}] ({sid}) {{$s_{idx+1}$}};"
            )

    # Etiquetas bajo nodos
    label_lines = []
    for idx in range(K):
        sid = f"s{idx+1}"
        label = STATE_NAMES[idx + 1]
        label_lines.append(
            f"\\node[below=0.3cm of {sid}, font=\\scriptsize\\itshape] {{{label}}};"
        )

    # Aristas con probabilidades reales > umbral
    threshold = 0.02
    edges_tex = []
    for i in range(K):
        for j in range(K):
            prob = P[i, j]
            if prob < threshold:
                continue
            prob_str = f"{prob:.2f}"
            si = f"s{i+1}"
            sj = f"s{j+1}"

            if i == j:
                lpos = loop_positions[i]
                edges_tex.append(
                    f"\\draw[->] ({si}) to[loop {lpos}] node {{{prob_str}}} ({si});"
                )
            else:
                reverse_prob = P[j, i]
                bend = "bend left=15" if reverse_prob >= threshold else "bend left=10"
                if i < j:
                    label_pos = "above" if abs(i - j) <= 1 else "right"
                else:
                    label_pos = "below" if abs(i - j) <= 1 else "left"
                edges_tex.append(
                    f"\\draw[->] ({si}) to[{bend}] "
                    f"node[{label_pos}, font=\\footnotesize] {{{prob_str}}} ({sj});"
                )

    tikz = f"""\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}[node distance=2.5cm and 3cm, >=Stealth,
  state/.style={{circle, draw, minimum size=1cm, font=\\large}}]
{chr(10).join(nodes_tex)}

{chr(10).join(label_lines)}

{chr(10).join(edges_tex)}
\\end{{tikzpicture}}
\\caption{{Diagrama de transici\\'on entre reg\\'imenes para combustible \\textbf{{{fuel_name}}} 
con $K={K}$ estados. Las probabilidades provienen de la matriz $\\hat{{P}}$ estimada 
mediante el modelo TCROC-Markov con datos reales (ene 2017 -- jul 2025).
Se muestran solo transiciones con probabilidad $\\geq {threshold}$.}}
\\label{{fig:apx_diagrama_{fuel_name.lower()}}}
\\end{{figure}}"""

    return tikz


def gen_table_P(fuel_name, P):
    """Genera tabla LaTeX con la matriz P real."""
    header = " & ".join([f"$s_{j+1}$" for j in range(K)])
    rows = []
    for i in range(K):
        vals = []
        for j in range(K):
            v = P[i, j]
            if v < 0.001:
                vals.append("$-$")
            elif v > 0.5:
                vals.append(f"\\textbf{{{v:.3f}}}")
            else:
                vals.append(f"{v:.3f}")
        rows.append(f"  $s_{i+1}$ & " + " & ".join(vals) + " \\\\")
    rows_str = "\n".join(rows)

    return f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Matriz de transici\\'on estimada $\\hat{{P}}$ para combustible \\textbf{{{fuel_name}}} ($K={K}$).}}
\\label{{tab:apx_P_{fuel_name.lower()}}}
\\begin{{tabular}}{{c|cccc}}
\\toprule
 & {header} \\\\
\\midrule
{rows_str}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Generando figuras Apendice A3 con datos REALES")
    print("=" * 60)

    sections = []

    for fuel in ["Super", "Regular"]:
        print(f"\n--- Procesando {fuel} ---")
        alpha, P, centroids, boundaries = load_fuel_data(fuel)
        states = assign_states(alpha, boundaries)

        print(f"  Alpha length: {len(alpha)}")
        print(f"  K = {K} (siempre 4 en el modelo)")
        print(f"  Centroids ({len(centroids)}): {np.round(centroids, 6)}")
        print(f"  P matrix ({P.shape}):")
        for row in P:
            print(f"    {np.round(row, 4)}")
        print(f"  States: { {i: int(np.sum(states==i)) for i in range(1, K+1)} }")

        # 1. Trayectorias
        traj_tikz = gen_tikz_trajectory(fuel, alpha, P, boundaries, window_size=20)
        sections.append(traj_tikz)

        # 2. Diagrama de transicion
        diag_tikz = gen_tikz_transition_diagram(fuel, P)
        sections.append(diag_tikz)

        # 3. Tabla P
        table_tex = gen_table_P(fuel, P)
        sections.append(table_tex)

        sections.append("")  # separador

    # ── Ensamblar archivo LaTeX ──
    header = r"""% ═══════════════════════════════════════════════════════════════════
% Ap\'endice A3 --- Figuras con datos REALES de combustibles
% Generado autom\'aticamente por generar_apendice_datos_reales.py
% Datos: TCROC-Markov, ene 2017 -- jul 2025
% ═══════════════════════════════════════════════════════════════════

"""
    content = header + "\n\n".join(sections)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n{'=' * 60}")
    print(f"Archivo generado: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
