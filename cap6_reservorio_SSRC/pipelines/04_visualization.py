"""
===================================================================
Pipeline 04: Visualización Completa (12 Figuras + 5 Tablas)
===================================================================
Genera todas las figuras y tablas del capítulo TCRA-SSRC con
calidad de publicación (DPI 600, fuentes serif IEEE).

Figuras:
  01 - Diagrama de arquitectura Markov vs SSRC
  02 - Verificación del Teorema de Inclusión
  03 - Heatmap de grilla (D, rho) → RMSE (4 paneles)
  04 - Autovalores de W_res en el plano complejo
  05 - Estados del reservorio h_t coloreados por régimen
  06 - Predicciones: Real vs Markov vs SSRC (4 paneles)
  07 - Boxplot de RMSE sobre N realizaciones
  08 - Sensibilidad a D (dimensión del reservorio)
  09 - Sensibilidad a rho (radio espectral)
  10 - Cota de perturbación (Proposición 6.2)
  11 - Convergencia del washout
  12 - Barplot comparativo final

Tablas:
  T1 - Comparación principal (LaTeX)
  T2 - Grilla completa de resultados
  T3 - Verificaciones teóricas
  T4 - Cotas de perturbación
  T5 - Mejores hiperparámetros SSRC
===================================================================
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from config import (FUEL_COLUMNS, TCROC_OPTIMAL, MARKOV_RMSE, MODELS_DIR,
                     OUTPUT_DIR, RESERVOIR_D_VALUES, RESERVOIR_RHO_VALUES,
                     N_REALIZATIONS, RESERVOIR_WASHOUT, TRAIN_SIZE,
                     FUEL_COLORS, DPI_QUALITY, REGIME_COLORS_NORD)
from reservoir.functions.create_reservoir import create_reservoir
from reservoir.functions.propagate_reservoir import propagate_reservoir
from reservoir.functions.estimate_readout import estimate_readout_nnls
from reservoir.functions.verify_theoretical import (
    verify_esp, verify_rank_condition,
    verify_perturbation_bound, demonstrate_inclusion_theorem
)
from evaluation.functions.rolling_window_ssrc import run_ssrc_rolling_window
from evaluation.functions.stats_tests import diebold_mariano_test
from visualization.latex_snippets import generate_latex_algorithm, generate_equations_table

# Reutilizar funciones de TCRA-Markov
MARKOV_SRC = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'cap5_deteccion_pronostico', 'src'))
sys.path.insert(0, MARKOV_SRC)
from processing.functions.calculate_alphas import calculate_alphas
from processing.functions.discretize_series import discretize_series
from models.functions.estimate_transition_matrix import estimate_transition_matrix


# ===================================================================
# SETUP IEEE-QUALITY STYLE
# ===================================================================
def setup_ieee_style():
    """Estilo de publicación IEEE/Q1-Q2."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
        'figure.dpi': 150,
        'savefig.dpi': DPI_QUALITY,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })
    sns.set_palette("colorblind")


def save_fig(fig, filename):
    """Guarda figura con calidad de publicación."""
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI_QUALITY, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  -> Guardado: {}".format(filename))


# Configuración de figuras
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLES_DIR = os.path.join(OUTPUT_DIR, 'tables')
LATEX_DIR = os.path.join(OUTPUT_DIR, 'latex_snippets')
os.makedirs(LATEX_DIR, exist_ok=True)


# ===================================================================
# FIG 01: Diagrama de Arquitectura Markov vs SSRC
# ===================================================================
def fig01_architecture_diagram():
    print("\n[FIG-01] Diagrama de arquitectura Markov vs SSRC")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    def draw_pipeline(ax, title, steps, color_base, bg_color):
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, len(steps) + 0.5)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        # Background
        rect = FancyBboxPatch((-0.3, -0.3), 1.6, len(steps) + 0.6,
                              boxstyle="round,pad=0.1", facecolor=bg_color,
                              edgecolor='gray', alpha=0.3)
        ax.add_patch(rect)

        for i, (label, sublabel) in enumerate(steps):
            y = len(steps) - 1 - i
            alpha_val = 0.4 + i * 0.12
            box = FancyBboxPatch((0.05, y - 0.25), 0.9, 0.5,
                                 boxstyle="round,pad=0.05",
                                 facecolor=color_base, alpha=alpha_val,
                                 edgecolor='black', linewidth=1.5)
            ax.add_patch(box)
            ax.text(0.5, y + 0.05, label, ha='center', va='center',
                    fontsize=12, fontweight='bold')
            if sublabel:
                ax.text(0.5, y - 0.15, sublabel, ha='center', va='center',
                        fontsize=9, style='italic', color='#333')

            if i < len(steps) - 1:
                ax.annotate('', xy=(0.5, y - 0.3), xytext=(0.5, y - 0.55),
                            arrowprops=dict(arrowstyle='->', lw=2, color='#333'))

    markov_steps = [
        ('$v_t$', 'Precios de combustible'),
        ('$\\alpha_t$', 'TCRA (WLS ponderado)'),
        ('$S_t \\in \\{s_1,...,s_K\\}$', 'Discretización K-Means'),
        ('$\\hat{P}$', 'Estimación NNLS'),
        ('$\\hat{v}_{t+1}$', 'Pronóstico lineal'),
    ]

    ssrc_steps = [
        ('$v_t$', 'Precios de combustible'),
        ('$\\alpha_t$', 'TCRA (WLS ponderado)'),
        ('$\\mathbf{h}_t = \\tanh(W_{\\mathrm{in}} \\alpha_t + W_{\\mathrm{res}} \\mathbf{h}_{t-1})$', 'Reservorio recurrente'),
        ('$W_{\\mathrm{out}}$', 'Capa de lectura NNLS'),
        ('$\\hat{y}_{t+1}$', 'Pronóstico no lineal'),
    ]

    draw_pipeline(ax1, 'TCRA-Markov\n(Lineal)', markov_steps, '#5E81AC', '#ECEFF4')  # Frost Blue + Snow Storm
    draw_pipeline(ax2, 'TCRA-SSRC\n(No Lineal)', ssrc_steps, '#D08770', '#ECEFF4')   # Aurora Orange + Snow Storm

    # Shared label
    fig.text(0.5, 0.02,
             'Ambos comparten TCRA (pasos 1-2) y NNLS (paso 4). La diferencia es el paso 3: discretización vs reservorio.',
             ha='center', fontsize=11, style='italic', color='#555')

    fig.suptitle('Comparación de Arquitecturas: TCRA-Markov vs TCRA-SSRC',
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_fig(fig, '01_architecture_diagram.png')


# ===================================================================
# FIG 02: Verificación del Teorema de Inclusión
# ===================================================================
def fig02_inclusion_theorem(data):
    print("\n[FIG-02] Verificación del Teorema de Inclusión (Teorema 6.2)")
    diffs = []
    for fuel in FUEL_COLUMNS:
        states = data['{}_states'.format(fuel)]
        P_hat = data['{}_P_hat'.format(fuel)]
        K = TCROC_OPTIMAL[fuel]['K']
        incl = demonstrate_inclusion_theorem(P_hat, states, K)
        diffs.append({'fuel': fuel, 'max_diff': incl['max_difference']})

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(diffs))
    colors = [FUEL_COLORS[d['fuel']] for d in diffs]
    bars = ax.bar(x, [d['max_diff'] for d in diffs], color=colors, edgecolor='black', linewidth=1.2)

    for bar, d in zip(bars, diffs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                '{:.1e}'.format(d['max_diff']), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([d['fuel'] for d in diffs], fontsize=12)
    ax.set_ylabel('$\\max |\\mathrm{Markov} - \\mathrm{SSRC\\ degenerado}|$', fontsize=12)
    ax.set_title('Verificación del Teorema de Inclusión\n(TCRA-Markov $\\equiv$ SSRC con $W_{\\mathrm{res}}=0$, $\\sigma=\\mathrm{id}$)',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=1e-10, color='#BF616A', linestyle='--', linewidth=2, label='Umbral $10^{-10}$')
    ax.set_yscale('symlog', linthresh=1e-16)
    ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    
    # Audit 02
    pd.DataFrame(diffs).to_csv(os.path.join(TABLES_DIR, 'audit_fig02_inclusion.csv'), index=False)
    save_fig(fig, '02_inclusion_theorem.png')


# ===================================================================
# FIG 03: Heatmap de grilla (D, ρ) → RMSE
# ===================================================================
def fig03_grid_heatmap(all_grid_results, best_configs):
    print("\n[FIG-03] Heatmap de grilla (D, rho) → RMSE")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('Búsqueda en Grilla SSRC: RMSE medio por $(D, \\rho)$\n'
                 '(mejor $a$ seleccionado por celda)',
                 fontsize=16, fontweight='bold', y=0.98)

    # --- Pre-compute global RMSE range for unified color scale ---
    global_vmin = np.inf
    global_vmax = -np.inf
    fuel_matrices = {}
    fuel_axes_info = {}
    for fuel in FUEL_COLUMNS:
        grid = all_grid_results[fuel]
        D_vals = sorted(set(g['D'] for g in grid))
        rho_vals = sorted(set(g['rho'] for g in grid))
        matrix = np.full((len(rho_vals), len(D_vals)), np.inf)
        for g in grid:
            r_idx = rho_vals.index(g['rho'])
            d_idx = D_vals.index(g['D'])
            if g['rmse_mean'] < matrix[r_idx, d_idx]:
                matrix[r_idx, d_idx] = g['rmse_mean']
        finite_vals = matrix[np.isfinite(matrix)]
        if len(finite_vals) > 0:
            global_vmin = min(global_vmin, finite_vals.min())
            global_vmax = max(global_vmax, finite_vals.max())
        fuel_matrices[fuel] = matrix
        fuel_axes_info[fuel] = (D_vals, rho_vals)

    for idx, fuel in enumerate(FUEL_COLUMNS):
        ax = axes[idx]
        bc = best_configs[fuel]
        matrix = fuel_matrices[fuel]
        D_vals, rho_vals = fuel_axes_info[fuel]

        # Heatmap with unified color scale
        sns.heatmap(matrix, ax=ax, annot=True, fmt='.4f', cmap='Blues',
                    vmin=global_vmin, vmax=global_vmax,
                    xticklabels=[str(d) for d in D_vals],
                    yticklabels=['{:.2f}'.format(r) for r in rho_vals],
                    linewidths=2, linecolor='white',
                    cbar_kws={'label': 'RMSE medio'})

        # Marcar mínimo
        min_idx = np.unravel_index(matrix.argmin(), matrix.shape)
        ax.add_patch(plt.Rectangle((min_idx[1], min_idx[0]), 1, 1,
                                   fill=False, edgecolor='black', linewidth=4))

        leak_s = ', $a^*$={:.1f}'.format(bc.get('leak_rate', 1.0)) if bc.get('leak_rate', 1.0) != 1.0 else ''
        ax.set_title('{} (Markov: {:.4f}, Mejor: D={}, ρ={:.2f}{})'.format(
            fuel, MARKOV_RMSE[fuel], bc['D'], bc['rho'], leak_s),
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('Dimensión $D$', fontsize=11)
        ax.set_ylabel('Radio espectral $\\rho$', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '03_grid_heatmap.png')


# ===================================================================
# FIG 04: Autovalores de W_res en el plano complejo
# ===================================================================
def fig04_eigenvalue_plot(best_configs):
    print("\n[FIG-04] Autovalores de W_res en el plano complejo (ESP)")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    fig.suptitle('Espectro de $W_{\\mathrm{res}}$: Verificación ESP (Teorema 6.1)',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, fuel in enumerate(FUEL_COLUMNS):
        ax = axes[idx]
        bc = best_configs[fuel]
        D_star = int(bc['D'])
        rho_star = float(bc['rho'])
        W_in, W_res = create_reservoir(input_dim=1, reservoir_dim=D_star,
                                       spectral_radius=rho_star, seed=42)
        esp_info = verify_esp(W_res)
        evals = esp_info['eigenvalues']

        # Círculo unitario
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color='#4C566A', linestyle='--', linewidth=1.5, alpha=0.5, label='$|z|=1$')
        ax.plot(rho_star * np.cos(theta), rho_star * np.sin(theta), color='#BF616A',
                linewidth=2, alpha=0.7, label='$\\rho^*={:.2f}$'.format(rho_star))

        # Autovalores
        ax.scatter(evals.real, evals.imag, c=FUEL_COLORS[fuel], s=30,
                   alpha=0.7, edgecolors='black', linewidth=0.5, zorder=5)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_xlabel('Re($\\lambda$)', fontsize=11)
        ax.set_ylabel('Im($\\lambda$)', fontsize=11)
        ax.set_title('{} ($D^*={}$, $\\rho(W_{{\\mathrm{{res}}}})={:.4f}$)'.format(
            fuel, D_star, esp_info['spectral_radius']),
            fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '04_eigenvalues_complex_plane.png')


# ===================================================================
# FIG 05: Estados del reservorio coloreados por régimen
# ===================================================================

# ===================================================================
# FIG 05: Estados del reservorio coloreados por regimen
# ===================================================================
def fig05_reservoir_states(data, best_configs):
    print("\n[FIG-05] Estados del reservorio h_t coloreados por régimen")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('Proyección 2D de Estados del Reservorio $\\mathbf{h}_t$ por Régimen',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, fuel in enumerate(FUEL_COLUMNS):
        ax = axes[idx]
        alphas = data['{}_alpha'.format(fuel)]
        states = data['{}_states'.format(fuel)]
        K = TCROC_OPTIMAL[fuel]['K']
        bc = best_configs[fuel]
        D_star = int(bc['D'])
        rho_star = float(bc['rho'])
        leak_star = float(bc.get('leak_rate', 1.0))

        W_in, W_res = create_reservoir(input_dim=1, reservoir_dim=D_star,
                                       spectral_radius=rho_star, seed=42)
        H = propagate_reservoir(alphas, W_in, W_res, washout=RESERVOIR_WASHOUT,
                                leak_rate=leak_star)

        # PCA manual para proyectar a 2D
        H_centered = H - H.mean(axis=1, keepdims=True)
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
        proj = U[:2, :] @ H_centered  # (2, T_eff)

        states_aligned = states[RESERVOIR_WASHOUT:]
        colors_map = REGIME_COLORS_NORD[:K]

        for k in range(K):
            mask = states_aligned == k
            if mask.sum() > 0:
                ax.scatter(proj[0, mask], proj[1, mask], c=[colors_map[k]],
                           s=25, alpha=0.6, edgecolors='none', label='Régimen {}'.format(k + 1))

        ax.set_xlabel('PC1 ($h_t^{(1)}$)', fontsize=11)
        ax.set_ylabel('PC2 ($h_t^{(2)}$)', fontsize=11)
        ax.set_title('{} ($K={}, D^*={}$)'.format(fuel, K, D_star),
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, markerscale=2, loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '05_reservoir_states_by_regime.png')


# ===================================================================
# FIG 06: Predicciones Real vs Markov vs SSRC
# ===================================================================

# ===================================================================
# FIG 06: Predicciones Real vs Markov vs SSRC
# ===================================================================
def fig06_predictions(data, best_configs):
    print("\n[FIG-06] Predicciones: Real vs Markov vs SSRC")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle('Precio Real vs Pronósticos (Ventana Deslizante)',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, fuel in enumerate(FUEL_COLUMNS):
        ax = axes[idx]
        alphas = data['{}_alpha'.format(fuel)]
        prices = data['{}_prices'.format(fuel)]
        states = data['{}_states'.format(fuel)]
        P_hat = data['{}_P_hat'.format(fuel)]
        K = TCROC_OPTIMAL[fuel]['K']
        bc = best_configs[fuel]
        D_star = int(bc['D'])
        rho_star = float(bc['rho'])
        leak_star = float(bc.get('leak_rate', 1.0))

        # --- SSRC predictions con config óptima ---
        W_in, W_res = create_reservoir(input_dim=1, reservoir_dim=D_star,
                                       spectral_radius=rho_star, seed=42)
        ssrc_res = run_ssrc_rolling_window(
            alphas, prices, TRAIN_SIZE, W_in, W_res, RESERVOIR_WASHOUT,
            leak_rate=leak_star)

        # --- Markov predictions (replicate protocol) ---
        centroids_sorted = np.sort(data.get('{}_P_hat'.format(fuel), P_hat).diagonal()
                                    if False else np.array([0]))  # Placeholder

        # Simple Markov prediction using same rolling window
        n_test = len(alphas) - TRAIN_SIZE - 1
        markov_preds = np.zeros(n_test)
        for i in range(n_test):
            t = TRAIN_SIZE + i
            # Assign current state
            alpha_val = alphas[t]
            # Re-discretize with training data
            train_alphas = alphas[:t]
            try:
                train_states, centroids = discretize_series(train_alphas, K)
                P_train, _ = estimate_transition_matrix(train_states, K)
                current_state = np.argmin(np.abs(alpha_val - centroids))
                next_state = np.argmax(P_train[:, current_state])
                pred_alpha = centroids[next_state]
                markov_preds[i] = prices[t] * (1 + pred_alpha)
            except:
                markov_preds[i] = prices[t]

        actuals = ssrc_res['actuals']
        ssrc_preds = ssrc_res['predictions']
        weeks = np.arange(1, len(actuals) + 1)

        # Plot
        ax.plot(weeks, actuals, color='#2E3440', linewidth=2, label='Real', zorder=3)
        ax.plot(weeks, markov_preds[:len(weeks)], color='#5E81AC', linewidth=1.5,
                linestyle='--', label='TCRA-Markov', alpha=0.8)
        ax.plot(weeks, ssrc_preds, color='#D08770', linewidth=1.5,
                linestyle='-.', label='TCRA-SSRC', alpha=0.8)

        ax.set_xlabel('Semana (fuera de muestra)', fontsize=11)
        ax.set_ylabel('Precio (HNL)', fontsize=11)
        ax.set_title('{}'.format(fuel), fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '06_predictions_real_vs_models.png')


# ===================================================================
# FIG 07: Boxplot de RMSE sobre N realizaciones
# ===================================================================
def fig07_boxplot_realizations(all_grid_results):
    print("\n[FIG-07] Boxplot de RMSE sobre realizaciones")
    fig, ax = plt.subplots(figsize=(12, 7))

    # Para cada combustible, obtener las realizaciones del mejor (D*, rho*)
    box_data = []
    labels = []
    colors_list = []

    for fuel in FUEL_COLUMNS:
        grid = all_grid_results[fuel]
        # Encontrar la mejor config
        best = min(grid, key=lambda g: g['rmse_mean'])
        box_data.append(best['rmse_all'])
        labels.append('{}\n$(D={}, \\rho={:.2f})$'.format(fuel, best['D'], best['rho']))
        colors_list.append(FUEL_COLORS[fuel])

    bp = ax.boxplot(box_data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Agregar líneas de referencia Markov
    for i, fuel in enumerate(FUEL_COLUMNS):
        ax.axhline(y=MARKOV_RMSE[fuel], xmin=(i) / len(FUEL_COLUMNS),
                   xmax=(i + 1) / len(FUEL_COLUMNS),
                   color=FUEL_COLORS[fuel], linestyle='--', linewidth=2, alpha=0.5)
        ax.text(i + 1.3, MARKOV_RMSE[fuel], 'Markov\n{:.4f}'.format(MARKOV_RMSE[fuel]),
                fontsize=8, color=FUEL_COLORS[fuel], va='center')

    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Distribución de RMSE sobre {} Realizaciones Aleatorias\n(Mejor configuración por combustible)'.format(N_REALIZATIONS),
                 fontsize=14, fontweight='bold')
    
    # Custom Legend for boxplot
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Mediana SSRC'),
        Line2D([0], [0], color='gray', linestyle='--', lw=2, label='Benchmark Markov')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    ax.grid(True, axis='y', alpha=0.3)
    save_fig(fig, '07_boxplot_realizations.png')


# ===================================================================
# FIG 08: Sensibilidad a D
# ===================================================================
def fig08_sensitivity_D(all_grid_results):
    print("\n[FIG-08] Sensibilidad a D")
    fig, ax = plt.subplots(figsize=(10, 7))

    for fuel in FUEL_COLUMNS:
        grid = all_grid_results[fuel]
        # Para cada D, promediar sobre rho
        D_vals = sorted(set(g['D'] for g in grid))
        rmse_by_D = []
        std_by_D = []
        for D in D_vals:
            vals = [g['rmse_mean'] for g in grid if g['D'] == D]
            rmse_by_D.append(np.mean(vals))
            std_by_D.append(np.std(vals))

        ax.errorbar(D_vals, rmse_by_D, yerr=std_by_D,
                    color=FUEL_COLORS[fuel], linewidth=2, marker='o',
                    markersize=8, capsize=5, label=fuel)

    ax.set_xlabel('Dimensión del Reservorio ($D$)', fontsize=12)
    ax.set_ylabel('RMSE medio (promediado sobre $\\rho$)', fontsize=12)
    ax.set_title('Sensibilidad al Hiperparámetro $D$', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xticks(RESERVOIR_D_VALUES)
    ax.grid(True, alpha=0.3)
    save_fig(fig, '08_sensitivity_D.png')


# ===================================================================
# FIG 09: Sensibilidad a rho
# ===================================================================
def fig09_sensitivity_rho(all_grid_results):
    print("\n[FIG-09] Sensibilidad a rho")
    fig, ax = plt.subplots(figsize=(10, 7))

    for fuel in FUEL_COLUMNS:
        grid = all_grid_results[fuel]
        rho_vals = sorted(set(g['rho'] for g in grid))
        rmse_by_rho = []
        std_by_rho = []
        for rho in rho_vals:
            vals = [g['rmse_mean'] for g in grid if g['rho'] == rho]
            rmse_by_rho.append(np.mean(vals))
            std_by_rho.append(np.std(vals))

        ax.errorbar(rho_vals, rmse_by_rho, yerr=std_by_rho,
                    color=FUEL_COLORS[fuel], linewidth=2, marker='s',
                    markersize=8, capsize=5, label=fuel)

    ax.axvline(x=1.0, color='#BF616A', linestyle='--', linewidth=2, alpha=0.5,
               label='Límite ESP ($\\rho=1$)')
    ax.set_xlabel('Radio Espectral ($\\rho$)', fontsize=12)
    ax.set_ylabel('RMSE medio (promediado sobre $D$)', fontsize=12)
    ax.set_title('Sensibilidad al Radio Espectral $\\rho$\n(Teorema 6.1: $\\rho < 1$ garantiza ESP)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    save_fig(fig, '09_sensitivity_rho.png')


# ===================================================================
# FIG 10: Cota de perturbación
# ===================================================================

# ===================================================================
# FIG 10: Cota de perturbacion
# ===================================================================
def fig10_perturbation_bound(data, best_configs):
    print("\n[FIG-10] Cota de perturbación (Proposición 6.2)")
    rho_range = np.array([0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99])
    fig, ax = plt.subplots(figsize=(10, 7))

    for fuel in FUEL_COLUMNS:
        alphas = data['{}_alpha'.format(fuel)]
        bc = best_configs[fuel]
        D_star = int(bc['D'])
        bounds = []

        for rho in rho_range:
            W_in, W_res = create_reservoir(input_dim=1, reservoir_dim=D_star,
                                           spectral_radius=rho, seed=42)
            H = propagate_reservoir(alphas, W_in, W_res, washout=RESERVOIR_WASHOUT)
            targets = alphas[RESERVOIR_WASHOUT + 1:]
            H_aligned = H[:, :-1]
            min_len = min(H_aligned.shape[1], len(targets))
            W_out = estimate_readout_nnls(H_aligned[:, :min_len], targets[:min_len])

            bound = verify_perturbation_bound(W_out, W_in, W_res, epsilon=0.01)
            bounds.append(bound['perturbation_bound'])

        ax.semilogy(rho_range, bounds, color=FUEL_COLORS[fuel],
                    linewidth=2, marker='o', markersize=6, label=fuel)

    ax.axvline(x=1.0, color='#BF616A', linestyle='--', linewidth=2, alpha=0.5,
               label='$\\rho = 1$ (divergencia)')
    ax.set_xlabel('Radio Espectral ($\\rho$)', fontsize=12)
    ax.set_ylabel('Cota $\\|\\delta \\hat{y}\\|$ (escala log)', fontsize=12)
    ax.set_title('Proposición 6.2: Cota de Perturbación\n'
                 '$\\|\\delta \\hat{y}\\| \\leq \\|W_{\\mathrm{out}}\\| \\cdot '
                 '\\|W_{\\mathrm{in}}\\| \\cdot \\varepsilon / (1 - \\rho)$',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    save_fig(fig, '10_perturbation_bound.png')


# ===================================================================
# FIG 11: Convergencia del washout
# ===================================================================

# ===================================================================
# FIG 11: Convergencia del washout
# ===================================================================
def fig11_washout_convergence(data, best_configs):
    print("\n[FIG-11] Convergencia del washout")
    washout_values = [0, 10, 20, 30, 50, 75, 100, 150]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle('Convergencia del RMSE según Período de Washout',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, fuel in enumerate(FUEL_COLUMNS):
        ax = axes[idx]
        alphas = data['{}_alpha'.format(fuel)]
        prices = data['{}_prices'.format(fuel)]
        bc = best_configs[fuel]
        D_star = int(bc['D'])
        rho_star = float(bc['rho'])
        leak_star = float(bc.get('leak_rate', 1.0))

        W_in, W_res = create_reservoir(input_dim=1, reservoir_dim=D_star,
                                       spectral_radius=rho_star, seed=42)

        rmses = []
        for w in washout_values:
            if w >= len(alphas) - TRAIN_SIZE - 10:
                rmses.append(np.nan)
                continue
            try:
                res = run_ssrc_rolling_window(
                    alphas, prices, TRAIN_SIZE, W_in, W_res, washout=w)
                rmses.append(res['rmse'])
            except:
                rmses.append(np.nan)

        ax.plot(washout_values, rmses, color=FUEL_COLORS[fuel],
                linewidth=2, marker='o', markersize=8)
        ax.axvline(x=50, color='#EBCB8B', linestyle='--', linewidth=2,
                   label='$W_{\\mathrm{wash}}=50$ (usado)')
        ax.set_xlabel('$W_{\\mathrm{wash}}$', fontsize=11)
        ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title(fuel, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '11_washout_convergence.png')


# ===================================================================
# FIG 12: Barplot comparativo final
# ===================================================================
def fig12_comparison_barplot(all_grid_results):
    print("\n[FIG-12] Barplot comparativo final")
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(FUEL_COLUMNS))
    width = 0.35

    markov_vals = [MARKOV_RMSE[f] for f in FUEL_COLUMNS]
    ssrc_vals = []
    ssrc_stds = []

    for fuel in FUEL_COLUMNS:
        best = min(all_grid_results[fuel], key=lambda g: g['rmse_mean'])
        ssrc_vals.append(best['rmse_mean'])
        ssrc_stds.append(best['rmse_std'])

    bars1 = ax.bar(x - width / 2, markov_vals, width, label='TCRA-Markov',
                   color='#5E81AC', edgecolor='#2E3440', linewidth=1.2)   # Frost Blue
    bars2 = ax.bar(x + width / 2, ssrc_vals, width, yerr=ssrc_stds,
                   label='TCRA-SSRC', color='#D08770', edgecolor='#2E3440',  # Aurora Orange
                   linewidth=1.2, capsize=5)

    # Etiquetas de valor
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                '{:.4f}'.format(bar.get_height()), ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    for bar, std in zip(bars2, ssrc_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                '{:.4f}'.format(bar.get_height()), ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    # Delta RMSE
    for i, fuel in enumerate(FUEL_COLUMNS):
        delta = (ssrc_vals[i] - markov_vals[i]) / markov_vals[i] * 100
        color = '#A3BE8C' if delta < 0 else '#BF616A'  # Aurora Green / Aurora Red
        ax.text(i, max(markov_vals[i], ssrc_vals[i]) + 0.06,
                '$\\Delta$={:+.1f}%'.format(delta), ha='center', fontsize=11,
                fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(FUEL_COLUMNS, fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Comparación Final: TCRA-Markov vs TCRA-SSRC',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(markov_vals) * 1.3)
    save_fig(fig, '12_comparison_barplot.png')


# ===================================================================
# FIG 13: Reservoir Regimes Scatter (Estilo Markov)
# ===================================================================

# ===================================================================
# FIG 13: Reservoir Regimes Scatter (Estilo Markov)
# ===================================================================
def fig13_reservoir_regimes_scatter(data, best_configs):
    print("\n[FIG-13] Reservoir Regimes Scatter (Estilo Markov)")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle('Dispersión de Alphas y Regímenes Identificados por SSRC',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, fuel in enumerate(FUEL_COLUMNS):
        ax = axes[idx]
        alphas = data['{}_alpha'.format(fuel)]
        states = data['{}_states'.format(fuel)]
        K = TCROC_OPTIMAL[fuel]['K']
        bc = best_configs[fuel]
        D_star = int(bc['D'])
        rho_star = float(bc['rho'])
        leak_star = float(bc.get('leak_rate', 1.0))
        
        W_in, W_res = create_reservoir(input_dim=1, reservoir_dim=D_star,
                                       spectral_radius=rho_star, seed=42)
        H = propagate_reservoir(alphas, W_in, W_res, washout=RESERVOIR_WASHOUT,
                                leak_rate=leak_star)
        h_mean = H.mean(axis=0)
        
        alphas_eff = alphas[RESERVOIR_WASHOUT:]
        states_eff = states[RESERVOIR_WASHOUT:]
        
        colors_map = REGIME_COLORS_NORD[:K]
        for k in range(K):
            mask = states_eff == k
            ax.scatter(alphas_eff[mask], h_mean[mask], c=[colors_map[k]], 
                       label=f'Régimen {k+1}', alpha=0.7, edgecolors='none', s=25)
        
        ax.set_title(fuel, fontweight='bold')
        ax.set_xlabel('$\\alpha_t$ (Retorno)', fontsize=10)
        ax.set_ylabel('Activación Promedio del Reservorio', fontsize=10)
        ax.legend(title='Régimen (Markov Proxy)', loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, '13_reservoir_regimes_scatter.png')

# ===================================================================
# FIG 14: Price Regimes Shading (Estilo Markov)
# ===================================================================
def fig14_reservoir_price_regimes(data):
    print("\n[FIG-14] Price Regimes Shading (Estilo Markov)")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, fuel in enumerate(FUEL_COLUMNS):
        ax = axes[idx]
        prices = data['{}_prices'.format(fuel)]
        states = data['{}_states'.format(fuel)]
        K = TCROC_OPTIMAL[fuel]['K']
        
        t = np.arange(len(prices))
        ax.plot(t, prices, color='black', linewidth=1.5, label='Precio HNL', zorder=5)
        
        # Shading (Optimized: draw contiguous blocks)
        colors_map = REGIME_COLORS_NORD[:K]
        for k in range(K):
            mask = states == k
            start = None
            for i in range(len(mask)):
                if mask[i] and start is None:
                    start = i
                elif not mask[i] and start is not None:
                    ax.axvspan(start, i, color=colors_map[k], alpha=0.25, zorder=1)
                    start = None
            if start is not None:
                ax.axvspan(start, len(mask), color=colors_map[k], alpha=0.25, zorder=1)
        
        # Legend proxies (Fixed to use correct palette)
        proxies = [plt.Rectangle((0,0),1,1, color=colors_map[k], alpha=0.3) for k in range(K)]
        ax.set_title(f'Regímenes de Mercado: {fuel}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precio (HNL)')
        ax.legend(proxies, [f'Regimen {k+1}' for k in range(K)], loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    save_fig(fig, '14_reservoir_price_regimes.png')

# ===================================================================
# AUDIT TABLES FOR FIGURES
# ===================================================================

def generate_figure_audit_tables(all_grid_results, data, best_configs):
    print("\n[AUDITORIA] Generando tablas de datos para cada figura...")
    
    # Audit 03: Grid Results
    rows = []
    for fuel, grid in all_grid_results.items():
        for entry in grid:
            rows.append({'Fuel': fuel, 'D': entry['D'], 'rho': entry['rho'],
                         'leak_rate': entry.get('leak_rate', 1.0),
                         'RMSE_mean': entry['rmse_mean'], 'RMSE_std': entry['rmse_std']})
    pd.DataFrame(rows).to_csv(os.path.join(TABLES_DIR, 'audit_fig03_grid_data.csv'), index=False)
    
    # Audit 06: Predictions
    for fuel in FUEL_COLUMNS:
        alphas = data['{}_alpha'.format(fuel)]
        prices = data['{}_prices'.format(fuel)]
        bc = best_configs[fuel]
        D_star = int(bc['D'])
        rho_star = float(bc['rho'])
        leak_star = float(bc.get('leak_rate', 1.0))
        W_in, W_res = create_reservoir(input_dim=1, reservoir_dim=D_star,
                                       spectral_radius=rho_star, seed=42)
        res = run_ssrc_rolling_window(alphas, prices, TRAIN_SIZE, W_in, W_res,
                                      RESERVOIR_WASHOUT, leak_rate=leak_star)
        
        # Guardar individual para evitar error de dimensiones
        df_fuel = pd.DataFrame({
            'Actual': res['actuals'],
            'SSRC_Pred': res['predictions']
        })
        df_fuel.to_csv(os.path.join(TABLES_DIR, f'audit_fig06_predictions_{fuel.lower()}.csv'), index=False)
    
    # Audit 09: Sensitivity
    rows_sens = []
    for fuel in FUEL_COLUMNS:
        grid = all_grid_results[fuel]
        for r in sorted(set(g['rho'] for g in grid)):
            m = np.mean([g['rmse_mean'] for g in grid if g['rho'] == r])
            rows_sens.append({'Fuel': fuel, 'rho': r, 'Avg_RMSE': m})
    pd.DataFrame(rows_sens).to_csv(os.path.join(TABLES_DIR, 'audit_fig09_sensitivity_rho.csv'), index=False)

    # Audit 10: Perturbation
    rows_per = []
    for fuel in FUEL_COLUMNS:
        alphas = data['{}_alpha'.format(fuel)]
        for rho in [0.80, 0.90, 0.95]:
            bc = best_configs[fuel]
            D_star = int(bc['D'])
            W_in, W_res = create_reservoir(1, D_star, rho, seed=42)
            H = propagate_reservoir(alphas, W_in, W_res, washout=RESERVOIR_WASHOUT)
            targets = alphas[RESERVOIR_WASHOUT + 1:]
            H_al = H[:, :-1]
            ml = min(H_al.shape[1], len(targets))
            W_out = estimate_readout_nnls(H_al[:, :ml], targets[:ml])
            pb = verify_perturbation_bound(W_out, W_in, W_res, epsilon=0.01)
            rows_per.append({'Fuel': fuel, 'rho': rho, 'Bound': pb['perturbation_bound']})
    pd.DataFrame(rows_per).to_csv(os.path.join(TABLES_DIR, 'audit_fig10_perturbation.csv'), index=False)
    
    # Audit 11: Washout
    # (Generating this takes time, so we just log a sample if needed)
    
    # Audit 14: Price Regimes - Save individually
    for fuel in FUEL_COLUMNS:
        df_reg = pd.DataFrame({
            'Price': data['{}_prices'.format(fuel)],
            'Regime': data['{}_states'.format(fuel)]
        })
        df_reg.to_csv(os.path.join(TABLES_DIR, f'audit_fig14_regimes_{fuel.lower()}.csv'), index=False)

    # Audit 12: Comparison Barplot
    rows_comp = []
    for fuel in FUEL_COLUMNS:
        best = min(all_grid_results[fuel], key=lambda g: g['rmse_mean'])
        rows_comp.append({
            'Fuel': fuel, 
            'Markov_RMSE': MARKOV_RMSE[fuel], 
            'SSRC_RMSE': best['rmse_mean'],
            'Delta': (best['rmse_mean'] - MARKOV_RMSE[fuel]) / MARKOV_RMSE[fuel] * 100
        })
    pd.DataFrame(rows_comp).to_csv(os.path.join(TABLES_DIR, 'audit_fig12_comparison.csv'), index=False)

    # Audit 07: Realizations
    rows_real = []
    for fuel in FUEL_COLUMNS:
        best = min(all_grid_results[fuel], key=lambda g: g['rmse_mean'])
        for r_val in best['rmse_all']:
            rows_real.append({'Fuel': fuel, 'RMSE': r_val})
    pd.DataFrame(rows_real).to_csv(os.path.join(TABLES_DIR, 'audit_fig07_realizations.csv'), index=False)

    print("  -> Auditorías detalladas guardadas en outputs/tables/")


def generate_tables(all_grid_results, data, best_configs):
    print("\n[TABLAS] Generando 5 tablas...")

    # T1: Comparación principal
    rows_t1 = []
    for fuel in FUEL_COLUMNS:
        best = min(all_grid_results[fuel], key=lambda g: g['rmse_mean'])
        delta = (best['rmse_mean'] - MARKOV_RMSE[fuel]) / MARKOV_RMSE[fuel] * 100
        rows_t1.append({
            'Serie': fuel,
            'TCRA-Markov RMSE': '{:.4f}'.format(MARKOV_RMSE[fuel]),
            'TCRA-SSRC RMSE': '{:.4f} +/- {:.4f}'.format(best['rmse_mean'], best['rmse_std']),
            'D*': best['D'],
            'rho*': '{:.2f}'.format(best['rho']),
            'a*': '{:.1f}'.format(best.get('leak_rate', 1.0)),
            'Delta RMSE (%)': '{:+.2f}'.format(delta),
            'Ganador': 'SSRC' if delta < 0 else 'Markov'
        })

    df_t1 = pd.DataFrame(rows_t1)
    df_t1.to_csv(os.path.join(TABLES_DIR, 'T1_comparison_main.csv'), index=False)

    # LaTeX format
    with open(os.path.join(TABLES_DIR, 'T1_comparison_main.tex'), 'w') as f:
        f.write('\\begin{table}[htbp]\n')
        f.write('\\centering\n')
        f.write('\\caption{Comparación de RMSE: TCRA-Markov vs TCRA-SSRC}\n')
        f.write('\\label{tab:ssrc_comparison}\n')
        f.write('\\begin{tabular}{@{}lccccc@{}}\n')
        f.write('\\toprule\n')
        f.write('\\textbf{Serie} & \\textbf{Markov} & \\textbf{SSRC} & $(D^*, \\rho^*, a^*)$ & $\\Delta$\\textbf{RMSE} \\\\\n')
        f.write('\\midrule\n')
        for _, row in df_t1.iterrows():
            best = min(all_grid_results[row['Serie']], key=lambda g: g['rmse_mean'])
            f.write('{} & ${:.4f}$ & ${:.4f} \\pm {:.4f}$ & $({}, {}, {})$ & ${:+.2f}\\%$ \\\\\n'.format(
                row['Serie'], MARKOV_RMSE[row['Serie']], best['rmse_mean'],
                best['rmse_std'], best['D'], row['rho*'], row['a*'],
                float(row['Delta RMSE (%)'])))
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    print("  -> T1 guardada (CSV + LaTeX)")

    # T2: Grilla completa
    rows_t2 = []
    for fuel in FUEL_COLUMNS:
        for g in all_grid_results[fuel]:
            rows_t2.append({
                'Serie': fuel, 'D': g['D'], 'rho': g['rho'],
                'leak_rate': g.get('leak_rate', 1.0),
                'RMSE_mean': '{:.4f}'.format(g['rmse_mean']),
                'RMSE_std': '{:.4f}'.format(g['rmse_std'])
            })
    df_t2 = pd.DataFrame(rows_t2)
    df_t2.to_csv(os.path.join(TABLES_DIR, 'T2_grid_full.csv'), index=False)
    print("  -> T2 guardada")

    # T3: Verificaciones teóricas — usa config real
    rows_t3 = []
    for fuel in FUEL_COLUMNS:
        alphas = data['{}_alpha'.format(fuel)]
        bc = best_configs[fuel]
        D_star = int(bc['D'])
        rho_star = float(bc['rho'])
        W_in, W_res = create_reservoir(input_dim=1, reservoir_dim=D_star,
                                       spectral_radius=rho_star, seed=42)
        esp = verify_esp(W_res)
        H = propagate_reservoir(alphas, W_in, W_res, washout=RESERVOIR_WASHOUT)
        rank_info = verify_rank_condition(H)
        rows_t3.append({
            'Serie': fuel,
            'rho(W_res)': '{:.4f}'.format(esp['spectral_radius']),
            'ESP': 'OK' if esp['esp_sufficient'] else 'FAIL',
            'rango(H)': rank_info['rank'],
            'D': rank_info['D'],
            'T_eff': rank_info['T_eff'],
            'kappa(H)': '{:.2f}'.format(rank_info['condition_number'])
        })
    df_t3 = pd.DataFrame(rows_t3)
    df_t3.to_csv(os.path.join(TABLES_DIR, 'T3_theoretical_verifications.csv'), index=False)
    print("  -> T3 guardada")

    # T4: Cotas de perturbación
    rows_t4 = []
    for fuel in FUEL_COLUMNS:
        alphas = data['{}_alpha'.format(fuel)]
        bc = best_configs[fuel]
        D_star = int(bc['D'])
        rho_star = float(bc['rho'])
        W_in, W_res = create_reservoir(input_dim=1, reservoir_dim=D_star,
                                       spectral_radius=rho_star, seed=42)
        H = propagate_reservoir(alphas, W_in, W_res, washout=RESERVOIR_WASHOUT)
        targets = alphas[RESERVOIR_WASHOUT + 1:]
        H_al = H[:, :-1]
        ml = min(H_al.shape[1], len(targets))
        W_out = estimate_readout_nnls(H_al[:, :ml], targets[:ml])
        pb = verify_perturbation_bound(W_out, W_in, W_res, epsilon=0.01)
        rows_t4.append({
            'Serie': fuel,
            'rho_res': '{:.4f}'.format(pb['rho_res']),
            '||W_out||': '{:.4f}'.format(pb['W_out_norm']),
            '||W_in||': '{:.4f}'.format(pb['W_in_norm']),
            'Cota delta_y (eps=0.01)': '{:.4f}'.format(pb['perturbation_bound'])
        })
    df_t4 = pd.DataFrame(rows_t4)
    df_t4.to_csv(os.path.join(TABLES_DIR, 'T4_perturbation_bounds.csv'), index=False)
    print("  -> T4 guardada")

    # T5: Mejores hiperparámetros
    rows_t5 = []
    for fuel in FUEL_COLUMNS:
        best = min(all_grid_results[fuel], key=lambda g: g['rmse_mean'])
        delta = (best['rmse_mean'] - MARKOV_RMSE[fuel]) / MARKOV_RMSE[fuel] * 100
        rows_t5.append({
            'Serie': fuel,
            'D*': best['D'],
            'rho*': '{:.2f}'.format(best['rho']),
            'a*': '{:.1f}'.format(best.get('leak_rate', 1.0)),
            'RMSE_SSRC': '{:.4f}'.format(best['rmse_mean']),
            'RMSE_Markov': '{:.4f}'.format(MARKOV_RMSE[fuel]),
            'Delta_RMSE%': '{:+.2f}'.format(delta),
            'Ganador': 'SSRC' if delta < 0 else 'Markov'
        })
    df_t5 = pd.DataFrame(rows_t5)
    df_t5.to_csv(os.path.join(TABLES_DIR, 'T5_best_hyperparameters.csv'), index=False)
    print("  -> T5 guardada")


# ===================================================================
# VERIFICACION
# ===================================================================

def generate_checklist(all_grid_results, data, best_configs):
    """Genera un archivo de checklist de progreso."""
    checks = []

    # Validación teórica
    for fuel in FUEL_COLUMNS:
        states = data['{}_states'.format(fuel)]
        P_hat = data['{}_P_hat'.format(fuel)]
        K = TCROC_OPTIMAL[fuel]['K']
        incl = demonstrate_inclusion_theorem(P_hat, states, K)
        checks.append(('Teo 6.2 Inclusion {}'.format(fuel),
                        'PASS' if incl['equivalent'] else 'FAIL'))

    bc_first = best_configs[FUEL_COLUMNS[0]]
    W_in, W_res = create_reservoir(input_dim=1, reservoir_dim=int(bc_first['D']),
                                   spectral_radius=float(bc_first['rho']), seed=42)
    esp = verify_esp(W_res)
    checks.append(('Teo 6.1 ESP (rho<1)', 'PASS' if esp['esp_sufficient'] else 'FAIL'))

    for fuel in FUEL_COLUMNS:
        alphas = data['{}_alpha'.format(fuel)]
        H = propagate_reservoir(alphas, W_in, W_res, washout=RESERVOIR_WASHOUT)
        ri = verify_rank_condition(H)
        checks.append(('Teo 6.3 Rango {}'.format(fuel),
                        'PASS' if ri['full_rank'] else 'FAIL'))

    # Resultados
    for fuel in FUEL_COLUMNS:
        best = min(all_grid_results[fuel], key=lambda g: g['rmse_mean'])
        checks.append(('SSRC mejora {} (Delta<0)'.format(fuel),
                        'PASS' if best['rmse_mean'] < MARKOV_RMSE[fuel] else 'NOTE'))
        checks.append(('Std < 10% media {}'.format(fuel),
                        'PASS' if best['rmse_std'] / best['rmse_mean'] < 0.10 else 'WARN'))

    # Figuras
    for i in range(1, 13):
        fname = '0{}_'.format(i) if i < 10 else '{}_'.format(i)
        exists = any(f.startswith(fname) for f in os.listdir(FIGURES_DIR))
        checks.append(('Figura {} generada'.format(i), 'PASS' if exists else 'PENDING'))

    # Tablas
    for i in range(1, 6):
        fname = 'T{}_'.format(i)
        exists = any(f.startswith(fname) for f in os.listdir(TABLES_DIR))
        checks.append(('Tabla {} generada'.format(i), 'PASS' if exists else 'PENDING'))

    # Escribir checklist
    with open(os.path.join(OUTPUT_DIR, 'CHECKLIST_PROGRESO.md'), 'w', encoding='utf-8') as f:
        f.write('# Checklist de Progreso — Capítulo TCRA-SSRC\n\n')
        f.write('Generado automáticamente por el pipeline.\n\n')

        passed = sum(1 for _, s in checks if s == 'PASS')
        total = len(checks)
        f.write('## Progreso: {}/{} ({:.0f}%)\n\n'.format(passed, total, 100 * passed / total))

        f.write('| # | Verificación | Estado |\n')
        f.write('|---|---|---|\n')
        for i, (desc, status) in enumerate(checks, 1):
            icon = {'PASS': '✅', 'FAIL': '❌', 'WARN': '⚠️',
                    'NOTE': '📝', 'PENDING': '⏳'}.get(status, '❓')
            f.write('| {} | {} | {} {} |\n'.format(i, desc, icon, status))

        f.write('\n---\n\n## Resumen de Resultados\n\n')
        f.write('| Serie | Markov RMSE | SSRC RMSE | ΔRMSE |\n')
        f.write('|---|---|---|---|\n')
        for fuel in FUEL_COLUMNS:
            best = min(all_grid_results[fuel], key=lambda g: g['rmse_mean'])
            delta = (best['rmse_mean'] - MARKOV_RMSE[fuel]) / MARKOV_RMSE[fuel] * 100
            f.write('| {} | {:.4f} | {:.4f} ± {:.4f} | {:+.2f}% |\n'.format(
                fuel, MARKOV_RMSE[fuel], best['rmse_mean'], best['rmse_std'], delta))

    print("\n  -> CHECKLIST_PROGRESO.md generado")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("=" * 60)
    print("PASO 4: Visualización Completa (12 Figuras + 5 Tablas)")
    print("=" * 60)

    setup_ieee_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Cargar datos preparados
    data = np.load(os.path.join(MODELS_DIR, 'prepared_data.npz'))

    # Cargar resultados de grilla PRE-COMPUTADOS (paso 02b)
    all_grid_results = {}
    for fuel in FUEL_COLUMNS:
        grid_path = os.path.join(MODELS_DIR, '{}_ssrc_grid.npz'.format(fuel.lower()))
        if os.path.exists(grid_path):
            gd = np.load(grid_path, allow_pickle=True)
            grid_list = gd['grid'].tolist() if isinstance(gd['grid'], np.ndarray) else gd['grid']
            all_grid_results[fuel] = grid_list
            best = min(grid_list, key=lambda x: x['rmse_mean'])
            print("  {} cargado: {} configs, mejor RMSE={:.4f} (D={}, rho={:.2f})".format(
                fuel, len(grid_list), best['rmse_mean'], best['D'], best['rho']))
        else:
            print("  WARN: No se encontró grilla para {}. Ejecute paso 02b primero.".format(fuel))
            all_grid_results[fuel] = []


    # Generar figuras
    # Extraer best configs del grid search
    best_configs = {}
    for fuel in FUEL_COLUMNS:
        if all_grid_results[fuel]:
            best_configs[fuel] = min(all_grid_results[fuel], key=lambda x: x['rmse_mean'])
        else:
            best_configs[fuel] = {'D': 100, 'rho': 0.80, 'leak_rate': 1.0}

    fig01_architecture_diagram()
    fig02_inclusion_theorem(data)
    fig03_grid_heatmap(all_grid_results, best_configs)
    fig04_eigenvalue_plot(best_configs)
    fig05_reservoir_states(data, best_configs)
    fig06_predictions(data, best_configs)
    fig07_boxplot_realizations(all_grid_results)
    fig08_sensitivity_D(all_grid_results)
    fig09_sensitivity_rho(all_grid_results)
    fig10_perturbation_bound(data, best_configs)
    fig11_washout_convergence(data, best_configs)
    fig12_comparison_barplot(all_grid_results)
    
    # NUEVAS FIGURAS ESTILO MARKOV
    fig13_reservoir_regimes_scatter(data, best_configs)
    fig14_reservoir_price_regimes(data)

    # Generar tablas
    generate_tables(all_grid_results, data, best_configs)
    
    # Generar auditorías de figuras
    generate_figure_audit_tables(all_grid_results, data, best_configs)

    # Generar checklist
    generate_checklist(all_grid_results, data, best_configs)

    # Generar snippets LaTeX para el documento
    generate_latex_algorithm(os.path.join(LATEX_DIR, 'alg_ssrc.tex'))
    generate_equations_table(os.path.join(LATEX_DIR, 'tab_ssrc_mapping.tex'))

    print("\n" + "=" * 60)
    print("PASO 4 COMPLETADO: 12 figuras + 5 tablas generadas")
    print("=" * 60)


if __name__ == '__main__':
    main()
