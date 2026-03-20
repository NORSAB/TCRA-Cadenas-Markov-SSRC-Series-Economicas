import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from scipy.stats import mode
from PIL import Image

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    FUEL_COLORS, FIGURES_DIR, REQUIRED_COLUMNS, 
    MODELS_DIR, TABLES_DIR, REGIME_COLORS_NORD
)
from src.visualization.functions.setup_plotting_style import setup_plotting_style
from src.visualization.functions.style_subplot import style_subplot
from src.visualization.functions.plot_transition_graph import plot_transition_graph, plot_quantile_transition_graph
from src.models.functions.simulate_markov_evolution import simulate_markov_evolution
from src.models.functions.assign_regime_names import assign_regime_names
from src.processing.functions.calculate_alphas import calculate_alphas

def plot_metric_over_k(optimization_results, metric_key, metric_label, plot_title, color, filename, is_higher_better):
    """
    Generates and saves a 2x2 subplot figure for a given optimization metric.
    Matching Articulol.py exactly.
    """
    print(f"Generating plot: {plot_title}...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle(plot_title, fontsize=18)

    # Iterate through each fuel series' results
    for i, fuel in enumerate(REQUIRED_COLUMNS):
        if fuel not in optimization_results: continue
        ax = axes[i]
        results = optimization_results[fuel]
        k_range = results['k_range']
        scores = results[metric_key]

        if is_higher_better:
            optimal_idx = np.argmax(scores)
        else:
            optimal_idx = np.argmin(scores)
        
        optimal_k = k_range[optimal_idx]
        optimal_score = scores[optimal_idx]

        ax.plot(k_range, scores, marker='o', linestyle='-', color=color)
        ax.axvline(x=optimal_k, color='gold', linestyle='--', linewidth=2, label=f'k Óptimo = {optimal_k}')
        ax.plot(optimal_k, optimal_score, '*', color='gold', markersize=20, markeredgecolor='black', zorder=5)

        ax.set_title(f'{metric_label} para {fuel}', fontsize=14)
        ax.set_xlabel('Número de Regímenes (k)', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True)
        
        if 'accuracies' in metric_key or 'Accuracy' in metric_label:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=600)
    plt.close()

def run_visualization():
    print("\n--- Pipeline 04: Visualización Completa (Diseño exacto Articulol.py) ---")
    setup_plotting_style()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # 1. FIG-00: Price Trends (Summary Chart)
    silver_path = "data/silver/fuel_data_standardized.csv"
    if os.path.exists(silver_path):
        data = pd.read_csv(silver_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').drop_duplicates('Date')
        date_min, date_max = data['Date'].min(), data['Date'].max()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        for idx, (fuel, color) in enumerate(FUEL_COLORS.items()):
            ax = axes[idx]
            y = data[fuel].values
            # Eliminar NaNs
            mask = ~np.isnan(y)
            if mask.sum() < 4: continue # Skip if not enough points
            y = y[mask]
            x_dates = data['Date'].values[mask]
            
            x_numeric = mdates.date2num(x_dates)
            if len(x_numeric) > 500: # Downsample if too many
                import scipy.interpolate
                x_smooth = np.linspace(x_numeric.min(), x_numeric.max(), 500)
                spline = scipy.interpolate.make_interp_spline(x_numeric, y, k=3)
                y_smooth = spline(x_smooth)
                x_smooth_dates = mdates.num2date(x_smooth)
                ax.fill_between(x_smooth_dates, y_smooth, color=color, alpha=0.3)
                ax.plot(x_smooth_dates, y_smooth, color=color, linewidth=3)
            else:
                ax.fill_between(x_dates, y, color=color, alpha=0.3)
                ax.plot(x_dates, y, color=color, linewidth=3)

            style_subplot(ax, title=fuel, date_min=date_min, date_max=date_max)
        plt.tight_layout(pad=2.0)
        plt.savefig(os.path.join(FIGURES_DIR, "fuel_prices_summary_600dpi.png"), dpi=600, bbox_inches='tight')
        plt.close()

    # 2. Optimization Result Retrieval (STEP 4 logic)
    csv_summary = os.path.join(TABLES_DIR, "21_grid_search_summary.csv")
    csv_best = os.path.join(TABLES_DIR, "22_best_hyperparameters.csv")
    
    optimization_results = {}
    optimal_ks = {}
    
    if os.path.exists(csv_best) and os.path.exists(csv_summary):
        best_df = pd.read_csv(csv_best)
        summary_df = pd.read_csv(csv_summary)
        
        for fuel_name in REQUIRED_COLUMNS:
            best_row = best_df[best_df['Combustible'] == fuel_name]
            if best_row.empty: continue
            
            W_opt, L_opt, k_opt = best_row.iloc[0]['W'], best_row.iloc[0]['Lambda'], best_row.iloc[0]['k']
            optimal_ks[fuel_name] = k_opt
            
            mask = (
                (summary_df['Combustible'] == fuel_name) &
                (summary_df['W'] == W_opt) &
                (np.isclose(summary_df['Lambda'], L_opt, atol=1e-9))
            )
            per_k_df = summary_df[mask].sort_values('k')
            
            optimization_results[fuel_name] = {
                'k_range': per_k_df['k'].tolist(),
                'aic_scores': per_k_df['AIC'].tolist(),
                'avg_accuracies': per_k_df['Exactitud'].tolist(),
                'avg_rmses': per_k_df['RMSE'].tolist()
            }

    if optimization_results:
        plot_metric_over_k(optimization_results, 'aic_scores', 'Puntaje AIC', 'Métrica 1: Puntaje AIC vs. Número de Regímenes (k)', '#BF616A', "1_aic_plot.png", is_higher_better=False)
        plot_metric_over_k(optimization_results, 'avg_accuracies', 'Exactitud Promedio', 'Métrica 2: Exactitud Promedio vs. Número de Regímenes (k)', '#5E81AC', "2_accuracy_plot.png", is_higher_better=True)
        plot_metric_over_k(optimization_results, 'avg_rmses', 'RMSE Promedio', 'Métrica 3: RMSE Promedio vs. Número de Regímenes (k)', '#A3BE8C', "3_rmse_plot.png", is_higher_better=False)

        # Plot 4 Bubble Chart
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        fig.subplots_adjust(top=0.91, hspace=0.35)
        fig.suptitle('Optimización Consolidada: Balance entre Métricas', fontsize=20, y=0.95)
        for i, fuel in enumerate(REQUIRED_COLUMNS):
            if fuel not in optimization_results: continue
            ax = axes[i]
            res = optimization_results[fuel]
            k_range, aic_scores, avg_accuracies, avg_rmses = np.array(res['k_range']), np.array(res['aic_scores']), np.array(res['avg_accuracies']), np.array(res['avg_rmses'])
            bubble_sizes = (1 / (avg_rmses + 0.01)) / (1 / (avg_rmses + 0.01)).max() * 1000
            idx_best = np.where(k_range == optimal_ks[fuel])[0][0]
            for j in range(len(k_range)):
                color = '#EBCB8B' if j == idx_best else '#4C566A'  # Aurora Yellow vs Nord gray
                ax.scatter(aic_scores[j], avg_accuracies[j], s=bubble_sizes[j], c=color, alpha=0.8, edgecolors='w', linewidth=1, zorder=3)
                ax.text(aic_scores[j], avg_accuracies[j], str(k_range[j]), ha='center', va='center', fontsize=12, fontweight='bold', zorder=4)
            ax.axvline(x=aic_scores[idx_best], color='#EBCB8B', linestyle='--', linewidth=2)
            ax.axhline(y=avg_accuracies[idx_best], color='#EBCB8B', linestyle='--', linewidth=2)
            ax.set_title(f'Métricas para {fuel}', fontsize=16)
            ax.set_xlabel('Puntaje AIC (Menor es Mejor)', fontsize=14)
            ax.set_ylabel('Exactitud Promedio', fontsize=14)
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.grid(True)
        fig.text(0.5, 0.02, "El tamaño de la burbuja es proporcional a 1 / RMSE (mayor = mejor).", ha='center', fontsize=12)
        plt.tight_layout(rect=[0, 0.05, 1, 0.89])
        plt.savefig(os.path.join(FIGURES_DIR, "4_bubble_chart.png"), dpi=600)
        plt.close()

    # 4. Alpha Distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle('Distribución de Tasas de Cambio Relativo ($\\alpha_t$)', fontsize=16, y=0.98)
    for idx, fuel in enumerate(REQUIRED_COLUMNS):
        alpha_path = f"data/gold/{fuel}_alpha.npy"
        if os.path.exists(alpha_path):
            sns.histplot(np.load(alpha_path), kde=True, ax=axes[idx], color=FUEL_COLORS[fuel], bins=50, stat="density")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(FIGURES_DIR, "25_alpha_distributions_optimal.png"), dpi=600)
    plt.close()

    # 5. Graph Panels
    sub_paths_km = {}
    for i, fuel in enumerate(REQUIRED_COLUMNS):
        p_km, c_km = os.path.join(MODELS_DIR, f"{fuel}_P_kmeans.npy"), os.path.join(MODELS_DIR, f"{fuel}_centroids.npy")
        if os.path.exists(p_km):
            plot_transition_graph(np.load(p_km), fuel, np.load(p_km).shape[0], np.load(c_km), FIGURES_DIR)
            sub_paths_km[fuel] = os.path.join(FIGURES_DIR, f"graph_kmeans_{fuel.lower()}.png")
    if len(sub_paths_km) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        for i, fuel in enumerate(REQUIRED_COLUMNS):
            axes.flatten()[i].imshow(Image.open(sub_paths_km[fuel])); axes.flatten()[i].set_title(f'({chr(97+i)}) K-Means {fuel}', fontsize=18); axes.flatten()[i].axis('off')
        plt.tight_layout(pad=0.5); plt.savefig(os.path.join(FIGURES_DIR, "6_final_transition_panel.png"), dpi=600, bbox_inches='tight'); plt.close()

    sub_paths_q = {}
    for i, fuel in enumerate(REQUIRED_COLUMNS):
        p_q, b_q = os.path.join(MODELS_DIR, f"{fuel}_P_quantiles.npy"), os.path.join(MODELS_DIR, f"{fuel}_boundaries.npy")
        if os.path.exists(p_q):
            plot_quantile_transition_graph(np.load(p_q), fuel, np.load(b_q), FIGURES_DIR)
            sub_paths_q[fuel] = os.path.join(FIGURES_DIR, f"graph_quantiles_{fuel.lower()}.png")
    if len(sub_paths_q) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        for i, fuel in enumerate(REQUIRED_COLUMNS):
            axes.flatten()[i].imshow(Image.open(sub_paths_q[fuel])); axes.flatten()[i].set_title(f'({chr(97+i)}) Quantiles {fuel}', fontsize=18); axes.flatten()[i].axis('off')
        plt.tight_layout(pad=0.5); plt.savefig(os.path.join(FIGURES_DIR, "7_final_quantiles_panel.png"), dpi=600, bbox_inches='tight'); plt.close()

    # STEP 6: K-Means Regimes Scatter (FIG-08)
    print("\n--- PASO 6: Visualizando regímenes finales encontrados por K-Means ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle('Regímenes Finales K-Means por Serie de Combustible (usando k Óptimo)', fontsize=20, y=0.98)
    for i, fuel in enumerate(REQUIRED_COLUMNS):
        a_p, audit_p, c_p = f"data/gold/{fuel}_alpha.npy", os.path.join(TABLES_DIR, f"{fuel}_discretization_audit.csv"), os.path.join(MODELS_DIR, f"{fuel}_centroids.npy")
        if os.path.exists(a_p) and os.path.exists(audit_p) and os.path.exists(c_p):
            alphas, audit_df, centroids = np.load(a_p), pd.read_csv(audit_p), np.load(c_p)
            states = audit_df['State_KMeans'].values
            k_optimal = len(centroids)
            
            df_plot = pd.DataFrame({'time': range(len(alphas)), 'alpha': alphas, 'regime': states})
            
            sns.scatterplot(
                data=df_plot, x='time', y='alpha', hue='regime',
                palette=REGIME_COLORS_NORD[:k_optimal],
                ax=axes[i], legend='full', s=25
            )
            
            for k_idx, centroid in enumerate(centroids):
                axes[i].axhline(y=centroid, color=REGIME_COLORS_NORD[k_idx % len(REGIME_COLORS_NORD)], linestyle='--', linewidth=2, label=f'Centroide s{k_idx+1}')
            
            axes[i].set_title(f'Regímenes K-Means para: {fuel} (k={k_optimal})', fontsize=16)
            axes[i].set_ylabel('Valor $\\alpha_t$ (Tasa de Cambio Relativo)', fontsize=14)
            axes[i].set_xlabel('Tiempo (semanas)', fontsize=14)
            axes[i].grid(True, linestyle=':', linewidth=0.5)
            
            handles, labels = axes[i].get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            legend_labels = [f'Régimen {int(l)+1}' if str(l).isdigit() else l for l in unique_labels.keys()]
            
            axes[i].legend(unique_labels.values(), legend_labels, title='Leyenda', bbox_to_anchor=(1.01, 1), loc='upper left', title_fontsize='13', fontsize='12')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(FIGURES_DIR, "8_kmeans_final_regimes_plot.png"), dpi=600, bbox_inches='tight'); plt.close()

    # STEP 7: Markov Simulation (FIG-09)
    print("\n--- PASO 7: Visualizando Evolución de Probabilidades de Estado (Simulación K-Means) ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    fig.suptitle('Evolución de Probabilidades de Estado (Cadena de Markov K-Means)', fontsize=20, y=0.98)
    for i, fuel in enumerate(REQUIRED_COLUMNS):
        p_p, c_p, audit_p = os.path.join(MODELS_DIR, f"{fuel}_P_kmeans.npy"), os.path.join(MODELS_DIR, f"{fuel}_centroids.npy"), os.path.join(TABLES_DIR, f"{fuel}_discretization_audit.csv")
        if os.path.exists(p_p) and os.path.exists(c_p) and os.path.exists(audit_p):
            P, centroids, states_seq = np.load(p_p), np.load(c_p), pd.read_csv(audit_p)['State_KMeans'].values
            n_s = P.shape[0]
            p0 = np.zeros(n_s); p0[mode(states_seq, keepdims=True).mode[0]] = 1.0
            num_steps = 400; trajectory = np.zeros((n_s, num_steps)); trajectory[:, 0] = p0; stability_step = -1
            for t in range(1, num_steps):
                trajectory[:, t] = P @ trajectory[:, t-1]
                if stability_step == -1 and np.linalg.norm(trajectory[:, t] - trajectory[:, t-1], 1) < 1e-6: stability_step = t
            colors = REGIME_COLORS_NORD[:n_s]
            state_labels = [f'Régimen {k+1} (α≈{c:.3f})' for k, c in enumerate(centroids)]
            for j in range(n_s): axes[i].plot(trajectory[j, :], label=state_labels[j], color=colors[j], linewidth=2.5)
            if stability_step != -1:
                axes[i].axvline(x=stability_step, color='#BF616A', linestyle='--', linewidth=2)
                axes[i].text(stability_step + 0.5, axes[i].get_ylim()[1] * 0.95, f'Equilibrio en paso {stability_step}', color='#BF616A', rotation=90, va='top', fontsize=12)
            axes[i].set_title(f'Evolución de Probabilidad de Estado: {fuel}', fontsize=16)
            axes[i].set_xlabel('Paso de Tiempo (Simulación)', fontsize=14)
            axes[i].set_ylabel('Probabilidad', fontsize=14)
            axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            axes[i].legend(title='Régimen', title_fontsize='13', fontsize='12')
            axes[i].grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(FIGURES_DIR, "9_kmeans_markov_evolution.png"), dpi=600, bbox_inches='tight'); plt.close()

    # PASO 8: Analisis de robustez (Excel)
    from src.evaluation.compute_predictive_metrics import compute_predictive_metrics
    from src.ingestion.load_fuel_data import load_fuel_data
    from src.config import CSV_PATH
    print("\n--- PASO 8: Ejecutando Análisis de Robustez Predictiva ---")
    _, fuel_series = load_fuel_data(CSV_PATH)
    if os.path.exists(csv_best):
        best_df = pd.read_csv(csv_best); predictive_summaries = {}
        for fuel in REQUIRED_COLUMNS:
            row = best_df[best_df['Combustible'] == fuel]
            if not row.empty:
                w, l, k = int(row.iloc[0]['W']), float(row.iloc[0]['Lambda']), int(row.iloc[0]['k'])
                alphas = calculate_alphas(fuel_series[fuel], W=w, lambda_decay=l)
                _, _, partition_results = compute_predictive_metrics(alphas, fuel_series[fuel], k, w, method='kmeans', return_splits=True)
                df_res = pd.DataFrame(partition_results).dropna()
                # Translate for output
                df_res.rename(columns={'Accuracy': 'Exactitud', 'Partition': 'Partición'}, inplace=True)
                predictive_summaries[fuel] = df_res
                print(f"\nRE-EVALUANDO {fuel.upper()}: W={w}, L={l}, k={k}"); print(df_res.to_string(index=False))
                print(f"Exactitud Promedio: {df_res['Exactitud'].mean():.10f} | RMSE Promedio: {df_res['RMSE'].mean():.10f}")
        with pd.ExcelWriter(os.path.join(TABLES_DIR, "10_predictive_summary_kmeans_consistent.xlsx")) as writer:
            for name, df in predictive_summaries.items(): df.to_excel(writer, sheet_name=name, index=False)
        print(f"\nResumen Excel guardado en {TABLES_DIR}")

    # STEP 9: Quantile Regimes (FIG-11)
    print("\n--- PASO 9: Visualizando Regímenes encontrados por Cuantiles ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    fig.suptitle('Regímenes Finales Basados en Cuantiles (Benchmark k=4)', fontsize=20, y=0.98)
    for i, fuel in enumerate(REQUIRED_COLUMNS):
        a_p, audit_p, b_p = f"data/gold/{fuel}_alpha.npy", os.path.join(TABLES_DIR, f"{fuel}_discretization_audit.csv"), os.path.join(MODELS_DIR, f"{fuel}_boundaries.npy")
        if os.path.exists(a_p) and os.path.exists(audit_p) and os.path.exists(b_p):
            alphas, audit_df, bounds = np.load(a_p), pd.read_csv(audit_p), np.load(b_p)
            states = audit_df['State_Quantiles'].values
            
            df_plot = pd.DataFrame({'time': range(len(alphas)), 'alpha': alphas, 'regime': states})
            
            sns.scatterplot(
                data=df_plot, x='time', y='alpha', hue='regime',
                palette=REGIME_COLORS_NORD[:4],
                ax=axes[i], legend='full', s=25
            )
            
            for b_idx, boundary in enumerate(bounds):
                label = 'Límite de Cuantil' if b_idx == 0 else None
                axes[i].axhline(y=boundary, color='#BF616A', linestyle='--', linewidth=1.5, label=label)
            
            axes[i].set_title(f'Regímenes Basados en Cuantiles: {fuel}', fontsize=16)
            axes[i].set_ylabel('Valor $\\alpha_t$ (Tasa de Cambio Relativo)', fontsize=14)
            axes[i].set_xlabel('Tiempo (semanas)', fontsize=14)
            axes[i].grid(True, linestyle=':', linewidth=0.5)
            
            handles, labels = axes[i].get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            legend_labels = [f'Régimen {int(l)+1}' for l in unique_labels.keys() if str(l).isdigit()] + \
                            [l for l in unique_labels.keys() if not str(l).isdigit()]
            
            axes[i].legend(unique_labels.values(), legend_labels, title='Leyenda', bbox_to_anchor=(1.01, 1), loc='upper left', title_fontsize='13', fontsize='12')
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(FIGURES_DIR, "11_quantile_final_regimes_plot.png"), dpi=600, bbox_inches='tight'); plt.close()

    # FIG-15: Price Shading (K-Means)
    print("\n--- Generando FIG-15: Sombreado de Precios con Regímenes K-Means ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
    axes = axes.flatten()
    fig.suptitle('Serie de Precios con Sombreado de Regímenes (K-Means)', fontsize=20, y=0.98)
    for i, fuel in enumerate(REQUIRED_COLUMNS):
        audit_p = os.path.join(TABLES_DIR, f"{fuel}_discretization_audit.csv")
        names_p = os.path.join(MODELS_DIR, f"{fuel}_regime_names.npy")
        if os.path.exists(audit_p):
            df = pd.read_csv(audit_p); dates = pd.to_datetime(df['Date']); states = df['State_KMeans'].values
            names = np.load(names_p) if os.path.exists(names_p) else [f"Régimen {x+1}" for x in range(10)]
            k_val = len(names); colors = REGIME_COLORS_NORD[:k_val]
            ax = axes[i]; curr_s, b_start = states[0], dates.iloc[0]
            for t in range(1, len(states)):
                if states[t] != curr_s: 
                    ax.axvspan(b_start, dates.iloc[t-1], color=colors[curr_s % k_val], alpha=0.3)
                    curr_s, b_start = states[t], dates.iloc[t]
            ax.axvspan(b_start, dates.iloc[-1], color=colors[curr_s % k_val], alpha=0.3)
            ax.plot(dates, df[fuel], color=FUEL_COLORS[fuel], linewidth=2, zorder=3)
            patches = [Patch(color=colors[idx], label=names[idx], alpha=0.5) for idx in range(k_val)]
            ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=k_val, fontsize=10, title="Regímenes")
            ax.set_title(f'{fuel}', pad=40, fontsize=15)
            ax.set_ylabel('Precio (HNL)', fontsize=12)
            ax.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(FIGURES_DIR, "15_kmeans_final_price_regimes.png"), dpi=600)
    plt.close()

    # FIG-16: Price Shading (Quantiles)
    print("--- Generando FIG-16: Sombreado de Precios con Regímenes Cuantiles ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
    axes = axes.flatten()
    fig.suptitle('Serie de Precios con Sombreado de Regímenes (Cuantiles)', fontsize=20, y=0.98)
    for i, fuel in enumerate(REQUIRED_COLUMNS):
        audit_p = os.path.join(TABLES_DIR, f"{fuel}_discretization_audit.csv")
        if os.path.exists(audit_p):
            df = pd.read_csv(audit_p); dates = pd.to_datetime(df['Date']); states = df['State_Quantiles'].values
            k_val = 4; colors = REGIME_COLORS_NORD[:k_val]
            ax = axes[i]; curr_s, b_start = states[0], dates.iloc[0]
            for t in range(1, len(states)):
                if states[t] != curr_s: 
                    ax.axvspan(b_start, dates.iloc[t-1], color=colors[curr_s % k_val], alpha=0.3)
                    curr_s, b_start = states[t], dates.iloc[t]
            ax.axvspan(b_start, dates.iloc[-1], color=colors[curr_s % k_val], alpha=0.3)
            ax.plot(dates, df[fuel], color=FUEL_COLORS[fuel], linewidth=2, zorder=3)
            patches = [Patch(color=colors[idx], label=f"Régimen {idx+1}", alpha=0.5) for idx in range(k_val)]
            ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=k_val, fontsize=10, title="Regímenes")
            ax.set_title(f'{fuel}', pad=40, fontsize=15)
            ax.set_ylabel('Precio (HNL)', fontsize=12)
            ax.grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(FIGURES_DIR, "16_quantile_final_price_regimes.png"), dpi=600)
    plt.close()

    # STEP 10: Markov Simulation (Quantiles) - FIG-12
    print("\n--- PASO 10: Visualizando Evolución de Probabilidades de Estado (Simulación Cuantiles) ---")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    fig.suptitle('Evolución de Probabilidades de Estado (Benchmark Cuantiles)', fontsize=20, y=0.98)
    for i, fuel in enumerate(REQUIRED_COLUMNS):
        p_p, b_p, audit_p = os.path.join(MODELS_DIR, f"{fuel}_P_quantiles.npy"), os.path.join(MODELS_DIR, f"{fuel}_boundaries.npy"), os.path.join(TABLES_DIR, f"{fuel}_discretization_audit.csv")
        if os.path.exists(p_p) and os.path.exists(b_p) and os.path.exists(audit_p):
            P, boundaries, states_seq = np.load(p_p), np.load(b_p), pd.read_csv(audit_p)['State_Quantiles'].values
            n_s = 4; p0 = np.zeros(n_s); p0[mode(states_seq, keepdims=True).mode[0]] = 1.0
            num_steps = 400; trajectory = np.zeros((n_s, num_steps)); trajectory[:, 0] = p0; stability_step = -1
            for t in range(1, num_steps):
                trajectory[:, t] = P @ trajectory[:, t-1]
                if stability_step == -1 and np.linalg.norm(trajectory[:, t] - trajectory[:, t-1], 1) < 1e-6: stability_step = t
            colors = REGIME_COLORS_NORD[:n_s]
            state_labels = [f'Régimen 1 (α < {boundaries[0]:.3f})', f'Régimen 2 ({boundaries[0]:.3f} ≤ α < {boundaries[1]:.3f})', f'Régimen 3 ({boundaries[1]:.3f} ≤ α < {boundaries[2]:.3f})', f'Régimen 4 (α ≥ {boundaries[2]:.3f})']
            for j in range(n_s): axes[i].plot(trajectory[j, :], label=state_labels[j], color=colors[j], linewidth=2.5)
            if stability_step != -1:
                axes[i].axvline(x=stability_step, color='#BF616A', linestyle='--', linewidth=2)
                axes[i].text(stability_step + 0.5, axes[i].get_ylim()[1] * 0.95, f'Equilibrio en paso {stability_step}', color='#BF616A', rotation=90, va='top', fontsize=12)
            axes[i].set_title(f'Evolución de Probabilidad de Estado: {fuel}', fontsize=16)
            axes[i].set_xlabel('Paso de Tiempo (Simulación)', fontsize=14)
            axes[i].set_ylabel('Probabilidad', fontsize=14)
            axes[i].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            axes[i].legend(title='Definición de Régimen', title_fontsize='13', fontsize='12')
            axes[i].grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(FIGURES_DIR, "12_quantiles_markov_evolution.png"), dpi=600, bbox_inches='tight'); plt.close()

    # PASO 11: Analisis de robustez (cuantiles)
    print("\n--- PASO 11: Ejecutando Análisis de Robustez Predictiva para Cuantiles ---")
    if os.path.exists(csv_best):
        best_df = pd.read_csv(csv_best); predictive_summaries_q = {}
        for fuel in REQUIRED_COLUMNS:
            row = best_df[best_df['Combustible'] == fuel]
            if not row.empty:
                w, l = int(row.iloc[0]['W']), float(row.iloc[0]['Lambda'])
                alphas = calculate_alphas(fuel_series[fuel], W=w, lambda_decay=l)
                # k=4 fixed for quantiles
                _, _, partition_results = compute_predictive_metrics(alphas, fuel_series[fuel], 4, w, method='quantiles', return_splits=True)
                df_res = pd.DataFrame(partition_results).dropna()
                # Translate for output
                df_res.rename(columns={'Accuracy': 'Exactitud', 'Partition': 'Partición'}, inplace=True)
                predictive_summaries_q[fuel] = df_res
                print(f"\nRE-EVALUANDO CUANTILES {fuel.upper()}: W={w}, L={l}, k=4"); print(df_res.to_string(index=False))
                print(f"Exactitud Promedio: {df_res['Exactitud'].mean():.10f} | RMSE Promedio: {df_res['RMSE'].mean():.10f}")
        with pd.ExcelWriter(os.path.join(TABLES_DIR, "13_predictive_summary_quantiles.xlsx")) as writer:
            for name, df in predictive_summaries_q.items(): df.to_excel(writer, sheet_name=name, index=False)
        print(f"\nResumen Excel guardado en {TABLES_DIR}")

    # STEP 16: Forecast Next Week
    print("\n--- PASO 16: Pronosticando Regímenes y Precios para la Próxima Semana ---")
    forecast_data = []
    # Dynamic date - Re-load data to ensure fresh context
    _, fuel_series = load_fuel_data(CSV_PATH)
    data_dates = pd.read_csv("data/silver/fuel_data_standardized.csv")
    last_date = pd.to_datetime(data_dates['Date']).max()
    
    # Spanish Date Formatting
    days_es = {'Monday': 'Lunes', 'Tuesday': 'Martes', 'Wednesday': 'Miércoles', 'Thursday': 'Jueves', 'Friday': 'Viernes', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    months_es = {'January': 'Enero', 'February': 'Febrero', 'March': 'Marzo', 'April': 'Abril', 'May': 'Mayo', 'June': 'Junio', 'July': 'Julio', 'August': 'Agosto', 'September': 'Septiembre', 'October': 'Octubre', 'November': 'Noviembre', 'December': 'Diciembre'}
    target_date = last_date + pd.Timedelta(days=7)
    d_name, m_name = target_date.strftime('%A'), target_date.strftime('%B')
    forecast_date_str = f"{days_es.get(d_name, d_name)}, {target_date.strftime('%d')} de {months_es.get(m_name, m_name)} de {target_date.strftime('%Y')}"
    
    for name in REQUIRED_COLUMNS:
        p_p, c_p, names_p = os.path.join(MODELS_DIR, f"{name}_P_kmeans.npy"), os.path.join(MODELS_DIR, f"{name}_centroids.npy"), os.path.join(MODELS_DIR, f"{name}_regime_names.npy")
        audit_p = os.path.join(TABLES_DIR, f"{name}_discretization_audit.csv")
        
        if os.path.exists(p_p) and os.path.exists(c_p) and os.path.exists(audit_p):
            P, centroids = np.load(p_p), np.load(c_p)
            regime_names = np.load(names_p) if os.path.exists(names_p) else [f"Régimen {x+1}" for x in range(len(centroids))]
            
            # Obtener ultimo estado
            last_state = pd.read_csv(audit_p)['State_KMeans'].iloc[-1]
            last_price = fuel_series[name][-1]
            
            # Predict
            next_state_idx = np.argmax(P[:, last_state])
            conf = P[next_state_idx, last_state]
            pred_alpha = centroids[next_state_idx]
            pred_price = last_price * (1 + pred_alpha)
            
            forecast_data.append({
                'Tipo de Combustible': name,
                'Precio Actual': f"{last_price:.2f}",
                'Régimen Actual': f"s{last_state+1} ({regime_names[last_state]})",
                'Próximo Régimen': f"s{next_state_idx+1} ({regime_names[next_state_idx]})",
                'Confianza': f"{conf:.1%}",
                'Cambio Esperado (α)': f"{pred_alpha:+.4f}",
                'Precio Pronosticado': f"{pred_price:.2f}"
            })
            
    if forecast_data:
        df_f = pd.DataFrame(forecast_data)
        print("\n" + "="*80 + f"\n{f'PRONÓSTICO DE PRECIOS PARA {forecast_date_str.upper()}'.center(80)}\n" + "="*80)
        print(df_f.to_string(index=False))
        df_f.to_excel(os.path.join(TABLES_DIR, "16_next_week_forecast.xlsx"), index=False)
        print(f"\n✅ Resumen de pronóstico guardado en: '16_next_week_forecast.xlsx'")

    # STEP 17: Statistical Significance
    print("\n--- PASO 17: Realizando Pruebas de Significancia Estadística (Corregidas) ---")
    from scipy import stats
    from sklearn.metrics import accuracy_score, mean_squared_error
    from src.processing.functions.discretize_series import discretize_series
    from src.models.functions.estimate_transition_matrix import estimate_transition_matrix

    def evaluate_model_performance(alphas, series, k, W, method):
        price_series_values = np.asarray(series)
        partition_accuracies, partition_rmses = [], []
        ratios = np.arange(0.60, 0.96, 0.05)
        splits = [(np.arange(0, int(len(alphas) * r)), np.arange(int(len(alphas) * r), len(alphas))) for r in ratios]

        for train_idx, test_idx in splits:
            if len(test_idx) < 2 or len(train_idx) < k:
                partition_accuracies.append(np.nan); partition_rmses.append(np.nan); continue

            alphas_train, alphas_test = alphas[train_idx], alphas[test_idx]
            if method == 'kmeans':
                train_states, centroids = discretize_series(alphas_train, k)
                centroids = np.sort(centroids) # Ensure sorted for consistent alpha mapping
                test_states = np.array([np.argmin(np.abs(alpha - centroids)) for alpha in alphas_test])
            else: # quantiles
                boundaries = np.quantile(alphas_train, np.linspace(0, 1, k + 1)[1:-1])
                train_states = np.digitize(alphas_train, boundaries)
                test_states = np.digitize(alphas_test, boundaries)
                centroids = np.array([alphas_train[train_states == i].mean() if np.any(train_states == i) else 0 for i in range(k)])

            P, _ = estimate_transition_matrix(train_states, k)
            if len(train_states) > 0:
                predicted_states = []
                last_state = train_states[-1]
                for _ in range(len(test_idx)):
                    pred_state = np.argmax(P[:, last_state])
                    predicted_states.append(pred_state)
                    last_state = pred_state
                
                acc = accuracy_score(test_states, predicted_states)
                predicted_alphas = np.array([centroids[s] for s in predicted_states])
                last_prices_idx = test_idx + W - 2; actual_prices_idx = test_idx + W - 1
                
                if np.max(actual_prices_idx) < len(price_series_values):
                    last_prices = price_series_values[last_prices_idx]
                    actual_prices = price_series_values[actual_prices_idx]
                    if len(predicted_alphas) == len(last_prices):
                        rmse = np.sqrt(mean_squared_error(actual_prices, last_prices * (1 + predicted_alphas)))
                        partition_accuracies.append(acc); partition_rmses.append(rmse); continue
            
            partition_accuracies.append(np.nan); partition_rmses.append(np.nan)
        return partition_accuracies, partition_rmses

    if os.path.exists(csv_best):
        df_best = pd.read_csv(csv_best)
        significance_results = []
        print("Ejecutando comparación justa usando parámetros óptimos...")
        for name in list(df_best['Combustible']):
            row = df_best[df_best['Combustible'] == name]
            if not row.empty:
                W, Lam, k = int(row.iloc[0]['W']), float(row.iloc[0]['Lambda']), int(row.iloc[0]['k'])
                series = fuel_series[name]
                if len(series) > 0:
                    alphas = calculate_alphas(series, W=W, lambda_decay=Lam)
                    km_acc, km_rmse = evaluate_model_performance(alphas, series, k, W, 'kmeans')
                    q_acc, q_rmse = evaluate_model_performance(alphas, series, 4, W, 'quantiles')
                    
                    km_r_arr, q_r_arr = np.array(km_rmse, dtype=float), np.array(q_rmse, dtype=float)
                    km_a_arr, q_a_arr = np.array(km_acc, dtype=float), np.array(q_acc, dtype=float)
                    
                    valid_r = ~np.isnan(km_r_arr) & ~np.isnan(q_r_arr)
                    valid_a = ~np.isnan(km_a_arr) & ~np.isnan(q_a_arr)
                    
                    if np.sum(valid_r) > 1 and np.sum(valid_a) > 1:
                        _, p_rmse = stats.ttest_rel(km_r_arr[valid_r], q_r_arr[valid_r], alternative='less')
                        _, p_acc = stats.ttest_rel(km_a_arr[valid_a], q_a_arr[valid_a], alternative='greater')
                        
                        significance_results.append({
                            'Combustible': name,
                            'RMSE (K-Means)': np.mean(km_r_arr[valid_r]), 'RMSE (Cuantiles)': np.mean(q_r_arr[valid_r]), 'Valor-p RMSE': p_rmse,
                            'Exactitud (K-Means)': np.mean(km_a_arr[valid_a]), 'Exactitud (Cuantiles)': np.mean(q_a_arr[valid_a]), 'Valor-p Exactitud': p_acc
                        })
        
        if significance_results:
            df_sig = pd.DataFrame(significance_results)
            print("\n" + "="*80 + "\nRESULTADOS DE PRUEBA DE SIGNIFICANCIA ESTADÍSTICA (T-TEST PAREADO)\n" + "="*80)
            print(df_sig.to_string())
            df_sig.to_excel(os.path.join(TABLES_DIR, "17_statistical_significance.xlsx"), index=False)

    # STEP 19: Transition Probability Heatmaps
    print("\n--- PASO 19: Generando Mapas de Calor de Probabilidades de Transición ---")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    fig.suptitle('Matrices de Transición de Probabilidad (Heatmaps)', fontsize=20, y=0.98)
    
    for i, fuel in enumerate(REQUIRED_COLUMNS):
        p_path = os.path.join(MODELS_DIR, f"{fuel}_P_kmeans.npy")
        c_path = os.path.join(MODELS_DIR, f"{fuel}_centroids.npy")
        
        if os.path.exists(p_path) and os.path.exists(c_path):
            P = np.load(p_path)
            centroids = np.load(c_path)
            k = len(centroids)
            
            # Labels with mean alpha change
            labels = [f"s{j+1}\n({centroids[j]:+.1%})" for j in range(k)]
            
            sns.heatmap(
                P, ax=axes[i], annot=True, fmt=".1%", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Probabilidad de Transición'}
            )
            
            axes[i].set_title(f'Matriz de Transición: {fuel}', fontsize=16)
            axes[i].set_xlabel('Estado Futuro ($t+1$)', fontsize=14)
            axes[i].set_ylabel('Estado Actual ($t$)', fontsize=14)
            
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(FIGURES_DIR, "19_transition_probability_heatmaps.png"), dpi=600, bbox_inches='tight')
    plt.close()
    print("✅ Mapas de Calor de Transición guardados.")

    # STEP 18: Sensitivity Analysis V6 (IEEE Style)
    print("\n--- Generando Análisis de Sensibilidad de Alta Calidad (V6) ---")
    if os.path.exists(csv_summary):
        df_summary = pd.read_csv(csv_summary)
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
        
        fig, axes = plt.subplots(2, 2, figsize=(17, 9))
        fig.suptitle('Análisis de Sensibilidad: Rendimiento vs. Tamaño de Ventana (W)', fontweight='bold')
        fuel_map = {'Regular': axes[0, 0], 'Super': axes[0, 1], 'Kerosene': axes[1, 0], 'Diesel': axes[1, 1]}
        
        from scipy.interpolate import make_interp_spline
        from matplotlib.ticker import FuncFormatter

        for fname, ax in fuel_map.items():
            df_s = df_summary[df_summary['Combustible'] == fname]
            df_real = df_s[df_s['Exactitud'] < 0.95].copy()
            
            if df_real.empty or len(df_real['W'].unique()) < 4:
                ax.text(0.5, 0.5, 'Datos Insuficientes', ha='center', va='center'); ax.set_title(f'{fname} (Sin Datos)'); continue
                
            best_rmse = df_real.loc[df_real.groupby('W')['RMSE'].idxmin()].sort_values('W')
            W, R, A = best_rmse['W'], best_rmse['RMSE'], best_rmse['Exactitud']
            
            # Suavizado (Smoothing)
            W_new = np.linspace(W.min(), W.max(), 300)
            try:
                spl_r = make_interp_spline(W, R, k=3); R_new = spl_r(W_new)
                spl_a = make_interp_spline(W, A, k=3); A_new = spl_a(W_new)
            except:
                W_new, R_new, A_new = W, R, A # Fallback

            ax.set_xlabel('Tamaño de Ventana (W)'); ax.set_ylabel('Error Cuadrático Medio (RMSE)')
            ax.plot(W_new, R_new, color='slategray', linestyle=':')
            
            ax2 = ax.twinx(); ax2.set_ylabel('Exactitud')
            ax2.plot(W_new, A_new, color='indianred', linestyle=':')
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100*y:.0f}%"))

            # Best point
            best_p = best_rmse.loc[best_rmse['RMSE'].idxmin()]
            bw, br, ba = best_p['W'], best_p['RMSE'], best_p['Exactitud']
            
            ax.axhline(y=br, color='gold', linestyle='--', linewidth=1); ax.axvline(x=bw, color='gold', linestyle='--', linewidth=1)
            ax2.axhline(y=ba, color='gold', linestyle='--', linewidth=1)
            ax.plot(bw, br, 'o', color='gold', markeredgecolor='black', markersize=6)
            
            info = f"$\\bf{{Parámetros\\ Óptimos}}$\nW={int(bw)}\nLambda={best_p['Lambda']:.2f}\nk={int(best_p['k'])}\n\n$\\bf{{Métricas}}$\nRMSE={br:.4f}\nExactitud={ba:.2%}"
            ax.text(1.2, 0.65, info, transform=ax.transAxes, fontsize=8, va='top')
            ax.text(1.2, 0.9, "$\\bf{Leyenda}$", transform=ax.transAxes, fontsize=8)
            ax.text(1.2, 0.85, "· · · · · RMSE", transform=ax.transAxes, fontsize=8, color='slategray')
            ax.text(1.2, 0.80, "· · · · · Exactitud", transform=ax.transAxes, fontsize=8, color='indianred')
            ax.set_title(f'Combustible: {fname}'); ax.grid(True, linestyle=':', alpha=0.6)
            
        plt.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.1, wspace=0.7, hspace=0.3)
        plt.savefig(os.path.join(FIGURES_DIR, "W_sensitivity_analysis_final_v6.png"), dpi=600)
        plt.close()
    print(f"✅ Análisis de Sensibilidad V6 guardado.")

    print("\nVisualizaciones completadas restituyendo el diseño exacto de Articulol.py.")

if __name__ == "__main__":
    run_visualization()
