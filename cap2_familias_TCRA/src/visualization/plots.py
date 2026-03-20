import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
import matplotlib as mpl
mpl.use('Agg')

plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12

from src.config import FIGURES_DIR, TABLES_DIR, SERIES_COLORS, ORIGINAL_COLOR

def smooth_curve(x_dates, y_values, points=500):
    """
    Returns smoothed x and y values for plotting series.
    Works with timestamp indices.
    """
    if len(y_values) < 4:
        return x_dates, y_values
        
    # Indice numerico para spline
    idx = np.arange(len(x_dates))
    idx_new = np.linspace(0, len(x_dates)-1, points)
    
    # Smooth y values
    spl = make_interp_spline(idx, y_values, k=3)
    y_smooth = spl(idx_new)
    
    # Map back to dates for plotting if needed, but plotting against numeric is safer for spline
    # We'll return the numeric index and the smoothed y
    # El llamador traduce the numeric index back to dates if using ax.plot(dates, ...)
    # Or ax.plot(idx_new, y_smooth) and fix labels
    
    # Alternative: interpolate the actual timestamps (as numbers)
    ts = x_dates.map(pd.Timestamp.timestamp).values
    spl_ts = make_interp_spline(idx, ts, k=1) # Linear for dates
    ts_new = spl_ts(idx_new)
    dates_new = pd.to_datetime(ts_new, unit='s')
    
    return dates_new, y_smooth

def apply_clean_style(ax):
    """Removes grid and sets layout."""
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.5)
    ax.set_facecolor('white')

def plot_multiseries_grid(df, dataset_name, suffix="original", show_adjustment=False, df_alphas=None):
    """
    Plots a 2x2 grid of the series in the dataframe.
    """
    # sns.set_theme(style="white")  # Removed to maintain ggplot theme
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, col in enumerate(df.columns):
        ax = axes[i]
        color = SERIES_COLORS.get(col, ORIGINAL_COLOR)
        
        # 1. Original Data (Smoothed Line Only, no markers)
        x_smooth, y_smooth = smooth_curve(df.index, df[col].values)
        ax.plot(x_smooth, y_smooth, color=ORIGINAL_COLOR, linewidth=1.5, alpha=0.8, label='Datos Originales')
        
        # 3. Adjustment (if requested)
        if show_adjustment and df_alphas is not None:
            # Look for best variant (highest suffix) or a specific one
            # For now, let's plot the "Best" which is usually in gold results
            # Columns in df_alphas are named like 'col_VARIANT'
            # We'll plot Just the TCROC or the best variant if we can identify it
            variants_available = [c for c in df_alphas.columns if c.startswith(col)]
            if variants_available:
                # Plot the very last one as it's often the 'Best' (ETCROCM)
                best_var_col = variants_available[-1]
                variant_name = best_var_col.replace(f"{col}_", "")
                
                # y_pred = x_{t-1} * (1 + alpha_t)
                y_pred = df[col].shift(1) * (1 + df_alphas[best_var_col])
                aligned = y_pred.dropna()
                if len(aligned) > 4:
                    xp, yp = smooth_curve(df.index[df.index.isin(aligned.index)], aligned.values)
                    ax.plot(xp, yp, color=color, linewidth=1.5, label=f'Ajuste {variant_name}')
        
        ax.set_title(col, fontsize=12, fontweight='bold', pad=15)
        apply_clean_style(ax)
        if show_adjustment:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    plt.tight_layout()
    output_path = FIGURES_DIR / f"grid_{dataset_name}_{suffix}.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved grid plot: {output_path}")

def plot_individual_comparison(dataset_name, series_name, df_original, df_alphas):
    """
    Plots a single series with its best variant.
    """
    plt.figure(figsize=(14, 7))
    color = SERIES_COLORS.get(series_name, '#2E86C1')
    
    # Original (No markers, just line)
    x_s, y_s = smooth_curve(df_original.index, df_original.values)
    plt.plot(x_s, y_s, color=ORIGINAL_COLOR, linestyle='-', linewidth=2.0, alpha=0.8, label='Datos Observados')
    
    # Extraer mejor variante absoluta from grid_search_best
    best_file = TABLES_DIR / f"grid_search_best_{dataset_name}.csv"
    best_variant = "ETCROCM" # fallback
    if best_file.exists():
        df_best = pd.read_csv(best_file)
        row = df_best[df_best['Series'] == series_name]
        if not row.empty:
            best_variant = row.iloc[0]['Variant']
            
    best_col = f"{series_name}_{best_variant}"
    if best_col in df_alphas.columns:
        # Prediction: x_{t-1} * (1 + alpha_t)
        y_pred = df_original.shift(1) * (1 + df_alphas[best_col])
        aligned = y_pred.dropna()
        if len(aligned) > 4:
            xp, yp = smooth_curve(df_original.index[df_original.index.isin(aligned.index)], aligned.values)
            plt.plot(xp, yp, color=color, linewidth=1.8, label=f'Modelo {best_variant} (Optimizado)')
    
    plt.title(f"Ajuste Dinámico TCROC: {series_name}", fontsize=15, fontweight='bold')
    ax = plt.gca()
    apply_clean_style(ax)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    safe_name = series_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("%", "")
    output_path = FIGURES_DIR / f"final_best_{dataset_name}_{safe_name}.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved individual plot: {output_path}")

def plot_alphas_distribution(dataset_name, series_name, df_alphas):
    """
    Plots the Alpha adjusters as an histogram to visualize the distribution of the model's dynamic behavior.
    """
    plt.figure(figsize=(10, 6))
    color = SERIES_COLORS.get(series_name, '#E74C3C')
    
    # Extraer variante ganadora de esta distribucion
    best_file = TABLES_DIR / f"grid_search_best_{dataset_name}.csv"
    best_variant = "ETCROCM" # Fallback
    w_val, lam_val = "?", "?"
    if best_file.exists():
        df_best = pd.read_csv(best_file)
        row = df_best[df_best['Series'] == series_name]
        if not row.empty:
            best_variant = row.iloc[0]['Variant']
            w_val = str(row.iloc[0]['W'])
            lam_val = f"{row.iloc[0]['Lambda']:.2f}"
            
    best_col = f"{series_name}_{best_variant}"
    if best_col in df_alphas.columns:
        alphas = df_alphas[best_col].dropna()
        if len(alphas) > 0:
            sns.histplot(alphas.values, bins=30, kde=True, color=color, stat="density", edgecolor='white', linewidth=1.2, alpha=0.7)
            
            # Línea vertical en 0 (ajuste neutro)
            plt.axvline(0, color='gray', linestyle='--', linewidth=1.5, label='Ajuste Nulo (0%)')
            
            # Extraer los parámetros ganadores para ilustrarlos
            plt.title(fr"Distribución dinámica ($\alpha_t$): {series_name}" + "\n" + fr"{best_variant} Óptimo (Ventana $W={w_val}$, Decaimiento $\lambda={lam_val}$)", fontsize=13, fontweight='bold')
            plt.xlabel(r"Magnitud del Ajuste ($\alpha_t$)")
            plt.ylabel("Densidad de Frecuencia")
            
            ax = plt.gca()
            apply_clean_style(ax)
            plt.legend(loc='upper right', frameon=False)
            
            safe_name = series_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("%", "")
            output_path = FIGURES_DIR / f"alphas_{dataset_name}_{safe_name}.png"
            plt.savefig(output_path, dpi=600, bbox_inches='tight')
            plt.close()
            print(f"Saved alphas distribution plot: {output_path}")

def plot_optimization_heatmap(df_detailed, dataset_name):
    """
    Generates heatmaps for each series and metric found in detailed results.
    """
    metrics = ["RMSE", "MAE", "MAPE", "R2"]
    series_list = df_detailed['Series'].unique()
    
    # We only plot ETCROCM because it's the one that varies in both W and Lambda
    df_plot = df_detailed[df_detailed['Variant'] == 'ETCROCM']
    
    if df_plot.empty:
        print("No ETCROCM data for heatmaps.")
        return

    for series in series_list:
        df_ser = df_plot[df_plot['Series'] == series]
        
        for metric in metrics:
            # Pivot for heatmap
            try:
                pivot = df_ser.pivot_table(index='W', columns='Lambda', values=metric)
                
                plt.figure(figsize=(10, 8))
                ax = sns.heatmap(pivot, annot=False, cmap='magma' if metric != 'R2' else 'viridis', cbar_kws={'label': metric})
                
                # Localizar el óptimo (min o max)
                if metric == 'R2':
                    mejor_valor = pivot.values.max()
                    idx_f, idx_c = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
                else:
                    mejor_valor = pivot.values.min()
                    # Reemplazar np.nan con infinito para encontrar el min correctamente
                    vals_safe = np.where(np.isnan(pivot.values), np.inf, pivot.values)
                    idx_f, idx_c = np.unravel_index(np.nanargmin(vals_safe), vals_safe.shape)
                    
                # Dibujar círculo lime en el óptimo (coordenadas son .5 del centro)
                ax.scatter(idx_c + 0.5, idx_f + 0.5, s=300, facecolors='none', edgecolors='lime', linewidth=3, label=f'Óptimo: W={pivot.index[idx_f]}, L={pivot.columns[idx_c]:.2f}')
                ax.legend(loc='upper right')
                
                plt.title(f"Superficie de Error ({metric}): {series}", fontsize=13, fontweight='bold')
                plt.ylabel("W (Ventana)")
                plt.xlabel(r"$\lambda$ (Decaimiento)")
                
                safe_name = series.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("%", "")
                output_path = FIGURES_DIR / f"heatmap_{dataset_name}_{safe_name}_{metric}.png"
                FIGURES_DIR.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path, dpi=600, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Could not generate heatmap for {series} - {metric}: {e}")

def plot_variant_comparison_bubble(df_detailed, dataset_name):
    """
    Plots a bubble chart (scatter) for each series comparing the 4 variants:
    X: RMSE, Y: MAE, Size: MAPE.
    """
    best_rows = []
    # Drop rows with NaN RMSE to avoid idxmin errors
    df_clean = df_detailed.dropna(subset=['RMSE', 'MAE', 'MAPE'])
    for (series, variant), group in df_clean.groupby(['Series', 'Variant']):
        if not group.empty:
            best = group.loc[group['RMSE'].idxmin()]
            best_rows.append(best)
            
    if not best_rows:
        return
        
    df_best = pd.DataFrame(best_rows)
    
    for series in df_best['Series'].unique():
        plt.figure(figsize=(9, 6))
        df_ser = df_best[df_best['Series'] == series].copy()
        
        # Scaling trick: we want the sizes to look reasonable and avoid giant overlaps.
        # We lower the sizes range to (50, 400) and turn down alpha to 0.5 so we can see through them.
        ax = sns.scatterplot(data=df_ser, x='RMSE', y='MAE', hue='Variant', 
                             size='MAPE', sizes=(50, 400), alpha=0.5, edgecolor='black', linewidth=1.2,
                             palette=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'])
        
        # Anotar los puntos con un pequeño margen dinámico en Y
        y_range = df_ser['MAE'].max() - df_ser['MAE'].min()
        offset = y_range * 0.08 if y_range > 0 else df_ser['MAE'].max() * 0.01
        
        for i in range(df_ser.shape[0]):
            plt.text(df_ser['RMSE'].iloc[i], df_ser['MAE'].iloc[i] + offset, 
                     df_ser['Variant'].iloc[i], horizontalalignment='center', 
                     size='medium', color='#333333', weight='bold')
                     
        plt.title(f"Dispersión y Mejor Variante: {series}", fontsize=14, fontweight='bold')
        apply_clean_style(ax)
        
        # Fix Legend Overlaps using labelspacing and borderpad
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
                   title="Variante y MAPE (%)", labelspacing=1.8, borderpad=1.2)
        
        safe_name = series.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("%", "")
        output_path = FIGURES_DIR / f"bubble_comparison_{dataset_name}_{safe_name}.png"
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved bubble comparison plot: {output_path}")

def plot_cross_validation_splits(df, dataset_name, series_name, max_W):
    """
    Plots the logic of TimeSeriesSplit for methodology chapter.
    """
    from sklearn.model_selection import TimeSeriesSplit
    
    plt.figure(figsize=(12, 8))
    y = df[series_name].values
    valid_start = max_W + 1
    
    if valid_start >= len(y):
        return
        
    y_eval = y[valid_start:]
    x_eval = df.index[valid_start:]
    
    x_hist = df.index[:valid_start]
    y_hist = y[:valid_start]
    
    tscv = TimeSeriesSplit(n_splits=4)
    
    for i, (train_index, test_index) in enumerate(tscv.split(y_eval)):
        plt.subplot(4, 1, i+1)
        
        # Histórico W
        plt.plot(x_hist, y_hist, color='darkgray', linewidth=1.5, label='Histórico Ignorado (Ventana W)' if i==0 else "")
        
        # Background: all data
        plt.plot(x_eval, y_eval, color='lightgray', linewidth=1, linestyle='--')
        
        # Train
        plt.plot(x_eval[train_index], y_eval[train_index], color='#3498DB', linewidth=2, label='Expansión de Conocimiento (Train)' if i==0 else "")
        
        # Test
        plt.plot(x_eval[test_index], y_eval[test_index], color='#E74C3C', linewidth=2.5, label='Evaluación Cruzada (Test)' if i==0 else "")
        
        plt.ylabel(f"Fold {i+1}")
        if i == 0:
            plt.title(f"Metodología Validación Cruzada Expansiva: {series_name}", fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
            
        ax = plt.gca()
        apply_clean_style(ax)
    
    plt.xlabel("Línea de Tiempo")
    plt.tight_layout()
    
    safe_name = series_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("%", "")
    output_path = FIGURES_DIR / f"cv_methodology_{dataset_name}_{safe_name}.png"
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved CV block plot: {output_path}")
