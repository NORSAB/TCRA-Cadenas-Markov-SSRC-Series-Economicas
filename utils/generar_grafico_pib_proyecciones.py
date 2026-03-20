import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from scipy.interpolate import make_interp_spline

# ======== PALETA NORD (thesis_colors.py) ========
NORD_0  = "#2E3440"   # Polar Night — texto
NORD_3  = "#4C566A"   # Bordes, texto secundario
NORD_5  = "#E5E9F0"   # Fondos intermedios
NORD_9  = "#81A1C1"   # Frost 3
NORD_10 = "#5E81AC"   # Frost Blue
NORD_12 = "#D08770"   # Aurora Orange
NORD_13 = "#EBCB8B"   # Aurora Yellow

# Configuraciones de estilo
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
mpl.rcParams['text.color'] = NORD_0
mpl.rcParams['axes.labelcolor'] = NORD_0
mpl.rcParams['xtick.color'] = NORD_3
mpl.rcParams['ytick.color'] = NORD_3

colors = {
    'Original': NORD_0,     # Polar Night (negro)
    'TCRA':     NORD_10,    # Frost Blue
    'TCRAM':    NORD_13,    # Aurora Yellow
    'ETCRA':    NORD_9,     # Frost 3
    'ETCRAM':   NORD_12     # Aurora Orange
}

# Parametros óptimos aproximados o canónicos
params = {
    'TCRA': {'W': None, 'lambd': 1.0},
    'TCRAM': {'W': 5, 'lambd': 1.0},
    'ETCRA': {'W': None, 'lambd': 0.9},
    'ETCRAM': {'W': 5, 'lambd': 0.9}
}

def compute_tcra(v, W=None, lambd=1.0):
    v = np.asarray(v, dtype=float)
    T = len(v) - 1
    if W is None or W > T:
        W = T
    start = T - W + 1
    v_target = v[start : T+1]
    v_lag    = v[start-1 : T]
    weights = lambd ** np.arange(W - 1, -1, -1) if lambd != 1.0 else 1.0
    num = np.sum(weights * v_target * v_lag)
    den = np.sum(weights * (v_lag ** 2))
    beta = num / den if den != 0 else 1.0
    return beta - 1.0, beta

def main():
    # Leer datos
    df = pd.read_csv(r"D:\2026\Tesis2026\Familias-TCROC\data\silver\pib_clean.csv")
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    years = df['Fecha'].dt.year.values
    cols = df.columns[1:]
    
    # Preparar proyecciones
    print("="*60)
    print("RESULTADOS NUMÉRICOS PARA TEX (2025 y 2026)")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=600)
    axes = axes.flatten()
    
    lines_for_legend = []
    labels_for_legend = []
    
    for idx, col in enumerate(cols):
        v = df[col].values
        
        # Calcular betas
        betas = {}
        for var, p in params.items():
            _, b = compute_tcra(v, W=p['W'], lambd=p['lambd'])
            betas[var] = b
            
        print(f"\n--- Variable: {col} ---")
        for var, b in betas.items():
            v_2025 = v[-1] * b
            v_2026 = v_2025 * b
            print(f"[{var}] beta={b:.4f}, alpha={b-1:.4f} | 2025: {v_2025:,.2f} | 2026: {v_2026:,.2f}")
            
        # Graficar
        ax = axes[idx]
        # Mostramos los ultimos datos para ver mejor el pronostico
        show_years = years[-10:]
        show_v = v[-10:]
        
        # Suavizar curva original usando Spline
        years_smooth = np.linspace(show_years.min(), show_years.max(), 300)
        spline = make_interp_spline(show_years, show_v, k=3)
        v_smooth = spline(years_smooth)
        
        line_orig, = ax.plot(years_smooth, v_smooth, color=colors['Original'], linestyle='-', linewidth=1.5, label='Histórico Original', zorder=1)
        # Trazar puntos reales sobre la linea suavizada (Rombos D)
        ax.scatter(show_years, show_v, color=colors['Original'], marker='D', s=20, zorder=2)
        
        if idx == 0:
            lines_for_legend.append(line_orig)
            labels_for_legend.append('Histórico Original')
        
        for idx_var, var in enumerate(['TCRA', 'TCRAM', 'ETCRA', 'ETCRAM']):
            b = betas[var]
            proy_years = [years[-1], 2025, 2026]
            proy_v = [v[-1], v[-1]*b, v[-1]*b*b]
            line_var, = ax.plot(proy_years, proy_v, color=colors[var], linestyle='--', marker='x', label=var)
            if idx == 0:
                lines_for_legend.append(line_var)
                labels_for_legend.append(var)
            
            # Etiqueta alfa especifica para cada sub-variante en el recuadro si es necesario, 
            # pero como la leyenda es global podemos añadir un minitexto.
            ax.annotate(f"$\\alpha$={b-1:.3f}", xy=(2026, v[-1]*b*b), xytext=(5,0), 
                        textcoords="offset points", color=colors[var], fontsize=7, va='center')
            
        ax.set_title(col, fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.5, color=NORD_5)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color(NORD_3)
            spine.set_linewidth(0.8)
        
    # Leyenda unificada arriba (top center)
    fig.legend(lines_for_legend, labels_for_legend, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False, fontsize=10)
    
    plt.tight_layout()
    # Guardar directo en figures de la tesis
    out_dir = r"D:\2026\Tesis2026\Nueva Tesis Marzo Entregable 2026\figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "EjemploPIB_TCRA_4Variantes.png")
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    print(f"\nImagen guardada en: {out_path}")

if __name__ == '__main__':
    main()
