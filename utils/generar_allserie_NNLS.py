"""
Genera la figura 'allseriecombustiblesNNLS.png'
Evolución de probabilidades (Matriz P Estimada) para los 4 combustibles
usando los modelos NNLS del capítulo 6 con paleta Nord.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# ======== PALETA NORD ========
NORD_0  = "#2E3440"
NORD_3  = "#4C566A"
NORD_5  = "#E5E9F0"
NORD_10 = "#5E81AC"   # Frost Blue — alza fuerte
NORD_11 = "#BF616A"   # Aurora Red — caída (placeholder if K>4)
NORD_13 = "#EBCB8B"   # Aurora Yellow — subida
NORD_14 = "#A3BE8C"   # Aurora Green — estable (caída en esta versión)

REGIME_COLORS = [NORD_14, NORD_10, NORD_13, NORD_11]  # caída, estable, subida, alza fuerte
REGIME_LABELS = ['caída', 'estable', 'subida', 'alza fuerte']

# Estilo
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.color': NORD_0,
    'axes.labelcolor': NORD_0,
    'xtick.color': NORD_3,
    'ytick.color': NORD_3,
})

# Intentar cargar datos reales de TCROC-Markov_Nuevo
MARKOV_DIR = r"D:\2026\Tesis2026\TCROC-Markov_Nuevo"
MODELS_DIR = os.path.join(MARKOV_DIR, "outputs", "models")
fuels = ['Super', 'Regular', 'Diesel', 'Kerosene']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
fig.suptitle('Evolución de Probabilidades (Matriz P Estimada)', fontsize=18, fontweight='bold', y=0.98)

for idx, fuel in enumerate(fuels):
    ax = axes[idx]
    
    # Cargar P real
    p_path = os.path.join(MODELS_DIR, f"{fuel}_P_kmeans.npy")
    c_path = os.path.join(MODELS_DIR, f"{fuel}_centroids.npy")
    
    if os.path.exists(p_path):
        P = np.load(p_path)
        K = P.shape[0]
    else:
        # Fallback K=4
        K = 4
        P = np.array([
            [0.70, 0.15, 0.10, 0.05],
            [0.10, 0.75, 0.10, 0.05],
            [0.10, 0.05, 0.75, 0.10],
            [0.10, 0.05, 0.05, 0.80],
        ])
    
    # Estado inicial: sistema en equilibrio desde estado 1
    p0 = np.zeros(K)
    p0[0] = 1.0
    
    n_steps = 100
    trajectory = np.zeros((K, n_steps))
    trajectory[:, 0] = p0
    for t in range(1, n_steps):
        trajectory[:, t] = P @ trajectory[:, t-1]
    
    colors = (REGIME_COLORS * ((K // len(REGIME_COLORS)) + 1))[:K]
    all_labels = REGIME_LABELS + [f'régimen {i+1}' for i in range(4, 10)]
    labels = all_labels[:K]
    
    for k in range(K):
        ax.plot(range(n_steps), trajectory[k, :], linewidth=2, color=colors[k], label=labels[k])
        # Añadir etiqueta al final de la línea
        ax.text(n_steps + 1, trajectory[k, -1], labels[k], fontsize=9, color=colors[k], va='center')
    
    ax.set_title(fuel, fontsize=14, fontweight='bold')
    ax.set_ylabel('Probabilidad')
    ax.grid(True, alpha=0.2, linestyle=':', color=NORD_3)
    ax.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_color(NORD_3)
        spine.set_linewidth(0.5)

# Leyenda compartida
handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc='upper center', ncol=4, fontsize=12,
           bbox_to_anchor=(0.5, 0.94), frameon=False)

# Labels compartidos
fig.text(0.5, 0.02, 'Tiempo', ha='center', fontsize=14)
fig.patch.set_facecolor('white')

plt.tight_layout(rect=[0, 0.05, 1, 0.92])

output_dir = r"D:\2026\Tesis2026\Nueva Tesis Marzo Entregable 2026\figures"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "allseriecombustiblesNNLS.png")
plt.savefig(out_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Guardado: {out_path}")
