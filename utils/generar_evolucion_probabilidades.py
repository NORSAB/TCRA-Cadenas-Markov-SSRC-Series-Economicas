"""
Genera la figura 'evolucion_probabilidades_markov.png'
Evolución de probabilidades de la Matriz P estimada (K=4 regímenes)
con paleta Nord para la tesis.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# ======== PALETA NORD ========
NORD_0  = "#2E3440"
NORD_3  = "#4C566A"
NORD_5  = "#E5E9F0"
NORD_10 = "#5E81AC"   # Frost Blue — alza fuerte
NORD_11 = "#BF616A"   # Aurora Red — caída
NORD_13 = "#EBCB8B"   # Aurora Yellow — subida moderada
NORD_14 = "#A3BE8C"   # Aurora Green — estable

# Estilo
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'text.color': NORD_0,
    'axes.labelcolor': NORD_0,
    'xtick.color': NORD_3,
    'ytick.color': NORD_3,
})

# Matriz P estimada — representativa (ejemplo Super, K=4)
# Estos son valores representativos del capítulo 6
P = np.array([
    [0.20, 0.30, 0.30, 0.05],  # desde s1 (caída)
    [0.35, 0.33, 0.43, 0.05],  # desde s2 (estable)
    [0.40, 0.32, 0.22, 0.05],  # desde s3 (subida)
    [0.05, 0.05, 0.05, 0.85],  # desde s4 (alza fuerte)
])
# Normalizar filas
P = P / P.sum(axis=0, keepdims=True)

# Estado inicial: comienza en estado 1 (caída)
K = P.shape[0]
p0 = np.zeros(K)
p0[0] = 1.0

# Simular evolución
n_steps = 20
trajectory = np.zeros((K, n_steps))
trajectory[:, 0] = p0
for t in range(1, n_steps):
    trajectory[:, t] = P @ trajectory[:, t-1]

# Graficar
fig, ax = plt.subplots(figsize=(12, 7))

colors = [NORD_11, NORD_14, NORD_13, NORD_10]  # caída, estable, subida, alza fuerte
labels = ['caída', 'estable', 'subida', 'alza fuerte']

for k in range(K):
    ax.plot(range(n_steps), trajectory[k, :], marker='D', markersize=5,
            linewidth=2.5, color=colors[k], label=labels[k])

ax.set_xlabel('Tiempo')
ax.set_ylabel('Probabilidad')
ax.set_title('Evolución de Probabilidades (Matriz P Estimada)')
ax.legend(fontsize=12, loc='upper right')
ax.grid(True, alpha=0.3, linestyle=':', color=NORD_3)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

# Bordes
for spine in ax.spines.values():
    spine.set_color(NORD_3)
    spine.set_linewidth(0.5)

output_dir = r"D:\2026\Tesis2026\Nueva Tesis Marzo Entregable 2026\figures"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "evolucion_probabilidades_markov.png")
plt.savefig(out_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Guardado: {out_path}")
