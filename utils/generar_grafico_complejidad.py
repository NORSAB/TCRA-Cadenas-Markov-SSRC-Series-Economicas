import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

# ======== PALETA NORD (thesis_colors.py) ========
NORD_0  = "#2E3440"   # Polar Night — texto
NORD_3  = "#4C566A"   # Bordes, texto secundario
NORD_5  = "#E5E9F0"   # Fondos intermedios
NORD_7  = "#8FBCBB"   # Frost 1
NORD_9  = "#81A1C1"   # Frost 3
NORD_10 = "#5E81AC"   # Frost Blue — acentos
NORD_11 = "#BF616A"   # Aurora Red
NORD_12 = "#D08770"   # Aurora Orange
NORD_13 = "#EBCB8B"   # Aurora Yellow
NORD_14 = "#A3BE8C"   # Aurora Green
NORD_15 = "#B48EAD"   # Aurora Purple

# Configuraciones de estilo
plt.style.use('default')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['text.color'] = NORD_0
mpl.rcParams['axes.labelcolor'] = NORD_0
mpl.rcParams['xtick.color'] = NORD_3
mpl.rcParams['ytick.color'] = NORD_3

# Directorio de Salida — guardar directo en figures de la tesis
output_dir = r"D:\2026\Tesis2026\Nueva Tesis Marzo Entregable 2026\figures"
os.makedirs(output_dir, exist_ok=True)

# Horizonte temporal T (Ambos ejes serán logarítmicos)
T = np.logspace(1, 6, 200) # De 10 a 1,000,000
W = 52  # Tamaño de ventana típica para TCRAM (Ej. 52 semanas)
L = 500 # Ponderación para Redes Neuronales (Representando hiperparámetros: Capas x Épocas x Lote)

# Definición de Ecuaciones de Complejidad Computacional (Entrenamiento/Estimación)
# Añadimos constantes para que las líneas no colapsen visualmente en una sola.

# Clase O(T) - Soluciones Analíticas Directas o Recursivas
O_TCRA    = 2 * T          
O_ETCRA   = 3 * T          
O_OLS     = 5 * T          

# Clase O(T * Constante Fija Menor) - Ventanas Móviles Locales
O_TCRAM   = 2 * T * W      
O_ETCRAM  = 3 * T * W      

# Clase O(T * Constante Masiva) - Backpropagation Empírico
# El tribunal te observó esto: Una RN por Backprop escala con T, pero multiplicada por 
# parámetros de arquitectura masivos (Épocas, Capas, Pesos iterativos). Es por eso que "L" 
# eleva la curva brutalmente frente a la TCROCM, a pesar de seguir siendo lineal respecto a T.
O_NN      = 5 * T * L      

# Clase iterativa Superior O(T^2), O(T^3) - Optimización tradicional y Cadenas de Markov
O_ARIMA   = 2 * T**2       
O_MLE     = 0.5 * T**2.5   

plt.figure(figsize=(10, 6.5))

# Plotting con paleta Nord
plt.plot(T, O_TCRA,    label=r'TCRA $\mathcal{O}(T)$', linewidth=2.5, color=NORD_10)  # Frost Blue
plt.plot(T, O_ETCRA,   label=r'ETCRA $\mathcal{O}(T)$', linewidth=2.5, linestyle='--', color=NORD_9)   # Frost 3
plt.plot(T, O_OLS,     label=r'OLS Clásico $\mathcal{O}(T)$', linewidth=1.5, linestyle=':', color=NORD_3) # Gris Nord

plt.plot(T, O_TCRAM,   label=r'TCRAM $\mathcal{O}(T \cdot W)$', linewidth=2.5, color=NORD_13)  # Aurora Yellow
plt.plot(T, O_ETCRAM,  label=r'ETCRAM $\mathcal{O}(T \cdot W)$', linewidth=2.5, linestyle='--', color=NORD_12) # Aurora Orange

# RN ahora tiene un salto visual claro justificando lo que te preguntó el jurado
plt.plot(T, O_NN,      label=r'Redes Neuronales $\mathcal{O}(T \cdot L)$', linewidth=2.5, linestyle='-.', color=NORD_15) # Aurora Purple

plt.plot(T, O_ARIMA,   label=r'ARIMA $\mathcal{O}(T^2)$', linewidth=2.5, color=NORD_14)  # Aurora Green
plt.plot(T, O_MLE,     label=r'MLE (Markov-S) $\mathcal{O}(T^3)$', linewidth=2.5, color=NORD_11) # Aurora Red

# Ajustes Log-Log
plt.xscale('log')
plt.yscale('log')

# Estética IEEE y Rejilla suavizada
plt.xlabel('Tamaño de la Serie Temporal $T$ (Escala Log)', fontweight='bold')
plt.ylabel('Operaciones Teóricas (Escala Log)', fontweight='bold')
plt.title('Comparativa de Complejidad Algorítmica', fontweight='bold', pad=15)

# Grilla suavizada color gris claro
plt.grid(True, which="major", ls="-", alpha=0.5, color=NORD_5)
plt.grid(True, which="minor", ls="--", alpha=0.25, color=NORD_5)

ax = plt.gca()
ax.set_facecolor('white')
for spine in ax.spines.values():
    spine.set_color(NORD_3)
    spine.set_linewidth(1.0)

# Leyenda sin marco
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, 
           title="Variantes y Métodos")

plt.tight_layout()
output_path = os.path.join(output_dir, 'CompTodosMeto.png')
# Se imprime en 800 DPI para resolución absoluta de prensa
plt.savefig(output_path, dpi=800, bbox_inches='tight')
plt.close()
