import matplotlib.pyplot as plt
import seaborn as sns

def setup_plotting_style():
    """
    Sets the global plotting style based on Articulol.py.
    Configura el estilo global para los gráficos basado en Articulol.py.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("colorblind")
