import os
import sys
from PIL import Image

def compose_graph_panel(graph_paths, output_path):
    """
    Composes 4 individual graph images into a 2x2 panel.
    Compone 4 imágenes de grafos en un panel de 2x2.
    """
    images = []
    for p in graph_paths:
        if os.path.exists(p):
            images.append(Image.open(p))
    
    if len(images) < 4:
        print("Not enough images to compose panel.")
        return
        
    # Assume all same size or resize
    w, h = images[0].size
    new_im = Image.new('RGB', (w*2, h*2), (255, 255, 255))
    
    new_im.paste(images[0], (0, 0))
    new_im.paste(images[1], (w, 0))
    new_im.paste(images[2], (0, h))
    new_im.paste(images[3], (w, h))
    
    new_im.save(output_path, dpi=(600, 600))
    print(f"Panel saved to {output_path}")

def run_graph_generation():
    # Helper to generate individual and panel
    from src.config import REQUIRED_COLUMNS, MODELS_DIR, FIGURES_DIR
    from src.visualization.functions.plot_transition_graph import plot_transition_graph
    
    kmeans_paths = []
    quantile_paths = []
    
    for fuel in REQUIRED_COLUMNS:
        p_km = os.path.join(MODELS_DIR, f"{fuel}_P_kmeans.npy")
        p_q = os.path.join(MODELS_DIR, f"{fuel}_P_quantiles.npy")
        n_km = os.path.join(MODELS_DIR, f"{fuel}_regime_names.npy")
        
        if os.path.exists(p_km):
            P = np.load(p_km)
            names = np.load(n_km)
            out = os.path.join(FIGURES_DIR, f"graph_km_{fuel}.png")
            plot_transition_graph(P, names, title=f"K-Means: {fuel}", output_path=out)
            kmeans_paths.append(out)
            
        if os.path.exists(p_q):
            P = np.load(p_q)
            # Use states indices as names for quantiles if no names
            out = os.path.join(FIGURES_DIR, f"graph_q_{fuel}.png")
            plot_transition_graph(P, [f"s{i+1}" for i in range(len(P))], title=f"Quantiles: {fuel}", output_path=out)
            quantile_paths.append(out)
            
    if len(kmeans_paths) == 4:
        compose_graph_panel(kmeans_paths, os.path.join(FIGURES_DIR, "6_markov_graph_kmeans_panel.png"))
    if len(quantile_paths) == 4:
        compose_graph_panel(quantile_paths, os.path.join(FIGURES_DIR, "7_markov_graph_quantiles_panel.png"))
