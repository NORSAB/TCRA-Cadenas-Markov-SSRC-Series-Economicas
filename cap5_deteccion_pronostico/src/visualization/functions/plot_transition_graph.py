from graphviz import Digraph
import seaborn as sns
import os
import numpy as np

def plot_transition_graph(Ap, series_name, k, centroids=None, output_dir='outputs/figures'):
    """
    Generates and saves a transition graph from a given transition matrix (Ap).
    Genera y guarda un grafo de transición a partir de una matriz de transición dada (Ap).
    """
    dot = Digraph(comment=f'Transition Graph for {series_name}')
    
    # --- Global layout control ---
    dot.attr(rankdir='LR', size='8,8', dpi='300', nodesep='1.0', ranksep='1.5')

    # --- Node and Edge Styling ---
    node_colors = sns.color_palette("pastel", n_colors=k).as_hex()
    
    # Define nodes
    for i in range(k):
        alpha_val = f"<br/> <font point-size='10'>α ≈ {centroids[i]:.3f}</font>" if centroids is not None else ""
        label = f'<<font point-size="16"><b>s<sub>{i+1}</sub></b></font>{alpha_val}>'
        dot.node(f's{i}', label=label, shape='circle', style='filled',
                 fillcolor=node_colors[i], fixedsize='true', width='1.5')

    # Define edges based on transition probabilities
    for i in range(k):  # From state i
        for j in range(k):  # To state j
            prob = Ap[j, i]
            if prob >= 0.01:
                # Use a different, less sensitive formula for self-loops (i == j)
                if i == j:
                    # Make self-loops significantly thinner
                    penwidth_val = 0.6 + prob 
                else:
                    # Keep other transitions more prominent
                    penwidth_val = 1.0 + prob * 2.5
                
                dot.edge(f's{i}', f's{j}', label=f'{prob:.1%}',
                         fontsize='12', penwidth=str(penwidth_val))

    # --- Export ---
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"graph_kmeans_{series_name.lower()}")
    try:
        dot.render(filename, format='png', cleanup=True)
    except Exception as e:
        print(f"  ERROR: Could not render graph for {series_name}. {e}")

def plot_quantile_transition_graph(Ap, series_name, boundaries, output_dir):
    """
    Generates a transition graph specifically for the 4-state quantile model,
    arranging the nodes in a 2x2 matrix layout.
    """
    k = 4  
    dot = Digraph(comment=f'Quantile Transition Graph for {series_name}')
    
    # --- Global layout control ---
    dot.attr(dpi='300', nodesep='0.8', ranksep='1.2')

    # --- Node Styling and 2x2 Layout using Subgraphs ---
    node_colors = sns.color_palette("pastel", n_colors=k).as_hex()
    
    # Fila superior (s1, s2)
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('s0', label=f'<<font point-size="16"><b>s<sub>1</sub></b></font><br/> <font point-size="10">α &lt; {boundaries[0]:.2f}</font>>', 
               shape='circle', style='filled', fillcolor=node_colors[0], fixedsize='true', width='1.8')
        s.node('s1', label=f'<<font point-size="16"><b>s<sub>2</sub></b></font><br/> <font point-size="10">{boundaries[0]:.2f} ≤ α &lt; {boundaries[1]:.2f}</font>>', 
               shape='circle', style='filled', fillcolor=node_colors[1], fixedsize='true', width='1.8')

    # Fila inferior (s3, s4)
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('s2', label=f'<<font point-size="16"><b>s<sub>3</sub></b></font><br/> <font point-size="10">{boundaries[1]:.2f} ≤ α &lt; {boundaries[2]:.2f}</font>>', 
               shape='circle', style='filled', fillcolor=node_colors[2], fixedsize='true', width='1.8')
        s.node('s3', label=f'<<font point-size="16"><b>s<sub>4</sub></b></font><br/> <font point-size="10">α ≥ {boundaries[2]:.2f}</font>>', 
               shape='circle', style='filled', fillcolor=node_colors[3], fixedsize='true', width='1.8')

    # --- Define Edges ---
    for i in range(k):  # From state i
        for j in range(k):  # To state j
            prob = Ap[j, i]
            if prob >= 0.01:
                penwidth_val = 0.6 + prob if i == j else 1.0 + prob * 2.5
                dot.edge(f's{i}', f's{j}', label=f'{prob:.1%}',
                         fontsize='12', penwidth=str(penwidth_val))

    # --- Export ---
    filename = os.path.join(output_dir, f"graph_quantiles_{series_name.lower()}")
    try:
        dot.render(filename, format='png', cleanup=True)
    except Exception as e:
        print(f"  ERROR: Could not render graph for {series_name}. {e}")
