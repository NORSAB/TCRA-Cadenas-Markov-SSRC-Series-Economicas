import sys
import os
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from pipelines.silver_processing import run_silver
from pipelines.grid_search import run_grid_search
from pipelines.gold_generation import run_gold
from pipelines.visualization import run_viz
from src.config import FIGURES_DIR

def clean_outputs():
    print("Limpiando gráficos y métricas anteriores...")
    if FIGURES_DIR.exists():
        for file in FIGURES_DIR.glob('*.png'):
            try:
                file.unlink()
            except Exception as e:
                print(f"No se pudo borrar {file}: {e}")

def main():
    root = tk.Tk()
    root.title("Monitor de Optimización: 8 Series TCROC")
    root.geometry("450x180")
    root.eval('tk::PlaceWindow . center')
    
    style = ttk.Style()
    style.theme_use('clam')
    
    lbl_title = ttk.Label(root, text="Optimizando y evaluando 8 series...", font=("Arial", 12, "bold"))
    lbl_title.pack(pady=10)
    
    lbl_status = ttk.Label(root, text="Inicializando...", font=("Arial", 10))
    lbl_status.pack(pady=5)
    
    progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=350, mode='determinate')
    progress.pack(pady=10)
    
    lbl_pct = ttk.Label(root, text="0%", font=("Arial", 10, "bold"))
    lbl_pct.pack()

    def update_gui(pct, text):
        progress['value'] = pct
        lbl_pct.config(text=f"{pct}%")
        lbl_status.config(text=text)
        root.update_idletasks()

    def pipeline_thread():
        try:
            update_gui(5, "Fase 1: Limpieza de salidas y preprocesamiento Silver...")
            clean_outputs()
            run_silver()
            
            # Las 8 series se procesan en los Grid Search
            update_gui(20, "Fase 2: Grid Search (Combustibles - 4 series)")
            run_grid_search("combustibles")
            
            update_gui(60, "Fase 2: Grid Search (PIB - 4 series)")
            run_grid_search("pib")
            
            update_gui(85, "Fase 3: Generación de tabla final Gold...")
            run_gold()
            
            update_gui(88, "Fase 3.5: Consolidando tablas resumen de análisis...")
            from pipelines.summary_tables import run_tables
            run_tables()
            
            update_gui(90, "Fase 4: Generando gráficos calidad IEEE (600 dpi)...")
            run_viz()
            
            update_gui(100, "¡Revisa la terminal para ver tus tablas de referencia!")
            root.after(3000, root.destroy)
            
            print("\n" + "="*90)
            print("===          TABLAS DE REFERENCIA: MEJOR VARIANTE Y PARÁMETROS POR MÉTRICA          ===")
            print("="*90)
            import pandas as pd
            from src.config import TABLES_DIR
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 2000)
            
            for ds in ["combustibles", "pib"]:
                file_path = TABLES_DIR / f"best_combinations_per_metric_{ds}.csv"
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    print(f"\n--- MEJORES COMBINACIONES PARA {ds.upper()} ---")
                    # Formato limpio de números
                    for m in ["RMSE", "MAE", "MedAE", "MSLE", "MAPE", "R2", "Lambda"]:
                        if m in df.columns:
                            try:
                                df[m] = df[m].apply(lambda x: f"{float(x):.4f}" if pd.notnull(x) else "NaN")
                            except:
                                pass
                    print(df.to_string(index=False))
                    
            print("\n" + "="*90)
            print("===                         Pipeline Completed Successfully                         ===")
            print("="*90)
        except Exception as e:
            update_gui(0, f"Error: {str(e)}")
            print(f"Error en pipeline: {e}")

    # Correr el pipeline en un hilo separado para que Tkinter no se bloquee
    threading.Thread(target=pipeline_thread, daemon=True).start()
    
    root.mainloop()

if __name__ == "__main__":
    main()
