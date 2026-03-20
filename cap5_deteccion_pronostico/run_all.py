import os
import sys
from datetime import datetime

# Orquestador principal de TCROC-Markov MLOps (Full Version)
# Main orchestrator for TCROC-Markov MLOps (Full Version)

def main():
    start_time = datetime.now()
    print(f"====================================================")
    print(f"TCROC-Markov MLOps: INICIO EJECUCIÓN COMPLETA")
    print(f"Timestamp: {start_time}")
    print(f"====================================================")

    # Function to clean output directories
    def clean_output_directories():
        dirs_to_clean = [
            "outputs/figures",
            "outputs/tables"
        ]
        print("\n--- Limpieza de Archivos Anteriores ---")
        for d in dirs_to_clean:
            path = os.path.abspath(d)
            if os.path.exists(path):
                print(f"Limpiando directorio: {d}")
                for filename in os.listdir(path):
                    file_path = os.path.join(path, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(f"Error al borrar {file_path}: {e}")
            else:
                os.makedirs(path, exist_ok=True)
                print(f"Directorio creado: {d}")

    clean_output_directories()

    # Ejecución secuencial de todos los pasos del notebook original
    pipelines = [
        "pipelines/01_ingestion.py",          # STEP 1 (ETL Bronze -> Silver)
        "pipelines/02_processing.py",         # STEP 2 (Silver -> Gold: Alphas)
        "pipelines/grid_search.py",           # STEP 3 & 4 (Optimization)
        "pipelines/03_modeling.py",           # STEP 5 (Matrices & Centroids)
        "pipelines/04_visualization.py",      # STEP 6-18 (Figures, Forecast, Stats)
    ]

    # Incluir src en PYTHONPATH
    src_dir = os.path.abspath("src")
    os.environ["PYTHONPATH"] = src_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

    for pipe in pipelines:
        print(f"\n>>> Ejecutando: {pipe}")
        # Cross-platform way to set PYTHONPATH and run
        # In Windows shell we use set, but for os.system it's better to use env vars in the call if possible
        # Or just rely on the os.environ we set above for subprocesses
        exit_code = os.system(f"python {pipe}")
        
        if exit_code != 0:
            print(f"!!! Error en {pipe}. Ejecución detenida.")
            break

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n====================================================")
    print(f"TCROC-Markov MLOps: EJECUCIÓN COMPLETADA")
    print(f"Duración total: {duration}")
    print(f"====================================================")

if __name__ == "__main__":
    main()
