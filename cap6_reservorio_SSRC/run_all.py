"""
===================================================================
TCROC-SSRC: Orquestador Principal (run_all.py)
===================================================================
Ejecuta todos los pasos del pipeline secuencialmente.
"""
import os
import sys
from datetime import datetime


def main():
    start_time = datetime.now()
    print("=" * 60)
    print("TCROC-SSRC MLOps: INICIO EJECUCION COMPLETA")
    print("Timestamp: {}".format(start_time))
    print("=" * 60)

    # Asegurar PYTHONPATH incluye src
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    os.environ["PYTHONPATH"] = src_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

    # Crear directorios de salida
    for d in ['outputs/figures', 'outputs/tables', 'outputs/models']:
        os.makedirs(d, exist_ok=True)

    # Ejecución secuencial
    pipelines = [
        "pipelines/01_prepare_data.py",     # Preparar datos (reutiliza TCROC-Markov)
        "pipelines/02_verify_theory.py",    # Verificaciones teóricas
        "pipelines/02b_grid_search_ssrc_gui.py", # Búsqueda en grilla con GUI (Nativo)
        "pipelines/03b_comparison_gui.py",      # Comparación Markov vs SSRC (GUI)
        "pipelines/04_visualization.py"     # Visualización Completa + Auditorías
    ]

    for pipe in pipelines:
        print("\n>>> Ejecutando: {}".format(pipe))
        exit_code = os.system("python {}".format(pipe))

        if exit_code != 0:
            print("!!! Error en {}. Ejecucion detenida.".format(pipe))
            break

    end_time = datetime.now()
    duration = end_time - start_time
    print("\n" + "=" * 60)
    print("TCROC-SSRC MLOps: EJECUCION COMPLETADA")
    print("Duracion total: {}".format(duration))
    print("=" * 60)


if __name__ == "__main__":
    main()
