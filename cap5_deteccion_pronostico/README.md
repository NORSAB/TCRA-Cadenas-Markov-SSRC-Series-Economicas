# TCROC-Markov_Nuevo — Detección y Pronóstico (Capítulo 5, artículo IEEE)

Pipeline completo de detección de regímenes y pronóstico con cadenas de Markov optimizadas. Incluye búsqueda en grilla de hiperparámetros (W, λ, K), tests estadísticos y generación de figuras para el artículo IEEE.

## Ejecución

```bash
pip install -r requirements.txt
python run_all.py
```

## Estructura

```
src/
├── config.py              # Configuración central
├── core/                  # Funciones TCRA y Markov
├── ingestion/             # Carga de datos
├── models/                # Estimación de matrices
├── processing/            # Discretización K-medias
├── evaluation/            # Métricas predictivas
└── visualization/         # Gráficos (heatmaps, grafos, etc.)

pipelines/
├── 01_ingestion.py        # Carga y limpieza
├── 02_processing.py       # Estandarización
├── 03_modeling.py         # Estimación de P̂
├── 04_visualization.py    # Figuras del artículo
├── 05_statistical_tests.py # ADF, Holm-Bonferroni
├── 06_forecasting.py      # Pronóstico one-step
└── grid_search.py         # Optimización (W, λ, K)
```

## Datos

Los datos de combustibles se encuentran en `data/`. Formato CSV con columnas: Fecha, Regular, Super, Diesel, Kerosene.
