# TCROC-SSRC — Reservorio Estocástico Estructurado (Capítulo 6)

Extensión del modelo TCROC-Markov mediante un Computador de Reservorio Estructurado Estocástico (SSRC), basado en la teoría de Echo State Networks.

## Ejecución

```bash
pip install -r requirements.txt
python run_all.py
```

## Estructura

```
src/
├── config.py              # Hiperparámetros y paleta Nord
├── reservoir/             # Implementación ESN/SSRC
├── evaluation/            # Métricas y test DM
└── visualization/         # Gráficos de comparación

pipelines/
├── 01_prepare_data.py     # Preparar datos (reutiliza TCROC-Markov)
├── 02_verify_theory.py    # Verificaciones teóricas (Teo. 6.1–6.5)
├── 02b_grid_search_ssrc_gui.py  # Búsqueda en grilla con interfaz
├── 03b_comparison_gui.py  # Comparación Markov vs SSRC
├── 04_visualization.py    # Visualización completa
└── 05_overfitting_analysis.py  # Análisis de sobreajuste
```

## Hiperparámetros

| Parámetro | Valores en grilla |
|-----------|:-----------------:|
| D (dimensión) | 20–150 |
| ρ (radio espectral) | 0.70–0.99 |
| a (tasa de fuga) | 0.1–1.0 |
| Realizaciones | 30 (ensemble) |
