# Familias-TCROC — Variantes TCRA (Capítulos 2–3)

Comparación de 4 variantes de la Tasa de Cambio Relativa y Acumulada (TCRA) aplicadas a series de precios de combustibles.

## Ejecución

```bash
pip install -r requirements.txt
python run_all.py
```

## Estructura

```
src/
├── config.py           # Hiperparámetros y rutas
├── core/               # Funciones TCRA
├── models/             # Modelos de transición
├── processing/         # Ingesta y discretización
├── evaluation/         # Métricas predictivas
└── visualization/      # Gráficos comparativos

pipelines/
├── grid_search.py      # Búsqueda en grilla (W, λ, K)
├── gold_generation.py  # Generación de resultados finales
└── ...
```

## Variantes comparadas

| Variante | W | λ |
|----------|:-:|:-:|
| Fija     | fijo | fijo |
| W-óptimo | óptimo | fijo |
| λ-óptimo | fijo | óptimo |
| Ambos    | óptimo | óptimo |
