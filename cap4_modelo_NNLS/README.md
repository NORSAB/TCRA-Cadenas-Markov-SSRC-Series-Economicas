# TCROC-NNLS — Modelo Híbrido (Capítulo 4)

Estimación de matrices de transición con **mínimos cuadrados no negativos (NNLS)** para series de precios de combustibles de Honduras.

## Ejecución

```bash
pip install numpy pandas scipy
python pipeline.py
```

## Estructura

| Archivo | Descripción |
|---------|-------------|
| `config.py` | Hiperparámetros: W=2, λ=1, K=4, umbrales fijos |
| `pipeline.py` | Pipeline completo: TCRA → discretización → NNLS → validación predictiva |
| `outputs/` | Tabla 4.3 y detalle por partición (60/40 a 95/5) |

## Método

1. Cálculo de tasas α_t (TCRA con ventana W=2)
2. Discretización en K=4 estados con umbrales fijos
3. Estimación MLE de P̂ y aproximación Ap vía SRep/NNLS
4. Predicción one-step-ahead con 8 particiones train/test
