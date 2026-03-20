# TCROC-Cadenas-Markov-SSRC-Series-Economicas

**Modelado de Series Temporales Económicas: De la Tasa de Cambio Relativa a los Modelos de Transición de Régimen Estocásticamente Estructurados**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18752541.svg)](https://doi.org/10.5281/zenodo.18752541)
[![License: MIT](https://img.shields.io/badge/License-MIT-5E81AC.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-A3BE8C.svg)](https://python.org)

---

## Descripción

Código fuente y datos de la tesis de maestría en Matemática (UNAH). Se propone una cadena metodológica para el análisis de regímenes en series económicas de Honduras:

```
Tasa de Cambio Relativa → K-Means → Cadena de Markov → NNLS → SSRC
```

1. **Tasa de Cambio Relativa**: operador variacional que estima tasas de cambio entre valores consecutivos.
2. **Discretización (K-Means)**: cuantización en K regímenes económicos.
3. **Cadena de Markov**: modelado de transiciones con análisis ergódico.
4. **NNLS**: estimación de matrices de transición con restricciones de no-negatividad.
5. **SSRC**: extensión no lineal mediante computación de reservorio.

## Estructura del repositorio

```
TCROC-Cadenas-Markov-SSRC-Series-Economicas/
│
├── cap2_familias_TCRA/           # Cap. 2-3: Variantes del operador
│   ├── src/                      # Módulos: config, modelos, evaluación
│   ├── pipelines/                # Grid search y generación de resultados
│   └── README.md
│
├── cap4_modelo_NNLS/             # Cap. 4: Modelo Híbrido NNLS
│   ├── config.py                 # Hiperparámetros (W=2, λ=1, K=4)
│   ├── pipeline.py               # Pipeline completo
│   └── README.md
│
├── cap5_deteccion_pronostico/    # Cap. 5: Detección de Regímenes y Pronóstico
│   ├── src/                      # Módulos: ingesta, modelos, evaluación
│   ├── pipelines/                # 6 pasos + grid search
│   ├── data/                     # Datos procesados
│   └── README.md
│
├── cap6_reservorio_SSRC/         # Cap. 6: Reservorio Estocástico (SSRC)
│   ├── src/                      # Reservorio ESN, evaluación
│   ├── pipelines/                # Verificación teórica, comparación
│   └── README.md
│
├── utils/                        # Scripts auxiliares (figuras, apéndices)
│   └── README.md
│
├── TCROC-Markov_Original/        # Notebook original del artículo IEEE
│   ├── Articulol.py
│   ├── Combustibles.csv
│   └── README.md
│
└── .gitignore
```

## Resultados principales

| Métrica | Valor |
|---------|:-----:|
| Exactitud predictiva (Cap. 4) | 82–95% |
| Error reconstrucción NNLS | ~10⁻¹³ |
| Tests estadísticos | T pareada, DM, Holm-Bonferroni |
| Estacionariedad (ADF + KPSS) | Confirmada 4 series |

## Requisitos

```bash
pip install numpy scipy matplotlib seaborn pandas scikit-learn statsmodels
```

## Autor

**Norman Reynaldo Sabillón Castro**
Maestría en Matemática — Universidad Nacional Autónoma de Honduras (UNAH)

**Asesor:** Dr. Fredy Vides — UNAH

## Cita

```bibtex
@mastersthesis{sabillon2026tesis,
  author  = {Sabillón Castro, Norman Reynaldo},
  title   = {Modelado de Series Temporales Económicas: De la Tasa de
             Cambio Relativa a los Modelos de Transición de Régimen
             Estocásticamente Estructurados},
  school  = {Universidad Nacional Autónoma de Honduras},
  year    = {2026},
  type    = {Tesis de Maestría},
  doi     = {10.5281/zenodo.18752541},
  url     = {https://doi.org/10.5281/zenodo.18752541}
}
```

## Licencia

Licencia MIT. Ver [LICENSE](LICENSE).
