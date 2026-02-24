# EcoSeries-RegimeSwitching

**Modelado de Series Temporales Económicas: De la Tasa de Cambio Relativa a los Modelos de Transición de Régimen Estocásticamente Estructurados**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18752541.svg)](https://doi.org/10.5281/zenodo.18752541)
[![License: MIT](https://img.shields.io/badge/License-MIT-5E81AC.svg)](LICENSE)
[![LaTeX](https://img.shields.io/badge/LaTeX-Thesis-2E3440.svg)]()
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-A3BE8C.svg)](https://python.org)

---

## Descripción

Este repositorio contiene el código fuente, los datos y la documentación completa de la tesis de maestría en Matemática con orientación en Ingeniería Matemática, presentada ante la Universidad Nacional Autónoma de Honduras (UNAH).

El trabajo propone una cadena metodológica original para el análisis de series temporales económicas:

```
TCROC → K-Means → Cadena de Markov → NNLS → SSRC
```

1. **TCROC** (Tasa de Cambio Relativa Óptima Conmutada): operador variacional que estima la tasa de cambio relativa entre valores consecutivos.
2. **Discretización con K-Means**: cuantización de la serie continua de tasas en K regímenes económicos.
3. **Cadena de Markov**: modelado de las transiciones entre regímenes con análisis ergódico y propiedades de mezcla.
4. **NNLS** (Mínimos Cuadrados No Negativos): estimación de la matriz de transición estocástica con restricciones de no-negatividad.
5. **SSRC** (Stochastically Structured Reservoir Computing): extensión no lineal mediante computación de reservorio.

## Estructura del repositorio

```
EcoSeries-RegimeSwitching/
├── Tesis_Final_UNAH/          # Documento LaTeX de la tesis
│   ├── Tesis.tex              # Archivo principal
│   ├── CAPITULOS/             # Capítulos 1-9, glosario, notación, apéndices
│   ├── figures/               # Figuras generadas
│   ├── references.bib         # Bibliografía (51 entradas)
│   ├── Portada.sty            # Estilo de portada UNAH
│   └── thesis_colors.py       # Paleta de colores Nord oficial
│
├── scripts/                   # Scripts de análisis y generación
│   ├── tcroc_optimizer.py     # Optimización de hiperparámetros (W, λ)
│   ├── markov_estimation.py   # Estimación de matrices de transición
│   ├── regime_analysis.py     # Análisis de regímenes económicos
│   └── generate_figures.py    # Regeneración de todas las figuras
│
├── data/                      # Datos de series temporales
│   ├── combustibles/          # Precios de combustibles Honduras 2017-2025
│   └── pib/                   # PIB trimestral Honduras
│
├── results/                   # Resultados y métricas
│   ├── matrices/              # Matrices de transición estimadas
│   ├── metrics/               # Métricas de evaluación (RMSE, exactitud)
│   └── dashboards/            # Dashboards HTML interactivos
│
└── README.md
```

## Metodología

### Operador TCROC y variantes

| Variante | Parámetros | Descripción             |
| -------- | ---------- | ----------------------- |
| TCROC    | α_T        | Tasa global canónica    |
| TCROCM   | α_T, W     | Ventana móvil           |
| ETCROC   | α_T, λ     | Decaimiento exponencial |
| ETCROCM  | α_T, λ, W  | Ventana + decaimiento   |

### Resultados principales

- Exactitud de predicción de régimen: **85-95%** según la serie
- Error de reconstrucción NNLS: **~10⁻¹³** (precisión de punto flotante)
- Tests estadísticos: T pareada (p < 0.01), Diebold-Mariano significativo
- Test ADF de estacionariedad: p < 0.01 para todas las series de α_t

## Requisitos

```bash
# Python
pip install numpy scipy matplotlib seaborn pandas scikit-learn

# LaTeX (para compilar la tesis)
# TinyTeX o TeX Live con paquetes: tikz, pgfplots, amsmath, natbib
```

## Compilación de la tesis

```bash
cd Tesis_Final_UNAH
pdflatex Tesis.tex
bibtex Tesis
pdflatex Tesis.tex
pdflatex Tesis.tex
```

## Paleta de colores

El proyecto utiliza la paleta **[Nord](https://nordtheme.com)** para garantizar consistencia visual:

| Color         | Hex       | Uso                       |
| ------------- | --------- | ------------------------- |
| Polar Night   | `#2E3440` | Texto, bordes principales |
| Snow Storm    | `#ECEFF4` | Fondos de diagramas       |
| Frost Blue    | `#5E81AC` | Acentos, enlaces          |
| Aurora Green  | `#A3BE8C` | Régimen estable           |
| Aurora Yellow | `#EBCB8B` | Régimen moderado          |
| Aurora Red    | `#BF616A` | Régimen de caída          |
| Aurora Purple | `#B48EAD` | Extensión SSRC            |

## Autor

**Norman Reynaldo Sabillón Castro**
Maestría en Matemática — Universidad Nacional Autónoma de Honduras (UNAH)

**Asesor:** Dr. Fredy Vides — UNAH

## Cita

```bibtex
@mastersthesis{sabillon2026ecoseries,
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

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.
