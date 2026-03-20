# TCROC-Markov: Un Marco de Dos Etapas para el Análisis de Regímenes Económicos

Este repositorio acompaña el artículo:

> **TCROC-Markov: Un Marco de Dos Etapas para el Análisis de Regímenes Económicos**  
> Norman Reynaldo Sabillon Castro  
> Escuela de Matemática y Ciencias de la Computación, UNAH  
> norman.sabillon@unah.edu.hn  
> Tegucigalpa, Honduras  
> [Enlace al artículo o preprint, si ya está disponible]  
>  
> **Borrador del artículo incluido en PDF**

## Descripción

Este repositorio contiene el notebook y los datos necesarios para **reproducir completamente** los experimentos y resultados presentados en el artículo, que propone un enfoque híbrido para el análisis de regímenes económicos usando la discretización TCROC y modelado de cadenas de Markov con estimación robusta NNLS.

## Archivos incluidos

- `TCROCMarkov A Two Stage Framework for Economic Regime Analysis.ipynb`  
  *Notebook principal* con toda la implementación, visualizaciones y análisis.

- `Combustibles.csv`  
  *Datos de precios semanales de combustibles en Honduras (2017–2025).*

- `TCROC_Markov__A_Two_Stage_Framework_for_Economic_Regime_Analysis.pdf`  
  *Borrador del artículo completo No disponible aun.*

## Cómo reproducir los resultados

1. **Clonar el repositorio**
    ```bash
    git clone https://github.com/NORSAB/TCROC-Markov-A-Two-Stage-Framework-for-Economic-Regime-Analysis
    cd TCROC-Markov-A-Two-Stage-Framework-for-Economic-Regime-Analysis
    ```

2. **Instalar las dependencias**
    - Instalar Python 3.8+ y Jupyter Notebook
    - Instalar paquetes requeridos (puedes crear un `requirements.txt` o usar el siguiente comando):
      ```bash
      pip install numpy pandas matplotlib scikit-learn jupyter
      ```

3. **Ejecutar el notebook**
    - Abre `TCROCMarkov A Two Stage Framework for Economic Regime Analysis.ipynb` con Jupyter.
    - Ejecuta las celdas paso a paso; el notebook carga automáticamente los datos y produce los resultados del artículo.

## Datos

Los datos incluidos (`Combustibles.csv`) corresponden a precios semanales de cuatro tipos de combustible en Honduras de 2017 a 2025, recolectados desde el portal Proceso Digital. Uso exclusivamente académico.

## Licencia

Este repositorio se distribuye bajo la licencia MIT para el código y CC-BY para los datos y notebook.

## Citar

Si utiliza este material, cite así:
