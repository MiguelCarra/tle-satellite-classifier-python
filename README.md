# Clasificación de Satélites por Grupo mediante Machine Learning
## Proyecto del Máster en Ingeniería de Telecomunicaciones - Asignatura de Procesado Avanzado de Señal para Multimedia - EPS/UAM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23F37626.svg?style=flat&logo=jupyter&logoColor=white)

Este proyecto desarrolla un sistema de clasificación de satélites basado en sus elementos orbitales de dos líneas (TLE). Utilizando técnicas de Machine Learning, se entrenan y evalúan diversos modelos para asignar satélites a grupos funcionales o de naturaleza similar, a partir de datos públicos de Space-Track y Celestrak.

**Desarrollado por:**
* [Miguel Carralero Lanchares](https://www.linkedin.com/in/miguel-carralero-lanchares/) <a href="https://www.linkedin.com/in/miguel-carralero-lanchares/" target="_blank"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="16" style="vertical-align:middle; margin-left:4px"/></a>
* [Luis Palomo de Onís Hernández](https://www.linkedin.com/in/luis-palomo-de-on%C3%ADs-5b1365203/) <a href="https://www.linkedin.com/in/luis-palomo-de-on%C3%ADs-5b1365203/" target="_blank"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="16" style="vertical-align:middle; margin-left:4px"/></a>

## Descripción General

El objetivo principal es construir un clasificador capaz de identificar el grupo al que pertenece un objeto espacial (satélite, debris, etc.) utilizando sus datos orbitales (TLE). Dada la complejidad de clasificar entre decenas de miles de objetos individuales, el problema se enfoca en la clasificación en subgrupos predefinidos que comparten características similares.

El flujo de trabajo incluye:
1.  **Adquisición y Preprocesamiento de Datos:** Recopilación de TLEs diarios de Space-Track y datos de grupos de Celestrak. Limpieza de datos TLE (eliminación de columnas constantes/checksums), fusión con etiquetas de grupo, y manejo de categorías problemáticas ("active", "unknown").
2.  **Balanceo de Clases:** Submuestreo de la clase mayoritaria "starlink" para mitigar el desbalanceo.
3.  **Entrenamiento de Modelos:** Implementación y entrenamiento de clasificadores como SVM, Random Forest y XGBoost. Los modelos entrenados se guardan para agilizar ejecuciones futuras.
4.  **Combinación de Modelos:** Exploración de estrategias de ensemble (promedio, máximo/mínimo de probabilidades, ponderado, votación mayoritaria) para mejorar el rendimiento.
5.  **Evaluación Exhaustiva:** Análisis de métricas como accuracy, reportes de clasificación, matrices de confusión, FAR/FRR por clase y curvas ROC/AUC.
6.  **Análisis de Importancia de Características:** Identificación de los parámetros orbitales más influyentes.

## Tecnologías Utilizadas

*   **Lenguaje:** Python 3
*   **Entorno:** Jupyter Notebook
*   **Análisis de Datos y Machine Learning:**
    *   Pandas (manipulación de datos)
    *   NumPy (cálculo numérico)
    *   Scikit-learn (preprocesamiento, modelos SVM y Random Forest, métricas, LabelEncoder, StandardScaler)
    *   XGBoost (modelo XGBClassifier)
    *   Joblib (guardado y carga de modelos)
*   **Visualización:** Matplotlib, Seaborn

## Estructura del Proyecto
```
.
+-- .gitignore
+-- .gitattributes
+-- LICENSE
+-- README.md
+-- requirements.txt
+-- DDBB/                                (Archivos CSV con datos TLE etiquetados)
| +-- 001_TLE_22ABR_J111_labelled.csv
| +-- ...                                (y otros archivos de datos)
+-- images/                              (Capturas de resultados para este README)
| +-- Evaluation of Individual models/
    +-- ConfMat_RF.png
    +-- ConfMat_SVM.png
    +-- ConfMat_XGBoost.png
    +-- ROC_RF.png
    +-- ROC_SVM.png
    +-- ROC_XGBoost.png
| +-- Evaluation of Model Combinations/
    +-- ...                              (archivos de matrices y ROC de combinaciones)
| +-- Importance of features/
    +-- ...                              (archivos de importancia de características)
+-- src/
|   +-- TLEClasificacionSatelites.ipynb  (Notebook principal con todo el código)
|   +-- SVM_model.joblib                 (Modelo SVM pre-entrenado)
|   +-- Random_Forest_model.joblib       (Modelo Random Forest pre-entrenado)
|   +-- XGBoost_model.joblib             (Modelo XGBoost pre-entrenado)
+-- ...
```

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.
