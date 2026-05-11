# 🎓 EduPredict: Analytics & Placement Machine Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit--Learn-orange.svg)
![Streamlit](https://img.shields.io/badge/Despliegue-Streamlit-red.svg)

Un proyecto universitario avanzado e integral sobre **Analítica Predictiva**, enfocado en determinar la probabilidad de colocación laboral (empleabilidad), estimar salarios y sugerir roles profesionales para estudiantes, aplicando estrictamente la metodología **CRISP-DM**.

---

## 📌 Descripción del Proyecto

El objetivo de esta solución es aplicar inteligencia artificial sobre perfiles de estudiantes (notas, habilidades en programación, tipo de universidad y puntajes de aptitud) para generar insights accionables. 

Este repositorio demuestra dominio completo del ciclo de vida del Machine Learning, incluyendo:
1. Análisis de viabilidad y calidad del dataset.
2. Feature Engineering, balanceo de clases sintético (SMOTE) y manejo de correlaciones (Multicolinealidad).
3. Experimentación sistemática con múltiples familias de algoritmos (Regresión Logística, Árboles de Decisión, SVM, KNN) y potentes métodos de ensamble (Random Forest, LightGBM, AdaBoost).
4. Rigurosidad estadística aplicando **ANOVA** y pruebas **Tukey HSD** sobre los resultados de validación cruzada.
5. Optimización de hiperparámetros híbrida (GridSearchCV y Optimización Bayesiana vía Optuna).
6. Creación de pipelines de inferencia y despliegue final a través de una atractiva Web App interactiva.

---

## 🏗 Arquitectura del Proyecto

```text
📦 Student_Placement_Project
 ┣ 📜 01_data_preparation.ipynb    # Fase 1: EDA, Limpieza, Correlaciones, SMOTE y Profiling
 ┣ 📜 02_predictive_modeling.ipynb # Fase 2: Modelado, CV, ANOVA, Optimización y Exportación
 ┣ 📜 app.py                       # Fase 3: Despliegue en Streamlit interactivo
 ┣ 📜 pipeline.pkl                 # Pipeline serializado de clasificación (Generado por los Notebooks)
 ┣ 📜 pipeline_salary.pkl          # Pipeline serializado de regresión salarial (Generado por los Notebooks)
 ┣ 📜 CRISP_DM_Documentation.md    # Documentación técnica metodológica en español
 ┣ 📜 requirements.txt             # Dependencias del proyecto
 ┗ 📜 student_placement_salary_elite_v2.csv # Dataset crudo
```

---

## 🚀 Instalación y Ejecución

Para clonar y correr esta aplicación localmente, asegúrate de tener Python 3.8+ instalado.

**1. Clonar el repositorio y navegar a la carpeta:**
```bash
git clone <tu-repositorio>
cd <nombre-carpeta>
```

**2. Instalar dependencias:**
```bash
pip install -r requirements.txt
```

**3. Entrenar Modelos (Importante):**
Para generar los archivos `.pkl` requeridos por la aplicación, debes ejecutar secuencialmente los notebooks o scripts.
```bash
# Si corres los scripts provistos en la terminal:
python create_nb_1.py
python create_nb_2.py

# Opcional: Entrar a Jupyter y correrlos manualmente
jupyter notebook
```

**4. Ejecutar la Web App (Streamlit):**
```bash
streamlit run app.py
```
*Se abrirá automáticamente tu navegador apuntando a `localhost:8501`.*

---

## 📈 Metodología CRISP-DM
Este proyecto fue construido usando CRISP-DM. Para una revisión detallada de las consideraciones de negocio, diseño de matriz de confusión, variables seleccionadas y conclusiones estadísticas, referirse al archivo adjunto: [CRISP_DM_Documentation.md](CRISP_DM_Documentation.md)

---

## 🛠 Tecnologías Utilizadas
- **Análisis y Manipulación:** Pandas, Numpy, YData-Profiling.
- **Visualización:** Matplotlib, Seaborn, Plotly.
- **Machine Learning:** Scikit-Learn, LightGBM, Imbalanced-Learn (SMOTE).
- **Estadística y Optimización:** SciPy, StatsModels, Optuna.
- **Despliegue:** Streamlit, Joblib.

---

*Desarrollado como proyecto universitario avanzado de Machine Learning.*
