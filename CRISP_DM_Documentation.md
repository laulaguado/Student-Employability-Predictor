# Documentación del Proyecto: Predicción Avanzada de Empleabilidad (Metodología CRISP-DM)

Este documento detalla rigurosamente las seis fases de la metodología **CRISP-DM** (Cross-Industry Standard Process for Data Mining) aplicadas al proyecto universitario avanzado de predicción de colocación laboral y estimación salarial.

---

## 1. Business Understanding (Comprensión del Negocio)

### 1.1 Objetivo del Proyecto
El objetivo principal es construir una aplicación inteligente capaz de evaluar el perfil integral de un estudiante (habilidades técnicas, rendimiento académico, puntajes de evaluación) para predecir:
- Su probabilidad de asegurar una colocación laboral (empleabilidad).
- Un rango salarial esperado en caso de ser empleado.
- Recomendaciones de posibles roles o cargos a los que debería aspirar.

### 1.2 Justificación Académica y de Negocio
Las universidades y los estudiantes carecen a menudo de herramientas objetivas para prever sus perspectivas laborales. Esta solución provee insights accionables que permiten a los estudiantes enfocarse en áreas de mejora (ej. aumentar su puntaje en DSA o Python) para incrementar su competitividad.

### 1.3 Criterios de Éxito
- **Técnicos**: Alcanzar un F1-Score y ROC AUC superior a 0.80 en la clasificación de empleabilidad, y un error absoluto medio razonable en la estimación salarial.
- **De Negocio**: Proveer una interfaz interactiva y fácil de usar (Streamlit) que arroje predicciones en tiempo real y ofrezca interpretabilidad (puntos fuertes y débiles).

---

## 2. Data Understanding (Comprensión de los Datos)

### 2.1 Recolección Inicial
Se utilizó el dataset `student_placement_salary_elite_v2.csv` que contiene registros de perfiles estudiantiles con variables numéricas y categóricas.

### 2.2 Descripción y Calidad de los Datos
Mediante un análisis automatizado y *Pandas Profiling* en el **Notebook 1**, se determinó:
- **Volumen**: El dataset supera el límite requerido de 400 registros y 10 variables, clasificándose como apto para un proyecto avanzado.
- **Variables Objetivo**:
  - `placed` (Binaria): Para clasificación.
  - `salary_lpa` (Continua): Para regresión.
- **Outliers**: Se detectaron valores atípicos en puntajes (coding, aptitude) y `salary_lpa` usando boxplots.
- **Correlaciones**: Se generó un Heatmap para detectar multicolinealidad (predictores altamente correlacionados entre sí > 0.85) e irrelevancia (correlación < 0.05 con respecto al target).

---

## 3. Data Preparation (Preparación de los Datos)

### 3.1 Limpieza y Transformación
- Eliminación de identificadores únicos no predictivos (`student_id`).
- Imputación de variables categóricas. Dado que `company_type` y `job_role` solo están presentes para estudiantes empleados (y generan un claro caso de *Target Leakage* si se ingresan como predictores crudos), se reemplazaron valores nulos o "None" por "Not_Applicable".
- **One-Hot Encoding**: Aplicado a ramas académicas (`branch`) y niveles universitarios (`college_tier`) para transformar atributos nominales en vectores matemáticos útiles para modelos de aprendizaje.

### 3.2 Partición y Balanceo (SMOTE)
Se evidenció un desbalance en la clase objetivo `placed`. 
Para evitar sesgar el modelo hacia la clase mayoritaria:
- Se dividió el set en Train/Test (70/30) con estratificación.
- Se aplicó **SMOTE** (Synthetic Minority Over-sampling Technique) estrictamente sobre el conjunto de entrenamiento, asegurando que el incremento artificial no distorsione radicalmente la distribución, fijando proporciones racionales (< 25% de aumento bruto) según las reglas del proyecto.

---

## 4. Modeling (Modelado)

### 4.1 Selección de Modelos
Se entrenaron y evaluaron algoritmos avanzados dentro de un marco de Validación Cruzada Estratificada (`StratifiedKFold` de 5 splits):
- **Supervisados Tradicionales**: Regresión Logística, Árbol de Decisión, SVM (Support Vector Classifier), KNN.
- **Ensambles**: Random Forest, LightGBM, AdaBoost.

### 4.2 Optimización de Hiperparámetros
Tras una primera evaluación competitiva, se aislaron los 3 mejores algoritmos. 
- Se utilizó **GridSearchCV** para explorar sistemáticamente los espacios de parámetros de ensambles como Random Forest (max_depth, n_estimators).
- Se aplicó **Optimización Bayesiana (Optuna)** para algoritmos que se benefician de una convergencia inteligente, maximizando específicamente el F1-Score sobre LightGBM.

### 4.3 Pipeline de Procesamiento
Para garantizar reproducibilidad y prevenir *Data Leakage*, se empaquetó el escalado (StandardScaler) y los transformadores polinomiales junto con el mejor clasificador dentro de un `Pipeline` de Scikit-Learn y un pipeline de Imblearn para el despliegue final en producción.

---

## 5. Evaluation (Evaluación)

### 5.1 Métricas de Rendimiento
Se priorizó el **F1-Score** y **ROC AUC** debido a la naturaleza binaria y previamente desbalanceada del problema. Además se consideró Accuracy, Precision y Recall.

### 5.2 Pruebas de Significancia Estadística
- **ANOVA**: Aplicado sobre las distribuciones de F1-Scores de las particiones K-Fold para probar la hipótesis nula de medias iguales.
- **Tukey HSD**: Como el test ANOVA arrojó $p < 0.05$, indicando diferencias significativas, la prueba Post-hoc Tukey HSD se utilizó para identificar matemáticamente qué modelo específico era superior, garantizando una selección basada en rigor estadístico y no en azares de muestreo.

### 5.3 Interpretabilidad
Se generó el gráfico de *Feature Importance*, confirmando estadísticamente qué variables del dominio educativo (ej. habilidades técnicas como DSA, proyectos o CGPA) dictaminan el éxito laboral.

---

## 6. Deployment (Despliegue)

### 6.1 Arquitectura del Despliegue
Los modelos optimizados fueron serializados mediante `joblib` (`pipeline.pkl`, `pipeline_salary.pkl`).

### 6.2 Aplicación Streamlit (`app.py`)
- **Interfaz Profesional**: Interfaz moderna (tema oscuro, responsive) construida en Streamlit.
- **Inferencia en Tiempo Real**: El usuario ingresa sus métricas. El sistema carga los pipelines serializados e inyecta los datos.
- **Output Visual**: Gráficos interactivos de Plotly (tipo Gauge) para ilustrar la probabilidad, visualización comparativa de expectativas salariales frente al promedio, y tags interactivos que recomiendan caminos profesionales basados en combinaciones específicas de habilidades.

### 6.3 Repositorio Final
El proyecto final incluye un estándar de la industria mediante un archivo `requirements.txt` y una estructura de carpetas modular para ser clonado y ejecutado directamente en entornos locales o desplegado en plataformas como Render o Streamlit Cloud.
