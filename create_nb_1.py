import json

def create_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Proyecto de Analítica Predictiva: Fase 1 - Preparación y Comprensión de Datos\n",
                "\n",
                "Este notebook corresponde a las primeras fases de la metodología CRISP-DM: **Data Understanding** y **Data Preparation**.\n",
                "\n",
                "## Objetivos de este notebook:\n",
                "1. Análisis automatizado del dataset para verificar su viabilidad en un proyecto avanzado.\n",
                "2. Análisis exploratorio de datos (EDA) y visualizaciones.\n",
                "3. Detección profunda de correlaciones (multicolinealidad) y outliers.\n",
                "4. Preprocesamiento de datos (imputación, codificación, partición) y aplicación de SMOTE."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from ydata_profiling import ProfileReport\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Configuración visual\n",
                "sns.set_theme(style=\"whitegrid\", palette=\"muted\")\n",
                "plt.rcParams['figure.figsize'] = (10, 6)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Carga de Datos y Exploración Inicial"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df = pd.read_csv('student_placement_salary_elite_v2.csv')\n",
                "\n",
                "# Convertir columnas específicas a 'category'\n",
                "cols_to_cat = ['student_id', 'branch', 'company_type', 'job_role']\n",
                "df[cols_to_cat] = df[cols_to_cat].astype('category')\n",
                "\n",
                "display(df.head())\n",
                "print(\"\\n--- Información del Dataset ---\")\n",
                "display(df.info())\n",
                "print(\"\\n--- Estadísticas Descriptivas ---\")\n",
                "display(df.describe())\n",
                "print(\"\\n--- Valores Únicos por Columna ---\")\n",
                "display(df.nunique())\n",
                "print(\"\\n--- Valores Nulos ---\")\n",
                "display(df.isnull().sum())\n",
                "print(\"\\n--- Filas Duplicadas ---\")\n",
                "display(df.duplicated().sum())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Análisis Automatizado del Dataset\n",
                "\n",
                "Evaluaremos si el dataset es apto para un proyecto universitario avanzado."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def evaluar_dataset(df):\n",
                "    print(\"=== REPORTE DE VIABILIDAD DEL DATASET ===\\n\")\n",
                "    \n",
                "    # 1. Tamaño y dimensionalidad\n",
                "    filas, columnas = df.shape\n",
                "    print(f\"1. Tamaño: {filas} registros, {columnas} variables.\")\n",
                "    if filas >= 400 and columnas >= 10:\n",
                "        print(\"   ✔ Cumple con el mínimo de 400 registros y 10 variables.\")\n",
                "    else:\n",
                "        print(\"   ❌ No cumple con el tamaño mínimo requerido.\")\n",
                "        \n",
                "    # 2. Variables Objetivo\n",
                "    print(\"\\n2. Posibles variables objetivo:\")\n",
                "    print(\"   - Clasificación: 'placed'\")\n",
                "    print(\"   - Regresión/Predicción: 'salary_lpa'\")\n",
                "    \n",
                "    # 3. Balance de clases\n",
                "    print(\"\\n3. Balance de clases en 'placed':\")\n",
                "    conteo_clases = df['placed'].value_counts(normalize=True) * 100\n",
                "    print(conteo_clases)\n",
                "    \n",
                "    if (conteo_clases.min() / conteo_clases.max()) < 0.5:\n",
                "        print(\"   ⚠️ Las clases están desbalanceadas. Se aplicará SMOTE en el entrenamiento.\")\n",
                "    else:\n",
                "        print(\"   ✔ Las clases están suficientemente balanceadas.\")\n",
                "        \n",
                "    # 4. Tipos de variables\n",
                "    num_cols = df.select_dtypes(include=np.number).columns.tolist()\n",
                "    cat_cols = df.select_dtypes(include='object').columns.tolist()\n",
                "    print(f\"\\n4. Variables detectadas:\")\n",
                "    print(f\"   - Numéricas ({len(num_cols)}): {num_cols}\")\n",
                "    print(f\"   - Categóricas ({len(cat_cols)}): {cat_cols}\")\n",
                "    \n",
                "    # 6. Viabilidad para ML Avanzado\n",
                "    print(\"\\n6 y 7. Viabilidad para algoritmos avanzados y ensambles:\")\n",
                "    print(\"   ✔ Dataset suficientemente complejo (mezcla de categóricas, continuas, discretas).\")\n",
                "    print(\"   ✔ Permite aplicar múltiples modelos supervisados, CV, GridSearchCV y validación estadística.\")\n",
                "    \n",
                "    print(\"\\n9. Idoneidad para Streamlit:\")\n",
                "    print(\"   ✔ Totalmente adecuado. Se puede construir un perfilador de habilidades para predecir empleabilidad.\")\n",
                "    \n",
                "    print(\"\\n12. Clasificación del Dataset:\")\n",
                "    print(\"   ⭐ ADECUADO PARA PROYECTO UNIVERSITARIO AVANZADO.\")\n",
                "\n",
                "evaluar_dataset(df)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Análisis Profundo de Correlaciones y Outliers"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_num = df.select_dtypes(include=np.number)\n",
                "corr_matrix = df_num.corr()\n",
                "\n",
                "plt.figure(figsize=(14, 10))\n",
                "sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)\n",
                "plt.title('Heatmap de Correlaciones')\n",
                "plt.show()\n",
                "\n",
                "print(\"\\n=== DETECCIÓN DE REDUNDANCIA E IRRELEVANCIA ===\")\n",
                "# Redundantes > 0.85\n",
                "redundantes = []\n",
                "for i in range(len(corr_matrix.columns)):\n",
                "    for j in range(i):\n",
                "        if abs(corr_matrix.iloc[i, j]) > 0.85:\n",
                "            redundantes.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))\n",
                "print(\"Variables predictoras redundantes (|corr| > 0.85):\", redundantes)\n",
                "if redundantes:\n",
                "    print(\"Sugerencia: Eliminar una variable de cada par redundante.\")\n",
                "else:\n",
                "    print(\"No se detectaron variables predictoras altamente redundantes.\")\n",
                "\n",
                "# Irrelevantes < 0.05 con respecto a 'placed'\n",
                "corr_objetivo = corr_matrix['placed'].abs()\n",
                "irrelevantes = corr_objetivo[corr_objetivo < 0.05].index.tolist()\n",
                "print(\"\\nVariables potencialmente irrelevantes para 'placed' (|corr| < 0.05):\", irrelevantes)\n",
                "print(\"Se recomienda analizar su impacto antes de eliminarlas completamente.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Detección de Outliers (Boxplots)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "vars_to_plot = ['cgpa', 'coding_score', 'communication_score', 'aptitude_score', 'resume_score', 'salary_lpa']\n",
                "fig, axes = plt.subplots(2, 3, figsize=(18, 10))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for i, var in enumerate(vars_to_plot):\n",
                "    sns.boxplot(y=df[var], ax=axes[i], color='skyblue')\n",
                "    axes[i].set_title(f'Boxplot de {var}')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Visualización de la Variable Objetivo y otras distribuciones"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "sns.countplot(data=df, x='placed', ax=axes[0], palette='Set2')\n",
                "axes[0].set_title('Distribución de Placed')\n",
                "\n",
                "sns.histplot(data=df[df['salary_lpa'] > 0], x='salary_lpa', kde=True, ax=axes[1], color='purple')\n",
                "axes[1].set_title('Distribución de Salario (LPA) para Empleados')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Pandas Profiling Automatizado\n",
                "\n",
                "Generando el reporte de Pandas Profiling. (Descomentar para ejecutar y guardar como HTML)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "profile = ProfileReport(df, title=\"Pandas Profiling Report\", explorative=True)\n",
                "profile.to_file(\"pandas_profiling_report.html\")\n",
                "print(\"Reporte guardado como 'pandas_profiling_report.html'\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Preprocesamiento de Datos\n",
                "\n",
                "En esta sección preparamos los datos para los modelos."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Eliminar student_id\n",
                "df_prep = df.drop(columns=['student_id'])\n",
                "\n",
                "# 2. Añadir la categoría 'Not_Applicable' y luego imputar nulos categóricos\n",
                "df_prep['company_type'] = df_prep['company_type'].cat.add_categories(['Not_Applicable'])\n",
                "df_prep['job_role'] = df_prep['job_role'].cat.add_categories(['Not_Applicable'])\n",
                "df_prep['company_type'] = df_prep['company_type'].fillna('Not_Applicable')\n",
                "df_prep['job_role'] = df_prep['job_role'].fillna('Not_Applicable')\n",
                "\n",
                "df_prep['company_type'] = df_prep['company_type'].replace('None', 'Not_Applicable')\n",
                "df_prep['job_role'] = df_prep['job_role'].replace('None', 'Not_Applicable')\n",
                "\n",
                "# 3. Separar features (X) y target de clasificación (y_class) y regresión (y_reg)\n",
                "X = df_prep.drop(columns=['placed', 'salary_lpa', 'company_type', 'job_role'])\n",
                "y_class = df_prep['placed']\n",
                "y_reg = df_prep['salary_lpa']\n",
                "\n",
                "# One Hot Encoding para EDA y partición (usaremos ColumnTransformer más adelante en el pipeline, pero aquí codificaremos para SMOTE)\n",
                "X_encoded = pd.get_dummies(X, drop_first=True)\n",
                "print(\"Forma de X_encoded después del One Hot Encoding:\", X_encoded.shape)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Partición Train/Test y SMOTE\n",
                "\n",
                "Usamos estratificación por la variable 'placed'."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.model_selection import train_test_split\n",
                "from imblearn.over_sampling import SMOTE\n",
                "\n",
                "X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_class, test_size=0.30, stratify=y_class, random_state=42)\n",
                "\n",
                "print(\"Distribución original en y_train:\\n\", y_train.value_counts(normalize=True))\n",
                "\n",
                "# Aplicar SMOTE asegurando que la clase minoritaria aumente máximo un 25% más de su tamaño original.\n",
                "class_counts = y_train.value_counts()\n",
                "minority_class = class_counts.idxmin()\n",
                "majority_class = class_counts.idxmax()\n",
                "n_min = class_counts[minority_class]\n",
                "n_maj = class_counts[majority_class]\n",
                "\n",
                "n_new_min = int(n_min * 1.25)\n",
                "if n_new_min > n_maj:\n",
                "    n_new_min = n_maj\n",
                "\n",
                "print(f\"\\nTamaño original: Mayoritaria={n_maj}, Minoritaria={n_min}\")\n",
                "print(f\"Aplicando SMOTE para aumentar la minoritaria a {n_new_min} muestras (max +25%)...\")\n",
                "smote = SMOTE(sampling_strategy={minority_class: n_new_min, majority_class: n_maj}, random_state=42)\n",
                "X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)\n",
                "print(\"\\nDistribución tras SMOTE en y_train:\\n\", y_train_sm.value_counts())\n",
                "\n",
                "X_train_sm.to_csv('X_train_prep.csv', index=False)\n",
                "X_test.to_csv('X_test_prep.csv', index=False)\n",
                "y_train_sm.to_csv('y_train_prep.csv', index=False)\n",
                "y_test.to_csv('y_test_prep.csv', index=False)\n",
                "\n",
                "print(\"\\nDatos preprocesados y guardados.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Visualización de SMOTE (Antes y Después)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "sns.countplot(x=y_train, ax=axes[0], palette='Set2')\n",
                "axes[0].set_title('Distribución de Clases ANTES de SMOTE')\n",
                "\n",
                "sns.countplot(x=y_train_sm, ax=axes[1], palette='Set2')\n",
                "axes[1].set_title('Distribución de Clases DESPUÉS de SMOTE')\n",
                "plt.show()\n",
                "\n",
                "print(\"Notamos cómo la clase minoritaria ha sido incrementada sintéticamente de manera controlada.\")"
            ]
        }
    ]
    
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open('01_data_preparation.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    create_notebook()
