import json

def create_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Proyecto de Analítica Predictiva: Fase 2 - Modelado y Evaluación\n",
                "\n",
                "Este notebook cubre las fases de **Modeling** y **Evaluation** de CRISP-DM.\n",
                "\n",
                "## Objetivos:\n",
                "1. Aplicar mínimo 4 modelos supervisados y 3 de ensamble.\n",
                "2. Evaluación rigurosa usando validación cruzada (StratifiedKFold).\n",
                "3. Comparación estadística usando ANOVA y Tukey HSD sobre F1-Score.\n",
                "4. Optimización de hiperparámetros (GridSearchCV y Bayesiana) para los 3 mejores.\n",
                "5. Evaluación del mejor modelo en test set.\n",
                "6. Creación y exportación del Pipeline final completo."
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
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Modelos\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.tree import DecisionTreeClassifier\n",
                "from sklearn.svm import SVC\n",
                "from sklearn.neighbors import KNeighborsClassifier\n",
                "from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier\n",
                "import lightgbm as lgb\n",
                "\n",
                "# Evaluación\n",
                "from sklearn.model_selection import StratifiedKFold, cross_validate\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report\n",
                "\n",
                "# Estadística\n",
                "from scipy.stats import f_oneway\n",
                "from statsmodels.stats.multicomp import pairwise_tukeyhsd\n",
                "\n",
                "# Optimización y Pipeline\n",
                "from sklearn.model_selection import GridSearchCV\n",
                "import optuna\n",
                "from sklearn.pipeline import Pipeline\n",
                "from sklearn.compose import ColumnTransformer\n",
                "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                "import joblib"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Carga de Datos Preprocesados\n",
                "Cargamos los datos que fueron divididos y balanceados (X_train_sm) en el notebook anterior."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "try:\n",
                "    X_train_sm = pd.read_csv('X_train_prep.csv')\n",
                "    X_test = pd.read_csv('X_test_prep.csv')\n",
                "    y_train_sm = pd.read_csv('y_train_prep.csv')['placed']\n",
                "    y_test = pd.read_csv('y_test_prep.csv')['placed']\n",
                "    print(\"Datos cargados correctamente.\")\n",
                "except Exception as e:\n",
                "    print(\"Error cargando datos. Asegúrese de correr el Notebook 1 primero.\")\n",
                "    print(e)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Entrenamiento y Evaluación con Validación Cruzada\n",
                "\n",
                "Aplicaremos 4 modelos base y 3 ensambles."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "models = {\n",
                "    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),\n",
                "    'Decision Tree': DecisionTreeClassifier(random_state=42),\n",
                "    'SVC': SVC(probability=True, random_state=42),\n",
                "    'KNN': KNeighborsClassifier(),\n",
                "    'Random Forest': RandomForestClassifier(random_state=42),\n",
                "    'LightGBM': lgb.LGBMClassifier(random_state=42),\n",
                "    'AdaBoost': AdaBoostClassifier(random_state=42)\n",
                "}\n",
                "\n",
                "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
                "scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']\n",
                "\n",
                "results = {}\n",
                "f1_scores_all = {}\n",
                "\n",
                "for name, model in models.items():\n",
                "    print(f\"Evaluando {name}...\")\n",
                "    # Escalar datos localmente dentro del CV ayuda a evitar data leakage, \n",
                "    # pero dado que los guardamos codificados usaremos el pipeline si lo requiere KNN/SVC.\n",
                "    # Por simplicidad y eficiencia, aplicaremos un StandardScaler si el modelo es sensible a la escala.\n",
                "    if name in ['Logistic Regression', 'SVC', 'KNN']:\n",
                "        pipe = Pipeline([('scaler', StandardScaler()), ('classifier', model)])\n",
                "        cv_res = cross_validate(pipe, X_train_sm, y_train_sm, cv=cv, scoring=scoring, n_jobs=-1)\n",
                "    else:\n",
                "        cv_res = cross_validate(model, X_train_sm, y_train_sm, cv=cv, scoring=scoring, n_jobs=-1)\n",
                "        \n",
                "    f1_scores_all[name] = cv_res['test_f1']\n",
                "    results[name] = {\n",
                "        'Accuracy': np.mean(cv_res['test_accuracy']),\n",
                "        'Precision': np.mean(cv_res['test_precision']),\n",
                "        'Recall': np.mean(cv_res['test_recall']),\n",
                "        'F1-Score': np.mean(cv_res['test_f1']),\n",
                "        'F1_Std': np.std(cv_res['test_f1']),\n",
                "        'ROC AUC': np.mean(cv_res['test_roc_auc'])\n",
                "    }\n",
                "\n",
                "df_results = pd.DataFrame(results).T\n",
                "display(df_results.sort_values(by='F1-Score', ascending=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Gráficos Comparativos"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_results_plot = df_results.reset_index().melt(id_vars='index', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'])\n",
                "plt.figure(figsize=(14, 8))\n",
                "sns.barplot(data=df_results_plot, x='index', y='value', hue='variable')\n",
                "plt.title('Comparación de Métricas por Modelo')\n",
                "plt.xticks(rotation=45)\n",
                "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Análisis Estadístico: ANOVA y Tukey HSD\n",
                "Verificaremos si las diferencias en los F1-Scores son estadísticamente significativas."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. ANOVA\n",
                "f1_lists = [f1_scores_all[m] for m in models.keys()]\n",
                "f_stat, p_value = f_oneway(*f1_lists)\n",
                "print(f\"ANOVA F-statistic: {f_stat:.4f}, p-value: {p_value:.4e}\")\n",
                "\n",
                "if p_value < 0.05:\n",
                "    print(\"\\nExisten diferencias significativas entre al menos dos modelos. Procediendo con Tukey HSD...\")\n",
                "    # Preparar datos para Tukey\n",
                "    tukey_data = []\n",
                "    tukey_labels = []\n",
                "    for model_name, scores in f1_scores_all.items():\n",
                "        tukey_data.extend(scores)\n",
                "        tukey_labels.extend([model_name] * len(scores))\n",
                "        \n",
                "    tukey_result = pairwise_tukeyhsd(endog=tukey_data, groups=tukey_labels, alpha=0.05)\n",
                "    print(\"\\n=== RESULTADOS PRUEBA TUKEY HSD ===\")\n",
                "    print(tukey_result)\n",
                "else:\n",
                "    print(\"\\nNo se detectaron diferencias estadísticamente significativas.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Selección de los 3 Mejores Modelos\n",
                "Basado en F1-Score, estabilidad (baja desviación estándar) y significancia, procedemos a seleccionar y optimizar los hiperparámetros. Optimzaremos: **Random Forest**, **LightGBM** y **Logistic Regression** (o los que hayan resultado mejor)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Seleccionando top 3 según F1-Score medio\n",
                "top_3_models = df_results.sort_values(by='F1-Score', ascending=False).head(3).index.tolist()\n",
                "print(\"Los 3 mejores modelos seleccionados para optimización son:\", top_3_models)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Optimización de Hiperparámetros (GridSearchCV y Optuna)\n",
                "Para fines académicos, implementaremos ambos métodos."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "best_models_optimized = {}\n",
                "\n",
                "for m in top_3_models:\n",
                "    print(f\"\\nOptimizando {m}...\")\n",
                "    if m == 'Random Forest':\n",
                "        params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}\n",
                "        grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=cv, scoring='f1', n_jobs=-1)\n",
                "        grid.fit(X_train_sm, y_train_sm)\n",
                "        best_models_optimized[m] = grid.best_estimator_\n",
                "        print(f\"Mejores parámetros: {grid.best_params_}, Mejor F1: {grid.best_score_}\")\n",
                "    elif m == 'LightGBM':\n",
                "        def objective(trial):\n",
                "            params = {'n_estimators': trial.suggest_int('n_estimators', 50, 200), 'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True)}\n",
                "            model = lgb.LGBMClassifier(**params, random_state=42)\n",
                "            return cross_validate(model, X_train_sm, y_train_sm, cv=cv, scoring='f1', n_jobs=-1)['test_score'].mean()\n",
                "        optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
                "        study = optuna.create_study(direction='maximize')\n",
                "        study.optimize(objective, n_trials=10)\n",
                "        best_models_optimized[m] = lgb.LGBMClassifier(**study.best_params, random_state=42).fit(X_train_sm, y_train_sm)\n",
                "        print(f\"Mejores parámetros: {study.best_params}, Mejor F1: {study.best_value}\")\n",
                "    elif m == 'Logistic Regression':\n",
                "        pipe = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression(random_state=42))])\n",
                "        params = {'classifier__C': [0.1, 1.0, 10.0], 'classifier__solver': ['liblinear', 'lbfgs']}\n",
                "        grid = GridSearchCV(pipe, params, cv=cv, scoring='f1', n_jobs=-1)\n",
                "        grid.fit(X_train_sm, y_train_sm)\n",
                "        best_models_optimized[m] = grid.best_estimator_\n",
                "        print(f\"Mejores parámetros: {grid.best_params_}, Mejor F1: {grid.best_score_}\")\n",
                "    elif m == 'Decision Tree':\n",
                "        params = {'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10]}\n",
                "        grid = GridSearchCV(DecisionTreeClassifier(random_state=42), params, cv=cv, scoring='f1', n_jobs=-1)\n",
                "        grid.fit(X_train_sm, y_train_sm)\n",
                "        best_models_optimized[m] = grid.best_estimator_\n",
                "        print(f\"Mejores parámetros: {grid.best_params_}, Mejor F1: {grid.best_score_}\")\n",
                "    elif m == 'SVC':\n",
                "        pipe = Pipeline([('scaler', StandardScaler()), ('classifier', SVC(probability=True, random_state=42))])\n",
                "        params = {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear', 'rbf']}\n",
                "        grid = GridSearchCV(pipe, params, cv=cv, scoring='f1', n_jobs=-1)\n",
                "        grid.fit(X_train_sm, y_train_sm)\n",
                "        best_models_optimized[m] = grid.best_estimator_\n",
                "        print(f\"Mejores parámetros: {grid.best_params_}, Mejor F1: {grid.best_score_}\")\n",
                "    elif m == 'KNN':\n",
                "        pipe = Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier())])\n",
                "        params = {'classifier__n_neighbors': [3, 5, 7, 9], 'classifier__weights': ['uniform', 'distance']}\n",
                "        grid = GridSearchCV(pipe, params, cv=cv, scoring='f1', n_jobs=-1)\n",
                "        grid.fit(X_train_sm, y_train_sm)\n",
                "        best_models_optimized[m] = grid.best_estimator_\n",
                "        print(f\"Mejores parámetros: {grid.best_params_}, Mejor F1: {grid.best_score_}\")\n",
                "    elif m == 'AdaBoost':\n",
                "        params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}\n",
                "        grid = GridSearchCV(AdaBoostClassifier(random_state=42), params, cv=cv, scoring='f1', n_jobs=-1)\n",
                "        grid.fit(X_train_sm, y_train_sm)\n",
                "        best_models_optimized[m] = grid.best_estimator_\n",
                "        print(f\"Mejores parámetros: {grid.best_params_}, Mejor F1: {grid.best_score_}\")\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Selección y Evaluación del Modelo Final sobre el Test Set NO Visto\n",
                "Seleccionaremos LightGBM (o Random Forest) como el modelo principal de producción."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Asumimos que LightGBM o Random Forest ganaron. Usamos el mejor modelo del top.\n",
                "best_model_name = top_3_models[0]\n",
                "final_model = best_models_optimized[best_model_name]\n",
                "\n",
                "print(f\"\\n=== EVALUANDO EL MEJOR MODELO FINAL ({best_model_name}) EN TEST ===\")\n",
                "y_pred = final_model.predict(X_test)\n",
                "if hasattr(final_model, \"predict_proba\"):\n",
                "    y_prob = final_model.predict_proba(X_test)[:, 1]\n",
                "else:\n",
                "    y_prob = y_pred # Failsafe\n",
                "\n",
                "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
                "print(\"Precision:\", precision_score(y_test, y_pred))\n",
                "print(\"Recall:\", recall_score(y_test, y_pred))\n",
                "print(\"F1-Score:\", f1_score(y_test, y_pred))\n",
                "print(\"ROC AUC:\", roc_auc_score(y_test, y_prob))\n",
                "\n",
                "print(\"\\nReporte de Clasificación:\\n\", classification_report(y_test, y_pred))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Matriz de Confusión y Curva ROC"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# Matriz de confusión\n",
                "cm = confusion_matrix(y_test, y_pred)\n",
                "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])\n",
                "axes[0].set_title('Matriz de Confusión')\n",
                "axes[0].set_ylabel('Real')\n",
                "axes[0].set_xlabel('Predicho')\n",
                "\n",
                "# Curva ROC\n",
                "fpr, tpr, _ = roc_curve(y_test, y_prob)\n",
                "axes[1].plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (area = {roc_auc_score(y_test, y_prob):.2f})')\n",
                "axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
                "axes[1].set_xlabel('False Positive Rate')\n",
                "axes[1].set_ylabel('True Positive Rate')\n",
                "axes[1].set_title('ROC Curve')\n",
                "axes[1].legend(loc=\"lower right\")\n",
                "\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Interpretabilidad (Feature Importance)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if hasattr(final_model, 'feature_importances_'):\n",
                "    importances = final_model.feature_importances_\n",
                "    indices = np.argsort(importances)[::-1]\n",
                "    features = X_train_sm.columns\n",
                "\n",
                "    plt.figure(figsize=(10, 6))\n",
                "    sns.barplot(x=importances[indices][:10], y=np.array(features)[indices][:10], palette='viridis')\n",
                "    plt.title(f'Top 10 Feature Importances ({best_model_name})')\n",
                "    plt.show()\n",
                "elif hasattr(final_model, 'coef_'):\n",
                "    # Para regresión logística / SVC lineal\n",
                "    importances = final_model.coef_[0]\n",
                "    indices = np.argsort(np.abs(importances))[::-1]\n",
                "    features = X_train_sm.columns\n",
                "\n",
                "    plt.figure(figsize=(10, 6))\n",
                "    sns.barplot(x=importances[indices][:10], y=np.array(features)[indices][:10], palette='vlag')\n",
                "    plt.title(f'Top 10 Coefficients ({best_model_name})')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Construcción y Exportación del Pipeline Final\n",
                "\n",
                "Para el despliegue en Streamlit, crearemos un pipeline completo que reciba los datos crudos, aplique `ColumnTransformer` (OneHotEncoding y Scaling si aplica) y prediga."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Volvemos a cargar el dataframe original para ver las features originales necesarias\n",
                "df_orig = pd.read_csv('student_placement_salary_elite_v2.csv')\n",
                "df_orig = df_orig.drop(columns=['student_id'])\n",
                "\n",
                "# Manejo de nulos en crudo para el entrenamiento del transformer\n",
                "df_orig['company_type'] = df_orig['company_type'].replace(['None', np.nan], 'Not_Applicable')\n",
                "df_orig['job_role'] = df_orig['job_role'].replace(['None', np.nan], 'Not_Applicable')\n",
                "\n",
                "X_raw = df_orig.drop(columns=['placed', 'salary_lpa', 'company_type', 'job_role'])\n",
                "y_raw = df_orig['placed']\n",
                "\n",
                "# Identificar categóricas y numéricas\n",
                "cat_features = X_raw.select_dtypes(include=['object']).columns.tolist()\n",
                "num_features = X_raw.select_dtypes(exclude=['object']).columns.tolist()\n",
                "\n",
                "# Preprocessor\n",
                "preprocessor = ColumnTransformer(\n",
                "    transformers=[\n",
                "        ('num', StandardScaler(), num_features),\n",
                "        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_features)\n",
                "    ])\n",
                "\n",
                "# Usamos el mejor modelo. Nota: para LightGBM lo instanciaremos con sus mejores parametros directamente.\n",
                "if best_model_name == 'LightGBM':\n",
                "    # Para evitar warnings de LGBM con feature names que tengan caracteres especiales JSON:\n",
                "    import re\n",
                "    X_raw = X_raw.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))\n",
                "    \n",
                "    final_classifier = final_model # Ya entrenado, pero lo reentrenaremos en el pipeline\n",
                "else:\n",
                "    final_classifier = final_model\n",
                "\n",
                "full_pipeline = Pipeline(steps=[\n",
                "    ('preprocessor', preprocessor),\n",
                "    ('classifier', final_classifier)\n",
                "])\n",
                "\n",
                "# Entrenamos el pipeline completo con los datos crudos completos para producción (o solo con el de entrenamiento)\n",
                "# Aquí lo entrenamos con X_raw y y_raw para aprovechar todo, o podríamos usar solo train.\n",
                "from sklearn.model_selection import train_test_split\n",
                "X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw, y_raw, test_size=0.3, stratify=y_raw, random_state=42)\n",
                "\n",
                "# Como usamos SMOTE antes, en un pipeline puro de sklearn estándar integrar SMOTE es complejo con imblearn.Pipeline.\n",
                "# Para producción usaremos un imblearn pipeline\n",
                "from imblearn.pipeline import Pipeline as ImbPipeline\n",
                "from imblearn.over_sampling import SMOTE\n",
                "\n",
                "production_pipeline = ImbPipeline(steps=[\n",
                "    ('preprocessor', preprocessor),\n",
                "    ('smote', SMOTE(sampling_strategy='auto', random_state=42)),\n",
                "    ('classifier', best_models_optimized[best_model_name])\n",
                "])\n",
                "\n",
                "production_pipeline.fit(X_train_raw, y_train_raw)\n",
                "\n",
                "# Guardamos el pipeline\n",
                "joblib.dump(production_pipeline, 'pipeline.pkl')\n",
                "print(\"\\nPipeline guardado exitosamente como 'pipeline.pkl'. Listo para producción en Streamlit.\")\n",
                "\n",
                "# BONUS: Pipeline de regresión para estimar SALARIO si fue empleado\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "df_emp = df_orig[df_orig['placed'] == 1]\n",
                "X_reg = df_emp.drop(columns=['placed', 'salary_lpa'])\n",
                "y_reg = df_emp['salary_lpa']\n",
                "\n",
                "reg_pipeline = Pipeline(steps=[\n",
                "    ('preprocessor', preprocessor),\n",
                "    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))\n",
                "])\n",
                "reg_pipeline.fit(X_reg, y_reg)\n",
                "joblib.dump(reg_pipeline, 'pipeline_salary.pkl')\n",
                "print(\"Pipeline de regresión salarial guardado como 'pipeline_salary.pkl'.\")"
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
    
    with open('02_predictive_modeling.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    create_notebook()
