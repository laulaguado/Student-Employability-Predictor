import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import time

# Configuración de página
st.set_page_config(
    page_title="EduPredict: Placement Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS moderno y profesional (Dark/Sleek)
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #1e2530;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #00ff9d;
    }
    .metric-label {
        font-size: 16px;
        color: #a0aec0;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    .recommendation-tag {
        display: inline-block;
        background: #2d3748;
        color: #63b3ed;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        font-weight: bold;
        border: 1px solid #4a5568;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado
st.markdown("<h1 style='text-align: center;'>🎓 Predicción Avanzada de Empleabilidad Estudiantil</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0aec0; font-size: 1.2rem; margin-bottom: 2rem;'>Utilizando Inteligencia Artificial para estimar tu probabilidad de colocación laboral y rango salarial esperado.</p>", unsafe_allow_html=True)

# Intentar cargar pipelines (Mock si no existen para que la app no rompa antes de correr notebooks)
@st.cache_resource
def load_models():
    try:
        clf_pipe = joblib.load('pipeline.pkl')
        reg_pipe = joblib.load('pipeline_salary.pkl')
        return clf_pipe, reg_pipe
    except:
        return None, None

clf_pipe, reg_pipe = load_models()

# Sidebar: Ingreso de Datos
with st.sidebar:
    st.header("📋 Perfil del Estudiante")
    st.markdown("Ingresa tus datos académicos y técnicos:")
    
    col1, col2 = st.columns(2)
    with col1:
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        college_tier = st.selectbox("Nivel Universitario (Tier)", [1, 2, 3])
        projects = st.number_input("Proyectos", min_value=0, max_value=20, value=3)
    with col2:
        branch = st.selectbox("Rama Académica", ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"])
        internships = st.number_input("Pasantías", min_value=0, max_value=10, value=1)
        backlogs = st.number_input("Backlogs", min_value=0, max_value=10, value=0)
    
    st.markdown("---")
    st.header("💻 Habilidades Técnicas")
    c1, c2, c3, c4 = st.columns(4)
    python_skill = 1 if c1.checkbox("Python") else 0
    dsa_skill = 1 if c2.checkbox("DSA") else 0
    ml_skill = 1 if c3.checkbox("ML") else 0
    web_dev_skill = 1 if c4.checkbox("Web Dev") else 0
    
    st.markdown("---")
    st.header("📊 Puntajes de Evaluación")
    coding_score = st.slider("Score de Programación", 0.0, 100.0, 65.0)
    communication_score = st.slider("Score de Comunicación", 0.0, 10.0, 7.5)
    aptitude_score = st.slider("Score de Aptitud", 0.0, 100.0, 70.0)
    resume_score = st.slider("Score del CV", 0.0, 100.0, 60.0)
    skill_score = st.slider("Score Global de Habilidades", 0, 5, 2)
    
    # Las variables company_type y job_role han sido eliminadas para evitar Target Leakage
    
    predict_button = st.button("🚀 Calcular Predicción")

def generate_recommendations(branch, python, dsa, ml, web):
    roles = []
    if ml == 1 and python == 1:
        roles.extend(["Data Scientist", "Machine Learning Engineer"])
    if web == 1:
        roles.extend(["Web Developer", "Full Stack Developer"])
    if python == 1 and dsa == 1:
        roles.extend(["Software Engineer", "Backend Developer"])
    if len(roles) == 0:
        if branch in ["CSE", "IT"]:
            roles.append("Software Analyst")
        else:
            roles.append("Core Engineer / Analyst")
    return list(set(roles))

if predict_button:
    if clf_pipe is None:
        st.error("⚠️ Los modelos no han sido entrenados. Por favor, ejecuta primero los notebooks de Jupyter para generar 'pipeline.pkl' y 'pipeline_salary.pkl'.")
    else:
        with st.spinner("Analizando perfil con Machine Learning..."):
            time.sleep(1) # Simular procesamiento para UX
            
            # Preparar datos de entrada
            input_data = pd.DataFrame([{
                'cgpa': cgpa, 'branch': branch, 'college_tier': college_tier,
                'python_skill': python_skill, 'dsa_skill': dsa_skill, 'ml_skill': ml_skill,
                'web_dev_skill': web_dev_skill, 'coding_score': coding_score,
                'communication_score': communication_score, 'aptitude_score': aptitude_score,
                'internships': internships, 'projects': projects, 'backlogs': backlogs,
                'resume_score': resume_score, 'skill_score': skill_score
            }])
            
            # Inferencia
            try:
                # Probabilidad
                if hasattr(clf_pipe, "predict_proba"):
                    prob = clf_pipe.predict_proba(input_data)[0][1]
                else:
                    pred = clf_pipe.predict(input_data)[0]
                    prob = 0.9 if pred == 1 else 0.1
                    
                is_placed = prob > 0.5
                
                # Salario
                if is_placed and reg_pipe is not None:
                    salario_pred = reg_pipe.predict(input_data)[0]
                    # Rango salarial (± 15%)
                    min_sal = salario_pred * 0.85
                    max_sal = salario_pred * 1.15
                else:
                    salario_pred = 0.0
                    min_sal, max_sal = 0.0, 0.0
                    
            except Exception as e:
                st.error(f"Error en la inferencia: {e}")
                st.stop()
        
        # Resultados UI
        st.markdown("---")
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            st.markdown("<h2 style='text-align: center;'>📊 Probabilidad de Empleo</h2>", unsafe_allow_html=True)
            
            # Gauge Chart con Plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                number = {'suffix': "%"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00ff9d" if is_placed else "#ff4b4b"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(255, 75, 75, 0.3)"},
                        {'range': [40, 70], 'color': "rgba(255, 165, 0, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(0, 255, 157, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100
                    }
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
        with col_res2:
            st.markdown("<h2 style='text-align: center;'>💼 Estimación Salarial</h2>", unsafe_allow_html=True)
            if is_placed:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>Rango Esperado (LPA)</div>
                    <div class='metric-value'>{min_sal:.1f} - {max_sal:.1f}</div>
                    <div style='color: #a0aec0; margin-top: 10px;'>Promedio estimado: {salario_pred:.1f} LPA</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Bar chart comparativo
                st.markdown("<h4 style='color: #a0aec0; text-align: center;'>Comparativa vs Promedio del Mercado</h4>", unsafe_allow_html=True)
                sal_data = pd.DataFrame({
                    "Categoría": ["Mínimo Esperado", "Tu Estimación", "Máximo Esperado"],
                    "LPA": [min_sal, salario_pred, max_sal]
                })
                fig_bar, ax = plt.subplots(figsize=(6, 2.5))
                fig_bar.patch.set_facecolor('#0e1117')
                ax.set_facecolor('#0e1117')
                bars = ax.barh(sal_data["Categoría"], sal_data["LPA"], color=['#ff9f43', '#00ff9d', '#00cfeb'])
                ax.tick_params(colors='white')
                ax.spines['bottom'].set_color('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('white')
                plt.tight_layout()
                st.pyplot(fig_bar)
            else:
                st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value' style='color: #ff4b4b;'>Baja Probabilidad</div>
                    <div class='metric-label'>Mejora tus habilidades técnicas e incrementa tus pasantías para acceder a ofertas salariales.</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<h2>🎯 Roles Recomendados para tu Perfil</h2>", unsafe_allow_html=True)
        recommended_roles = generate_recommendations(branch, python_skill, dsa_skill, ml_skill, web_dev_skill)
        
        tags_html = "".join([f"<span class='recommendation-tag'>✓ {role}</span>" for role in recommended_roles])
        st.markdown(f"<div style='text-align: center;'>{tags_html}</div>", unsafe_allow_html=True)
        
        # Interpretación del modelo
        with st.expander("🔍 ¿Por qué obtuve este resultado? (Interpretabilidad)"):
            st.write("El modelo evalúa múltiples factores. En tu caso, las variables que más impactan positivamente o que podrías mejorar son:")
            
            # Simple heuristic explanation (since we can't extract SHAP natively from standard scikit pipeline easily here without extra dependencies)
            strong_points = []
            weak_points = []
            
            if cgpa > 8.0: strong_points.append("Alto rendimiento académico (CGPA)")
            else: weak_points.append("CGPA puede mejorar")
            
            if coding_score > 70: strong_points.append("Excelente puntaje en programación")
            else: weak_points.append("Puntaje de programación bajo")
                
            if internships > 1: strong_points.append("Experiencia práctica (Pasantías)")
            elif internships == 0: weak_points.append("Falta de experiencia en pasantías")
                
            if backlogs > 0: weak_points.append("Presencia de backlogs afecta negativamente")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.success("**Puntos Fuertes:**\n" + ("\n".join([f"- {p}" for p in strong_points]) if strong_points else "- Ninguno destacable"))
            with col_b:
                st.warning("**Áreas de Mejora:**\n" + ("\n".join([f"- {p}" for p in weak_points]) if weak_points else "- Perfil muy sólido"))
else:
    # Estado inicial
    st.info("👈 Ajusta los parámetros en la barra lateral y presiona 'Calcular Predicción'.")
    
    # Agregando un gráfico dummy visualmente atractivo para el estado vacío
    st.markdown("### 📈 Tendencias del Mercado (Visualización de Ejemplo)")
    dummy_data = pd.DataFrame({
        'Habilidad': ['Python', 'DSA', 'Machine Learning', 'Web Dev'],
        'Demanda (%)': [85, 90, 75, 80]
    })
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    sns.barplot(data=dummy_data, x='Habilidad', y='Demanda (%)', palette='mako', ax=ax)
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('white')
    st.pyplot(fig)
