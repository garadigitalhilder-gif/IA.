import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json
import time

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y DISEÑO
# ==============================================================================
st.set_page_config(
    page_title="Regresión Lineal y Clasificación - Plataforma Educativa IA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para estética oscura premium (Glassmorphism)
st.markdown("""
<style>
    /* Estilo del contenedor principal */
    .reportview-container {
        background: #090d16;
    }
    
    /* Títulos y fuentes */
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Banner de cabecera */
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .header-title {
        color: #ffffff;
        font-weight: 800;
        margin: 0;
        font-size: 2.2rem;
    }
    .header-subtitle {
        color: #94a3b8;
        font-weight: 500;
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    .author-badge {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.3);
        display: inline-block;
        margin-top: 0.5rem;
    }
    
    /* Tarjetas de métricas */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: bold;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .metric-desc {
        font-size: 0.75rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# Render de la cabecera
st.markdown("""
<div class="header-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="header-title"> Regresión vs Clasificación</h1>
            <p class="header-subtitle">Plataforma Interactiva para la Enseñanza de Clasificadores Supervisados</p>
        </div>
        <div style="text-align: right;">
            <span class="author-badge">Desarrollado por: Marleinis Orozco y Yuleisi Carranza</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# ESTADO GLOBAL (SESSION STATE)
# ==============================================================================
if 'points' not in st.session_state:
    # Puntos por defecto en 2D (Separable)
    st.session_state.points = pd.DataFrame([
        {"X": 0.15, "Y": 0.20, "Clase": 0},
        {"X": 0.20, "Y": 0.12, "Clase": 0},
        {"X": 0.25, "Y": 0.30, "Clase": 0},
        {"X": 0.35, "Y": 0.22, "Clase": 0},
        {"X": 0.30, "Y": 0.40, "Clase": 0},
        {"X": 0.65, "Y": 0.70, "Clase": 1},
        {"X": 0.70, "Y": 0.85, "Clase": 1},
        {"X": 0.80, "Y": 0.65, "Clase": 1},
        {"X": 0.85, "Y": 0.80, "Clase": 1},
        {"X": 0.90, "Y": 0.72, "Clase": 1}
    ])

if 'model_params_2d' not in st.session_state:
    st.session_state.model_params_2d = {"m": 0.0, "b": 0.5}

if 'm2_raw_data' not in st.session_state:
    st.session_state.m2_raw_data = None
if 'm2_train_data' not in st.session_state:
    st.session_state.m2_train_data = None
if 'm2_test_data' not in st.session_state:
    st.session_state.m2_test_data = None
if 'm2_features' not in st.session_state:
    st.session_state.m2_features = []
if 'm2_target' not in st.session_state:
    st.session_state.m2_target = ""
if 'm2_weights' not in st.session_state:
    st.session_state.m2_weights = []
if 'm2_bias' not in st.session_state:
    st.session_state.m2_bias = 0.0
if 'm2_feature_stats' not in st.session_state:
    st.session_state.m2_feature_stats = {}
if 'm2_trained' not in st.session_state:
    st.session_state.m2_trained = False
if 'm2_cost_history' not in st.session_state:
    st.session_state.m2_cost_history = []

# Datasets precargados de ejemplo en formato CSV
DATASET_DIABETES_CSV = """Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigree,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0.0,0.232,54,1
4,110,92,0,0,37.6,0.191,30,0
10,168,74,0,0,38.0,0.537,34,1
1,189,60,23,846,30.1,0.398,59,1
5,166,72,19,175,25.8,0.587,51,1
7,100,0,0,0,30.0,0.484,32,1
1,103,30,38,83,43.3,0.183,33,0
3,126,88,41,235,39.3,0.704,27,0
8,99,84,0,0,35.4,0.388,50,0
7,196,90,0,0,39.8,0.451,41,1
3,162,52,38,0,37.2,0.652,24,1
1,100,66,29,196,32.0,0.444,42,0
13,129,82,0,0,39.9,0.569,42,1
4,129,86,20,270,35.1,0.231,23,0
3,79,80,25,37,25.4,0.583,22,0
4,110,111,0,0,37.6,0.191,30,0
5,117,92,0,0,34.1,0.337,38,0
5,109,75,26,0,36.0,0.546,60,0
10,101,86,37,0,45.6,1.136,38,1
10,122,78,31,0,27.6,0.512,45,0
10,125,70,26,115,31.1,0.205,41,1"""

DATASET_ADMISSION_CSV = """Examen1,Examen2,NotaPreparatoria,Admitido
34.6,78.0,3.2,0
30.2,43.8,2.1,0
35.8,72.9,3.5,0
60.1,86.3,4.2,1
79.0,75.3,4.0,1
90.2,96.2,4.8,1
61.1,96.5,4.5,1
75.0,46.5,3.0,0
76.0,87.4,4.1,1
84.4,43.5,3.3,0
95.8,38.2,3.1,0
75.0,30.6,2.5,0
82.3,79.0,4.2,1
93.1,91.5,4.9,1
55.3,64.2,3.5,0
60.2,70.1,3.8,1
45.5,50.2,2.8,0
89.2,90.5,4.7,1
72.0,81.1,4.1,1
68.5,40.2,3.0,0
52.1,75.0,3.9,0
99.1,92.5,5.0,1
90.0,42.5,3.5,0
78.3,82.5,4.3,1
61.2,72.2,3.8,1"""

# ==============================================================================
# SELECCIÓN DE MÓDULO (SIDEBAR)
# ==============================================================================
st.sidebar.markdown("###  Menú de Navegación")
module = st.sidebar.radio(
    "Selecciona un Módulo:",
    [
        " Módulo 1: Simulador 2D (Regresión Lineal)",
        " Módulo 2: Clasificación Multivariable (Regresión Logística)",
        " Módulo 3: Metodología y Fundamentos Teóricos"
    ]
)

# ==============================================================================
# MÓDULO 1: SIMULADOR INTERACTIVO 2D
# ==============================================================================
if module == " Módulo 1: Simulador 2D (Regresión Lineal)":
    st.markdown("##  Módulo 1: Simulador Interactivo 2D (Regresión Lineal)")
    st.markdown("""
    Este simulador permite analizar de forma visual e intuitiva la hipótesis de **Regresión Lineal como Clasificador** ($h(x) = mx + b$). 
    Podrás observar la recta continua ajustada, las desviaciones cuadráticas (residuos) y cómo un **umbral de decisión** se usa para segregar el plano.
    """)

    col_side, col_plot = st.columns([1, 2])

    with col_side:
        st.markdown("###  Control de Datos y Modelo")
        
        # Selección de Dataset predefinido o carga
        dataset_choice = st.selectbox(
            "Cargar plantilla de puntos:",
            ["Seleccionar...", "Separable", "Traslapado", "Con Outliers", "Limpiar todo"]
        )
        
        if dataset_choice == "Separable":
            st.session_state.points = pd.DataFrame([
                {"X": 0.15, "Y": 0.20, "Clase": 0},
                {"X": 0.20, "Y": 0.12, "Clase": 0},
                {"X": 0.25, "Y": 0.30, "Clase": 0},
                {"X": 0.35, "Y": 0.22, "Clase": 0},
                {"X": 0.30, "Y": 0.40, "Clase": 0},
                {"X": 0.65, "Y": 0.70, "Clase": 1},
                {"X": 0.70, "Y": 0.85, "Clase": 1},
                {"X": 0.80, "Y": 0.65, "Clase": 1},
                {"X": 0.85, "Y": 0.80, "Clase": 1},
                {"X": 0.90, "Y": 0.72, "Clase": 1}
            ])
            st.session_state.model_params_2d = {"m": 0.0, "b": 0.5}
        elif dataset_choice == "Traslapado":
            st.session_state.points = pd.DataFrame([
                {"X": 0.20, "Y": 0.30, "Clase": 0},
                {"X": 0.30, "Y": 0.25, "Clase": 0},
                {"X": 0.40, "Y": 0.50, "Clase": 0},
                {"X": 0.45, "Y": 0.35, "Clase": 0},
                {"X": 0.50, "Y": 0.20, "Clase": 0},
                {"X": 0.60, "Y": 0.45, "Clase": 0},
                {"X": 0.42, "Y": 0.62, "Clase": 1},
                {"X": 0.50, "Y": 0.75, "Clase": 1},
                {"X": 0.55, "Y": 0.52, "Clase": 1},
                {"X": 0.60, "Y": 0.80, "Clase": 1},
                {"X": 0.70, "Y": 0.60, "Clase": 1},
                {"X": 0.80, "Y": 0.70, "Clase": 1}
            ])
            st.session_state.model_params_2d = {"m": 0.0, "b": 0.5}
        elif dataset_choice == "Con Outliers":
            st.session_state.points = pd.DataFrame([
                {"X": 0.10, "Y": 0.20, "Clase": 0},
                {"X": 0.15, "Y": 0.30, "Clase": 0},
                {"X": 0.20, "Y": 0.22, "Clase": 0},
                {"X": 0.25, "Y": 0.15, "Clase": 0},
                {"X": 0.28, "Y": 0.35, "Clase": 0},
                {"X": 0.45, "Y": 0.70, "Clase": 1},
                {"X": 0.50, "Y": 0.78, "Clase": 1},
                {"X": 0.55, "Y": 0.65, "Clase": 1},
                {"X": 0.60, "Y": 0.80, "Clase": 1},
                # Outliers de clase 1 en Y muy alto pero X también extremo derecho
                {"X": 0.90, "Y": 0.85, "Clase": 1},
                {"X": 0.95, "Y": 0.90, "Clase": 1}
            ])
            st.session_state.model_params_2d = {"m": 0.0, "b": 0.5}
        elif dataset_choice == "Limpiar todo":
            st.session_state.points = pd.DataFrame(columns=["X", "Y", "Clase"])
            st.session_state.model_params_2d = {"m": 0.0, "b": 0.5}

        # Cargar puntos desde CSV
        csv_sim = st.file_uploader("O sube un CSV para el plano 2D (Columnas: X, Y, Clase)", type=["csv"])
        if csv_sim:
            try:
                df_loaded = pd.read_csv(csv_sim)
                if all(c in df_loaded.columns for c in ["X", "Y", "Clase"]):
                    st.session_state.points = df_loaded[["X", "Y", "Clase"]].dropna()
                    st.success("CSV cargado con éxito en el simulador.")
                else:
                    st.error("El CSV debe contener exactamente las columnas 'X', 'Y' y 'Clase'.")
            except Exception as e:
                st.error(f"Error al leer CSV: {e}")

        # st.data_editor interactivo
        st.markdown("** Editar Puntos Directamente (X e Y entre 0 y 1, Clase 0 o 1):**")
        edited_df = st.data_editor(
            st.session_state.points,
            num_rows="dynamic",
            width="stretch",
            column_config={
                "X": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
                "Y": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.01),
                "Clase": st.column_config.SelectboxColumn(options=[0, 1])
            }
        )
        st.session_state.points = edited_df

        # Parámetros del modelo y entrenamiento
        st.markdown("---")
        st.markdown("###  Entrenamiento")
        method = st.radio("Método de Ajuste:", ["Mínimos Cuadrados (OLS)", "Descenso de Gradiente (GD)"])
        
        lr_2d = st.slider("Tasa de Aprendizaje (α):", 0.001, 0.5, 0.05, 0.001, disabled=(method == "Mínimos Cuadrados (OLS)"))
        epochs_2d = st.slider("Iteraciones (GD):", 10, 1000, 100, 10, disabled=(method == "Mínimos Cuadrados (OLS)"))
        
        train_btn = st.button(" Ejecutar Entrenamiento")
        
        st.markdown("###  Clasificación")
        threshold_2d = st.slider("Umbral de Decisión de Clase:", 0.0, 1.0, 0.5, 0.01)

        # Opciones visuales
        st.markdown("###  Opciones del Gráfico")
        show_residuals = st.checkbox("Mostrar Residuos (Líneas de error)", value=True)
        show_regions = st.checkbox("Mostrar Regiones de Clasificación", value=True)

    # Lógica de Entrenamiento 2D
    pts = st.session_state.points
    m_val, b_val = st.session_state.model_params_2d["m"], st.session_state.model_params_2d["b"]

    if train_btn:
        if len(pts) < 2:
            st.warning("Se requieren al menos 2 puntos para entrenar el modelo.")
        else:
            X_2d = pts["X"].values
            Y_2d = pts["Y"].values
            
            if method == "Mínimos Cuadrados (OLS)":
                mean_x = np.mean(X_2d)
                mean_y = np.mean(Y_2d)
                num = np.sum((X_2d - mean_x) * (Y_2d - mean_y))
                den = np.sum((X_2d - mean_x) ** 2)
                
                if den == 0:
                    m_val = 0.0
                    b_val = mean_y
                else:
                    m_val = num / den
                    b_val = mean_y - m_val * mean_x
                
                st.session_state.model_params_2d = {"m": m_val, "b": b_val}
                st.success(f"OLS completado. Parámetros: m = {m_val:.4f}, b = {b_val:.4f}")
                
            elif method == "Descenso de Gradiente (GD)":
                # Animación del Descenso de Gradiente
                m_val = 0.0
                b_val = 0.5
                N = len(pts)
                
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                for epoch in range(epochs_2d):
                    # Predicciones lineales
                    y_pred = m_val * X_2d + b_val
                    error = y_pred - Y_2d
                    
                    # Cálculo de gradientes
                    dm = (2.0 / N) * np.sum(error * X_2d)
                    db = (2.0 / N) * np.sum(error)
                    
                    # Actualización
                    m_val -= lr_2d * dm
                    b_val -= lr_2d * db
                    
                    # Guardar parámetros
                    st.session_state.model_params_2d = {"m": m_val, "b": b_val}
                    
                    # Actualizar UI cada cierto número de épocas
                    if epoch % max(1, epochs_2d // 10) == 0:
                        progress_bar.progress(int((epoch + 1) / epochs_2d * 100))
                        status_text.text(f"Época {epoch + 1}/{epochs_2d} - Ajustando recta...")
                        time.sleep(0.01)
                
                progress_bar.progress(100)
                status_text.text("¡Descenso de Gradiente completado!")
                st.success(f"GD finalizado: m = {m_val:.4f}, b = {b_val:.4f}")

    # Cálculos de Métricas
    mse_2d = 0.0
    r2_2d = 0.0
    acc_2d = 0.0
    prec_2d = 0.0
    rec_2d = 0.0
    f1_2d = 0.0
    tn_2d, fp_2d, fn_2d, tp_2d = 0, 0, 0, 0

    if len(pts) > 0:
        X_2d = pts["X"].values
        Y_2d = pts["Y"].values
        Clase_2d = pts["Clase"].values.astype(int)
        
        preds_cont = m_val * X_2d + b_val
        # Error cuadrático
        mse_2d = np.mean((Y_2d - preds_cont)**2)
        # R2
        mean_y = np.mean(Y_2d)
        ss_res = np.sum((Y_2d - preds_cont)**2)
        ss_tot = np.sum((Y_2d - mean_y)**2)
        r2_2d = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Predicción binaria según umbral en Y continuo
        preds_bin = (preds_cont >= threshold_2d).astype(int)
        
        acc_2d = accuracy_score(Clase_2d, preds_bin)
        prec_2d = precision_score(Clase_2d, preds_bin, zero_division=0)
        rec_2d = recall_score(Clase_2d, preds_bin, zero_division=0)
        f1_2d = f1_score(Clase_2d, preds_bin, zero_division=0)
        
        cm_2d = confusion_matrix(Clase_2d, preds_bin, labels=[0, 1])
        tn_2d, fp_2d, fn_2d, tp_2d = cm_2d.ravel()

    with col_plot:
        st.markdown("###  Plano Cartesiano Interactivo 2D")
        
        fig = go.Figure()

        # 1. Regiones de clasificación de fondo
        if show_regions and len(pts) > 0 and abs(m_val) > 0.0001:
            boundary_x = (threshold_2d - b_val) / m_val
            boundary_x_clamped = max(0.0, min(1.0, boundary_x))
            
            # Determinar el color de cada mitad
            # Si m > 0, para x menor a la frontera, y_pred < threshold (Clase 0)
            color_left = "rgba(244, 63, 94, 0.05)" if m_val > 0 else "rgba(16, 185, 129, 0.05)"
            color_right = "rgba(16, 185, 129, 0.05)" if m_val > 0 else "rgba(244, 63, 94, 0.05)"
            
            if boundary_x_clamped > 0:
                fig.add_vrect(x0=0, x1=boundary_x_clamped, fillcolor=color_left, line_width=0, layer="below")
            if boundary_x_clamped < 1:
                fig.add_vrect(x0=boundary_x_clamped, x1=1, fillcolor=color_right, line_width=0, layer="below")
        elif show_regions and len(pts) > 0:
            # Caso m aproximado a 0
            color_all = "rgba(16, 185, 129, 0.05)" if b_val >= threshold_2d else "rgba(244, 63, 94, 0.05)"
            fig.add_vrect(x0=0, x1=1, fillcolor=color_all, line_width=0, layer="below")

        # 2. Residuales (Líneas verticales de error)
        if show_residuals and len(pts) > 0:
            for i, row in pts.iterrows():
                pred_y = m_val * row["X"] + b_val
                fig.add_trace(go.Scatter(
                    x=[row["X"], row["X"]],
                    y=[row["Y"], pred_y],
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip"
                ))

        # 3. Puntos de datos
        if len(pts) > 0:
            c0_pts = pts[pts["Clase"] == 0]
            c1_pts = pts[pts["Clase"] == 1]
            
            fig.add_trace(go.Scatter(
                x=c0_pts["X"], y=c0_pts["Y"],
                mode="markers",
                marker=dict(size=12, color="#f43f5e", line=dict(color="#be123c", width=2), symbol="circle"),
                name="Clase 0 (Rosado)"
            ))
            
            fig.add_trace(go.Scatter(
                x=c1_pts["X"], y=c1_pts["Y"],
                mode="markers",
                marker=dict(size=12, color="#10b981", line=dict(color="#047857", width=2), symbol="circle"),
                name="Clase 1 (Verde)"
            ))

        # 4. Recta continua de regresión
        x_range = np.linspace(0, 1, 100)
        y_range = m_val * x_range + b_val
        fig.add_trace(go.Scatter(
            x=x_range, y=y_range,
            mode="lines",
            line=dict(color="#6366f1", width=3.5),
            name=f"Regresión Lineal: y = {m_val:.3f}x + {b_val:.3f}"
        ))

        # 5. Frontera de decisión (Línea discontinua vertical)
        if abs(m_val) > 0.0001:
            bx = (threshold_2d - b_val) / m_val
            if 0 <= bx <= 1:
                fig.add_vline(
                    x=bx,
                    line=dict(color="#ffffff", width=2, dash="dash"),
                    annotation_text="Frontera de Decisión",
                    annotation_position="bottom right"
                )

        fig.update_layout(
            template="plotly_dark",
            xaxis=dict(title="Variable de Entrada (X)", range=[-0.05, 1.05], showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Salida Continua / Clase (Y)", range=[-0.05, 1.05], showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=40, r=40, t=20, b=40),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Dashboard de métricas del Simulador
        st.markdown("###  Métricas de Evaluación del Simulador")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Coste Medio (MSE)</div>
                <div class="metric-value">{mse_2d:.4f}</div>
                <div class="metric-desc">Error cuadrático de la recta</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Coeficiente R²</div>
                <div class="metric-value">{r2_2d:.3f}</div>
                <div class="metric-desc">Bondad de ajuste continuo</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Exactitud (Accuracy)</div>
                <div class="metric-value" style="color: #6366f1;">{acc_2d * 100:.1f}%</div>
                <div class="metric-desc">Predicciones correctas totales</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">F1-Score</div>
                <div class="metric-value" style="color: #a855f7;">{f1_2d * 100:.1f}%</div>
                <div class="metric-desc">Media armónica (P y R)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#####  Detalle de Clasificación:")
        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.write(f"**Precisión (Precision):** {prec_2d*100:.1f}%")
            st.write(f"**Sensibilidad (Recall):** {rec_2d*100:.1f}%")
        with d_col2:
            st.write(f"**Verdaderos Negativos (TN):** {tn_2d}")
            st.write(f"**Falsos Positivos (FP):** {fp_2d}")
        with d_col3:
            st.write(f"**Falsos Negativos (FN):** {fn_2d}")
            st.write(f"**Verdaderos Positivos (TP):** {tp_2d}")

        st.warning(" **Impacto de Outliers:** Carga la plantilla **'Con Outliers'** y entrena el modelo. Observa cómo los dos puntos lejanos a la derecha fuerzan a la línea a subir, moviendo la frontera de decisión hacia la izquierda, lo que causa clasificaciones incorrectas en la zona central. Esto demuestra que minimizar errores cuadráticos continuos no es el enfoque ideal para clasificación binaria.")

# ==============================================================================
# MÓDULO 2: CLASIFICACIÓN MULTIVARIABLE (REGRESIÓN LOGÍSTICA)
# ==============================================================================
elif module == " Módulo 2: Clasificación Multivariable (Regresión Logística)":
    st.markdown("##  Módulo 2: Clasificación Multivariable (Regresión Logística)")
    st.markdown("""
    En este módulo entrenaremos un clasificador de **Regresión Logística** utilizando **Descenso de Gradiente Multivariable** con **Función de Pérdida Log-Loss (Entropía Cruzada)**.
    Todo el preprocesamiento, normalización Z-Score, optimización iterativa y evaluación de rendimiento con curva ROC se calcula paso a paso.
    """)

    tabs = st.tabs([
        "A, B. Carga y Selección de Datos", 
        "C, D. Entrenamiento y Coeficientes", 
        "E. Validación en Conjunto de Test", 
        "F. Predicción de Nuevos Registros"
    ])

    # --------------------------------------------------------------------------
    # Tab 1: Carga y Procesamiento de Datos
    # --------------------------------------------------------------------------
    with tabs[0]:
        st.markdown("### A y B. Carga de Datos y División Train/Test")
        st.markdown("Sube tu archivo CSV o selecciona uno de nuestros datasets de prueba prediseñados:")
        
        sample_choice = st.selectbox(
            "Seleccionar dataset precargado:",
            ["...", "Ejemplo 1: Diabetes (Salud)", "Ejemplo 2: Admisiones Universitarias"]
        )

        csv_uploaded = st.file_uploader("O sube un archivo CSV personalizado:", type=["csv"])

        df_raw = None
        if csv_uploaded:
            try:
                df_raw = pd.read_csv(csv_uploaded)
                st.session_state.m2_raw_data = df_raw
                st.success(f"Archivo '{csv_uploaded.name}' cargado con éxito.")
            except Exception as e:
                st.error(f"Error al leer el CSV: {e}")
        elif sample_choice == "Ejemplo 1: Diabetes (Salud)":
            from io import StringIO
            df_raw = pd.read_csv(StringIO(DATASET_DIABETES_CSV))
            st.session_state.m2_raw_data = df_raw
            st.success("Cargado dataset de Diabetes (Pima Indians).")
        elif sample_choice == "Ejemplo 2: Admisiones Universitarias":
            from io import StringIO
            df_raw = pd.read_csv(StringIO(DATASET_ADMISSION_CSV))
            st.session_state.m2_raw_data = df_raw
            st.success("Cargado dataset de Admisiones Universitarias.")

        if st.session_state.m2_raw_data is not None:
            df = st.session_state.m2_raw_data
            
            # Filtro automático para columnas con datos numéricos
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(num_cols) < 2:
                st.error("El dataset debe poseer al menos 2 columnas numéricas para el entrenamiento.")
            else:
                col_sel1, col_sel2 = st.columns(2)
                
                with col_sel1:
                    target = st.selectbox(
                        "Selecciona la columna objetivo (Target):",
                        options=num_cols,
                        index=len(num_cols)-1
                    )
                    st.session_state.m2_target = target
                    
                    # Validar si el target tiene más de 2 clases
                    unique_vals = sorted(df[target].dropna().unique())
                    is_multiclass = len(unique_vals) > 2
                    
                    positive_class = None
                    if is_multiclass:
                        st.info(f" El target '{target}' contiene {len(unique_vals)} clases ({unique_vals}). Para aplicar Regresión Logística Binaria, selecciona cuál de estas clases consideras como Clase Positiva (1). Las demás se agruparán automáticamente como Clase Negativa (0).")
                        positive_class = st.selectbox(
                            "Selecciona la Clase Positiva (1):",
                            options=unique_vals,
                            index=0
                        )
                    else:
                        if len(unique_vals) != 2 or unique_vals != [0, 1]:
                            st.warning(f" ¡Cuidado! Regresión Logística requiere una variable binaria (0 y 1). Los valores únicos detectados en '{target}' son: {unique_vals}. Los mapearemos automáticamente a [0, 1].")

                with col_sel2:
                    split_ratio = st.slider(
                        "Porcentaje de datos para Entrenamiento (Train Set):",
                        50, 95, 80, step=5
                    )
                    test_ratio = 100 - split_ratio
                    st.write(f"Conjunto de Entrenamiento: **{split_ratio}%** | Conjunto de Prueba (Test): **{test_ratio}%**")

                # Selector de características (Features)
                pos_features = [c for c in num_cols if c != target]
                selected_feats = st.multiselect(
                    "Selecciona las características (Features) predictoras:",
                    options=pos_features,
                    default=pos_features
                )
                st.session_state.m2_features = selected_feats

                # Botón de Procesamiento
                st.markdown("---")
                process_btn = st.button(" Procesar y Separar Datos")
                
                if process_btn:
                    if len(selected_feats) == 0:
                        st.error("Debes seleccionar al menos una característica (Feature).")
                    else:
                        # Limpieza y filtrado
                        df_clean = df[[target] + selected_feats].dropna().copy()
                        # Re-mapear target a 0 y 1
                        if positive_class is not None:
                            df_clean[target] = (df_clean[target] == positive_class).astype(int)
                        else:
                            val_map = {val: idx for idx, val in enumerate(sorted(df_clean[target].unique()))}
                            df_clean[target] = df_clean[target].map(val_map)
                            
                            # Mezcla y separación aleatoria (Train/Test Split)
                            shuffled_df = df_clean.sample(frac=1.0, random_state=42).reset_index(drop=True)
                            split_idx = int(len(shuffled_df) * (split_ratio / 100))
                            
                            st.session_state.m2_train_data = shuffled_df.iloc[:split_idx]
                            st.session_state.m2_test_data = shuffled_df.iloc[split_idx:]
                            st.session_state.m2_trained = False # Resetear bandera de entrenamiento
                            
                            # Calcular estadísticas de normalización Z-Score (usando solo el conjunto de entrenamiento)
                            stats = {}
                            for feat in selected_feats:
                                mean_val = st.session_state.m2_train_data[feat].mean()
                                std_val = st.session_state.m2_train_data[feat].std()
                                std_val = 1.0 if std_val == 0 else std_val # Prevenir división entre cero
                                stats[feat] = {"mean": mean_val, "std": std_val}
                            st.session_state.m2_feature_stats = stats
                            
                            st.success(f"Datos procesados. Entrenamiento (Train): {len(st.session_state.m2_train_data)} filas | Prueba (Test): {len(st.session_state.m2_test_data)} filas.")

            # Mostrar Estadísticas y Preview
            st.markdown("### 📋 Vista Previa y Estadísticas Descriptivas")
            prev_col, stats_col = st.columns([1, 1])
            with prev_col:
                st.markdown("**Primeras 5 filas del Dataset:**")
                st.dataframe(df.head())
            with stats_col:
                st.markdown("**Estadísticas Descriptivas:**")
                st.dataframe(df[num_cols].describe().T[["mean", "std", "min", "max"]])

            # Visualización Scatter Plot
            if len(pos_features) >= 1:
                st.markdown("---")
                st.markdown("### 🔍 Exploración Visual de Relación de Variables")
                sc_col1, sc_col2 = st.columns(2)
                with sc_col1:
                    x_feat = st.selectbox("Eje X para Scatter:", pos_features, index=0)
                with sc_col2:
                    y_feat = st.selectbox("Eje Y para Scatter:", pos_features, index=min(1, len(pos_features)-1))

                fig_sc = go.Figure()
                
                # Separar por clases reales
                c0_data = df[df[target] == df[target].unique()[0]]
                c1_data = df[df[target] == df[target].unique()[1]] if len(df[target].unique()) > 1 else pd.DataFrame()
                
                fig_sc.add_trace(go.Scatter(
                    x=c0_data[x_feat], y=c0_data[y_feat],
                    mode="markers",
                    marker=dict(size=8, color="#f43f5e", line=dict(color="#be123c", width=1)),
                    name="Clase 0"
                ))
                
                if not c1_data.empty:
                    fig_sc.add_trace(go.Scatter(
                        x=c1_data[x_feat], y=c1_data[y_feat],
                        mode="markers",
                        marker=dict(size=8, color="#10b981", line=dict(color="#047857", width=1)),
                        name="Clase 1"
                    ))

                # Si el modelo ya está entrenado, intentar dibujar la frontera de decisión lineal 2D
                if st.session_state.m2_trained and x_feat in st.session_state.m2_features and y_feat in st.session_state.m2_features:
                    idx_x = st.session_state.m2_features.index(x_feat)
                    idx_y = st.session_state.m2_features.index(y_feat)
                    w_x = st.session_state.m2_weights[idx_x]
                    w_y = st.session_state.m2_weights[idx_y]
                    b_m2 = st.session_state.m2_bias
                    
                    if w_y != 0:
                        # Ecuación de la frontera de decisión: w_x * x_norm + w_y * y_norm + b = 0
                        # w_x * ((x - mean_x)/std_x) + w_y * ((y - mean_y)/std_y) + b = 0
                        # Resolviendo para y en función de x:
                        # y = mean_y + std_y * [ - (w_x * (x - mean_x) / std_x - b) / w_y ]
                        mean_x = st.session_state.m2_feature_stats[x_feat]["mean"]
                        std_x = st.session_state.m2_feature_stats[x_feat]["std"]
                        mean_y = st.session_state.m2_feature_stats[y_feat]["mean"]
                        std_y = st.session_state.m2_feature_stats[y_feat]["std"]
                        
                        x_line = np.linspace(df[x_feat].min(), df[x_feat].max(), 100)
                        y_line = mean_y + std_y * ( - (w_x * (x_line - mean_x) / std_x + b_m2) / w_y )
                        
                        # Filtrar límites
                        fig_sc.add_trace(go.Scatter(
                            x=x_line, y=y_line,
                            mode="lines",
                            line=dict(color="#6366f1", width=3, dash="dash"),
                            name="Frontera Proyectada"
                        ))

                fig_sc.update_layout(
                    template="plotly_dark",
                    xaxis=dict(title=x_feat, showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title=y_feat, showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(l=40, r=40, t=20, b=40),
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_sc, use_container_width=True)

    # --------------------------------------------------------------------------
    # Tab 2: Entrenamiento y Coeficientes
    # --------------------------------------------------------------------------
    with tabs[1]:
        st.markdown("### C y D. Configuración del Algoritmo y Entrenamiento")
        
        if st.session_state.m2_train_data is None:
            st.info("Primero carga y procesa los datos en la pestaña anterior.")
        else:
            st.write(f"**Target:** `{st.session_state.m2_target}`")
            st.write(f"**Características:** `{st.session_state.m2_features}`")
            
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                lr_m2 = st.slider("Tasa de Aprendizaje (α) - Logística:", 0.01, 1.0, 0.1, 0.01)
            with c_col2:
                epochs_m2 = st.slider("Épocas de Entrenamiento - Logística:", 50, 2000, 500, 50)
            
            # Botones
            train_m2_btn = st.button(" Iniciar Descenso del Gradiente Multivariable")
            
            # Lógica del entrenamiento multivariable manual
            if train_m2_btn:
                # Obtener matrices de entrenamiento normalizadas
                train_df = st.session_state.m2_train_data
                feats = st.session_state.m2_features
                target = st.session_state.m2_target
                
                # Normalizar
                X = np.zeros((len(train_df), len(feats)))
                for idx, feat in enumerate(feats):
                    mean_f = st.session_state.m2_feature_stats[feat]["mean"]
                    std_f = st.session_state.m2_feature_stats[feat]["std"]
                    X[:, idx] = (train_df[feat].values - mean_f) / std_f
                
                y = train_df[target].values
                m, n = X.shape
                
                # Inicializar parámetros
                weights = np.zeros(n)
                bias = 0.0
                cost_history = []
                
                # Bucle de optimización
                for epoch in range(epochs_m2):
                    # Combinación lineal
                    z = np.dot(X, weights) + bias
                    # Sigmoide estable
                    z_clipped = np.clip(z, -500, 500)
                    a = 1.0 / (1.0 + np.exp(-z_clipped))
                    
                    # Log-Loss (Entropía Cruzada)
                    a_clipped = np.clip(a, 1e-15, 1.0 - 1e-15)
                    cost = - (1.0 / m) * np.sum(y * np.log(a_clipped) + (1 - y) * np.log(1.0 - a_clipped))
                    cost_history.append(cost)
                    
                    # Gradientes
                    dz = a - y
                    dw = (1.0 / m) * np.dot(X.T, dz)
                    db = (1.0 / m) * np.sum(dz)
                    
                    # Actualización de pesos
                    weights -= lr_m2 * dw
                    bias -= lr_m2 * db
                
                # Almacenar en session_state
                st.session_state.m2_weights = weights.tolist()
                st.session_state.m2_bias = float(bias)
                st.session_state.m2_cost_history = cost_history
                st.session_state.m2_trained = True
                st.success(f"Entrenamiento completado exitosamente. Costo final (Log-Loss): {cost_history[-1]:.5f}")

            # Mostrar resultados si ya está entrenado
            if st.session_state.m2_trained:
                st.markdown("---")
                res_col1, res_col2 = st.columns([1, 1])
                
                with res_col1:
                    st.markdown("#### Curva de Aprendizaje (Log-Loss vs Épocas)")
                    fig_cost = go.Figure()
                    fig_cost.add_trace(go.Scatter(
                        y=st.session_state.m2_cost_history,
                        mode="lines",
                        line=dict(color="#6366f1", width=2.5),
                        name="Costo"
                    ))
                    fig_cost.update_layout(
                        template="plotly_dark",
                        xaxis=dict(title="Épocas", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(title="Log-Loss", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                        margin=dict(l=40, r=40, t=10, b=40),
                        height=250,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_cost, use_container_width=True)
                
                with res_col2:
                    st.markdown("#### D. Interpretación de Coeficientes ($W$)")
                    coef_df = pd.DataFrame({
                        "Característica": st.session_state.m2_features,
                        "Peso Estandarizado (W)": st.session_state.m2_weights
                    })
                    
                    def interpretar_peso(w):
                        if abs(w) < 0.1:
                            return "Poco impacto en la predicción."
                        elif w > 0:
                            return "Aumenta la probabilidad de ser Clase 1."
                        else:
                            return "Disminuye la probabilidad de ser Clase 1."
                            
                    coef_df["Interpretación"] = coef_df["Peso Estandarizado (W)"].apply(interpretar_peso)
                    st.dataframe(coef_df, width="stretch")
                    st.info(f"**Sesgo / Intercepto (bias):** {st.session_state.m2_bias:.5f}")

                # Exportar modelo JSON
                st.markdown("####  Exportar Modelo Entrenado")
                model_state = {
                    "m2_weights": st.session_state.m2_weights,
                    "m2_bias": st.session_state.m2_bias,
                    "m2_features": st.session_state.m2_features,
                    "m2_target": st.session_state.m2_target,
                    "m2_feature_stats": st.session_state.m2_feature_stats
                }
                model_json = json.dumps(model_state, indent=2)
                st.download_button(
                    label="Descargar archivo del modelo (JSON)",
                    data=model_json,
                    file_name=f"modelo_logistico_{st.session_state.m2_target}.json",
                    mime="application/json"
                )

            # Importar modelo JSON
            st.markdown("---")
            st.markdown("####  Importar Modelo Entrenado (JSON)")
            imported_file = st.file_uploader("Sube un archivo de modelo previamente entrenado:", type=["json"])
            if imported_file:
                try:
                    imp_data = json.load(imported_file)
                    st.session_state.m2_weights = imp_data["m2_weights"]
                    st.session_state.m2_bias = imp_data["m2_bias"]
                    st.session_state.m2_features = imp_data["m2_features"]
                    st.session_state.m2_target = imp_data["m2_target"]
                    st.session_state.m2_feature_stats = imp_data["m2_feature_stats"]
                    st.session_state.m2_trained = True
                    st.success("Modelo JSON importado con éxito.")
                except Exception as e:
                    st.error(f"Error al parsear el archivo JSON: {e}")

# --------------------------------------------------------------------------
# Tab 3: Evaluación en Test Set
# --------------------------------------------------------------------------
    with tabs[2]:
        st.markdown("### E. Evaluación del Modelo con Datos de Test")
        
        if not st.session_state.m2_trained:
            st.info("Primero debes entrenar el modelo en la pestaña anterior para evaluarlo.")
        else:
            test_df = st.session_state.m2_test_data
            feats = st.session_state.m2_features
            target = st.session_state.m2_target
            
            # Parámetros del modelo
            weights = np.array(st.session_state.m2_weights)
            bias = st.session_state.m2_bias
            stats = st.session_state.m2_feature_stats
            
            # Normalizar conjunto de prueba
            X_test = np.zeros((len(test_df), len(feats)))
            for idx, feat in enumerate(feats):
                X_test[:, idx] = (test_df[feat].values - stats[feat]["mean"]) / stats[feat]["std"]
            
            y_test = test_df[target].values
            
            # Predecir probabilidades
            z_test = np.dot(X_test, weights) + bias
            z_test_clipped = np.clip(z_test, -500, 500)
            probs_test = 1.0 / (1.0 + np.exp(-z_test_clipped))
            
            # Selector de umbral interactivo de clasificación
            st.markdown("Ajusta el **umbral de clasificación** (threshold) y observa cómo varían la matriz de confusión y las métricas en tiempo real:")
            th_m2 = st.slider("Umbral de Clasificación:", 0.05, 0.95, 0.50, 0.05)
            
            preds_bin = (probs_test >= th_m2).astype(int)
            
            acc = accuracy_score(y_test, preds_bin)
            prec = precision_score(y_test, preds_bin, zero_division=0)
            rec = recall_score(y_test, preds_bin, zero_division=0)
            f1 = f1_score(y_test, preds_bin, zero_division=0)
            
            cm = confusion_matrix(y_test, preds_bin, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            # Tarjetas de Métricas de Evaluación
            ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
            with ev_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Accuracy (Exactitud)</div>
                    <div class="metric-value">{acc * 100:.1f}%</div>
                    <div class="metric-desc">Aciertos totales en test</div>
                </div>
                """, unsafe_allow_html=True)
            with ev_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Precision (Precisión)</div>
                    <div class="metric-value" style="color: #10b981;">{prec * 100:.1f}%</div>
                    <div class="metric-desc">TP / (TP + FP)</div>
                </div>
                """, unsafe_allow_html=True)
            with ev_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Recall (Sensibilidad)</div>
                    <div class="metric-value" style="color: #6366f1;">{rec * 100:.1f}%</div>
                    <div class="metric-desc">TP / (TP + FN)</div>
                </div>
                """, unsafe_allow_html=True)
            with ev_col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">F1-Score</div>
                    <div class="metric-value" style="color: #a855f7;">{f1 * 100:.1f}%</div>
                    <div class="metric-desc">Media armónica</div>
                </div>
                """, unsafe_allow_html=True)

            # Matriz de Confusión y Curva ROC
            st.markdown("---")
            bot_col1, bot_col2 = st.columns([1, 1])
            
            with bot_col1:
                st.markdown("#### Matriz de Confusión")
                # Formatear la matriz de confusión como una tabla bonita HTML
                st.markdown(f"""
                <table style="width:100%; border-collapse: collapse; text-align: center; font-size:1rem; border: 1px solid rgba(255,255,255,0.1);">
                    <tr style="background-color: rgba(99, 102, 241, 0.1);">
                        <th style="padding: 10px; border: 1px solid rgba(255,255,255,0.1);">Real \\ Pred.</th>
                        <th style="padding: 10px; border: 1px solid rgba(255,255,255,0.1);">Predicción 0 (Negativo)</th>
                        <th style="padding: 10px; border: 1px solid rgba(255,255,255,0.1);">Predicción 1 (Positivo)</th>
                    </tr>
                    <tr>
                        <td style="padding: 15px; font-weight: bold; background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1);">Real 0</td>
                        <td style="padding: 15px; color:#f43f5e; font-weight:bold; border: 1px solid rgba(255,255,255,0.1);">TN: {tn}</td>
                        <td style="padding: 15px; color:#f43f5e; border: 1px solid rgba(255,255,255,0.1);">FP: {fp}</td>
                    </tr>
                    <tr>
                        <td style="padding: 15px; font-weight: bold; background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1);">Real 1</td>
                        <td style="padding: 15px; color:#10b981; border: 1px solid rgba(255,255,255,0.1);">FN: {fn}</td>
                        <td style="padding: 15px; color:#10b981; font-weight:bold; border: 1px solid rgba(255,255,255,0.1);">TP: {tp}</td>
                    </tr>
                </table>
                """, unsafe_allow_html=True)
                st.write("")
                st.write(f"*Total registros evaluados (Test Set):* **{len(y_test)}**")

            with bot_col2:
                st.markdown("#### Curva ROC (Receiver Operating Characteristic)")
                
                # Calcular tasas TPR y FPR en diferentes umbrales
                thresholds_roc = np.linspace(0.0, 1.001, 100)
                tpr_list = []
                fpr_list = []
                
                for t in thresholds_roc:
                    preds_t = (probs_test >= t).astype(int)
                    cm_t = confusion_matrix(y_test, preds_t, labels=[0, 1])
                    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
                    
                    tpr_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
                    fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
                    
                    tpr_list.append(tpr_t)
                    fpr_list.append(fpr_t)
                
                # Ordenar
                sorted_idx = np.argsort(fpr_list)
                fpr_arr = np.array(fpr_list)[sorted_idx]
                tpr_arr = np.array(tpr_list)[sorted_idx]
                
                # Calcular AUC usando regla trapezoidal (con soporte para numpy 2.0+)
                if hasattr(np, "trapezoid"):
                    auc = np.trapezoid(tpr_arr, fpr_arr)
                else:
                    auc = np.trapz(tpr_arr, fpr_arr)
                
                # Dibujar Curva ROC con Plotly
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr_arr, y=tpr_arr,
                    mode="lines",
                    line=dict(color="#6366f1", width=3),
                    name=f"Curva ROC (AUC = {auc:.4f})"
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    line=dict(color="rgba(255,255,255,0.2)", width=1.5, dash="dash"),
                    name="Línea de Azar (AUC = 0.5)"
                ))
                fig_roc.update_layout(
                    template="plotly_dark",
                    xaxis=dict(title="Tasa de Falsos Positivos (FPR)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(title="Tasa de Verdaderos Positivos (TPR)", showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(l=40, r=40, t=10, b=40),
                    height=250,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                st.write(f"**Área Bajo la Curva (AUC-ROC):** `{auc:.4f}`")

# --------------------------------------------------------------------------
# Tab 4: Predicción de Nuevos Registros
# --------------------------------------------------------------------------
    with tabs[3]:
        st.markdown("### F. Predicción de Nuevos Registros")
        
        if not st.session_state.m2_trained:
            st.info("Primero debes entrenar el modelo en la pestaña anterior para poder realizar predicciones.")
        else:
            st.markdown("Ingresa los valores de las variables explicativas para estimar la probabilidad y clasificar:")
            
            # Parámetros del modelo
            weights = np.array(st.session_state.m2_weights)
            bias = st.session_state.m2_bias
            stats = st.session_state.m2_feature_stats
            feats = st.session_state.m2_features
            
            # Autogenerar formulario según variables del modelo
            input_vals = {}
            st.markdown("#### Valores de Entrada:")
            f_cols = st.columns(min(3, len(feats)))
            
            for idx, feat in enumerate(feats):
                col_idx = idx % len(f_cols)
                mean_f = stats[feat]["mean"]
                std_f = stats[feat]["std"]
                
                with f_cols[col_idx]:
                    input_vals[feat] = st.number_input(
                        f"{feat}:", 
                        value=float(mean_f), 
                        step=0.1
                    )
            
            pred_btn = st.button(" Realizar Predicción")
            
            if pred_btn:
                # Normalizar entradas
                x_norm = np.zeros(len(feats))
                for idx, feat in enumerate(feats):
                    x_norm[idx] = (input_vals[feat] - stats[feat]["mean"]) / stats[feat]["std"]
                
                # Predecir
                z = np.dot(x_norm, weights) + bias
                z_clipped = np.clip(z, -500, 500)
                prob = 1.0 / (1.0 + np.exp(-z_clipped))
                pred_class = 1 if prob >= 0.5 else 0 # Usar 0.50 estándar
                
                st.markdown("---")
                st.markdown("###  Resultado del Modelo:")
                res_box_color = "rgba(16, 185, 129, 0.15)" if pred_class == 1 else "rgba(244, 63, 94, 0.15)"
                border_color = "#10b981" if pred_class == 1 else "#f43f5e"
                text_color = "#10b981" if pred_class == 1 else "#f43f5e"
                
                st.markdown(f"""
                <div style="background: {res_box_color}; border: 2px solid {border_color}; padding: 1.5rem; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.1rem; color: #94a3b8; text-transform: uppercase; font-weight: bold;">Clase Predicha</div>
                    <div style="font-size: 2.8rem; font-weight: 800; color: {text_color}; margin: 0.5rem 0;">Clase {pred_class}</div>
                    <div style="font-size: 1.2rem; color: #ffffff;">Probabilidad de ser Clase 1: <strong>{prob * 100:.2f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)

# ==============================================================================
# MÓDULO 3: DOCUMENTACIÓN Y USO DE IA
# ==============================================================================
elif module == " Módulo 3: Metodología y Fundamentos Teóricos":
    st.markdown("##  Módulo 3: Metodología y Fundamentos Teóricos Generativa")
    st.markdown("### Entregables y Justificación Pedagógica")
    
  

    st.markdown(f"""
    <div style="background: rgba(99, 102, 241, 0.1); border-left: 4px solid #6366f1; padding: 1rem; border-radius: 4px; margin-bottom: 1.5rem;">
        <strong>Estudiantes:</strong> Marleinis Orozco y Yuleisi Carranza<br>
        <strong>Curso:</strong> Inteligencia Artificial y Aprendizaje de Máquina (Ingeniería Informática)<br>
        <strong>Docente:</strong> Docente Jorge Rudas
    </div>
    """, unsafe_allow_html=True)

    tab_doc1, tab_doc2, tab_doc3, tab_doc4 = st.tabs([
        "1. Diseño Estructural y Plantillas de Código", 
        "2. Depuración y Adaptación de Lógica", 
        "3. Análisis Teórico y Conceptos Consolidados",
        "4. Guía de Despliegue en Servidor Cloud"
    ])

    with tab_doc1:
        st.markdown("####  Diseño Estructural y Plantillas de Código")
        st.markdown("""
        * **Estructura base del Algoritmo:** El esqueleto del algoritmo de descenso de gradiente multivariable para Regresión Logística y las operaciones matriciales vectorizadas con NumPy.
        * **Interactividad y Gráficos:** Plantilla de dibujo de trazos con Plotly, tales como el trazado de la curva ROC, la línea de regresión continua y la sombra de regiones de clasificación según límites.
        * **Estilo CSS de la Interfaz:** Estilos CSS oscuro premium "Glassmorphism" con gradientes para la visualización fluida de métricas y cards adaptables.
        * **Normalización:** La automatización del escalado Z-score guardando estadísticas de entrenamiento para ser consumidas por las muestras de validación y de predicción.
        """)

    with tab_doc2:
        st.markdown("####  Depuración y Adaptación de Lógica")
        st.markdown("""
        * **Compatibilidad de Entrada en Predicción:** Se corrigió el casting de los valores en los widgets `st.number_input` a tipo float directo para prevenir colisiones de tipado con los arrays de NumPy.
        * **Normalización del Target:** Se implementó una lógica de mapeo dinámica en caso de que la variable objetivo ingresada tuviera etiquetas de clase distintas de `[0, 1]` (ej. `[1, 2]`), asegurando la validez del logaritmo en la Entropía Cruzada.
        * **Estabilidad Numérica:** Se incluyó un límite (*clip*) en la entrada de la función sigmoide (`np.clip(z, -500, 500)`) y en los valores predictivos de probabilidad (`np.clip(prob, 1e-15, 1 - 1e-15)`) para erradicar errores de tipo `NaN` y desbordamiento aritmético (*overflow*).
        * **Diseño del Gráfico 2D:** Ajuste de las áreas sombreadas del plano cartesiano interactivo 2D para que cambiaran de orientación de forma consistente cuando la pendiente del clasificador ($m$) fuera negativa.
        """)

    with tab_doc3:
        st.markdown("####  Análisis Teórico y Conceptos Consolidados")
        st.markdown("""
        1. **Regresión Lineal como Clasificador (Límites):** Comprender que modelar etiquetas de clase de forma lineal discreta es susceptible a desviaciones drásticas por valores atípicos (*outliers*) debido a la función de coste cuadrática (MSE), la cual prioriza el ajuste general sobre la optimización del límite de separación.
        2. **Función de Pérdida Log-Loss (Entropía Cruzada):** Aprender que la pérdida logarítmica penaliza de forma logarítmica e infinita las predicciones seguras e incorrectas, promoviendo la máxima separación probabilística de clases sin verse perturbada por outliers lejanos en features.
        3. **Importancia del Scaler en Producción:** Comprender que cualquier dato que ingrese a predecir debe ser normalizado empleando la **media y desviación estándar del conjunto de entrenamiento**, y no de la muestra de entrada, para garantizar que la ecuación se evalúe dentro del mismo espacio vectorial en el que el modelo aprendió.
        4. **Compromiso en Métricas (Precision/Recall):** Analizar que modificar el umbral de decisión permite ajustar el modelo según el dominio del problema: un umbral bajo mejora la sensibilidad (*Recall*) capturando más casos positivos (ideal para salud), mientras que un umbral alto maximiza la *Precisión* evitando falsos positivos (ideal para filtros de spam).
        """)

    with tab_doc4:
        st.markdown("####  Cómo desplegamos la app")
        st.markdown("""

        
        ##### Paso 1: Subir tu código a GitHub
        1. Crea una cuenta gratuita en [GitHub](https://github.com/) si aún no la tienes.
        2. Crea un nuevo repositorio público con el nombre: `Proyecto_Final_Inteligencia_Artificial`.
        3. Sube los siguientes archivos de tu carpeta local al repositorio:
           * `app.py`
           * `requirements.txt`
           * `README.md`
           * `wine.csv` (opcional, para que los usuarios puedan descargarlo de allí)
        
        ##### Paso 2: Conectar con Streamlit Community Cloud
        1. Entramos a [Streamlit Share](https://share.streamlit.io/) y nos registramos iniciando sesión con nuestra cuenta de **GitHub**.
        2. Hacemos clic en el botón azul **"Create app"** (o **"Deploy an app"**).
        3. Rellenamos los campos con los datos de nuestro repositorio:
           * **Repository:** `tu-usuario-github/Proyecto_Final_Inteligencia_Artificial` (puedes buscarlo en la lista)
           * **Branch:** `main` (o la rama principal que uses)
           * **Main file path:** `app.py`
        4. (Opcional) En el campo **App URL** de la derecha, puedes cambiar el nombre para personalizar el enlace, por ejemplo: `marleinis-orozco-ia.streamlit.app`.
        5. Hacemos clic en **"Deploy!"**.
        
        ##### Paso 3: ¡Listo!
        
        """)
