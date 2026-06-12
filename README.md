# Plataforma Completa de Clasificación (Regresión Logística y Lineal)

Este repositorio contiene el Proyecto Final de la asignatura de Inteligencia Artificial desarrollado por **Marleinis Orozco** y **Yuleisi Carranza**.

La aplicación es una plataforma web interactiva de **Machine Learning** que ejecuta todo su procesamiento (parseo de datos, separación, entrenamiento matemático, predicción) directamente utilizando **Python y Streamlit** en una interfaz interactiva de alto rendimiento.

## Cumplimiento de Objetivos Específicos (Rúbrica)

La aplicación demuestra la comprensión integral del ciclo de vida de un modelo supervisado:

*   **A. Estructuración del Problema:** La pestaña *1. Datos y Selección* del Módulo 2 permite cargar un dataset real (CSV) y seleccionar dinámicamente qué columna representa el objetivo a clasificar (Target) y cuáles son las características predictoras (Features).
*   **B. Separación de Datos (Train/Test Split):** Antes del entrenamiento, el sistema baraja y separa el dataset según un porcentaje definido por el usuario (ej. 80% entrenamiento, 20% prueba) para evaluar de forma honesta el sobreajuste (overfitting).
*   **C. Entrenamiento de Regresión Logística:** El modelo se entrena en la pestaña *2. Entrenamiento* utilizando el algoritmo de **Gradiente Descendiente Multivariable** programado en Python puro. Se optimiza la función de pérdida *Log-Loss* (Entropía Cruzada) aplicando la función de activación Sigmoide a las características previamente normalizadas mediante *Z-Score*.
*   **D. Interpretación de Coeficientes:** Una vez finalizado el entrenamiento, el sistema despliega una tabla con los pesos ($W$) aprendidos para cada característica e interpreta automáticamente si influyen positiva o negativamente hacia la clase 1. También grafica la frontera de decisión en 2D sobre el gráfico de dispersión de variables.
*   **E. Evaluación del Modelo:** En la pestaña *3. Evaluación*, se calculan las proyecciones del modelo sobre el Conjunto de Prueba. Se genera la **Matriz de Confusión** y se derivan métricas como *Accuracy*, *Precision*, *Recall*, *F1-Score*, además de calcular y graficar la curva ROC y el área bajo la curva (AUC).
*   **F. Predicción con Nuevos Datos:** La pestaña *4. Predicciones* lee dinámicamente el esquema de variables y genera un formulario. Permite introducir nuevos registros manuales, normalizarlos bajo la escala del modelo, y retornar una probabilidad y clasificación final.
*   **G. Despliegue en la Nube:** La aplicación está lista para ser desplegada en **Streamlit Community Cloud** de forma estable y accesible públicamente sin requerir mantenimiento de servidores complejos.
*   **H. Metodología de Desarrollo y Herramientas de Apoyo:** Documentado internamente en el Módulo 3 de la propia plataforma, donde se especifica cómo se utilizó Google DeepMind Antigravity como copiloto para estructurar la UI, corregir las matemáticas del gradiente multivariable y afianzar los conceptos matemáticos del estudiante.

---

## Ejecución Local

Para ejecutar esta aplicación localmente en tu máquina:

1. Clona el repositorio:
   ```bash
   git clone https://github.com/Candresh9/Proyecto_Final_Inteligencia_Artificial.git
   ```
2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta el servidor local de Streamlit:
   ```bash
   streamlit run app.py
   ```

La aplicación se abrirá automáticamente en tu navegador web (por defecto en `http://localhost:8501`).

## Despliegue en la Nube

Para desplegar esta aplicación públicamente en **Streamlit Cloud**:
1. Sube tu código a un repositorio público en GitHub.
2. Inicia sesión en [share.streamlit.io](https://share.streamlit.io/).
3. Haz clic en "New app", selecciona tu repositorio, rama y especifica `app.py` como el archivo principal.
4. Haz clic en "Deploy". La app estará en línea en un par de minutos.
