# Guía y Documentación Completa de la Plataforma Educativa de Machine Learning

Esta documentación detalla de forma exhaustiva el funcionamiento teórico, las ecuaciones matemáticas y el flujo lógico de la aplicación web interactiva desarrollada por **Marleinis Orozco** y **Yuleisi Carranza** para el proyecto final del curso de **Inteligencia Artificial y Aprendizaje de Máquina** (dirigido por el **Ph.D. Jorge Rudas**).

La plataforma es una herramienta pedagógica diseñada para comparar dos paradigmas fundamentales en el aprendizaje supervisado: la **Regresión Lineal como Clasificador** (en 2D) y la **Regresión Logística** (en problemas multivariables de clasificación binaria real).

---

##  Arquitectura General del Sistema

La aplicación está construida utilizando el framework **Streamlit** en **Python**. Toda la lógica está contenida en el archivo `app.py`. La app está dividida en tres secciones lógicas accesibles desde la barra lateral:
1. ** Módulo 1: Simulador Interactivo 2D (Regresión Lineal)**
2. ** Módulo 2: Clasificación Multivariable (Regresión Logística)**
3. ** Módulo 3: Metodología y Fundamentos Teóricos**

---

##  Módulo 1: Simulador Interactivo 2D (Regresión Lineal)

Este módulo está destinado a demostrar de manera gráfica cómo funciona la regresión lineal continua cuando se le impone un umbral artificial para resolver un problema de clasificación de dos clases (Clase 0: Rosado y Clase 1: Verde).

### 1. Interacción con los Datos
* **Edición en Tabla:** A través de un componente `st.data_editor`, el usuario puede ingresar y modificar coordenadas $(X, Y)$ y la etiqueta de la clase (0 o 1).
* **Plantillas de Datos:** Se ofrecen tres escenarios predefinidos para analizar de inmediato:
  * **Separable:** Puntos ordenados donde las clases no se cruzan.
  * **Traslapado:** Puntos donde las clases se mezclan en la zona central (frontera no lineal).
  * **Con Outliers:** Puntos adicionales colocados lejos del grupo principal para demostrar la inestabilidad de la regresión lineal.

### 2. Fundamento Matemático del Modelo Lineal
La hipótesis lineal modela una salida continua a partir de una variable de entrada $x$:
$$h(x) = mx + b$$
donde $m$ representa la pendiente y $b$ representa el intercepto con el eje Y.

#### Ajuste por Mínimos Cuadrados Ordinarios (OLS)
Si el usuario selecciona OLS, la app calcula analíticamente la pendiente y el intercepto óptimos en un solo paso basándose en minimizar la suma de errores al cuadrado:
$$m = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2}$$
$$b = \bar{y} - m\bar{x}$$
donde $\bar{x}$ y $\bar{y}$ representan las medias de las variables $X$ y $Y$.

#### Ajuste por Descenso de Gradiente (GD)
Si el usuario prefiere Descenso de Gradiente, la app ejecuta un bucle iterativo (animado en tiempo real) para minimizar la función de costo del **Error Cuadrático Medio (MSE)**:
$$\text{MSE}(m, b) = \frac{1}{N} \sum_{i=1}^N \left( (mx_i + b) - y_i \right)^2$$

En cada iteración (época), se calculan las derivadas parciales de la pérdida respecto a $m$ y $b$ (gradientes):
$$\frac{\partial \text{MSE}}{\partial m} = \frac{2}{N}\sum_{i=1}^N ((mx_i + b) - y_i)x_i$$
$$\frac{\partial \text{MSE}}{\partial b} = \frac{2}{N}\sum_{i=1}^N ((mx_i + b) - y_i)$$

Los parámetros se actualizan simultáneamente multiplicando los gradientes por la tasa de aprendizaje $\alpha$:
$$m := m - \alpha \frac{\partial \text{MSE}}{\partial m}$$
$$b := b - \alpha \frac{\partial \text{MSE}}{\partial b}$$

### 3. Umbral y Frontera de Decisión
Dado que la recta continua proyecta valores reales en el rango $(-\infty, \infty)$, se impone un **umbral de clasificación** ($t$) de modo que:
$$\text{Clase Predicha} = \begin{cases} 1 & \text{si } h(x) \ge t \\ 0 & \text{si } h(x) < t \end{cases}$$

La **Frontera de Decisión** es el punto vertical en el plano X donde la hipótesis es exactamente igual al umbral ($mx + b = t$). Despejando $x$:
$$x_{\text{frontera}} = \frac{t - b}{m}$$

Este límite separa el plano en dos regiones sombreadas de fondo (rosa para predicción de Clase 0, verde para Clase 1).

### 4. Métricas Computadas
* **MSE (Error Cuadrático Medio):** Mide la desviación de la recta respecto a los valores del eje Y.
* **$R^2$ (Coeficiente de Determinación):** Proporción de la varianza en Y que es explicable por X:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
* **Accuracy, Precision, Recall y F1-Score:** Métricas binarias calculadas sobre los aciertos de clasificación resultantes tras aplicar el umbral.

### 5. Demostración Pedagógica del Outlier
La regresión lineal minimiza los errores cuadráticos verticales. Si se introduce un outlier de clase 1 muy alejado en el eje X, la diferencia cuadrática $(y_i - y_{\text{pred}})^2$ es inmensa. Para minimizar este único error gigante, la línea de regresión se ve forzada a rotar severamente hacia arriba, lo cual arrastra la frontera de decisión hacia la izquierda, clasificando erróneamente puntos de la clase 0 que estaban perfectamente situados en el centro. Esto demuestra por qué **la pérdida cuadrática no es adecuada para clasificación**.

---

##  Módulo 2: Clasificación Multivariable (Regresión Logística)

Este módulo implementa un clasificador real multivariable utilizando el algoritmo matemático nativo de la Regresión Logística.

### 1. Carga, Filtrado y Mapeo Multiclase (wine.csv)
* **Ingreso:** El usuario carga un CSV (como `wine.csv`).
* **Mapeo Multiclase a Binario:** Si la variable objetivo elegida (como `Class` en `wine.csv`) tiene más de dos etiquetas (ej. 1, 2 y 3), la aplicación lo identifica e introduce un selector para que el usuario elija cuál es la **Clase Positiva (1)**. El sistema mapea esa clase como `1` y re-agrupa automáticamente las demás clases como `0`. Esto hace que el modelo de clasificación binaria sea robusto ante cualquier archivo externo.

### 2. Normalización de Características (Z-Score)
Para evitar que las variables con rangos grandes dominen a las de rango pequeño y provoquen que el Descenso de Gradiente diverja, la app normaliza las variables explicativas seleccionadas.
La media ($\mu_j$) y la desviación estándar ($\sigma_j$) se calculan **únicamente en el conjunto de entrenamiento (Train Set)** para evitar la fuga de información (*data leakage*):
$$X_{\text{norm}, j} = \frac{X_j - \mu_j}{\sigma_j}$$

---

### 3. Entrenamiento Matemático de la Regresión Logística
Dado un conjunto de entrenamiento con $m$ muestras y $n$ características, el modelo calcula una probabilidad para cada registro.

#### A. Hipótesis Probabilística (Sigmoide)
Primero, calcula la combinación lineal de los datos con los pesos ($W$) y el sesgo ($bias$):
$$z = W^T X_{\text{norm}} + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$
Luego, aplica la **Función Sigmoide** para acotar el resultado en el rango $(0, 1)$:
$$a = \sigma(z) = \frac{1}{1 + e^{-z}}$$
Esta salida se interpreta como la probabilidad de que la muestra pertenezca a la clase positiva: $P(y=1|X)$.

#### B. Optimización mediante Log-Loss (Entropía Cruzada)
En lugar del MSE, la regresión logística optimiza la **Función de Pérdida Log-Loss** (que es convexa al aplicarse sobre la sigmoide, evitando mínimos locales):
$$J(W, b) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) \right]$$

* Si la clase real es $y=1$ y la predicción es $a \to 0$, el costo tiende a infinito.
* Si la clase real es $y=0$ y la predicción es $a \to 1$, el costo tiende a infinito.

#### C. Descenso de Gradiente Multivariable
Para minimizar $J$, se calculan las derivadas parciales respecto a cada peso y al sesgo:
$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m \left( a^{(i)} - y^{(i)} \right) x_j^{(i)}$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m \left( a^{(i)} - y^{(i)} \right)$$

En cada época, todos los pesos y el sesgo se actualizan simultáneamente:
$$w_j := w_j - \alpha \frac{\partial J}{\partial w_j}$$
$$b := b - \alpha \frac{\partial J}{\partial b}$$

Se grafica la curva de costo resultante para verificar visualmente que el algoritmo converge (el costo desciende a medida que pasan las épocas).

---

### 4. Interpretación de Coeficientes ($W$)
Los pesos finales representan el grado de asociación de cada variable con la probabilidad de la clase positiva:
* **$w_j > 0$:** Indica una relación positiva. Al aumentar la característica $x_j$, aumenta la probabilidad de pertenecer a la clase 1.
* **$w_j < 0$:** Indica una relación negativa. Al aumentar la característica $x_j$, disminuye la probabilidad de la clase 1.
* **$w_j \approx 0$:** La característica no aporta información relevante para la discriminación del modelo.

---

### 5. Evaluación en el Conjunto de Test
Para medir el desempeño real, el modelo hace predicciones sobre el conjunto de prueba (datos no vistos).

#### Matriz de Confusión
Clasifica los aciertos y errores en 4 cuadrantes según el umbral ($t$) seleccionado por el usuario:
* **TN (Verdaderos Negativos):** Reales 0 predichos como 0.
* **FP (Falsos Positivos):** Reales 0 predichos como 1 (Error Tipo I).
* **FN (Falsos Negativos):** Reales 1 predichos como 0 (Error Tipo II).
* **TP (Verdaderos Positivos):** Reales 1 predichos como 1.

#### Métricas Derivadas
* **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$ (Acierto global).
* **Precision:** $\frac{TP}{TP + FP}$ (De los que predije positivos, ¿cuántos lo eran?).
* **Recall (Sensibilidad):** $\frac{TP}{TP + FN}$ (De los positivos reales, ¿cuántos capturé?).
* **F1-Score:** Media armónica: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.

#### Curva ROC (Receiver Operating Characteristic) & AUC
* **Curva ROC:** Grafica la Tasa de Verdaderos Positivos (Recall / TPR) frente a la Tasa de Falsos Positivos (FPR = $\frac{FP}{FP + TN}$) evaluando todos los posibles umbrales de decisión entre 0 y 1.
* **AUC (Área Bajo la Curva):** Mide la capacidad de separación del clasificador. Se calcula mediante la **regla trapezoidal** integrando numéricamente el área bajo la curva ROC:
  $$\text{AUC} \approx \sum_{k=1}^{M} \frac{\text{TPR}_k + \text{TPR}_{k-1}}{2} (\text{FPR}_k - \text{FPR}_{k-1})$$
  donde $M$ es el número de umbrales evaluados. Un clasificador perfecto tiene $\text{AUC}=1.0$; uno puramente aleatorio tiene $\text{AUC}=0.5$.

---

### 6. Predicción e Inferencia Dinámica
Al introducir nuevos valores numéricos en el formulario:
1. El sistema lee las medias ($\mu_j$) y desviaciones estándar ($\sigma_j$) guardadas durante el entrenamiento del modelo.
2. Normaliza los valores del usuario: $x_{\text{user\_norm}, j} = \frac{x_{\text{user}, j} - \mu_j}{\sigma_j}$.
3. Evalúa la combinación lineal: $z = \sum w_j x_{\text{user\_norm}, j} + bias$.
4. Pasa $z$ por la sigmoide para calcular la probabilidad final y asume la predicción de clase.

---

##  Guía de Despliegue con un Solo Link (Streamlit Cloud)

Para alojar la aplicación de manera pública en la nube:

1. **Subir a GitHub:** Sube `app.py`, `requirements.txt`, `README.md` y `wine.csv` a un repositorio público en tu cuenta de GitHub (ej: `Proyecto_Final_Inteligencia_Artificial`).
2. **Conectar a Streamlit Share:** Regístrate en [share.streamlit.io](https://share.streamlit.io/) ingresando con tu cuenta de GitHub.
3. **Desplegar:** Haz clic en **Deploy an app**, busca tu repositorio y rama, escribe `app.py` en "Main file path" y haz clic en **Deploy**.
4. **URL Única:** En un minuto obtendrás un enlace único público (ej: `https://marleinis-orozco-ia.streamlit.app/`) listo para ser compartido y evaluado por el docente.
