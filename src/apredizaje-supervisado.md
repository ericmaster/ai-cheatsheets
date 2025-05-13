# Aprendizaje Supervisado

## 🔍 Diagnóstico Visual: Bias vs Variance

| Escenario                   | Curva de Pérdida (CV vs Entrenamiento) | Sugerencias                                |
| --------------------------- | -------------------------------------- | ------------------------------------------ |
| **High Bias (Underfit)**    | Ambas curvas altas, pequeña diferencia | Agregar features, modelo más complejo, disminuir regularización      |
| **High Variance (Overfit)** | Entrenamiento bajo, CV alto            | Regularizar, simplificar modelo, más datos |
| **Buen Fit**                | Ambas bajas, diferencia pequeña        | Ideal, no cambiar                          |

## 🛠️ Técnicas de Optimización de Hiperparámetros

| Técnica                         | Aplicación                             | Uso recomendado              |
| ------------------------------- | -------------------------------------- | ---------------------------- |
| **K-Fold CV**                   | División equilibrada                   | Evaluar generalización       |
| **Grid Search / Random Search** | Combinaciones de hiperparámetros       | Selección de hiperparámetros |
| **Nested CV**                   | Selección de modelo + evaluación real  | Model selection + tuning     |
| **Early stopping**              | Detener entrenamiento antes de overfit | Redes neuronales             |
| **Regularización (L1/L2)**      | Penaliza parámetros grandes            | Evita overfitting            |

⚠️ No uses el conjunto de test para ajustar hiperparámetros; usa validación cruzada.

## 🧮 Algoritmos de Clasificación/Regresión y Cuándo Usarlos

| Algoritmo               | Problemas comunes                    | Pros                                            | Contras / Riesgos                                       |
| ----------------------- | ------------------------------------ | ----------------------------------------------- | ------------------------------------------------------- |
| **Regresión Lineal**    | Predicción continua, pocos atributos | Simplicidad, interpretabilidad                  | Supone linealidad, sensible a outliers                  |
| **Regresión Logística** | Clasificación binaria/multiclase     | Probabilística, rápida                          | Inadecuado en relaciones no lineales                    |
| **k-NN**                | Problemas locales con muchos datos   | No paramétrico, simple                          | Costoso computacionalmente, sensible al ruido           |
| **SVM**                 | Texto, imagen, margen definido       | Robusto, kernel trick para no linealidad        | Escalable difícil, sin probabilidades directas          |
| **Árboles (DT)**        | Datos tabulares, interpretabilidad   | Interpretables, rápido                          | Sensibles a cambios en datos                            |
| **Random Forest**       | Datos tabulares, mezcla compleja     | Reducción de varianza, robustez                 | Menos interpretables, costoso en memoria                |
| **Boosting (XGBoost)**  | Competencias de precisión            | Mejores scores en test                          | Overfitting si no se controla regularización            |
| **Redes Neuronales**    | Imágenes, audio, texto               | Flexible, autoexplotación de features complejas | Requieren muchos datos, caja negra, entrenamiento lento |
| **Transfer Learning**   | Imágenes, NLP con pocos datos        | Uso de modelos preentrenados                    | Dependiente del dominio fuente y fine-tuning cuidadoso  |

## 📊 Selección de Modelo

```
TIPO DE PROBLEMA ─► ¿SALIDA?
                    │
                    ├─► Continua (Precio, Edad)
                    │     ├─► Lineal (pocas features) → Regresión Lineal
                    │     └─► No lineal / compleja → kNN, SVR, NN
                    │
                    └─► Discreta (Clase)
                          ├─► Binaria → Regresión Logística, SVM
                          ├─► Multiclase → Árboles, SVM, NN
                          └─► Multietiqueta → Árboles, NN (con salida sigmoid)
```

## 🔄 Técnicas de Validación Cruzada

| **Técnica**               | **Descripción**                                         | **Uso recomendado**                               |
| ------------------------- | ------------------------------------------------------- | ------------------------------------------------- |
| **Hold-out**              | División en train/validation/test (ej. 60/20/20)        | Simple, rápida, menos robusta                     |
| **K-Fold CV**             | Divide en k subconjuntos; cada uno rota como validación | Balance entre sesgo y varianza (típico: k=5 o 10) |
| **Stratified K-Fold**     | Igual que K-Fold pero conserva proporción de clases     | Clasificación con clases desbalanceadas           |
| **Leave-One-Out (LOOCV)** | Cada muestra se usa una vez como validación             | Muy costoso computacionalmente, alta varianza     |
| **Nested CV**             | CV interna para tuning y externa para evaluación        | Evita optimismo al ajustar hiperparámetros        |

## 📊 Métricas de Evaluación

### 🔹 Clasificación

| **Métrica** | **Fórmula / Significado**                                   | **Uso recomendado**                  |
| ----------- | ----------------------------------------------------------- | ------------------------------------ |
| Accuracy    | $\frac{TP + TN}{TP + TN + FP + FN}$                         | Datasets balanceados                 |
| Precision   | $\frac{TP}{TP + FP}$                                        | Alta penalización a falsos positivos |
| Recall      | $\frac{TP}{TP + FN}$                                        | Alta penalización a falsos negativos |
| F1-Score    | $2 \cdot \frac{precision \cdot recall}{precision + recall}$ | Compromiso entre precision y recall  |
| ROC-AUC     | Área bajo curva ROC (TPR vs FPR)                            | Comparación global de clasificadores |
| PR-AUC      | Área bajo curva Precisión vs Recall                         | Más útil en datasets desbalanceados  |


### 🔹 Regresión

| **Métrica** | **Fórmula / Significado**              | **Uso recomendado**                           |   |                           |
| ----------- | -------------------------------------- | --------------------------------------------- | - | ------------------------- |
| MSE         | $\frac{1}{n} \sum (y_i - \hat{y}_i)^2$ | Penaliza errores grandes, sensible a outliers |   |                           |
| RMSE        | $\sqrt{MSE}$                           | Interpretable en unidades originales          |   |                           |
| MAE         | $\frac{1}{n} \sum \|y_i - \hat{y}_i\|$ | Más robusto ante outliers |
| $R^2$ Score | $1 - \frac{SS_{res}}{SS_{tot}}$        | Proporción de varianza explicada              |   |                           |

### 🧭 Guía para Toma de Decisiones

| **Escenario**                            | **Decisión recomendada**                                  |
| ---------------------------------------- | --------------------------------------------------------- |
| Datos balanceados, clasificación binaria | Usa **accuracy** y **F1** si importa la precisión general |
| Datos desbalanceados                     | Usa **Precision/Recall**, **F1**, y **PR-AUC**            |
| Regresión con outliers                   | Usa **MAE**                                               |
| Optimización de hiperparámetros          | Usa **K-Fold CV** o **Nested CV**                         |
| Comparar modelos con métricas similares  | Usa test estadístico o curvas ROC/PR                      |
| Detectar overfitting                     | Comparar métricas entre training y validation             |


## 📋 Regularización

| **Aspecto**                    | **L1 (Lasso)**                                                                         | **L2 (Ridge)**                                           | **Uso Común / Consejos**                                                                |                           |                                                                |
| ------------------------------ | -------------------------------------------------------------------------------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------- |
| **Definición**                 | Penaliza la suma de valores absolutos de los coeficientes                              | Penaliza la suma de los cuadrados de los coeficientes    | Ambas técnicas reducen la complejidad del modelo para evitar overfitting                |                           |                                                                |
| **Efecto en los coeficientes** | Tiende a hacer que algunos coeficientes sean exactamente cero (selección de variables) | Reduce los coeficientes sin hacerlos cero                | L1 útil si se busca **sparse model**, L2 útil cuando todas las variables aportan        |                           |                                                                |
| **Diagnóstico aplicado**       | Alto **variance** → aplicar L1 o L2                                                    | Alto **variance** → aplicar L2                           | En casos con high bias, usar **menos regularización**                                   |                           |                                                                |
| **Tipo de problema**           | Modelos interpretables con muchas features                                             | Modelos con colinealidad                                 | L1 mejor para selección automática de variables, L2 mejor para evitar multicolinealidad |                           |                                                                |
| **Modelos aplicables**         | Regresión lineal/logística, SVM, NN (con dropout)                                      | Igual que L1                                             | Regularización puede usarse en la mayoría de los modelos paramétricos                   |                           |                                                                |
| **Riesgos si se aplica mal**   | Demasiado grande λ puede causar **underfitting**                                       | Igual: penalización excesiva reduce capacidad del modelo | Usar **curvas de validación** para ajustar λ                                            |                           |                                                                |

### 🧭 Sugerencias Prácticas para Toma de Decisiones
- ¿Demasiado buen rendimiento en entrenamiento pero malo en validación?
→ Prueba regularización (aumentar λ)

- ¿Modelo muy complejo o lento?
→ Prueba L1 para seleccionar features relevantes

- ¿Datos altamente correlacionados?
→ Usa L2 (Ridge) para estabilizar los coeficientes

- ¿Problemas con overfitting en redes neuronales?
→ Combina L2 con early stopping o usa dropout

## 📋 Redes Neuronales (NN)

| **Aspecto**                    | **Descripción / Detalles**                                                                         |
| ------------------------------ | -------------------------------------------------------------------------------------------------- |
| **Objetivo principal**         | Modelar relaciones complejas y no lineales entre entradas y salidas                                |
| **Arquitectura básica**        | Capas: entrada → ocultas (hidden layers) → salida<br>Neuronas con funciones de activación          |
| **Función de activación**      | ReLU (default en capas ocultas), Sigmoid/Softmax (salidas), Tanh, Leaky ReLU                       |
| **Cost function**              | - Regresión: MSE<br>- Clasificación binaria: BCE<br>- Multiclase: Categorical Cross-Entropy        |
| **Capacidades clave**          | - Aprende representaciones jerárquicas<br>- Adecuado para datos no estructurados (imágenes, texto) |
| **Técnicas de optimización**   | Descenso por gradiente con variantes: SGD, Adam, RMSProp                                           |
| **Ventajas**                   | - Flexible y poderosa para datos grandes y complejos<br>- Automatiza extracción de features        |
| **Desventajas**                | - Requiere muchos datos<br>- Tiempo de entrenamiento alto<br>- Difícil interpretabilidad           |
| **¿Cuándo usar NN?**           | - Visión por computador, NLP, secuencias, relaciones no lineales                                   |
| **¿Cuándo evitar NN?**         | - Pocos datos<br>- Necesidad alta de interpretabilidad<br>- Modelos tabulares simples              |
| **Técnicas de regularización** | Dropout, L2 (weight decay), batch normalization, early stopping                                    |
| **Diagnóstico común**          | - High bias → más neuronas/capas, cambiar activación<br>- High variance → regularización           |
| **Frameworks populares**       | TensorFlow, Keras, PyTorch                                                                         |

### 🧭 Sugerencias para la Toma de Decisiones

| **Situación**                                        | **Recomendación**                                                |
| ---------------------------------------------------- | ---------------------------------------------------------------- |
| Muchas features no lineales o datos no estructurados | Usa NN: puede aprender patrones complejos (imagen, texto, audio) |
| Alta precisión en clasificación con pocos datos      | Mejor usar modelos como SVM, Random Forest                       |
| Entrenamiento lento o sin GPU                        | Considera modelos más simples (Regresión, Árboles)               |
| Modelo con sobreajuste                               | Aplica dropout, L2, o early stopping                             |
| Predicción secuencial o multietiqueta compleja       | Usa arquitecturas específicas: RNN, LSTM, Transformers           |

## 📋 Transfer Learning

| **Aspecto**                | **Descripción / Detalles**                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Definición**             | Técnica que reutiliza un modelo preentrenado en una tarea diferente pero relacionada                         |
| **Objetivo**               | Aprovechar conocimiento existente para evitar entrenar modelos desde cero                                    |
| **Arquitecturas comunes**  | Imagen: VGG, ResNet, MobileNet, Inception, YOLO<br>Texto: BERT, GPT, RoBERTa                                 |
| **Tipos de transferencia** | - **Feature Extraction**: usar pesos congelados<br>- **Fine-Tuning**: reentrenar capas                       |
| **¿Cuándo usar?**          | - Datos limitados<br>- Tarea similar a dataset grande existente (ej. ImageNet, BERT)                         |
| **¿Cuándo evitar?**        | - Tarea muy diferente a la del modelo fuente<br>- Dominio altamente especializado                            |
| **Beneficios clave**       | - Ahorro computacional<br>- Menor necesidad de datos<br>- Mejores resultados iniciales                       |
| **Requerimientos**         | - Modelo base preentrenado<br>- Conjunto de datos etiquetados para tarea destino                             |
| **Desventajas**            | - Riesgo de **negative transfer** si dominios no coinciden<br>- Sensible a overfitting si no se congela bien |
| **Decisión entre métodos** | - **Pocos datos** → feature extraction<br>- **Datos medianos y similares** → fine-tuning                     |
| **Criterios para ajuste**  | - Reducir learning rate en fine-tuning<br>- Evitar sobreajuste al descongelar                                |
| **Usos populares**         | Clasificación de imágenes, análisis de sentimientos, detección de objetos, clasificación médica              |

### 🧭 Sugerencias para la Toma de Decisiones

| **Escenario**                                                 | **Decisión recomendada**                                    |
| ------------------------------------------------------------- | ----------------------------------------------------------- |
| Tienes pocos datos pero tarea similar (e.g. imagenes médicas) | Usa modelo preentrenado con **feature extraction**          |
| Datos moderados y dominio similar                             | Usa **fine-tuning**: descongela capas superiores            |
| Tu tarea es muy distinta (imagen → audio)                     | Transfer learning **no recomendado**                        |
| Problemas de sobreajuste en fine-tuning                       | Disminuir learning rate, congelar capas iniciales           |
| Requiere entrenamiento rápido y eficaz                        | Transfer learning con capas personalizadas y base congelada |

## 📋 k-Nearest Neighbors (k-NN)

| **Aspecto**                    | **Descripción / Detalles**                                                                                        |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| **Definición**                 | Algoritmo basado en memoria: clasifica o estima según los k puntos más cercanos                                   |
| **Tipo de modelo**             | No paramétrico, aprendizaje "perezoso" (lazy learning), sin entrenamiento explícito                               |
| **Función de decisión**        | - Clasificación: mayoría entre vecinos<br>- Regresión: promedio de los valores de vecinos                         |
| **Métricas de distancia**      | Euclidiana, Manhattan, Coseno, Pearson — según el dominio y escala                                                |
| **Preprocesamiento necesario** | Normalización/estandarización de características (para distancias significativas)                                 |
| **¿Cuándo usar?**              | - Dataset pequeño a mediano<br>- Relaciones locales fuertes<br>- Pocas dimensiones                                |
| **¿Cuándo evitar?**            | - Muchos atributos irrelevantes<br>- Alta dimensionalidad (curse of dimensionality)<br>- Datos escasos o ruidosos |
| **Ventajas**                   | Simple, sin suposiciones sobre la distribución, adaptable a problemas multiclase                                  |
| **Desventajas**                | Costoso en predicción, sensible a ruido y escalamiento, requiere almacenamiento completo                          |
| **Parámetro clave (k)**        | - k pequeño: alta varianza, más sensible<br>- k grande: más estable, pero puede suavizar demasiado                |
| **Pesado vs. No pesado**       | Ponderación por distancia mejora rendimiento en datasets con densidad variable                                    |
| **Uso en regresión**           | Promedio de valores de los k vecinos más cercanos                                                                 |
| **Uso en clasificación**       | Voto de mayoría entre clases de los k vecinos                                                                     |
### 🧭 Sugerencias para la Toma de Decisiones

| **Escenario**                            | **Decisión recomendada**                                         |
| ---------------------------------------- | ---------------------------------------------------------------- |
| Pocos datos y relaciones locales simples | Usa **k-NN con k=3 o 5**                                         |
| Muchas features no escaladas             | Aplica **escalado de datos** antes de usar k-NN                  |
| Datos ruidosos o outliers                | Usa **k más grande o versión ponderada**                         |
| Predicción lenta o muchos datos          | Considera modelos más rápidos: SVM, árboles, regresión logística |
| Altas dimensiones (>20-30 features)      | Aplica **PCA** o selección de características                    |

## 📋 Support Vector Machines (SVM)

| **Aspecto**                     | **Descripción / Detalles**                                                                                     |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Objetivo**                    | Encontrar el hiperplano óptimo que **maximiza el margen** entre clases                                         |
| **Tipo de modelo**              | Modelo discriminativo, **margin-based**, eficaz para clasificación lineal y no lineal                          |
| **Casos de uso típicos**        | Texto, imágenes, bioinformática, clasificación binaria                                                         |
| **¿Lineal o no lineal?**        | Lineal: cuando los datos son separables<br>No lineal: usa el **truco del kernel**                              |
| **Kernel Trick**                | Computa productos internos en espacio de alta dimensión sin mapear explícitamente                              |
| **Kernels comunes**             | - Lineal<br>- Polinomial<br>- RBF (Gaussian)<br>- Sigmoide                                                     |
| **Hiperparámetros clave**       | - **C**: penaliza errores (soft margin)<br>- **γ**: ancho de RBF (si se usa RBF)                               |
| **¿Cuándo usar?**               | - Dataset con pocas features<br>- Margen claro entre clases<br>- Datos no muy grandes                          |
| **¿Cuándo evitar?**             | - Datos ruidosos con muchos outliers<br>- Problemas multiclase complejos<br>- Escalabilidad                    |
| **Ventajas**                    | - Precisión alta<br>- Efectivo en espacios de alta dimensión<br>- Robusto al overfitting                       |
| **Desventajas**                 | - Costoso en entrenamiento<br>- No produce probabilidades directamente<br>- Difícil de ajustar para multiclase |
| **SVM vs. Regresión Logística** | SVM se enfoca en el margen geométrico, RL en la probabilidad y entropía cruzada                                |
| **Aplicación en regresión**     | **SVR**: minimiza un margen tolerante alrededor del valor real                                                 |

### 🧭 Sugerencias para la Toma de Decisiones

| **Situación**                                  | **Recomendación**                                                      |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| Pocos datos y gran separación entre clases     | Usa **SVM con kernel lineal**                                          |
| Clasificación no lineal con patrones complejos | Usa **kernel RBF o polinomial**                                        |
| Hay muchos outliers en los datos               | Ajusta el parámetro **C (menor valor)** para permitir margen suave     |
| Dataset muy grande (>10,000 muestras)          | Considera modelos más escalables: árboles, regresión logística         |
| Necesitas probabilidades                       | Usa regresión logística o calibración externa (Platt scaling para SVM) |

## 📋 Decision Trees

| **Aspecto**                   | **Descripción / Detalles**                                                                     |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| **Objetivo principal**        | Dividir el espacio de decisiones en regiones homogéneas mediante condiciones lógicas           |
| **Tipo de modelo**            | Modelo no paramétrico, interpretable, basado en reglas if-then                                 |
| **Aplicaciones comunes**      | Clasificación, regresión, segmentación, análisis exploratorio                                  |
| **Criterios de división**     | - Clasificación: **Entropía**, **Índice Gini**<br>- Regresión: **Varianza / MSE**              |
| **Componentes del árbol**     | - Nodo raíz, nodos internos (condiciones), hojas (predicciones)                                |
| **Ventajas**                  | - Muy interpretable<br>- No requiere escalado de features<br>- Maneja datos categóricos        |
| **Desventajas**               | - Sensible a pequeñas variaciones en los datos<br>- Propenso a **overfitting**                 |
| **¿Cuándo usar?**             | - Reglas de decisión claras<br>- Datos tabulares con relaciones no lineales                    |
| **¿Cuándo evitar?**           | - Datos ruidosos<br>- Requiere alta generalización sin ensemble                                |
| **Regularización**            | - Profundidad máxima (max\_depth)<br>- Número mínimo de muestras por nodo (min\_samples)       |
| **Detención del crecimiento** | - Cuando se alcanza pureza total<br>- Cuando no hay ganancia significativa de información      |
| **Mejoras comunes**           | Usar **pruning** (poda), **early stopping**, o emplear ensembles como Random Forest o Boosting |

### 🧭 Sugerencias para la Toma de Decisiones

| **Situación**                                     | **Decisión recomendada**                                                |
| ------------------------------------------------- | ----------------------------------------------------------------------- |
| Se requiere un modelo **altamente interpretable** | Usa Decision Tree con poda y profundidad limitada                       |
| Modelo sobreajusta mucho                          | Ajusta **max\_depth**, **min\_samples\_split**, o usa **Random Forest** |
| Problema con muchas features irrelevantes         | Considera selección de características o usar árbol con regularización  |
| ¿Métricas similares en divisiones posibles?       | Prefiere la que maximice **ganancia de información**                    |
| Clasificación binaria con clases balanceadas      | Usa **índice Gini** o **entropía**; ambos funcionan bien                |
