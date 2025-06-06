# Aprendizaje No Supervisado

| **Tópico**                          | **Concepto / Definición**                                                          | **Casos de Uso**                                                        | **Ejemplos**                                                             | **Guía para Toma de Decisiones**                                                          |
| ----------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| **Aprendizaje Supervisado (SL)**    | Algoritmos entrenados con datos etiquetados para tareas específicas.               | Clasificación, regresión, visión por computador.                        | Clasificar imágenes de gatos vs. perros.                                 | Úsalo cuando tienes una gran cantidad de datos etiquetados y una tarea bien definida.     |
| **Aprendizaje No Supervisado (UL)** | Descubre patrones ocultos en datos no etiquetados.                                 | Segmentación, reducción de dimensionalidad, detección de anomalías.     | Cluster de clientes, PCA en genómica.                                    | Ideal cuando no se cuenta con etiquetas. Bueno para exploración inicial y descubrimiento. |
| **Sistemas Basados en Reglas**      | Toman decisiones con reglas “if-then”.                                             | Automatización simple, control de procesos.                             | Filtro de spam con palabras clave como “BUY NOW”.                        | Útiles para problemas estáticos, con lógica clara. No escalan ni se adaptan fácilmente.   |
| **SL vs UL**                        | SL requiere etiquetas; UL no. SL optimiza funciones de error; UL busca estructura. | SL: Diagnóstico médico. UL: Exploración de datos médicos sin etiquetas. | SL: clasificar tipo de cáncer. UL: detectar grupos de tumores similares. | Si el problema es abierto y exploratorio, usa UL. Si es claro y medible, SL.              |
| **Ventajas de SL**                  | Precisión en tareas específicas. Buenas métricas de evaluación.                    | Reconocimiento de imágenes, NLP.                                        | Clasificar sentimientos en reseñas.                                      | Requiere muchos datos etiquetados; mejor en tareas acotadas.                              |
| **Ventajas de UL**                  | Generaliza, detecta nuevas estructuras. Ahorra en etiquetado.                      | Segmentación de clientes, compresión de datos, agrupamiento genético.   | Identificar nuevos segmentos de mercado.                                 | Mejor en ambientes con alta variabilidad o desconocimiento del dominio.                   |
| **Clustering**                      | Agrupamiento de objetos similares.                                                 | Segmentación de mercado, análisis de comportamiento.                    | K-means para agrupar clientes de un retail.                              | Usar cuando se requiere explorar la estructura subyacente de los datos.                   |
| **Reducción de Dimensionalidad**    | Representar datos en menos dimensiones manteniendo estructura relevante.           | Visualización, preprocesamiento.                                        | PCA en imágenes, t-SNE en texto.                                         | Útil para visualización y preprocesamiento de datos con muchas variables.                 |
| **Autoencoders**                    | Redes neuronales que aprenden compresión y reconstrucción de datos.                | Detección de anomalías, compresión, reducción de ruido.                 | Detectar fraudes en tarjetas de crédito.                                 | Útil si se requiere generar representaciones compactas o preentrenar modelos.             |
| **UL para mejorar ML**              | Usar UL para preentrenar, detectar outliers, reducir dimensiones.                  | Aumentar generalización, reducir overfitting.                           | Preprocesar datos con PCA antes de regresión logística.                  | Cuando los modelos SL sobreajustan o hay muchos outliers, aplicar UL como etapa previa.   |
| **Detección de Anomalías**          | Identificación de instancias que se desvían del patrón normal.                     | Fraude, mantenimiento predictivo, ciberseguridad.                       | Transacción de tarjeta atípica.                                          | Útil cuando se desea vigilar comportamientos raros sin necesidad de etiquetas.            |
| **Datos secuenciales con UL**       | Aplicación de UL a series de tiempo o secuencias.                                  | Análisis de sensores, logs, salud.                                      | Series temporales de sensores de maquinaria.                             | Ideal cuando no se dispone de etiquetas pero se busca detectar segmentos o anomalías.     |
| **Semi-supervisado**                | Combina datos etiquetados y no etiquetados.                                        | Clasificación con pocos datos.                                          | Etiquetar con clusters y refinar con SL.                                 | Muy útil cuando etiquetar es costoso pero se tiene una pequeña muestra etiquetada.        |

## Consejos Prácticos
| **Situación**                            | **Recomendación**                              |
| ---------------------------------------- | ---------------------------------------------- |
| Pocos datos y sin etiquetas              | Unsupervised Learning (clustering, PCA)        |
| Muchos datos etiquetados                 | Supervised Learning (regresión, clasificación) |
| Mezcla de ambos                          | Semi-supervised Learning                       |
| Alta dimensionalidad                     | Reducción de dimensionalidad (PCA, t-SNE)      |
| Análisis de comportamiento extraño       | Detección de anomalías                         |
| Necesidad de entender estructura interna | Clustering o autoencoders                      |

Segmentación de clientes: makreting
Detección de Anomalías: transacciones comportamientos anomalos
Recomendaciones de productos: Contenido Streaming
Agrupación de documentos: Biblioteca (Clusterización natural)
Bioinformatica: Agrupar genes
Reducción de Dimensionalidad
Identificación de patrones en datos de sensores: Detección de anomlías


| **Criterio**                          | **Rules-Based Systems**                                            | **Machine Learning (ML)**                                                                      |
| ------------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| **Definición**                        | Sistema con reglas explícitas codificadas manualmente (if-then).   | Algoritmos que aprenden patrones a partir de datos.                                            |
| **Fuente de conocimiento**            | Conocimiento humano explícito convertido en reglas.                | Conocimiento inferido automáticamente desde los datos.                                         |
| **Adaptabilidad**                     | Baja. Requiere reprogramar ante cambios.                           | Alta. Aprende y se adapta con nuevos datos.                                                    |
| **Escalabilidad**                     | Difícil de escalar con gran cantidad de reglas o variables.        | Escalable con grandes volúmenes de datos.                                                      |
| **Complejidad que puede manejar**     | Limitada a casos simples y estructurados.                          | Capaz de modelar relaciones complejas y no lineales.                                           |
| **Costo de mantenimiento**            | Alto: requiere expertos para modificar reglas.                     | Menor: requiere reentrenamiento, no reprogramación.                                            |
| **Transparencia (interpretabilidad)** | Alta: decisiones rastreables a reglas específicas.                 | Variable: algunos modelos (como árboles) son interpretables, otros no (como redes neuronales). |
| **Requisitos de datos**               | No necesita datos de entrenamiento.                                | Requiere datos (etiquetados o no) para entrenar modelos.                                       |
| **Ejemplo típico**                    | Filtro de spam que detecta palabras clave como “BUY NOW”.          | Clasificador de spam entrenado con miles de correos etiquetados.                               |
| **Ventajas**                          | - Simplicidad <br> - Control total <br> - Transparencia            | - Adaptabilidad <br> - Detección de patrones <br> - Eficiencia en tareas complejas             |
| **Desventajas**                       | - Difícil de actualizar <br> - No escala bien <br> - Poco flexible | - Requiere datos de calidad <br> - Puede ser una caja negra <br> - Costo computacional alto    |
| **Aplicaciones comunes**              | - Validación de formularios <br> - Automatización de tareas fijas  | - Reconocimiento facial <br> - Recomendadores <br> - Detección de fraudes                      |
| **Mejor cuando...**                   | ...el entorno es estático y las reglas son claras.                 | ...hay muchos datos y el entorno cambia o es incierto.                                         |

---

