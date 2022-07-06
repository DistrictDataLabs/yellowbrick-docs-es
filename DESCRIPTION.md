# Yellowbrick

[![Visualizers](https://github.com/DistrictDataLabs/yellowbrick/raw/develop/docs/images/readme/banner.png)](https://www.scikit-yb.org/)

Yellowbrick es un conjunto de herramientas de análisis visual y diagnóstico diseñadas para facilitar el machine learning con scikit-learn. La biblioteca implementa un nuevo objeto de API central, el `Visualizer` que es un estimador scikit-learn &mdash; un objeto que aprende de los datos. Al igual que los transformadores o modelos, los visualizadores aprenden de los datos creando una representación visual del flujo de trabajo de sección del modelo.

Los visualizadores permiten a los usuarios dirigir el proceso de selección de modelos, creando intuición en torno a la ingeniería de características, la selección de algoritmos y el ajuste de hiperparámetros. Por ejemplo, pueden ayudar a diagnosticar problemas comunes relacionados con la complejidad y el sesgado del modelo, la heterocedasticidad, el bajo ajuste y el sobreentrenamiento, o los problemas de equilibrio de clases. Al aplicar visualizadores al flujo de trabajo de selección de modelos, Yellowbrick le permite dirigir los modelos predictivos hacia resultados más exitosos, más rápido.

La documentación completa se puede encontrar en [scikit-yb.org](https://scikit-yb.org/) e incluye una [Guía de inicio rápido](https://www.scikit-yb.org/en/latest/quickstart.html) para nuevos usuarios.

## Visualizadores

Los visualizadores son estimadores &mdash; objetos que aprenden de los datos &mdash; cuyo objetivo principal es crear visualizaciones que permitan conocer el proceso de selección del modelo. En términos de scikit-learn, pueden ser similares a los transformadores al visualizar el espacio de datos o envolver un estimador de modelo similar a cómo funcionan los métodos `ModelCV` (e.g. [`RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html), [`LassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)). El objetivo principal de Yellowbrick es crear una API sensorial similar a scikit-learn. Algunos de nuestros visualizadores más populares incluyen:

### Visualización de clasificación

- **Classification Report**: un informe de clasificación visual que muestra la precisión, el recall y las puntuaciones de F1 por clase de un modelo como un mapa de calor
- **Confusion Matrix**: una vista de mapa de calor de la matriz de confusión de pares de clases en la clasificación multiclase
- **Discrimination Threshold**: una visualización de la precisión, el recall, la puntuación F1 y la cola con respecto al umbral de discriminación de un clasificador binario
- **Precision-Recall Curve**: traza las puntuaciones de precisión vs recall para diferentes umbrales de probabilidad
- **ROCAUC**: grafica la característica del operador receptor (ROC) y el área bajo la curva (AUC)

### Visualización de agrupaciones

- **Intercluster Distance Maps**: visualiza la distancia relativa y el tamaño de los grupos
- **KElbow Visualizer**: visualiza la agrupación según la función de puntuación especificada, buscando el "codo" en la curva.
- **Silhouette Visualizer**: selecciona `k` visualizando las puntuaciones del coeficiente de silueta de cada grupo en un solo modelo

### Visualización de características

- **Manifold Visualization**: visualización de alta dimensión con aprendizaje de variedades
- **Parallel Coordinates**: visualización horizontal de instancias
- **PCA Projection**: proyección de instancias basadas en componentes principales
- **RadViz Visualizer**: separación de instancias alrededor de una gráfica circular
- **Rank Features**: clasificación de características individuales o por pares para detectar relaciones

### Visualización de la selección del modelo

- **Cross Validation Scores**: muestra las puntuaciones validadas cruzadamente como un gráfico de barras con la puntuación media trazada como una línea horizontal
- **Feature Importances**: clasifica las características en función de su rendimiento en el modelo
- **Learning Curve**: muestra si un modelo podría beneficiarse de más datos o menos complejidad
- **Recursive Feature Elimination**: encuentra el mejor subconjunto de características en función de la importancia
- **Validation Curve**: ajusta un modelo con respecto a un solo hiperparámetro

### Visualización de regresión

- **Alpha Selection**: muestra cómo la elección del alfa influye en la regularización
- **Cook's Distance**: muestra la influencia de las instancias en la regresión lineal
- **Prediction Error Plots**: encuentra interrupciones de modelos a lo largo del dominio del objetivo
- **Residuals Plot**: muestra la diferencia en los residuos de los datos de entrenamiento y prueba

### Visualización de objetivos

- **Balanced Binning Reference**: genera un histograma con líneas verticales que muestren el punto de valor recomendado a los datos binarios distribuidos uniformemente
- **Class Balance**: muestra la relación del soporte para cada clase tanto en los datos de entrenamiento como de prueba mostrando la frecuencia con la que se produce cada clase como un gráfico de barras, la frecuencia de la representación de las clases en el conjunto de datos
- **Feature Correlation**: visualiza la correlación entre las variables dependientes y el objetivo

### Visualización de texto

- **Dispersion Plot**: visualiza cómo se dispersan los términos clave a lo largo de un cuerpo
- **PosTag Visualizer**: traza los recuentos de diferentes partes del habla a lo largo de un cuerpo etiquetado
- **Token Frequency Distribution**: visualizar la distribución de frecuencias de los términos en el cuerpo
- **t-SNE Corpus Visualization**: utiliza la incrustación estocástica de vecinos en los documentos del proyecto
- **UMAP Corpus Visualization**: traza documentos similares más juntos para descubrir agrupaciones

... ¡y más! Yellowbrick está agregando nuevos visualizadores todo el tiempo, así que asegúrate de consultar nuestra [galería de ejemplos](https://github.com/DistrictDataLabs/yellowbrick/tree/develop/examples) &mdash; o incluso la rama [desarrollar](https://github.com/districtdatalabs/yellowbrick/tree/develop) &mdash; ¡y siéntete libre de contribuir con tus ideas para nuevos Visualizadores!

## Afiliaciones
[![District Data Labs](https://github.com/DistrictDataLabs/yellowbrick/raw/develop/docs/images/readme/affiliates_ddl.png)](https://www.districtdatalabs.com/) [![NumFOCUS Affiliated Project](https://github.com/DistrictDataLabs/yellowbrick/raw/develop/docs/images/readme/affiliates_numfocus.png)](https://numfocus.org/)
