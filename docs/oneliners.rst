.. -*- mode: rst -*-

Los "Oneliners"
===============

¡Los métodos rápidos de Yellowbrick son visualizadores de una sola línea de código!

Yellowbrick está diseñado para darte tanto control como desees sobre las gráficas que crees, ofreciendo parámetros para ayudarte a personalizar todo, desde el color, el tamaño y el título hasta la evaluación preferida o la medida de correlación, las líneas de mejor ajuste opcionales o histogramas y las técnicas de validación cruzada. Para obtener más información sobre cómo personalizar sus visualizaciones utilizando esos parámetros, consulte :doc:`api/index`.

Pero... ¡a veces solo quieres construir un gráfico con una sola línea de código!

En esta página exploraremos los métodos rápidos de Yellowbrick (también conocidos como "oneliners"), que devuelven un objeto visualizador completamente ajustado y listo en una sola línea.

.. note:: Esta página ilustra oneliners para algunos de nuestros visualizadores más populares para análisis de características, clasificación, regresión, agrupación y evaluación de objetivos, pero no es una lista completa. ¡Casi todos los visualizadores de Yellowbrick tienen un método rápido asociado!

Análisis de características
---------------------------

Rank2D
~~~~~~


Los gráficos ``rank1d`` y ``rank2d`` muestran clasificaciones por pares o características para ayudarte a detectar relaciones. Conoce más en:doc:`api/features/rankd`.

.. plot::
    :context: close-figs
    :alt: Rank2D Quick Method

    from yellowbrick.features import rank2d
    from yellowbrick.datasets import load_credit


    X, _ = load_credit()
    visualizer = rank2d(X)

.. plot::
    :context: close-figs
    :alt: Rank1D Quick Method

    from yellowbrick.features import rank1d
    from yellowbrick.datasets import load_energy


    X, _ = load_energy()
    visualizer = rank1d(X, color="r")


Coordenadas paralelas
~~~~~~~~~~~~~~~~~~~~~

La gráfica ``parallel_coordinates`` es una visualización horizontal de instancias, clasificadas por las características que las describen. Conoce más en :doc:`api/features/pcoords`.

.. plot::
    :context: close-figs
    :alt: Parallel Coordinates Quick Method

    from sklearn.datasets import load_wine
    from yellowbrick.features import parallel_coordinates


    X, y = load_wine(return_X_y=True)
    visualizer = parallel_coordinates(X, y, normalize="standard")


Visualización radial
~~~~~~~~~~~~~~~~~~~~

La gráfica ``radviz`` muestra la separación de instancias alrededor de un círculo unitario. Conoce más en :doc:`api/features/radviz`.

.. plot::
    :context: close-figs
    :alt: Radviz Quick Method

    from yellowbrick.features import radviz
    from yellowbrick.datasets import load_occupancy


    X, y = load_occupancy()
    visualizer = radviz(X, y, colors=["maroon", "gold"])


PCA
~~~

La ``pca_decomposition`` es una proyección de instancias basada en componentes principales. Conoce más en :doc:`api/features/pca`.

.. plot::
    :context: close-figs
    :alt: PCA Quick Method

    from yellowbrick.datasets import load_spam
    from yellowbrick.features import pca_decomposition


    X, y = load_spam()
    visualizer = pca_decomposition(X, y)


Múltiple
~~~~~~~~

La gráfica ``manifold_embedding`` es una visualización de alta dimensión con aprendizaje múltiple, que puede mostrar relaciones no lineales en las características. Conoce más en :doc:`api/features/manifold`.

.. plot::
    :context: close-figs
    :alt: Manifold Quick Method

    from sklearn.datasets import load_iris
    from yellowbrick.features import manifold_embedding


    X, y = load_iris(return_X_y=True)
    visualizer = manifold_embedding(X, y)


Clasificación
-------------

Error de predicción de clase
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

El gráfico ``class_prediction_error`` ilustra el error y la ayuda en una clasificación como un gráfico de barras. Conoce más en :doc:`api/classifier/class_prediction_error`.

.. plot::
    :context: close-figs
    :alt: Class Prediction Error Quick Method

    from yellowbrick.datasets import load_game
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from yellowbrick.classifier import class_prediction_error


    X, y = load_game()
    X = OneHotEncoder().fit_transform(X)
    visualizer = class_prediction_error(
        RandomForestClassifier(n_estimators=10), X, y
    )


Informe de clasificación
~~~~~~~~~~~~~~~~~~~~~~~~

El ``classification_report`` es una representación visual de la precisión, recall, y la puntuación F1. Conoce más en :doc:`api/classifier/classification_report`.

.. plot::
    :context: close-figs
    :alt: Classification Report Quick Method

    from yellowbrick.datasets import load_credit
    from sklearn.ensemble import RandomForestClassifier
    from yellowbrick.classifier import classification_report


    X, y = load_credit()
    visualizer = classification_report(
        RandomForestClassifier(n_estimators=10), X, y
    )


Matriz de confusión
~~~~~~~~~~~~~~~~~~~

La ``confusion_matrix`` es una descripción visual de la toma de decisiones por clase. Conoce más en :doc:`api/classifier/confusion_matrix`.

.. plot::
    :context: close-figs
    :alt: Confusion Matrix Quick Method

    from yellowbrick.datasets import load_game
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import RidgeClassifier
    from yellowbrick.classifier import confusion_matrix


    X, y = load_game()
    X = OneHotEncoder().fit_transform(X)
    visualizer = confusion_matrix(RidgeClassifier(), X, y, cmap="Greens")


Precisión en el Recall
~~~~~~~~~~~~~~~~~~~~~~

La ``precision_recall_curve`` muestra la compensación entre la precisión y el recall para diferentes umbrales de probabilidad. Conoce más en :doc:`api/classifier/prcurve`.

.. plot::
    :context: close-figs
    :alt: Precision Recall Quick Method

    from sklearn.naive_bayes import GaussianNB
    from yellowbrick.datasets import load_occupancy
    from yellowbrick.classifier import precision_recall_curve


    X, y = load_occupancy()
    visualizer = precision_recall_curve(GaussianNB(), X, y)


ROCAUC
~~~~~~

El gráfico ``roc_auc`` muestra las características del operador del receptor y el área debajo de la curva. Conoce más en:doc:`api/classifier/rocauc`.

.. plot::
    :context: close-figs
    :alt: ROCAUC Quick Method

    from yellowbrick.classifier import roc_auc
    from yellowbrick.datasets import load_spam
    from sklearn.linear_model import LogisticRegression


    X, y = load_spam()
    visualizer = roc_auc(LogisticRegression(), X, y)


Umbral de discriminación
~~~~~~~~~~~~~~~~~~~~~~~~

El gráfico ``discrimination_threshold`` puede ayudar a encontrar un umbral que mejor separe las clases binarias. Conoce más en :doc:`api/classifier/threshold`.

.. plot::
    :context: close-figs
    :alt: Discrimination Threshold Quick Method

    from yellowbrick.classifier import discrimination_threshold
    from sklearn.linear_model import LogisticRegression
    from yellowbrick.datasets import load_spam

    X, y = load_spam()
    visualizer = discrimination_threshold(
        LogisticRegression(multi_class="auto", solver="liblinear"), X, y
    )


Regresión
---------

Gráfico de residuos
~~~~~~~~~~~~~~~~~~~

El ``residuals_plot`` muestra la diferencia en los residuos entre los datos de entrenamiento y prueba. Conoce más en :doc:`api/regressor/residuals`.

.. plot::
    :context: close-figs
    :alt: Residuals Quick Method

    from sklearn.linear_model import Ridge
    from yellowbrick.datasets import load_concrete
    from yellowbrick.regressor import residuals_plot


    X, y = load_concrete()
    visualizer = residuals_plot(
        Ridge(), X, y, train_color="maroon", test_color="gold"
    )

Error de predicción
~~~~~~~~~~~~~~~~~~~

El ``prediction_error`` ayuda a encontrar dónde la regresión está cometiendo la mayoría de los errores. Conoce más en :doc:`api/regressor/peplot`.

.. plot::
    :context: close-figs
    :alt: Prediction Error Quick Method

    from sklearn.linear_model import Lasso
    from yellowbrick.datasets import load_bikeshare
    from yellowbrick.regressor import prediction_error


    X, y = load_bikeshare()
    visualizer = prediction_error(Lasso(), X, y)


Distancia de Cook
~~~~~~~~~~~~~~~~~

El gráfico ``cooks_distance`` muestra la influencia de las instancias en la regresión lineal. Conoce más en :doc:`api/regressor/influence`.

.. plot::
    :context: close-figs
    :alt: Cooks Distance Quick Method

    from sklearn.datasets import load_diabetes
    from yellowbrick.regressor import cooks_distance


    X, y = load_diabetes(return_X_y=True)
    visualizer = cooks_distance(X, y)


Agrupamiento
------------

Valores de la silueta
~~~~~~~~~~~~~~~~~~~~~

El ``silhouette_visualizer`` puede ayudarte a seleccionar ``k`` visualizando los valores del coeficiente de la silueta. Conoce más en :doc:`api/cluster/silhouette`.

.. plot::
    :context: close-figs
    :alt: Silhouette Scores Quick Method

    from sklearn.cluster import KMeans
    from yellowbrick.datasets import load_nfl
    from yellowbrick.cluster import silhouette_visualizer

    X, y = load_nfl()
    visualizer = silhouette_visualizer(KMeans(5, random_state=42), X)


Distancia entre grupos
~~~~~~~~~~~~~~~~~~~~~~

La ``intercluster_distance`` muestra el tamaño y la distancia relativa entre los grupos. Conoce más en :doc:`api/cluster/icdm`.

.. plot::
    :context: close-figs
    :alt: ICDM Quick Method

    from yellowbrick.datasets import load_nfl
    from sklearn.cluster import MiniBatchKMeans
    from yellowbrick.cluster import intercluster_distance


    X, y = load_nfl()
    visualizer = intercluster_distance(MiniBatchKMeans(5, random_state=777), X)


Análisis de objetivos
---------------------

ClassBalance
~~~~~~~~~~~~

El gráfico ``class_balance`` puede hacer que sea más fácil ver cómo la distribución de clases puede afectar al modelo. Conoce más en :doc:`api/target/class_balance`.

.. plot::
    :context: close-figs
    :alt: ClassBalance Quick Method

    from yellowbrick.datasets import load_game
    from yellowbrick.target import class_balance


    X, y = load_game()
    visualizer = class_balance(y, labels=["draw", "loss", "win"])

