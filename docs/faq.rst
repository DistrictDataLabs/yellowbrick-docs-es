.. -*- mode: rst -*-

Preguntas Frecuentes
====================

Bienvenido a nuestra página de preguntas frecuentes. ¡Nos alegra que uses Yellowbrick! Si tu pregunta no la encuentras aquí, envíala a nuestro `Google Groups Listserv <https://groups.google.com/forum/#!forum/yellowbrick>`_. Esta es una lista de correo electrónico / foro al que usted, como usuario de Yellowbrick, puede unirse e interactuar con otros usuarios para abordar y solucionar problemas de Yellowbrick. Google Groups Listserv es donde puedes recibir la respuesta más rápida. ¡Le damos la bienvenida y te alentamos a que te unas al grupo para que puedas responder a las preguntas de los demás! También puede hacer preguntas en `Stack Overflow <http://stackoverflow.com/questions/tagged/yellowbrick>`_ y coloca la etiqueta "yellowbrick". Finalmente, puedes agregar problemas en GitHub y puedes twittear o enviarnos un mensaje en Twitter `@scikit_yb <https://twitter.com/scikit_yb>`_.


¿Cómo puedo cambiar el tamaño de una gráfica en Yellowbrick?
------------------------------------------------------------

Puedes cambiar el ``size`` de una gráfica pasando las dimensiones deseadas en pixeles en la instancia del visualizador:

.. code:: python

    # Importa el visualizador
    from yellowbrick.features import RadViz

    # Crear una instancia del visualizador usando el parámetro ``size``
    visualizer = RadViz(
        classes=classes, features=features, size=(1080, 720)
    )

    ...


Nota: estamos considerando agregar soporte para cambiar el ``size`` en pulgadas en una futura versión de Yellowbrick. Si necesitas un conveniente convertidor de pulgada a píxel, echa un vistazo a `www.unitconversion.org <http://www.unitconversion.org/typography/inchs-to-pixels-y-conversion.html>`_.

¿Cómo puedo cambiar el título de una trama de Yellowbrick?
----------------------------------------------------------

Puedes cambiar el ``title`` de una gráfica colocando el título deseado como una cadena de caracteres en la creación de instancias:


.. code:: python

    from yellowbrick.classifier import ROCAUC
    from sklearn.linear_model import RidgeClassifier

    # Crea tu título personalizado
    my_title = "ROCAUC Curves for Multiclass RidgeClassifier"

    # Crea una instancia del visualizador colocando el título personalizado
    visualizer = ROCAUC(
        RidgeClassifier(), classes=classes, title=my_title
    )

    ...



¿Cómo puedo cambiar el color de una gráfica de Yellowbrick?
-----------------------------------------------------------

Yellowbrick utiliza colores para hacer que los visualizadores sean lo más interpretables posible para tener diagnósticos intuitivos de aprendizaje automático. Generalmente, el color se especifica por la variable objetivo, ``y`` que puede pasar a un método de ajuste de un estimador. Por lo tanto, Yellowbrick considera el color en función del tipo de datos de la variable objetivo:

- **Discreto**: cuando el objetivo está representado por clases discretas, Yellowbrick utiliza colores categóricos que son fáciles de discriminar entre sí.
- **Continuo**: cuando el objetivo está representado por valores continuos, Yellowbrick utiliza un mapa de colores secuencial para mostrar el rango de datos.

*La mayoría* de los visualizadores aceptan los argumentos ``colors`` y ``colormap`` cuando se inicializan. En términos generales, si la variable objetivo es discreta, especifique ``colors`` como una lista de colores matplotlib válidos; de lo contrario, si la variable objetivo es continua, especifique un mapa de colores o el nombre de un mapa de colores matplotlib. La mayoría de los visualizadores de Yellowbrick son lo suficientemente inteligentes como para elegir los colores para cada uno de sus puntos de datos en función de lo que colocas; por ejemplo, si colocas un mapa de colores para una variable objetivo discreta, el visualizador creará una lista de colores discretos a partir de los colores secuenciales.

.. note:: Aunque la mayoría de los visualizadores admiten estos argumentos, asegúrese de verificar el visualizador, ya que puede tener requisitos de color específicos. Por ejemplo, el :doc:`ResidualsPlot <api/regressor/residuals>` acepta el ``train_color``, ``test_color``, y el ``line_color`` para modificar su visualización. Para ver los argumentos de un visualizador, puedes usar ``help(Visualizer)`` o ``visualizer.get_params()``.


Por ejemplo, el :doc:`Manifold <api/features/manifold>` puede visualizar variables objetivos discretas y secuenciales. En el caso de la discreta, coloque una lista de `valores de color válidos <https://matplotlib.org/api/colors_api.html>`_ de la siguiente manera:


.. code:: python

    from yellowbrick.features.manifold import Manifold

    visualizer = Manifold(
        manifold="tsne", target="discrete", colors=["teal", "orchid"]
    )

    ...


... mientras que para las variables objetivos``continuous`, es mejor especificar un `matplotlib colormap <https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html>`_:


.. code:: python

    from yellowbrick.features.manifold import Manifold

    visualizer = Manifold(
        manifold="isomap", target="continuous", colormap="YlOrRd"
    )

    ...


Finalmente, tenga en cuenta que puede manipular los colores predeterminados que utiliza Yellowbrick modificando los `matplotlib styles <https://matplotlib.org/3.1.1/tutorials/introductory/customizing.html>`_, particularmente el ciclo de color predeterminado. Yellowbrick también tiene herramientas para la gestión de estilos, consulte :doc:`api/palettes` para obtener más información.


¿Cómo puedo guardar una gráfica de Yellowbrick?
-----------------------------------------------

Guarda tu gráfica de Yellowbrick colocando un ``outpath`` en ``show()``:

.. code:: python

    from sklearn.cluster import MiniBatchKMeans
    from yellowbrick.cluster import KElbowVisualizer

    visualizer = KElbowVisualizer(MiniBatchKMeans(), k=(4,12))

    visualizer.fit(X)
    visualizer.show(outpath="kelbow_minibatchkmeans.png")

    ...


¡La mayoría de los backends admiten png, pdf, ps, eps y svg para guardar tu trabajo!


¿Cómo puedo hacer que los puntos superpuestos se muestren mejor?
----------------------------------------------------------------

Puedes usar el parámetro ``alpha`` para cambiar la opacidad de los puntos trazados (donde ``alpha=1`` es opacidad completa y ``alpha=0`` es transparencia completa):

.. code:: python

    from yellowbrick.contrib.scatter import ScatterVisualizer

    visualizer = ScatterVisualizer(
        x="light", y="C02", classes=classes, alpha=0.5
    )


¿Cómo puedo acceder a los conjuntos de datos de muestra utilizados en los ejemplos?
-----------------------------------------------------------------------------------

Visita la página :doc:`api/datasets/index`.


¿Puedo usar Yellowbrick con librerías que no sean scikit-learn?
---------------------------------------------------------------

¡Potencialmente! Los visualizadores de Yellowbrick se basan en el modelo interno que implementa la API scikit-learn (por ejemplo, tener un método ``fit()`` y ``predict()``), y a menudo se espera poder recuperar los atributos aprendidos del modelo (por ejemplo, ``coef_``). Algunos estimadores de terceros implementan completamente la API scikit-learn, pero no todos.

Cuando utilice bibliotecas de terceros con Yellowbrick, le recomendamos, ``wrap`` el modelo con el módulo ``yellowbrick.contrib.wrapper``. ¡Visita la página :doc:`api/contrib/wrapper` para obtener todos los detalles!
