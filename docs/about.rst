Acerca de nosotros
==================

.. image:: images/yellowbrickroad.jpg

Imagen de QuatroCinco_, usada con permiso de Flickr Creative Commons.

Yellowbrick es un proyecto Python puro de código abierto que extiende el API_ scikit-learn con herramientas de análisis visual y diagnóstico. La API de Yellowbrick también envuelve matplotlib para crear figuras listas para su publicación y exploraciones de datos interactivas, al mismo tiempo que permite a los desarrolladores un control detallado de las figuras. Para los usuarios, Yellowbrick puede ayudar a evaluar el rendimiento, la estabilidad y el valor predictivo de los modelos de machine learning y ayudar a diagnosticar problemas a lo largo del flujo de trabajo de machine learning.

Recientemente, gran parte de este flujo de trabajo se ha automatizado a través de métodos de búsqueda en cuadrícula, API estandarizadas y aplicaciones basadas en GUI. En la práctica, sin embargo, la intuición y la orientación humanas pueden perfeccionar los modelos de calidad de manera más efectiva que la búsqueda exhaustiva. Al visualizar el proceso de selección de modelos, los datos científicos pueden dirigirse hacia modelos finales y explicables y evitar dificultades y trampas.

La biblioteca Yellowbrick es una plataforma de visualización de diagnóstico para el machine learning que permite a los científicos de datos dirigir el procedimiento de selección de modelos. Amplía la API scikit-learn con un nuevo objeto central: el Visualizador. Los visualizadores permiten que los modelos visuales se ajusten y transformen como parte del proceso de canalización scikit-learn, proporcionando diagnósticos visuales a lo largo de la transformación de datos de alta dimensión.

Selección de modelos
--------------------
Las discusiones sobre el machine learning se caracterizan con frecuencia por un enfoque singular en la selección de modelos. Ya sea regresión logística, bosques aleatorios, métodos bayesianos o redes neuronales artificiales, los profesionales del machine learning a menudo se apresuran a expresar sus preferencias. La razón de esto es principalmente histórica. Aunque las bibliotecas modernas de machine learning de terceros han hecho que la implementación de múltiples modelos parezca casi trivial, tradicionalmente la aplicación y el desarrollo de incluso uno de estos algoritmos, los cuales requirieron muchos años de estudio. Como resultado, los profesionales del machine learning tendían a tener fuertes preferencias por modelos particulares (y probablemente más familiares) sobre otros.

Sin embargo, la selección de modelos es un poco más matizada que simplemente elegir el algoritmo "correcto" o "incorrecto". En la práctica, el flujo de trabajo incluye:

  1. Seleccionar y/o diseñar el conjunto de características más pequeño y predictivo
  2. Elegir un conjunto de algoritmos de una familia de modelos
  3. Ajuste de los hiperparámetros del algoritmo para optimizar el rendimiento

La **triple de selección de modelos** fue descrita por primera vez en un artículo 2015 SIGMOD_ por Kumar y otros. En su artículo, que se refiere al desarrollo de sistemas de bases de datos de próxima generación construidos para anticipar el modelado predictivo, los autores expresan convincentemente que tales sistemas son muy necesarios debido a la naturaleza altamente experimental del machine learning en la práctica. "La selección de modelos", explican, "es iterativa y exploratoria porque el espacio de [triples de selección de modelos] suele ser infinito, y generalmente es imposible para los analistas saber a priori qué [combinación] producirá una precisión y/o conocimientos satisfactorios".


¿Para quién es Yellowbrick?
---------------------------

Los ``Visualizers`` Yellowbrick tienen múltiples casos de uso:

 - Para los científicos de datos, pueden ayudar a evaluar la estabilidad y el valor predictivo de los modelos de machine learning y mejorar la velocidad del flujo de trabajo experimental.
 - Para los ingenieros de datos, Yellowbrick proporciona herramientas visuales para monitorear el rendimiento del modelo en aplicaciones del mundo real.
 - Para los usuarios de modelos, Yellowbrick proporciona una interpretación visual del comportamiento del modelo en el espacio de características de alta dimensión.
 - Para profesores y estudiantes, Yellowbrick es un marco para la enseñanza y la comprensión de una gran variedad de algoritmos y métodos.


Origen del nombre
-----------------
El paquete Yellowbrick recibe su nombre del elemento ficticio en la novela infantil de 1900 **El maravilloso mago de Oz** del autor estadounidense L. Frank Baum. En el libro, el camino de ladrillo amarillo es el camino que la protagonista, Dorothy Gale, debe recorrer para llegar a su destino en la Ciudad Esmeralda.

Desde Wikipedia_:
    "El camino se introduce por primera vez en el tercer capítulo de El maravilloso mago de Oz. El camino comienza en el corazón del cuadrante oriental llamado Munchkin Country en la Tierra de Oz. Funciona como una guía que lleva a todos los que la siguen, al destino final de la carretera: la capital imperial de Oz llamada Ciudad Esmeralda que se encuentra en el centro exacto de todo el continente. En el libro, la protagonista principal de la novela, Dorothy, se ve obligada a buscar el camino antes de que pueda comenzar su búsqueda para buscar al Mago. Esto se debe a que el ciclón de Kansas no soltó su granja cerca de ella como lo hizo en las diversas adaptaciones cinematográficas. Después del concilio con los nativos Munchkins y su querida amiga la Bruja Buena del Norte, Dorothy comienza a buscarla y ve muchos caminos y caminos cercanos, (todos los cuales conducen en varias direcciones). Afortunadamente, no le lleva demasiado tiempo ver el pavimentado con ladrillos amarillos brillantes".

Equipo
------

Yellowbrick está desarrollado por científicos de datos voluntarios que creen en el código abierto y el proyecto disfruta de contribuciones de desarrolladores de Python de todo el mundo. El Proyecto fue iniciado por `@rebeccabilbro`_ y `@bbengfort`_ como un intento de explicar mejor los conceptos de machine learning a sus estudiantes en la Universidad de Georgetown, donde enseñan un programa de certificación de ciencia de datos. Sin embargo, rápidamente se dieron cuenta de que el potencial para la dirección visual podría tener un gran impacto en la ciencia de datos práctica y lo desarrollaron en una biblioteca python lista para la producción.

Yellowbrick fue incubado por District Data Labs (DDL) en asociación con la Universidad de Georgetown. District Data Labs es una organización que se dedica al desarrollo de código abierto y la educación en ciencia de datos y proporcionó recursos para ayudar a Yellowbrick a crecer. Yellowbrick se introdujo por primera vez en la comunidad python en `PyCon 2016 <https://youtu.be/c5DaaGZWQqY>`_ tanto en charlas como durante los sprints de desarrollo. El proyecto se llevó a cabo a través de DDL Research Labs, sprints de un semestre de duración donde los miembros de la comunidad DDL contribuyen a varios proyectos relacionados con los datos.

Desde entonces, Yellowbrick ha disfrutado de la participación de un gran número de colaboradores de todo el mundo y un creciente apoyo en la comunidad PyData. Yellowbrick ha aparecido en charlas en eventos organizados por PyData, Scipy, NumFOCUS y PSF, así como en publicaciones de blog y competiciones de Kaggle. Estamos tan motivados por tener una comunidad tan dedicada involucrada en contribuciones activas, tanto grandes como pequeñas.

Para obtener una lista completa de los servicios de mantenimiento actuales y los contribuyentes principales, consulta `MAINTAINERS.md <https://github.com/DistrictDataLabs/yellowbrick/blob/develop/MAINTAINERS.md>`_ en el almacén de datos de GitHub. ¡Muchas gracias a todos los que han `contribuido a Yellowbrick <https://github.com/DistrictDataLabs/yellowbrick/graphs/contributors>`_!

Afiliaciones
------------

Yellowbrick se enorgullece de estar afiliado a varias organizaciones que brindan apoyo institucional al proyecto. Tal apoyo es a veces financiero, a menudo material, y siempre en el espíritu del software libre y de código abierto. No podemos agradecerles lo suficiente por su papel en hacer de Yellowbrick lo que es hoy.

`District Data Labs`_: District Data Labs incubó Yellowbrick y patrocina laboratorios de investigación mediante la compra de alimentos y la organización de eventos. Los laboratorios de investigación son sprints de un semestre de duración que permiten a los productores de Yellowbrick reunirse en persona, compartir una comida y participar en el proyecto. DDL también patrocina viajes a las conferencias PyCon y PyData para los que ofrecen servicio de mantenimiento en Yellowbrick y nos ayuda a comprar material promocional como stickers y camisetas.

`NumFOCUS`_: Yellowbrick es un proyecto afiliado a NumFOCUS (no un proyecto patrocinado fiscalmente). Nuestra relación con NumFOCUS nos ha dado mucha credibilidad en la ciencia de datos en la comunidad al aparecer en su sitio web. También somos elegibles para solicitar pequeñas campañas de desarrollo y apoyo a la infraestructura. A menudo participamos en la lista de correo de desarrolladores de proyectos y otras actividades como Google Summer of Code.

`Georgetown University`_: Georgetown proporciona principalmente espacio para eventos de Yellowbrick, incluidos los laboratorios de investigación. Además, los estudiantes del Certificado de Ciencia de Datos de Georgetown son introducidos a Yellowbrick al comienzo de su educación de machine learning y a menudo realizamos pruebas de usuario de nuevas características en ellos.

Cómo apoyar a Yellowbrick
~~~~~~~~~~~~~~~~~~~~~~~~~
Yellowbrick es desarrollado por voluntarios que trabajan en el proyecto en su tiempo libre y no como parte de su trabajo regular a tiempo completo. Si Yellowbrick se ha vuelto crítico para el éxito de su organización, considere retribuir a Yellowbrick.

    "... El código abierto prospera con recursos humanos en lugar de financieros. Allí
    hay muchas maneras de hacer crecer los recursos humanos, como la distribución de los
    carga de trabajo entre más contribuyentes o animar a las empresas a
    hacer que el código abierto forme parte del trabajo de sus empleados. Un
    La estrategia de soporte debe incluir múltiples formas de generar tiempo y
    recursos además de financiar directamente el desarrollo. Debe comenzar desde
    el principio de que el enfoque de código abierto no es intrínsecamente defectuoso,
    sino más bien de origen insuficiente".

    -- `'Carreteras y puentes: el trabajo invisible detrás de nuestra infraestructura digital <https://www.fordfoundation.org/about/library/reports-and-studies/roads-and-bridges-the-unseen-labor-behind-our-digital-infrastructure/>`_

Lo principal que necesitan los servicios de mantenimiento de Yellowbrick es *tiempo*. Hay muchas maneras de proporcionar ese tiempo a través de mecanismos no financieros como:

- Crear una política escrita en el manual de su empresa que dedique tiempo para que sus empleados contribuyan a proyectos de código abierto como Yellowbrick.
- Interactuar con nuestra comunidad dando aliento y asesoramiento, particularmente para la planificación a largo plazo y actividades no relacionadas con el código, como el diseño y la documentación.
- Abogar y evangelizar el uso de Yellowbrick y otro software de código abierto através de publicaciones de blog y redes sociales.
- Considerar estrategias de apoyo a largo plazo en lugar de acciones ad hoc o únicas.
- Enseñe a sus estudiantes Machine Learning con Yellowbrick.

Un apoyo más concreto y financiero también es bienvenido, especialmente si se dirige a través de un esfuerzo específico. Si está interesado en este tipo de apoyo, considere:

- Hacer una donación a NumFOCUS en nombre de Yellowbrick.
- Involucrar a District Data Labs para capacitación corporativa sobre machine learning visual con Yellowbrick (que apoyará directamente a los servicios de mantenimiento de Yellowbrick).
- Apoyar la educación profesional continua de su empleado en el Certificado de Ciencia de Datos de Georgetown.
- Proporcionar apoyo a largo plazo para costes fijos como el alojamiento o hosting.

La misión de Yellowbrick es mejorar el flujo de trabajo de machine learning a través de la dirección visual y el diagnóstico de código abierto. Si estás interesado en una relación de afiliación más formal para apoyar esta misión, contáctese con nosotros directamente.

Licencia
--------

Yellowbrick es un proyecto de código abierto y su `license <https://github.com/DistrictDataLabs/yellowbrick/blob/master/LICENSE.txt>`_ es una implementación de la licencia FOSS `Apache 2.0 <http://www.apache.org/licenses/LICENSE-2.0>`_ license by the Apache Software Foundation. `En lenguaje simple <https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)>`_ esto significa que puede usar Yellowbrick con fines comerciales, modificar y distribuir el código fuente e incluso sublicenciarlo. Queremos que uses Yellowbrick, te beneficies de él y contribuyas si haces cosas geniales con él.

Hay, sin embargo, un par de requisitos que te pedimos. Primero, cuando copies o distribuyas el código fuente de Yellowbrick, incluye nuestros derechos de autor y licencia que se encuentran en la `LICENSE.txt <https://github.com/DistrictDataLabs/yellowbrick/blob/master/LICENSE.txt>`_ en la raíz del almacenamiento de datos de software. Además, si creamos un archivo llamado "AVISO" en nuestro proyecto también debes incluirlo en tu distribución de origen. ¡El archivo "AVISO" incluirá atribución y agradecimiento a aquellos que han trabajado tan duro en el proyecto! Ten en cuenta que no puede usar nuestros nombres, marcas comerciales o logotipos para promocionar su trabajo o de ninguna otra manera que no sea para hacer referencia a Yellowbrick. Finalmente, proporcionamos a Yellowbrick sin garantía y no puede responsabilizar a ningún colaborador o afiliado de Yellowbrick por su uso de nuestro software.

Creemos que es un trato bastante justo, y somos grandes creyentes en el código abierto. Si realiza algún cambio en su software, lo usa comercial o académicamente, o tiene algún otro interés, nos encantaría conocerlo.

Presentaciones
--------------

Yellowbrick ha disfrutado del centro de atención en varias presentaciones en conferencias recientes. Esperamos que estos libros, charlas y diapositivas te ayuden a entender Yellowbrick un poco mejor.

Documentos:
    - `Yellowbrick: Visualizing the Scikit-Learn Model Selection Process <http://joss.theoj.org/papers/10.21105/joss.01075>`_

Conferencias (videos):
    - `Visual Diagnostics for More Informed Machine Learning: Within and Beyond Scikit-Learn (PyCon 2016) <https://youtu.be/c5DaaGZWQqY>`_
    - `Yellowbrick: Steering Machine Learning with Visual Transformers (PyData London 2017) <https://youtu.be/2ZKng7pCB5k>`_

Cuadernos Jupyter:
    - `Data Science Delivered: ML Regression Predications <https://github.com/ianozsvald/data_science_delivered/blob/master/ml_explain_regression_prediction.ipynb>`_

Diapositivas:
    - `Machine Learning Libraries You'd Wish You'd Known About (PyData Budapest 2017) <https://speakerdeck.com/ianozsvald/machine-learning-libraries-youd-wish-youd-known-about-1>`_
    - `Visualizing the Model Selection Process <https://www.slideshare.net/BenjaminBengfort/visualizing-the-model-selection-process>`_
    - `Visualizing Model Selection with Scikit-Yellowbrick <https://www.slideshare.net/BenjaminBengfort/visualizing-model-selection-with-scikityellowbrick-an-introduction-to-developing-visualizers>`_
    - `Visual Pipelines for Text Analysis (Data Intelligence 2017) <https://speakerdeck.com/dataintelligence/visual-pipelines-for-text-analysis>`_


Citar a Yellowbrick
-------------------

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.1206239.svg
   :target: https://doi.org/10.5281/zenodo.1206239

.. image:: http://joss.theoj.org/papers/10.21105/joss.01075/status.svg
   :target: https://doi.org/10.21105/joss.01075

Esperamos que Yellowbrick facilite el machine learning de todo tipo y nos gusta especialmente el trabajo académico y la investigación. Si estás escribiendo una publicación científica que usa Yellowbrick, puedes citar *Bengfort et al. (2018)* con el siguiente BibTex:

.. code-block:: bibtex

    @software{bengfort_yellowbrick_2018,
        title = {Yellowbrick},
        rights = {Apache License 2.0},
        url = {http://www.scikit-yb.org/en/latest/},
        abstract = {Yellowbrick is an open source, pure Python project that
            extends the Scikit-Learn {API} with visual analysis and
            diagnostic tools. The Yellowbrick {API} also wraps Matplotlib to
            create publication-ready figures and interactive data
            explorations while still allowing developers fine-grain control
            of figures. For users, Yellowbrick can help evaluate the
            performance, stability, and predictive value of machine learning
            models, and assist in diagnosing problems throughout the machine
            learning workflow.},
        version = {0.9.1},
        author = {Bengfort, Benjamin and Bilbro, Rebecca and Danielsen, Nathan and
            Gray, Larry and {McIntyre}, Kristen and Roman, Prema and Poh, Zijie and
            others},
        date = {2018-11-14},
        year = {2018},
        doi = {10.5281/zenodo.1206264}
    }

También puede encontrar DOI (identificadores de objetos digitales) para cada versión de Yellowbrick en `zenodo.org <https://doi.org/10.5281/zenodo.1206239>`_; utilizar el BibTeX en este sitio para hacer referencia a versiones específicas o cambios realizados en el software.

También hemos publicado un artículo en el `Journal of Open Source Software (JOSS) <http://joss.theoj.org/papers/10.21105/joss.01075>`_ que analiza cómo Yellowbrick está diseñado para influir en el flujo de trabajo de selección de modelos. Puedes citar este documento si estás discutiendo Yellowbrick de manera más general en su investigación (en lugar de una versión específica) o si estás interesado en discutir el análisis visual o la visualización para el machine learning. Por favor, cite *Bengfort and Bilbro (2019)* con el siguiente BibTex:

.. code-block:: bibtex

    @article{bengfort_yellowbrick_2019,
        title = {Yellowbrick: {{Visualizing}} the {{Scikit}}-{{Learn Model Selection Process}}},
        journaltitle = {The Journal of Open Source Software},
        volume = {4},
        number = {35},
        series = {1075},
        date = {2019-03-24},
        year = {2019},
        author = {Bengfort, Benjamin and Bilbro, Rebecca},
        url = {http://joss.theoj.org/papers/10.21105/joss.01075},
        doi = {10.21105/joss.01075}
    }

Contáctanos
-----------

La mejor manera de contactar con el equipo de Yellowbrick es enviarnos un correo o nota en una de las siguientes plataformas:

- Enviar un correo electrónico a través `mailing list`_.
- Envíenos un mensaje directo en `Twitter`_.
- Haga una pregunta en `Stack Overflow`_.
- Reportar un problema en nuestro `GitHub Repo`_.


.. _`GitHub Repo`: https://github.com/DistrictDataLabs/yellowbrick
.. _`mailing list`: http://bit.ly/yb-listserv
.. _`Stack Overflow`: https://stackoverflow.com/questions/tagged/yellowbrick
.. _`Twitter`: https://twitter.com/scikit_yb

.. _QuatroCinco: https://flic.kr/p/2Yj9mj
.. _API: http://scikit-learn.org/stable/modules/classes.html
.. _SIGMOD: http://cseweb.ucsd.edu/~arunkk/vision/SIGMODRecord15.pdf
.. _Wikipedia: https://en.wikipedia.org/wiki/Yellow_brick_road
.. _`@rebeccabilbro`: https://github.com/rebeccabilbro
.. _`@bbengfort`: https://github.com/bbengfort
.. _`District Data Labs`: http://www.districtdatalabs.com/
.. _`Georgetown University`: https://scs.georgetown.edu/programs/375/certificate-in-data-science/
.. _`NumFOCUS`: https://numfocus.org/