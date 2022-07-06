# Yellowbrick


[![Build Status](https://github.com/DistrictDataLabs/yellowbrick/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/DistrictDataLabs/yellowbrick/actions/workflows/ci.yml)
[![Coverage Status](https://codecov.io/gh/DistrictDataLabs/yellowbrick/branch/develop/graph/badge.svg?token=BnaSECZz2r)](https://codecov.io/gh/DistrictDataLabs/yellowbrick)
[![Total Alerts](https://img.shields.io/lgtm/alerts/g/DistrictDataLabs/yellowbrick.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/DistrictDataLabs/yellowbrick/alerts/)
[![Language Grade: Python](https://img.shields.io/lgtm/grade/python/g/DistrictDataLabs/yellowbrick.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/DistrictDataLabs/yellowbrick/context:python)
[![PyPI version](https://badge.fury.io/py/yellowbrick.svg)](https://badge.fury.io/py/yellowbrick)
[![Documentation Status](https://readthedocs.org/projects/yellowbrick/badge/?version=latest)](http://yellowbrick.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1206239.svg)](https://doi.org/10.5281/zenodo.1206239)
[![JOSS](http://joss.theoj.org/papers/10.21105/joss.01075/status.svg)](https://doi.org/10.21105/joss.01075)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/DistrictDataLabs/yellowbrick/develop?filepath=examples%2Fexamples.ipynb)


**Herramientas de análisis visual y diagnóstico para facilitar la selección de modelos de machine learning.**

[![Banner](docs/images/readme/banner.png)](https://www.scikit-yb.org/en/latest/gallery.html)

## ¿Qué es Yellowbrick?

Yellowbrick es un conjunto de herramientas de diagnóstico visual llamadas "Visualizadores" que extienden la API scikit-learn para permitir la dirección humana del proceso de selección del modelo. En pocas palabras, Yellowbrick combina scikit-learn con matplotlib en la mejor tradición de la documentación scikit-learn, ¡pero para producir visualizaciones para _su_ flujo de trabajo de machine learning!

Para obtener documentación completa sobre la API de Yellowbrick, una galería de visualizadores disponibles, la guía del colaborador, tutoriales y recursos de enseñanza, preguntas frecuentes y más, visite nuestra documentación en [www.scikit-yb.org](https://www.scikit-yb.org/).

## Instalación de Yellowbrick

Yellowbrick es compatible con Python 3.4 o posterior y también depende de scikit-learn y matplotlib. La forma más sencilla de instalar Yellowbrick y sus dependencias es desde PyPI con pip, el instalador de paquetes preferido de Python.

    $ pip install yellowbrick

Ten en cuenta que Yellowbrick es un proyecto activo y publica rutinariamente nuevas versiones con más visualizadores y actualizaciones. Para actualizar Yellowbrick a la última versión, use pip de la siguiente manera.

    $ pip install -U yellowbrick

También puedes usar la bandera `-U` para actualizar scikit-learn, matplotlib o cualquier otra utilidad de terceros que funcione bien con Yellowbrick en sus últimas versiones.

Si estás utilizando Anaconda (recomendado para usuarios de Windows), puedes aprovechar la utilidad conda para instalar Yellowbrick:

    conda install -c districtdatalabs yellowbrick

## Usando Yellowbrick

La API de Yellowbrick está diseñada específicamente para llevarse bien con scikit-learn. Aquí hay un ejemplo de una secuencia de flujo de trabajo típica con scikit-learn y Yellowbrick:

### Visualización de características

En este ejemplo, vemos cómo Rank2D realiza comparaciones por pares de cada característica del conjunto de datos con una métrica o algoritmo específico y luego las devuelve clasificadas como un diagrama de triángulo izquierdo inferior.

```python
from yellowbrick.features import Rank2D

visualizer = Rank2D(
    features=features, algorithm='covariance'
)
visualizer.fit(X, y)                # Ajuste los datos al visualizador
visualizer.transform(X)             # Transformar los datos
visualizer.show()                   # Finalizar y representar la figura
```

### Visualización del modelo

En este ejemplo, creamos una instancia de un clasificador scikit-learn y luego usamos la clase ROCAUC de Yellowbrick para visualizar la compensación entre la sensibilidad y la especificidad del clasificador.

```python
from sklearn.svm import LinearSVC
from yellowbrick.classifier import ROCAUC

model = LinearSVC()
visualizer = ROCAUC(model)
visualizer.fit(X,y)
visualizer.score(X,y)
visualizer.show()
```

Para obtener información adicional sobre cómo comenzar con Yellowbrick, consulte la [Guía de inicio rápido](https://www.scikit-yb.org/en/latest/quickstart.html) en la [documentación](https://www.scikit-yb.org/en/latest/) y consulte nuestro [libro de ejemplos](https://github.com/DistrictDataLabs/yellowbrick/blob/develop/examples/examples.ipynb).

## Contribuyendo a Yellowbrick

Yellowbrick es un proyecto de código abierto que cuenta con el apoyo de una comunidad que aceptará con gratitud y humildad cualquier contribución que hagas al proyecto. Grande o pequeña, cualquier contribución hace una gran diferencia; y si nunca antes has contribuido a un proyecto de código abierto, ¡esperamos que comiences con Yellowbrick!

Si estás interesado en contribuir, consulte nuestra [guía del colaborador](https://www.scikit-yb.org/en/latest/contributing/index.html). Más allá de crear visualizadores, hay muchas formas de contribuir:

- Enviar un informe de error o una solicitud de función en [Problemas de GitHub](https://github.com/DistrictDataLabs/yellowbrick/issues).
- Contribuir con un cuaderno Jupyter a nuestros ejemplos [galería](https://github.com/DistrictDataLabs/yellowbrick/tree/develop/examples).
- Ayudarnos con [pruebas de usuario](https://www.scikit-yb.org/en/latest/evaluation.html).
- Ayudar con la documentación o ayudar con nuestro sitio web, [scikit-yb.org](https://www.scikit-yb.org).
- Escribir [pruebas unitarias o de integración](https://www.scikit-yb.org/en/latest/contributing/developing_visualizers.html#integration-tests) para nuestro proyecto.
- Responda preguntas sobre nuestros problemas, lista de correo, Stack Overflow y otros lugares.
- Traducir nuestra documentación a otro idioma.
- Escribir una publicación de blog, tuitear o compartir nuestro proyecto con otros.
- [Enseñar](https://www.scikit-yb.org/en/latest/teaching.html) a alguien cómo usar Yellowbrick.

Como puedes ver, hay muchas maneras de involucrarse y estaremos muy contentos de que te unas a nosotros. Lo único que le pedimos es que cumplas con los principios de apertura, respeto y consideración de los demás como se describe en el [Código de conducta de Python Software Foundation](https://www.python.org/psf/codeofconduct/).

Para obtener más información, consulte el archivo `CONTRIBUTING.md` en la raíz del almacén de datos o la documentación detallada en [Contribución a Yellowbrick](https://www.scikit-yb.org/en/latest/contributing/index.html)

## Conjuntos de datos de Yellowbrick

Yellowbrick proporciona un fácil acceso a varios conjuntos de datos que se utilizan para los ejemplos en la documentación y las pruebas. Estos conjuntos de datos están alojados en nuestra CDN y deben descargarse para su uso. Por lo general, cuando un usuario llama a una de las funciones de carga de datos, por ejemplo, `load_bikeshare()` los datos se descargan automáticamente si aún no están en la computadora del usuario. Sin embargo, para el desarrollo y las pruebas, o si sabe que trabajará sin acceso a Internet, podría ser más fácil simplemente descargar todos los datos de una vez.

El script del descargador de datos se puede ejecutar de la siguiente manera:

    $ python -m yellowbrick.download

Esto descargará los datos al directorio de accesorios dentro de los paquetes del sitio Yellowbrick. Puedes especificar la ubicación de la descarga como un argumento para el script del descargador (use `--help` para obtener más detalles) o estableciendo la variable de entorno `$YELLOWBRICK_DATA` . Este es el mecanismo preferido porque también influirá en cómo se cargan los datos en Yellowbrick.

_Nota: Los desarrolladores que han descargado datos de versiones anteriores a v1.0 pueden experimentar algunos problemas con el formato de datos anterior. Si esto ocurre, puedes borrar la memoria caché de datos de la siguiente manera:_

    $ python -m yellowbrick.download --cleanup

_ Esto eliminará los conjuntos de datos antiguos y descargará los nuevos. También puedes usar la bandera `--no-download` para simplemente borrar la caché sin volver a descargar datos. Los usuarios que tienen dificultades con los conjuntos de datos también pueden usar esto o pueden desinstalar y reinstalar Yellowbrick usando `pip`._

## Citando a Yellowbrick

¡Estaríamos encantados de que usaras Yellowbrick en tus publicaciones científicas! Si lo haces, por favor cita nuestro proyecto usando las [pautas de citación](https://www.scikit-yb.org/en/latest/about.html#citing-yellowbrick).

## Afiliaciones

[![District Data Labs](docs/images/readme/affiliates_ddl.png)](https://districtdatalabs.com/) [![NumFOCUS Affiliated Project](docs/images/readme/affiliates_numfocus.png)](https://numfocus.org)