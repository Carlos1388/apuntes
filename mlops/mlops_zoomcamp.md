## INFO
## INFO
- **Name**: [Carlos Poveda]
- **Email**: [carlospov4@gmail.com]

Apuntes sobre serie de videos de MLOps - Zoomcamp

[Link to video](https://youtu.be/s0uaFZSzwfI)

## 1. Introducción a MLFlow

MLFlow es una plataforma de código abierto para gestionar el ciclo de vida de Machine Learning. Permite a los equipos de datos gestionar experimentos, paquetes de código, modelos y despliegues en diferentes entornos.

Para instalar MLFlow, se puede hacer con pip:

```bash
pip install mlflow
```

Para iniciar un web ui de MLFlow, se puede hacer con el siguiente comando:

```bash
mlflow ui
```

Esto iniciará antes un servidor en `http://localhost:5000` o algún otro puerto. En este servidor se pueden ver los experimentos, modelos, etc. Es un servidor local apuntando a un directorio local que se crea automáticamente.

Es mejor si se trabaja con otras personas, tener un servidor centralizado. Para esto se puede usar SQLite, MySQL, PostgreSQL, etc. Para esto se puede configurar la conexión a la base de datos con el siguiente comando. Esto hará que la ui de MLFlow apunte a la base de datos centralizada.

```python
mlflow.set_tracking_uri(uri="sqlite:///mlflow.db")
```

## 2. Experimentos

Un experimento sirve para agrupar ejecuciones de modelos. Se pueden comparar diferentes modelos, parámetros, etc. Para crear un experimento se puede hacer con el siguiente comando:

```python
mlflow.create_experiment(name="experiment_name")
```
También podemos dar etiquetas a los experimentos o definir path para guardar los artefactos (modelos, datasets, etc). En las etiquetas podemos poner lo que queramos, por ejemplo, el nombre del proyecto, el nombre del programador, versión, etc.

```python
tags = {"tag_key": "tag_value",
        "tag_key2": "tag_value2",
        ...
        "tag_keyN": "tag_valueN"}

artifact_location = "path/to/artifact"

mlflow.create_experiment(name="experiment_name",
                        artifact_location=artifact_location,
                        tags=tags)
```

Una vez creado el experimento, se pueden guardar runs en el experimento. Para esto se puede hacer con el siguiente comando:

```python
mlflow.set_experiment(experiment_name)
```

En el caso de no existir el experimento, se creará automáticamente. Para guardar un run en el experimento, se puede hacer con el siguiente comandodespués de haber definido el experimento en el que se quiere guardar el run:

```python
with mlflow.start_run():
    
    code_here
```

Es decir, se inicia un run y se ejecuta el código que se quiera guardar. Por ejemplo, podemos definir etiquetas para la run, métricas, parámetros, etc. Estos datos se pueden guardar manualmente en la run o se pueden guardar automáticamente. Para guardar manualmente se puede hacer con el siguiente comando:

```python
with mlflow.start_run():
    
    mlflow.set_tag("tag_key", "tag_value") # <- registramos una etiqueta para la run
    mlflow.log_param("train-data-path", "path/to/train/data") # <- registramos un parámetro para la run que es la ruta de los datos de entrenamiento
    mlflow.log_param("test-data-path", "path/to/test/data") # <- registramos un parámetro para la run que es la ruta de los datos de test

    alpha = 0.5
    beta = 0.1
    mlflow.log_param("alpha", alpha) # <- registramos un parámetro para la run que es el valor de alpha
    mlflow.log_param("beta", beta) # <- registramos un parámetro para la run que es el valor de beta

    code_here # <- código con el entrenamiento del modelo, cálculo de métricas, etc

    rmse = mean_squared_error(y_true, y_pred, squared=False) # por ejemplo
    mlflow.log_metric("rmse", rmse) # <- registramos una métrica para la run que es el rmse
```

Guardar runs uno a uno es tedioso y puede no dar información suficiente como para comparar resultados, es mejor crear bucles de optimización de hiperparámetros e ir guardando cada run en un experimento. Por ejemplo, podemos pensar en definir unas métricas de evaluación, una lista de posibles modelos y para cada uno una lista de posibles hiperparámetros. Todo ello se puede etiquetar para poder hacer búsquedas más fácilmente.

![MLFlow](https://mlflow.org/docs/latest/_images/tag-exp-run-relationship.svg)

Para crear bucles de optimización de hiperparámetros, se puede hacer utilizando [`hyperopt`](https://hyperopt.github.io/hyperopt/) junto con `mlflow`. Para instalar `hyperopt` se puede hacer con el siguiente comando:

```python	
pip install hyperopt
```	

Para crear un bucle de optimización de hiperparámetros, se puede hacer con el siguiente código:

```python
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope

def objective(params):

    with mlflow.start_run():
        mlflow.set_tag("model_name", "model_name")
        mlflow.log_params(params) # <- registramos los hiperparámetros que le vayamos a meter al modelo

        code_here # <- código con el entrenamiento del modelo, cálculo de métricas, etc

        rmse = mean_squared_error(y_true, y_pred, squared=False) # por ejemplo
        mlflow.log_metric("rmse", rmse) # <- registramos la métrica

    return {"loss": rmse, "status": STATUS_OK} # <- devolvemos la métrica que queremos optimizar

space = {  
    'n_estimators': hp.choice('n_estimators', range(10, 100)),                      # <- hiperparámetros que queremos optimizar con hyperopt
    'max_depth': hp.choice('max_depth', range(1, 20)),                              # (este es un ejemplo para RandomForest)
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None])
}

trials = Trials() # <- objeto para guardar los resultados de cada iteración

best = fmin(fn=objective,   # <- función objetivo
            space=space,    # <- espacio de hiperparámetros
            algo=tpe.suggest,  # <- algoritmo de optimización
            max_evals=100,   # <- número de iteraciones
            trials=trials)   # <- objeto para guardar los resultados de cada iteración

```

