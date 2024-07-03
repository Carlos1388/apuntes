## INFO

- **Name**: [Carlos Poveda]
- **Email**: [carlospovedat1388@gmail.com]

Apuntes sobre serie de videos de MLOps - Zoomcamp

[Link a videos](https://youtu.be/s0uaFZSzwfI)

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

## 2. Experimentos y Runs

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

## 3. Optimización de Hiperparámetros

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

En la UI de MLFlow se pueden ver los resultados de cada iteración y seleccionar el mejor modelo, ya que se pueden ver los hiperparámetros que se han ido probando y las métricas que se han ido obteniendo. Después de la búsqueda de hiperparámetros lo mejor es guardar el modelo específico, entrenándolo de nuevo y registrándolo en MLFlow.




## 4. Autologging

Para algunas librerías, podemos registrar todos los parámetros de este proceso automáticamente.

Lista de librerías que soportan esto:

- `Fastai`,
- `Gluon`,
- `Keras/TensorFlow`,
- `LangChain`,
- `LightGBM`,
- `OpenAI`,
- `Paddle`,
- `PySpark`,
- `PyTorch`,
- `Scikit-learn`,
- `Spark`,
- `Statsmodels`,
- `XGBoost`.

Para registrar automáticamente los parámetros de un modelo de estas librerías, se puede utilizar el autolog de `mlflow`. Para ello, primero hay que definir qué librerías se quiere autologgear:

```python
# Opción 1: Autologging solo para PyTorch
mlflow.pytorch.autolog()

# Opción 2: Autologging para todo excepto scikit-learn
mlflow.sklearn.autolog(disable=True) # <- deshabilitamos el autologging de scikit-learn
mlflow.autolog()   # <- autologging para todas las librerías que soportan autologging
```

Se puede ver qué se registra en cada caso en la [documentación de MLFlow](https://mlflow.org/docs/latest/tracking/autolog.html).

Para ver los resultados de las métricas y parámetros registrados automáticamente, se puede hacer con `mlflow.last_active_run()`. Un ejemplo de esto sería:

```python
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Crear y entrenar el modelo
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Usamos el modelo para hacer predicciones
predictions = rf.predict(X_test)
autolog_run = mlflow.last_active_run()
print(autolog_run)
# <Run:
#    data=<RunData:
#        metrics={'accuracy': 0.0},
#        params={'n_estimators': '100', 'max_depth': '6', 'max_features': '3'},
#        tags={'estimator_class': 'sklearn.ensemble._forest.RandomForestRegressor', 'estimator_name': 'RandomForestRegressor'}
#    >,
#    info=<RunInfo:
#        artifact_uri='file:///Users/andrew/Code/mlflow/mlruns/0/0c0b.../artifacts',
#        end_time=163...0,
#        run_id='0c0b...',
#        run_uuid='0c0b...',
#        start_time=163...0,
#        status='FINISHED',
#        user_id='ME'>
#    >
# >
```

## 5. Model Registry

El Model Registry es una funcionalidad de MLFlow que permite a los equipos de datos gestionar los modelos a lo largo de su ciclo de vida.

La forma más básica de registrar un modelo es guardándolo como artifact en MLFlow. Para ello, se puede hacer con el siguiente código:

```python
modelo = ... # modelo entrenado

with open("path/to/model", "wb") as f:
    pickle.dump(modelo, f)

mlflow.log_artifact(local_path = "path/to/model", artifact_path = "models_pickle/")
```

o también se puede hacer con el siguiente código:

```python
modelo = ... # modelo entrenado

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "mi_modelo_artifact")
```

en este caso se podría encontrar el modelo en el directorio de la run concreta desde la que se ha guardado el modelo.

```python
# Obtener la URI del modelo registrado como artefacto
artifact_model_uri = f"runs:/{run.info.run_id}/mi_modelo_artifact"
print("Modelo registrado como artefacto en:", artifact_model_uri)
```

También podemos registrar el modelo con nombres y apellidos, en este caso se guardará en una carpeta de modelos y se le puede asignar un estado del ciclo de vida (`staging`,`production`, etc). Para ello, se puede hacer con el siguiente código:	

```python
...
model.fit(X_train, y_train)

# Iniciar una ejecución de MLflow y registrar el modelo en el Registro de Modelos
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "random_forest_model_registered", registered_model_name="Nombre_modelo")

# Obtener la URI del modelo registrado en el Registro de Modelos
registered_model_uri = "models:/Nombre_modelo/1"
print("Modelo registrado en el Registro de Modelos en:", registered_model_uri)
```

Para utilizar el modelo registrado, se puede hacer con los siguientes snippets.

En el caso de querer cargar el modelo registrado como artefacto:

```python

# Cargar datos
...
X_train, X_test, y_train, y_test = train_test_split(...)

# URI del modelo registrado como artefacto
artifact_model_uri = "runs:/<run_id>/mi_modelo_artifact"

# Cargar el modelo registrado como artefacto
model_artifact = mlflow.sklearn.load_model(artifact_model_uri)

# Hacer predicciones
predictions_artifact = model_artifact.predict(X_test)
print("Predicciones (artefacto):", predictions_artifact)
```

En el caso de querer cargar el modelo registrado en el Registro de Modelos:

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Cargar datos
...
X_train, X_test, y_train, y_test = train_test_split(...)

# URI del modelo registrado en el Registro de Modelos
registered_model_uri = "models:/Nombre_modelo/1" # <- Aquí la diferencia, el path es diferente

# Cargar el modelo registrado en el Registro de Modelos
model_registered = mlflow.sklearn.load_model(registered_model_uri)

# Hacer predicciones
predictions_registered = model_registered.predict(X_test)
print("Predicciones (Registro de Modelos):", predictions_registered)
```

## 6. Inferencia

Podemos utilizar PyFunc, que ya viene como API dentro de MLFlow, para hacer inferencia con los modelos registrados. Si desde la UI de MLFlow se selecciona un modelo registrado, se puede ver el código que se necesita para hacer inferencia con ese modelo. Por ejemplo, para tener definido el modelo en pyfunc haríamos algo como esto:

```python
logged_model = "runs:/<run_id>/mi_modelo_artifact"

loaded_model = mlflow.pyfunc.load_model(logged_model)
```

Una vez cargado, podemos hacer inferencia sobre dataframes de pandas. Por ejemplo:

```python
loaded_model.predict(pd.DataFrame(data))
```

