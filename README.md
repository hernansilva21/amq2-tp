# TP Final de Aprendizaje de Maquina II - CEIA - FIUBA

## Descripción
Este proyecto es la entrega final de la asignatura de Aprendizaje de Máquinas 2 del curso de posgrado de CEIA de la Facultad de Ingenieria de la Universidad de Buenos Aires.
Su objetivo es construir y desplegar un modelo de aprendizaje automático para predecir el precio de autos usados en Bolivia. El proyecto consiste en la integración de diversas herramientas de MLOps para gestionar el ciclo de vida completo del modelo, desde el entrenamiento hasta el despliegue.

## Herramientas Utilizadas
En este proyecto se han utilizado las siguientes herramientas y tecnologías:

- **Docker**: Para contenerizar las aplicaciones y configurar el entorno.
- **Docker Compose**: Para orquestar múltiples contenedores, incluyendo los siguientes servicios:
  - **FastAPI**: Para construir y exponer la API para consumir el modelo.
  - **MLflow**: Usado para el seguimiento de experimentos, versionado de modelos y registro de métricas.
  - **Airflow**: Orquesta las pipelines de aprendizaje automático y programa tareas (como training y ETLs).
  - **MinIO**: Almacenamiento de objetos compatible con S3 para diferentes servicios.
  - **PostgreSQL**: Utilizado como base de datos relacional.
  - **Streamlit**: Framework utilizado para construir una UI web para consumir la API.

## Estructura del Proyecto

```bash
├── airflow/              # DAGs y configuraciones de Airflow
├── data/                 # Dataset de inicialización
├── docker-compose.yml    # Archivo de Docker Compose para orquestar los servicios
├── dockerfiles/
│   ├── airflow/            # Archivos de servicio docker de Airflow
│   ├── fastapi/            # Archivos de servicio docker de API
│   ├── mlflow/             # Archivos de servicio docker de MLFlow
│   ├── postgres/           # Archivos de servicio docker de postgres
│   ├── streamlit/          # Archivos de servicio docker de Streamlit
├── .env                  # Archivo con variables de entorno (*)
├── .gitignore            # Lista de carpetas/archivos ignorados por git
└── README.md             # El documento que estás leyendo ahora mismo
# (*): Para este proyecto, las contraseñas son almacenadas aquí por cuestiones de practicidad y no es una práctica recomendable en absoluto.
```

## Requisitos Previos
 - Disponer de Docker
 - Disponer de Docker Compose

## Ejecución del Proyecto

1. Clonar el repositorio

```bash
git clone https://github.com/hernansilva21/amq2-tp.git
cd amq2-tp
```

2. Levantar los servicios con Docker Compose

```bash
docker compose --profile all up
```

3. Acceder a los servicios

 - **FastAPI**: ``localhost:8800/docs``
 - **Airflow**: ``localhost:8080``
 - **MLFlow**:  ``localhost:5000``
 - **Streamlit**: ``localhost:8501``

## Resumen Pipeline del Modelo:

1. **Ingesta de datos**: Un DAG de ETL toma información cruda del datasource original (en este caso un .xlsx subido al bucket de s3 al inicializar los servicios) y realiza limpieza, feature selection, estandarización y demás procedimientos. Luego lo versiona y se almacena.

2. **(Entrenamiento/reentrenamiento del modelo**: Un segundo DAG lee los últimos datos almacenados para train y test de modelo y entrena un CatBooster con GridSearch de hiperparámetros para guardarlo en s3 (versionado a través de experimentos de MLFlow). Si no es la primera vez que se ejecuta el DAG, este procede con una modalidad champion-challenger para actualizar la versión champion de ser necesaria.

3. **Despliegue**: El modelo champion se encuentra disponible automáticamente tras el DAG de entrenamiento, ya que se actualiza el bucket s3 que la aplicación de FastAPI consulta para actualizar el modelo de predicción.

4. **Interfaz de predicción**: La API expone un endpoint que consumimos con una app web de Streamlit.

## Uso

Una vez levantados los servicios, es necesario entrar a airflow para activar los dos DAGs (es necesario que el proceso ETL se ejecute al menos uan vez antes de ejecutar el DAG de entrenamiento).

Luego se puede o bien consumir la API directamente (puede encontrar documentación de parámetros necesarios en el endpoint /docs de la misma) o bien ingresar a la aplicación web de Streamlit, con modalidad de formulario.