from airflow.decorators import dag, task
from datetime import datetime

@dag(
    schedule='@daily',
    start_date=datetime(2023, 9, 14),
    catchup=False,
    dag_id='retrain_the_model',
    tags=['ML model', 'MLFlow']
)
def processing_dag():
    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=[
            "scikit-learn==1.3.2",
            "awswrangler==3.6.0",
            "mlflow==2.10.2",
            "catboost==1.2"
        ],
        system_site_packages=True
    )
    def train_the_challenger_model():
        import mlflow
        import awswrangler as wr
        from catboost import CatBoostRegressor
        import datetime
        from sklearn.base import clone
        from sklearn.metrics import mean_squared_error
        from mlflow.models.signature import infer_signature

        mlflow.set_tracking_uri('http://mlflow:5000')

        def load_the_champion_model():
            model_name = "used_cars_model_prod"
            client = mlflow.MlflowClient()
            try:
                model_data = client.get_model_version_by_alias(model_name, "champion")
                model = mlflow.catboost.load_model(model_data.source)
                return model
            except Exception as e:
                print(f"Error loading champion model: {e}")
                # No champion model exists
                return None

        def load_the_train_test_data():
            X_train = wr.s3.read_csv("s3://data/final/train/used_cars_X_train.csv")
            y_train = wr.s3.read_csv("s3://data/final/train/used_cars_y_train.csv")
            X_test = wr.s3.read_csv("s3://data/final/test/used_cars_X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/used_cars_y_test.csv")
            return X_train, y_train, X_test, y_test

        def mlflow_track_experiment(model, X):
            # Track the experiment
            experiment = mlflow.set_experiment("Used Cars")
            mlflow.start_run(
                run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                experiment_id=experiment.experiment_id,
                tags={"experiment": "challenger models", "dataset": "Used cars"},
                log_system_metrics=True
            )
            params = model.get_params()
            params["model"] = type(model).__name__
            mlflow.log_params(params)

            # Save the artifact of the challenger model
            artifact_path = "model"
            signature = infer_signature(X, model.predict(X))

            mlflow.catboost.log_model(
                cb_model=model,
                artifact_path=artifact_path,
                signature=signature,
                registered_model_name="used_cars_model_dev",
                metadata={"model_data_version": 1}
            )

            # Obtain the model URI
            return mlflow.get_artifact_uri(artifact_path)

        def register_challenger(model, mse, model_uri):
            client = mlflow.MlflowClient()
            name = "used_cars_model_prod"

            # Ensure the registered model exists
            try:
                client.get_registered_model(name)
            except mlflow.exceptions.RestException:
                client.create_registered_model(name)

            # Save the model params as tags
            tags = model.get_params()
            tags["model"] = type(model).__name__
            tags["mse"] = mse

            # Save the version of the model
            result = client.create_model_version(
                name=name,
                source=model_uri,
                run_id=model_uri.split("/")[-3],
                tags=tags
            )

            # Save the alias as challenger
            client.set_registered_model_alias(name, "challenger", result.version)

        def fit_challenger(model, X_train, y_train):
            from sklearn.model_selection import GridSearchCV

            # Define the hyperparameter grid
            param_grid = {
                'iterations': [500, 1000, 1500],
                'learning_rate': [0.01, 0.1],
                'depth': [6, 8],
                'l2_leaf_reg': [3, 5],
                'border_count': [32, 64]
            }

            # Configure GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=4,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )

            # Fit GridSearchCV
            grid_search.fit(X_train, y_train)
            return grid_search.best_estimator_

        # Load the champion model
        champion_model = load_the_champion_model()

        if champion_model is not None:
            # Clone the model
            challenger_model = clone(champion_model)
        else:
            # Define a new model
            challenger_model = CatBoostRegressor(
                random_seed=42,
                verbose=False
            )

        # Load the dataset
        X_train, y_train, X_test, y_test = load_the_train_test_data()

        # Fit the challenger model
        challenger_model = fit_challenger(challenger_model, X_train, y_train)

        # Obtain the metric of the model
        y_pred = challenger_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        # Track the experiment
        artifact_uri = mlflow_track_experiment(challenger_model, X_train)

        # Record the model
        register_challenger(challenger_model, mse, artifact_uri)

    @task.virtualenv(
        task_id="evaluate_champion_challenger",
        requirements=[
            "scikit-learn==1.3.2",
            "mlflow==2.10.2",
            "awswrangler==3.6.0",
            "catboost==1.2"
        ],
        system_site_packages=True
    )
    def evaluate_champion_challenger():
        import mlflow
        import awswrangler as wr

        mlflow.set_tracking_uri('http://mlflow:5000')

        def load_the_model(alias):
            model_name = "used_cars_model_prod"
            client = mlflow.MlflowClient()
            try:
                model_data = client.get_model_version_by_alias(model_name, alias)
                model = mlflow.catboost.load_model(model_data.source)
                return model
            except Exception as e:
                print(f"Error loading model with alias '{alias}': {e}")
                # No model with the alias exists
                return None

        def load_the_test_data():
            X_test = wr.s3.read_csv("s3://data/final/test/used_cars_X_test.csv")
            y_test = wr.s3.read_csv("s3://data/final/test/used_cars_y_test.csv")
            return X_test, y_test

        def promote_challenger(name):
            client = mlflow.MlflowClient()
            # Demote the champion if exists
            try:
                client.delete_registered_model_alias(name, "champion")
            except mlflow.exceptions.RestException:
                pass

            # Load the challenger from registry
            challenger_version = client.get_model_version_by_alias(name, "challenger")

            # Delete the alias of challenger
            client.delete_registered_model_alias(name, "challenger")

            # Transform into champion
            client.set_registered_model_alias(name, "champion", challenger_version.version)

        def demote_challenger(name):
            client = mlflow.MlflowClient()
            # Delete the alias of challenger if exists
            try:
                client.delete_registered_model_alias(name, "challenger")
            except mlflow.exceptions.RestException:
                pass

        # Load the models
        champion_model = load_the_model("champion")
        challenger_model = load_the_model("challenger")

        # Load the dataset
        X_test, y_test = load_the_test_data()

        # Obtain the metric of the models
        from sklearn.metrics import mean_squared_error

        if champion_model is not None:
            y_pred_champion = champion_model.predict(X_test)
            mse_champion = mean_squared_error(y_test, y_pred_champion)
        else:
            mse_champion = float('inf')  # Set to infinity if champion doesn't exist

        y_pred_challenger = challenger_model.predict(X_test)
        mse_challenger = mean_squared_error(y_test, y_pred_challenger)

        experiment = mlflow.set_experiment("Used Cars")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_metric("test_mse_challenger", mse_challenger)
            mlflow.log_metric("test_mse_champion", mse_champion)

            name = "used_cars_model_prod"

            if mse_challenger < mse_champion:
                mlflow.log_param("Winner", 'Challenger')
                promote_challenger(name)
            else:
                mlflow.log_param("Winner", 'Champion')
                demote_challenger(name)

    train_the_challenger_model() >> evaluate_champion_challenger()

my_dag = processing_dag()
