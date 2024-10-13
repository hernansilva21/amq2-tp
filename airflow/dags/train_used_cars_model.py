import datetime

from airflow.decorators import dag, task

default_args = {
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=60)
}

@dag(
    dag_id="train_used_cars_model",
    description="DAG to train the used cars price prediction model with champion-challenger logic.",
    tags=["Training", "Used cars"],
    default_args=default_args,
    catchup=False,
)
def train_used_cars_model():

    @task.virtualenv(
        task_id="train_the_challenger_model",
        requirements=[
            "awswrangler==3.6.0",
            "pandas",
            "numpy",
            "scikit-learn==1.3.2",
            "catboost",
            "mlflow==2.10.2",
        ],
        system_site_packages=True
    )
    def train_the_challenger_model():
        """
        Trains the challenger model using CatBoost and GridSearchCV, and logs the trained model to MLflow.
        """
        import awswrangler as wr
        import pandas as pd
        import numpy as np
        from catboost import CatBoostRegressor
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import mean_squared_error
        import mlflow
        from mlflow.models import infer_signature

        # S3 paths
        X_train_path = "s3://data/final/train/used_cars_X_train.csv"
        X_test_path = "s3://data/final/test/used_cars_X_test.csv"
        y_train_path = "s3://data/final/train/used_cars_y_train.csv"
        y_test_path = "s3://data/final/test/used_cars_y_test.csv"

        # Load data directly from S3
        X_train = wr.s3.read_csv(X_train_path)
        X_test = wr.s3.read_csv(X_test_path)
        y_train = wr.s3.read_csv(y_train_path)
        y_test = wr.s3.read_csv(y_test_path)

        # Convert target variable to Series if it's a DataFrame
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

        # Define the base model
        model = CatBoostRegressor(
            random_seed=42,
            verbose=False
        )

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

        # Get the best parameters and the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        print("Best parameters:", best_params)

        # Evaluate the model on the test set
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"MSE on the test set: {mse}")

        # Log the trained model in MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Used Cars")

        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Challenger_Model_Training"):
            mlflow.log_params(best_params)
            mlflow.log_metric("test_mse", mse)
            # Log the model
            signature = infer_signature(X_train, best_model.predict(X_train))
            mlflow.catboost.log_model(
                cb_model=best_model,
                artifact_path="model",
                registered_model_name="used_cars_model_prod",
                signature=signature
            )

            # Get the current run ID
            run_id = mlflow.active_run().info.run_id

            # Obtain the model URI
            model_uri = mlflow.get_artifact_uri("model")

            # Register the model version with alias 'challenger'
            client = mlflow.MlflowClient()
            model_versions = client.search_model_versions(f"name='used_cars_model_prod'")

            challenger_version = None
            for mv in model_versions:
                if mv.run_id == run_id:
                    challenger_version = mv.version
                    break

            if challenger_version is None:
                raise Exception("Failed to find the model version just registered.")

            # Set the alias 'challenger' to the model version
            client.set_registered_model_alias(name="used_cars_model_prod", alias="challenger", version=challenger_version)

    @task.virtualenv(
        task_id="evaluate_champion_challenger",
        requirements=[
            "awswrangler==3.6.0",
            "pandas",
            "numpy",
            "scikit-learn==1.3.2",
            "catboost",
            "mlflow==2.10.2",
        ],
        system_site_packages=True
    )
    def evaluate_champion_challenger():
        """
        Evaluates the champion and challenger models, and promotes the challenger if it performs better.
        Logs comparison metrics to MLflow for visualization.
        """
        import awswrangler as wr
        import pandas as pd
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import mlflow
        from mlflow.tracking import MlflowClient
        import sys

        # Load test data
        X_test_path = "s3://data/final/test/used_cars_X_test.csv"
        y_test_path = "s3://data/final/test/used_cars_y_test.csv"

        X_test = wr.s3.read_csv(X_test_path)
        y_test = wr.s3.read_csv(y_test_path)

        # Convert target variable to Series if it's a DataFrame
        y_test = y_test.squeeze()

        # Set MLflow tracking URI
        mlflow.set_tracking_uri('http://mlflow:5000')

        client = MlflowClient()

        model_name = "used_cars_model_prod"

        try:
            # Try to load the champion model
            champion_model_info = client.get_model_version_by_alias(model_name, "champion")
            champion_model = mlflow.catboost.load_model(f"models:/{model_name}/{champion_model_info.version}")
            # Evaluate the champion model
            y_pred_champion = champion_model.predict(X_test)
            mse_champion = mean_squared_error(y_test, y_pred_champion)
            mae_champion = mean_absolute_error(y_test, y_pred_champion)
            r2_champion = r2_score(y_test, y_pred_champion)
        except Exception:
            # No champion model exists
            print("No champion model found.")
            mse_champion = None
            mae_champion = None
            r2_champion = None

        # Load the challenger model
        challenger_model_info = client.get_model_version_by_alias(model_name, "challenger")
        challenger_model = mlflow.catboost.load_model(f"models:/{model_name}/{challenger_model_info.version}")
        # Evaluate the challenger model
        y_pred_challenger = challenger_model.predict(X_test)
        mse_challenger = mean_squared_error(y_test, y_pred_challenger)
        mae_challenger = mean_absolute_error(y_test, y_pred_challenger)
        r2_challenger = r2_score(y_test, y_pred_challenger)

        print(f"Champion MSE: {mse_champion}")
        print(f"Challenger MSE: {mse_challenger}")

        # Start an MLflow run to log comparison metrics
        experiment = mlflow.set_experiment("Used Cars")
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="Champion_Challenger_Evaluation") as run:
            if mse_champion is not None:
                mlflow.log_metric("mse_champion", mse_champion)
                mlflow.log_metric("mae_champion", mae_champion)
                mlflow.log_metric("r2_champion", r2_champion)
            else:
                mlflow.log_metric("mse_champion", sys.maxsize)
                mlflow.log_metric("mae_champion", sys.maxsize)
                mlflow.log_metric("r2_champion", 0.0)
            mlflow.log_metric("mse_challenger", mse_challenger)
            mlflow.log_metric("mae_challenger", mae_challenger)
            mlflow.log_metric("r2_challenger", r2_challenger)

            # Log comparison metrics
            comparison_metrics = {
                "mse_difference": (mse_champion - mse_challenger) if mse_champion is not None else None,
                "mae_difference": (mae_champion - mae_challenger) if mae_champion is not None else None,
                "r2_difference": (r2_challenger - r2_champion) if r2_champion is not None else None,
            }
            mlflow.log_metrics(comparison_metrics)

            # Log the decision
            if (mse_champion is None) or (mse_challenger < mse_champion):
                # Challenger is better, promote it to champion
                print("Challenger model performs better. Promoting to champion.")
                # Demote the current champion if exists
                if mse_champion is not None:
                    client.delete_registered_model_alias(model_name, "champion")
                # Promote challenger to champion
                client.set_registered_model_alias(model_name, "champion", challenger_model_info.version)
                # Remove 'challenger' alias
                client.delete_registered_model_alias(model_name, "challenger")
                mlflow.log_param("Model Decision", "Challenger promoted to Champion")
            else:
                # Challenger is worse, keep the champion
                print("Champion model performs better. Keeping the current champion.")
                # Remove 'challenger' alias
                client.delete_registered_model_alias(model_name, "challenger")
                mlflow.log_param("Model Decision", "Champion remains")

            # Tag the run with model versions for traceability
            mlflow.set_tag("champion_version", champion_model_info.version if mse_champion is not None else "None")
            mlflow.set_tag("challenger_version", challenger_model_info.version)

    # Define the task flow
    train_the_challenger_model() >> evaluate_champion_challenger()

dag = train_used_cars_model()
