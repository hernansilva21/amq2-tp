import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Process for used car sale ads data

This DAG extracts information from the original XLSX file stored in the S3 bucket data. 
It preprocesses the data by creating dummy variables and scaling numerical features.
    
After preprocessing, the data is saved back into a S3 bucket as two separate CSV files: one for training and one for 
testing. The split between the training and testing datasets is 70/30 and they are stratified.
"""


default_args = {
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="process_etl_used_cars_data",
    description="ETL process for used car sale ads data, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "Used cars"],
    default_args=default_args,
    catchup=False,
)
def process_etl_used_cars_data():

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["awswrangler==3.6.0", "s3fs", "openpyxl"],
        system_site_packages=True
    )
    def get_data():
        """
        Load the raw data from S3 bucket
        """
        import awswrangler as wr
        import pandas as pd

        # fetch dataset
        raw_data_path = "s3://data/raw/merged_data_03_07.xlsx"
        used_cars = pd.read_excel(raw_data_path)

        data_path = "s3://data/raw/used_cars_data.csv"

        wr.s3.to_csv(df=used_cars,
                     path=data_path,
                     index=False)


    @task.virtualenv(
        task_id="make_dummies_variables",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def make_dummies_variables():
        """
        Convert categorical variables into one-hot encoding.
        """
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow

        import awswrangler as wr
        import pandas as pd
        import numpy as np

        from airflow.models import Variable

        data_original_path = "s3://data/raw/used_cars_data.csv"
        data_end_path = "s3://data/raw/used_cars_data.csv"
        dataset = wr.s3.read_csv(data_original_path)

        # Clean duplicates
        dataset.drop_duplicates(inplace=True, ignore_index=True)
        # Drop NaN
        dataset.dropna(inplace=True, ignore_index=True)
        # Feature selection
        dataset.drop(columns=['id_auto', 'publicacion', 'modelo', 'Combustible', 'Color', 'precio_nuevo'], inplace=True)

        # Convert categorical variables into dummies
        categories_list = ["marca", "Motor", "Tipo", "Transmision"]
        dataset_with_dummies = pd.get_dummies(data=dataset,
                                              columns=categories_list,
                                              drop_first=True)

        wr.s3.to_csv(df=dataset_with_dummies,
                     path=data_end_path,
                     index=False)

        # Save information of the dataset
        client = boto3.client('s3')

        data_dict = {}
        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                # Something else has gone wrong.
                raise e

        target_col = Variable.get("target_col_used_cars")
        dataset_log = dataset.drop(columns=target_col)
        dataset_with_dummies_log = dataset_with_dummies.drop(columns=target_col)

        # Upload JSON String to an S3 Object
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['columns_after_dummy'] = dataset_with_dummies_log.columns.to_list()
        data_dict['target_col'] = target_col
        data_dict['categorical_columns'] = categories_list
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_log.dtypes.to_dict().items()}
        data_dict['columns_dtypes_after_dummy'] = {k: str(v) for k, v in dataset_with_dummies_log.dtypes
                                                                                                 .to_dict()
                                                                                                 .items()}

        category_dummies_dict = {}
        for category in categories_list:
            category_dummies_dict[category] = np.sort(dataset_log[category].unique()).tolist()

        data_dict['categories_values_per_categorical'] = category_dummies_dict

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Used Cars")

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "etl", "dataset": "Used Cars"},
                         log_system_metrics=True)

        mlflow_dataset = mlflow.data.from_pandas(dataset,
                                                 source="s3://data/raw/merged_data_03_07.xlsx",
                                                 targets=target_col,
                                                 name="used_cars_data_complete")
        mlflow_dataset_dummies = mlflow.data.from_pandas(dataset_with_dummies,
                                                         source="s3://data/raw/merged_data_03_07.xlsx",
                                                         targets=target_col,
                                                         name="used_cars_data_complete_with_dummies")
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_dummies, context="Dataset")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2"],
        system_site_packages=True
    )
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        data_original_path = "s3://data/raw/used_cars_data.csv"
        dataset = wr.s3.read_csv(data_original_path)

        test_size = Variable.get("test_size_used_cars")
        target_col = Variable.get("target_col_used_cars")

        X = dataset.drop(columns=target_col)
        y = dataset[[target_col]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # Clean duplicates
        dataset.drop_duplicates(inplace=True, ignore_index=True)

        save_to_csv(X_train, "s3://data/final/train/used_cars_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/used_cars_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/used_cars_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/used_cars_y_test.csv")

    @task.virtualenv(
        task_id="normalize_numerical_features",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2",
                      "mlflow==2.10.2"],
        system_site_packages=True
    )
    def normalize_data():
        """
        Standardization of numerical columns
        """
        import json
        import mlflow
        import boto3
        import botocore.exceptions

        import awswrangler as wr
        import pandas as pd

        from sklearn.preprocessing import StandardScaler

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        X_train = wr.s3.read_csv("s3://data/final/train/used_cars_X_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/used_cars_X_test.csv")

        sc_X = StandardScaler(with_mean=True, with_std=True)
        X_train_arr = sc_X.fit_transform(X_train)
        X_test_arr = sc_X.transform(X_test)

        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

        save_to_csv(X_train, "s3://data/final/train/used_cars_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/used_cars_X_test.csv")

        # Save information of the dataset
        client = boto3.client('s3')

        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
                # Something else has gone wrong.
                raise e

        # Upload JSON String to an S3 Object
        data_dict['standard_scaler_mean'] = sc_X.mean_.tolist()
        data_dict['standard_scaler_std'] = sc_X.scale_.tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Used Cars")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):

            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Standard Scaler feature names", sc_X.feature_names_in_)
            mlflow.log_param("Standard Scaler mean values", sc_X.mean_)
            mlflow.log_param("Standard Scaler scale values", sc_X.scale_)


    get_data() >> make_dummies_variables() >> split_dataset() >> normalize_data()


dag = process_etl_used_cars_data()