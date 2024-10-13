import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated


def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "used_cars_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass


class ModelInput(BaseModel):
    """
    Input schema for the used cars prediction model.

    This class defines the input fields required by the used cars prediction model along with their descriptions
    and validation constraints.

    :param marca: car brand
    :param Motor: engine displacement 
    :param Ano: model year
    :param Tipo: vehicle segment
    :param Transmision: Vehicle transmission. .
    """

    marca: str = Field(
        description="Car brand"
    )
    Motor: str = Field(
        description="Engine displacement"
    )
    Ano: int = Field(
        description="Model year",
        ge=1900,
        le=2100,
    )
    Tipo: str = Field(
        description="Vehicle segment"
    )
    Transmision: str = Field(
        description="Vehicle transmission"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "marca": "Nissan",
                    "Motor": "1.4 Lt.",
                    "Ano": 2022,
                    "Tipo": "SUV",
                    "Transmision": "manual", 
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output schema for the used cars prediction model.

    This class defines the output fields returned by the used cars prediction model along with their descriptions
    and possible values.

    :param int_output: Output of the model. Car price.
    """

    int_output: int = Field(
        description="Output of the model. Car price",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": 10000
                }
            ]
        }
    }


# Load the model before start
model, version_model, data_dict = load_model("used_cars_model_prod", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the Used Cars API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Used Cars Price API"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting used cars price.

    This endpoint receives features related to a used car and predicts the car price
    using a trained model. It returns the prediction result in integer format.
    """

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)

    # Process categorical features
    for categorical_col in data_dict["categorical_columns"]:
        features_df[categorical_col] = features_df[categorical_col].astype(int)
        categories = data_dict["categories_values_per_categorical"][categorical_col]
        features_df[categorical_col] = pd.Categorical(features_df[categorical_col], categories=categories)

    # Convert categorical features into dummy variables
    features_df = pd.get_dummies(data=features_df,
                                 columns=data_dict["categorical_columns"],
                                 drop_first=True)

    # Reorder DataFrame columns
    features_df = features_df[data_dict["columns_after_dummy"]]

    # Scale the data using standard scaler
    features_df = (features_df-data_dict["standard_scaler_mean"])/data_dict["standard_scaler_std"]

    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=int(prediction[0].item()))