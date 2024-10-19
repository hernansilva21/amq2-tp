import json
import pickle
import boto3
import mlflow
import traceback

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from typing import List



def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.
    """
    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        print("27")
        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        print("exito")
        print("30")
        model_ml = mlflow.catboost.load_model(model_data_mlflow.source)
        print("exito")
        print("33")
        version_model_ml = int(model_data_mlflow.version)
        print("exito")
    except Exception as e:
        print("excepcion")
        traceback.print_exc() 
        # If there is no registry in MLflow, open the default model
        with open('/app/files/model.pkl', 'rb') as file_ml:
            print("load pickle local")
            model_ml = pickle.load(file_ml)
            print("exito")
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
    except Exception as e:
        print(f"Error loading data dictionary from S3: {e}")
        # If data dictionary is not found in S3, load it from local file
        with open('/app/files/data.json', 'r') as file_s3:
            data_dictionary = json.load(file_s3)

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.
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

    except Exception as e:
        print(f"Error checking for model updates: {e}")
        # If an error occurs during the process, pass silently
        pass


class ModelInput(BaseModel):
    """
    Input schema for the used cars prediction model.
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
print("Cargar modelo")
model, version_model, data_dict = load_model("used_cars_model_prod", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the Used Cars API.
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
    """
    # Extract features from the request and convert them into a list and dictionary
    features_dict = features.dict()
    features_df = pd.DataFrame([features_dict])

    # Preprocess the features
    # Ensure the data types are correct
    features_df['Ano'] = features_df['Ano'].astype(int)

    # If you used dummy variables and scaling during training, apply the same transformations
    # Process categorical features
    for categorical_col in data_dict["categorical_columns"]:
        # Map categorical variables to categories used during training
        categories = data_dict["categories_values_per_categorical"][categorical_col]
        features_df[categorical_col] = pd.Categorical(features_df[categorical_col], categories=categories)

    # Convert categorical features into dummy variables
    features_df = pd.get_dummies(
        data=features_df,
        columns=data_dict["categorical_columns"],
        drop_first=True
    )

    # Reorder DataFrame columns to match the model's expected input
    features_df = features_df.reindex(columns=data_dict["columns_after_dummy"], fill_value=0)

    # Scale the data using standard scaler if it was used during training
    features_df = (features_df - data_dict["standard_scaler_mean"]) / data_dict["standard_scaler_std"]

    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=int(prediction[0]))


class MarcasOutput(BaseModel):
    """
    Output schema for the list of car brands.
    """
    marcas: List[str] = Field(
        description="List of available car brands",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "marcas": ["Nissan", "Toyota", "Honda"]
                }
            ]
        }
    }

@app.get("/marcas/", response_model=MarcasOutput)
def get_marcas():
    """
    Endpoint to get the list of available car brands.
    """
    try:
        # Extract the list of car brands from the data dictionary
        marcas = data_dict["categories_values_per_categorical"]["marca"]
        return MarcasOutput(marcas=marcas)
    except KeyError:
        # Handle the case where 'marca' is not in the data dictionary
        return JSONResponse(
            status_code=500,
            content={"message": "Error retrieving car brands from data dictionary."}
        )
