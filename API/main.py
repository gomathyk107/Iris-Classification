from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np

app = FastAPI()

# Load model from MLflow Model Registry
model_name = "IrisBestModel"
model_stage = "Staging"  # or "Production"

model_uri = f"models:/{model_name}/{model_stage}"
mlflow.set_tracking_uri("file:../models/mlruns")
# mlflow.set_tracking_uri("file:/app/models/mlruns")
model = mlflow.pyfunc.load_model("models:/IrisBestModel/2")
# model = mlflow.pyfunc.load_model("file:/app/models/mlruns/709871268375005215/models/m-c24f54082aec4688b1a70e3982650b90/artifacts")

# Define input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    SepalArea  : float
    PetalArea  : float

@app.get("/")
def read_root():
    return {"message": "Iris ML model API is live!"}

@app.post("/predict")
def predict(data: IrisInput):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
        data.SepalArea,
        data.PetalArea
    ]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
