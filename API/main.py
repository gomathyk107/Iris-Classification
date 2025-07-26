import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
from pathlib import Path

app = FastAPI()

# Load model from MLflow Model Registry
model_name = "IrisBestModel"
model_stage = "Staging"  # or "Production"

model_uri = f"models:/{model_name}/{model_stage}"

env = os.environ.get("ENVIRONMENT")
if env == "tests":
        # Relative test path
    mlruns_path = Path(__file__).resolve().parent.parent / "models" / "mlruns"
    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    model_path = os.path.join(mlruns_path ,"709871268375005215/models/m-c24f54082aec4688b1a70e3982650b90/artifacts")
    model = mlflow.pyfunc.load_model(model_path)
else:
    # Docker/Prod path
    mlflow.set_tracking_uri("file:/app/models/mlruns")
    model = mlflow.pyfunc.load_model("file:/app/models/mlruns/709871268375005215/models/m-c24f54082aec4688b1a70e3982650b90/artifacts")
# mlflow.set_tracking_uri("file:../models/mlruns")
# mlflow.set_tracking_uri("file:/app/models/mlruns")
# model = mlflow.pyfunc.load_model("models:/IrisBestModel/2")
# model = mlflow.pyfunc.load_model("file:/app/models/mlruns/709871268375005215/models/m-c24f54082aec4688b1a70e3982650b90/artifacts")
# mlruns_path = Path(__file__).resolve().parent.parent / "models" / "mlruns"
# mlflow.set_tracking_uri(f"file:{mlruns_path}")
# model_path = "D:/MLOPS/Iris-Classification/models/mlruns/709871268375005215/models/m-c24f54082aec4688b1a70e3982650b90/artifacts"
# model = mlflow.pyfunc.load_model(model_path)


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
