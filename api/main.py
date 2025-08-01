import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi import FastAPI,Request
from pydantic import BaseModel,Field,validator
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow.pyfunc
import numpy as np
from pathlib import Path
from api.logger import log_to_db, log_to_file

app = FastAPI()

Instrumentator().instrument(app).expose(app)
from prometheus_client import start_http_server, Summary
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST,Gauge
from starlette.responses import Response

REQUEST_COUNT = Counter('request_count', 'Total HTTP Requests')
@app.middleware("http")
async def count_requests(request, call_next):
    REQUEST_COUNT.inc()
    response = await call_next(request)
    return response

# @app.get("/metrics")
# def metrics():
    # return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Counter: total predictions made
TOTAL_PREDICTIONS = Counter("total_predictions", "Total number of predictions made")

# Gauge: last predicted class (we'll use 0, 1, 2 for iris classes)
LAST_PREDICTION = Gauge("last_prediction", "Last predicted class")

# Optional: Custom gauge values for inputs, if needed
LAST_SEPAL_LENGTH = Gauge("last_sepal_length", "Sepal length of last input")

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
    sepal_length: float = Field(..., gt=0, lt=10, description="Sepal length in cm (0-10)")
    sepal_width: float  = Field(..., gt=0, lt=10, description="Sepal width in cm (0-10)")
    petal_length: float = Field(..., gt=0, lt=10, description="Petal length in cm (0-10)")
    petal_width: float = Field(..., gt=0, lt=10, description="Petal width in cm (0-10)")
    SepalArea  : float = Field(..., gt=0, lt=100, description="Calculated Sepal area")
    PetalArea  : float = Field(..., gt=0, lt=100, description="Calculated Petal area")

    @validator("*", pre=True)
    def check_not_nan(cls, value):
        if value is None:
            raise ValueError("Value must not be null")
        return value

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

    TOTAL_PREDICTIONS.inc()
    LAST_PREDICTION.set(int(prediction[0]))  # assuming prediction is 0,1,2
    # LAST_SEPAL_LENGTH.set(features[[0]])

    # Log to file + DB
    log_to_file(features, prediction[0])
    log_to_db(features, prediction[0])
    return {"prediction": int(prediction[0])}

@app.get("/metrics")
def get_metrics():
    from sqlalchemy import func
    from api.logger import PredictionLog, session

    total_preds = session.query(func.count(PredictionLog.id)).scalar()
    last_pred = session.query(PredictionLog).order_by(PredictionLog.id.desc()).first()

    return {
        "total_predictions": total_preds,
        "last_prediction": last_pred.prediction[0] if last_pred else None,
        "last_input": last_pred.input_data if last_pred else None
    }

@app.get("/prometheus-metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

