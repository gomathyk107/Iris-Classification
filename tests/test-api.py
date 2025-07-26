# tests/test_api.py

from fastapi.testclient import TestClient
from api.main import app  # adjust path if main.py is elsewhere

client = TestClient(app)

def test_iris_prediction():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "SepalArea":8.6,
        "PetalArea":1.6
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200, "Prediction failed with wrong status code"
    assert "prediction" in response.json(), "No prediction key in response"
    assert isinstance(response.json()["prediction"], str), "Prediction is not a string"
