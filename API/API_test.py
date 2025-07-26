import requests

data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
    "SepalArea":8.6,
    "PetalArea":1.6
}

res = requests.post("http://127.0.0.1:8000/predict", json=data)
print(res.json())
