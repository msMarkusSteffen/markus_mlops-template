import mlflow
import mlflow.sklearn
from fastapi import FastAPI
import pandas as pd

# MLflow Tracking-Server setzen
mlflow.set_tracking_uri("http://localhost:5001")

MODEL_NAME = "FlowerPower"
STAGE = "None"  # "Staging" oder "Production", wenn du willst

# Model aus der Registry laden
def load_model(model_name: str, stage: str = "None"):
    model_uri = f"models:/{model_name}/{stage}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model '{model_name}' from stage '{stage}'")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model(MODEL_NAME, STAGE)

# FastAPI Endpoint
app = FastAPI()

@app.get("/predict")
def predict(data: list):
    if model is None:
        return {"error": "Model not ready"}
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}
