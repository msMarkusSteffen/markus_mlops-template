import os
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# S3 / MinIO Credentials im Skript setzen
os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
os.environ["AWS_SECRET_ACCESS_KEY"] = "mlflow123"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

# 1. Daten vom ETL-Service holen
response = requests.get("http://localhost:8000/data")
data = response.json()
df = pd.DataFrame(data)

print(df.head())
# 2. Features/Labels definieren
X = df.drop("target", axis=1)
y = df["target"]

# 3. Split + Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. MLflow Logging
mlflow.set_tag("Dataset", "iris") # Im Experiment gibt es eine Tabelle wo auch die Spalte Dataset auftaucht
mlflow.set_tracking_uri("http://localhost:5001") 
mlflow.set_experiment("irisflower")
with mlflow.start_run(nested=True):
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.sklearn.log_model(model, "FlowerPower")
    score = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", score)
    print("Accuracy:", score)
    #mlflow.end_run() # NOTE muss vor log model sonst Beule !

    #mlflow.sklearn.log_model(model, "FlowerPower") # NOTE optional wenn man log_model macht wird auch das Model hochgeladen, 
    model_name = "FlowerPower"
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=model_name  # <-- Modell wird registriert
    )
    # NOTE für echte Versionierung muss man register_model machen
    
