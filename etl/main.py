# etl/main.py
from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = FastAPI()

# ETL vorbereiten
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

scaler = StandardScaler()
df[iris.feature_names] = scaler.fit_transform(df[iris.feature_names])

@app.get("/data")
def get_data():
    return df.to_dict(orient="records")
