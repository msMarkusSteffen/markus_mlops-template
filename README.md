# ML-Ops Iris Flower Template

## Objectives
The goal of this project is to provide a **simple ML-Ops template** based on the Iris Flower dataset. It demonstrates an end-to-end workflow for **training, versioning, and serving machine learning models**, allowing easy experimentation and reuse.

## Architecture & Setup
This project is structured with Docker and consists of multiple services:

- **ETL Service**: FastAPI service that exposes the Iris dataset as JSON for downstream training or visualization.
- **Training Scripts**: Python scripts that train a scikit-learn model, log metrics, and register the model in MLflow.
- **Prediction Service**: FastAPI service that loads the latest registered model from MLflow and provides a `/predict` endpoint.
- **MLflow**: Tracks experiments, metrics, and model versions.
- **MinIO**: Object storage to persist model artifacts.
- **PostgreSQL**: Backend database for MLflow metadata.
- **Dashboard**: (Optional) Plotly Dash dashboard consuming data from the ETL service for visualization.

### Flow
1. **ETL Service** exposes dataset endpoints.
2. **Training Script** fetches data, trains a model, logs metrics & artifacts to MLflow, and optionally registers the model.
3. **Prediction Service** loads the registered model and serves predictions.
4. **MLflow Web UI** allows monitoring experiments and model versions.
5. **MinIO** stores artifacts like model files and environment dependencies.
6. **Dashboard** visualizes data and predictions for quick insights.

### Technology Stack
- Python 3.12
- FastAPI for API services
- scikit-learn for ML models
- MLflow for experiment tracking and model registry
- MinIO for object storage
- PostgreSQL for MLflow metadata
- Docker & Docker Compose for containerized deployment
- Plotly Dash for visualization

## How to setup 
install docker and clone this project into a folder of your choice. Navigate into the folder with command line and run `docker compose up -d` for running and setting the container up. Check the docker-compose.yaml file for the ports, you can access the services via http:\\localshost:PORT and maybe add \data or \predict for the ETL and predict container. `docker compose down`for shutting down the services.

# Ressources 
https://ruhyadi.github.io/blog/mlflow-docker/

# TODO's
- predict container does not deliver dataset nor does it respond with anything
- isolate Ports and only let them use by internal network
- add rabbitmq 