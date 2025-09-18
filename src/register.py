# src/register.py
import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "mlops_assignment_model"
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri)

with open("results/best_run.txt") as f:
    name, run_id, f1 = f.read().strip().split(",")

model_uri = f"runs:/{run_id}/model"
print("Registering model:", model_uri)

try:
    # registers a new model version from the model URI
    mv = mlflow.register_model(model_uri, MODEL_NAME)
    print("Registered model version:", mv.version)
except Exception as e:
    print("Register error (maybe already exists) :", e)
