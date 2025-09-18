# src/train.py
import os
import argparse
import yaml
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import mlflow
import mlflow.sklearn

from src.utils import ensure_dirs, save_model

def main(config_path="config.yaml", tracking_uri=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("random_seed", 42)
    test_size = cfg.get("test_size", 0.2)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    ensure_dirs(["models", "results"])

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=seed, stratify=data.target
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=cfg["models"]["LogisticRegression"]["max_iter"]),
        "RandomForest": RandomForestClassifier(n_estimators=cfg["models"]["RandomForestClassifier"]["n_estimators"], random_state=seed),
        "SVC": SVC(probability=cfg["models"]["SVC"]["probability"])
    }

    best_f1 = -1.0
    best_run_id = None
    best_model_name = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="macro", zero_division=0)
            rec = recall_score(y_test, preds, average="macro", zero_division=0)
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)

            # Log some params (safe subset)
            try:
                params_to_log = {}
                p = model.get_params()
                for key in ["C","kernel","n_estimators","max_iter","probability","random_state"]:
                    if key in p:
                        params_to_log[key] = p[key]
                if params_to_log:
                    mlflow.log_params(params_to_log)
            except Exception:
                pass

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1", f1)

            # confusion matrix
            cm = confusion_matrix(y_test, preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            cm_path = f"results/confusion_matrix_{name}.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")

            # save model locally and log via mlflow
            local_model_path = f"models/{name}.joblib"
            save_model(model, local_model_path)
            mlflow.log_artifact(local_model_path, artifact_path="saved_models")
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_id = mlflow.active_run().info.run_id
            print(f"Completed {name} run_id={run_id} f1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_run_id = run_id
                best_model_name = name

    with open("results/best_run.txt", "w") as f:
        f.write(f"{best_model_name},{best_run_id},{best_f1}")

    print("Best:", best_model_name, best_run_id, best_f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tracking_uri", default=None)
    args = parser.parse_args()
    main(config_path=args.config, tracking_uri=args.tracking_uri)
