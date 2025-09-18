# src/utils.py
import os
import joblib

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
