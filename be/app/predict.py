import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR.parent / "model" / "diabetes_pipeline.pkl"

print("Loading model from:", MODEL_PATH)

model = joblib.load(MODEL_PATH)

def predict_diabetes(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]
    return int(pred), float(prob)
