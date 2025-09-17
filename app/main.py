from fastapi import FastAPI

# create a FastAPI app
app = FastAPI()

import joblib
from pathlib import Path
import numpy as np

MODEL_PATH = Path("models/model.joblib")
model = joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return {"message": "Hello, the API is running!"}

import pandas as pd

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

@app.post("/predict")
def predict(features: dict):
    # turn dict into DataFrame with column names
    x = pd.DataFrame([features], columns=FEATURES)
    yhat = float(model.predict(x)[0])
    return {"prediction": yhat}
