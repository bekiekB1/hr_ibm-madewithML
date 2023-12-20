from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sklearn.ensemble import RandomForestClassifier

# Assuming you have a trained RandomForestClassifier
rf_model = RandomForestClassifier()

# Define application
app = FastAPI(title="Made With ML Tutorial", description="Employee Attrition Prediction", version="0.1")

# Replace with Best run ID path
weight_path = r"log/mlflow/234382354845373750/bc12b4ac1d7d4bb19ab6c33c63995422/artifacts/model_weight/random_forest_model.joblib"

rf_model = joblib.load(weight_path)


# Define a root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Employee Attrition Prediction API"}


# Endpoint for making predictions
@app.post("/predict/")
async def predict(request: Request, data: dict):
    try:
        data_unseen = data["data"]
        # Assuming sample_test is a list of dictionaries
        df = pd.DataFrame([data_unseen])
        # Make predictions using the trained model
        rf_predictions = rf_model.predict(df)
        rf_probabilities = rf_model.predict_proba(df)

        # Format results
        results = [
            {"prediction": int(prediction), "probabilities": probabilities.tolist()}
            for prediction, probabilities in zip(rf_predictions, rf_probabilities)
        ]

        return JSONResponse(content={"results": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
