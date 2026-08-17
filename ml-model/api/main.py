"""
FastAPI application for Iris flower prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ml_model.predict import predict_flower


app = FastAPI(
    title="Iris Flower Prediction API",
    description="REST API for predicting Iris flower species.",
    version="1.0.0",
)


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    """Return API information."""
    return {
        "message": "Iris Flower Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: IrisFeatures):
    """Predict Iris flower species."""

    try:
        values = [
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]

        prediction = predict_flower(values)

        return {
            "prediction": prediction,
            "features": values,
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )
