"""
Iris Flower Classification - Prediction Module

Loads the trained Iris model and provides
a function for making predictions.
"""

from pathlib import Path

import joblib


MODEL_PATH = Path(__file__).parent / "model" / "iris_model.joblib"


def load_model():
    """Load the trained Iris classification model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run train.py first to create the model."
        )

    return joblib.load(MODEL_PATH)


def predict_flower(features):
    """
    Predict the Iris flower species.

    Args:
        features: Four measurements:
            sepal length
            sepal width
            petal length
            petal width

    Returns:
        Predicted flower species.
    """

    model_data = load_model()

    model = model_data["model"]
    target_names = model_data["target_names"]

    prediction = model.predict([features])[0]

    return target_names[prediction]


if __name__ == "__main__":
    sample = [5.1, 3.5, 1.4, 0.2]

    result = predict_flower(sample)

    print(f"Prediction: {result}")
