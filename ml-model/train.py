"""
Iris Flower Classification - Model Training

This module:
1. Loads the Iris dataset
2. Splits the data into training and testing sets
3. Trains a Logistic Regression model
4. Evaluates the model
5. Saves the trained model
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


MODEL_DIR = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "iris_model.joblib"


def train_model():
    """Train and evaluate the Iris classification model."""

    # Load dataset
    iris = load_iris()

    X = iris.data
    y = iris.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Create model
    model = LogisticRegression(max_iter=200)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names,
        )
    )

    # Create model directory
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(
        {
            "model": model,
            "target_names": iris.target_names.tolist(),
            "feature_names": iris.feature_names,
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
