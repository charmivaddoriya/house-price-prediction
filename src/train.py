from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression

from data_preprocessing import load_data, prepare_data, split_data


def train_model():
    """Train the Linear Regression model and save it."""

    # Load and prepare data
    df = load_data()
    X, y = prepare_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create models directory
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save trained model
    model_path = models_dir / "house_price_model.pkl"
    joblib.dump(model, model_path)

    print("Model training completed successfully")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    train_model()