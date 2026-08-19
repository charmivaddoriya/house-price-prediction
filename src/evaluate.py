from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import load_data, prepare_data, split_data


def evaluate_model():
    """Load the trained model and evaluate its performance."""

    # Load and prepare data
    df = load_data()
    X, y = prepare_data(df)

    # Use the same train/test split as training
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Load trained model
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "house_price_model.pkl"

    model = joblib.load(model_path)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("=" * 50)
    print("HOUSE PRICE MODEL EVALUATION")
    print("=" * 50)
    print(f"MAE      : {mae:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"R² Score : {r2:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_model()