from pathlib import Path

import joblib
import pandas as pd


def predict_house_price():
    """Predict the price of a house using user-provided features."""

    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "house_price_model.pkl"

    # Load trained model
    model = joblib.load(model_path)

    print("=" * 50)
    print("HOUSE PRICE PREDICTION")
    print("=" * 50)

    # Collect property information
    med_inc = float(input("Median income: "))
    house_age = float(input("House age: "))
    ave_rooms = float(input("Average rooms: "))
    ave_bedrms = float(input("Average bedrooms: "))
    population = float(input("Population: "))
    ave_occup = float(input("Average occupancy: "))
    latitude = float(input("Latitude: "))
    longitude = float(input("Longitude: "))

    # Create input DataFrame
    new_house = pd.DataFrame({
        "MedInc": [med_inc],
        "HouseAge": [house_age],
        "AveRooms": [ave_rooms],
        "AveBedrms": [ave_bedrms],
        "Population": [population],
        "AveOccup": [ave_occup],
        "Latitude": [latitude],
        "Longitude": [longitude],
    })

    # Generate prediction
    prediction = model.predict(new_house)[0]

    # Convert from $100,000 units to dollars
    predicted_price = prediction * 100000

    print("\n" + "=" * 50)
    print(f"Predicted House Value: ${predicted_price:,.2f}")
    print("=" * 50)


if __name__ == "__main__":
    predict_house_price()