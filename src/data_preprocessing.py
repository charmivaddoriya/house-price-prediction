from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    """Load the raw housing dataset."""
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "housing.csv"

    return pd.read_csv(data_path)


def prepare_data(df):
    """Prepare features and target for model training."""
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    return X, y


def split_data(X, y, test_size=0.20, random_state=42):
    """Split the dataset into training and testing sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Data preparation successful")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")