from pathlib import Path

from sklearn.datasets import fetch_california_housing
import pandas as pd


def main():
    # Project root directory
    project_root = Path(__file__).resolve().parent.parent

    # Raw data directory
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # Download California Housing dataset
    print("Downloading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)

    # Create DataFrame
    df = housing.frame

    # Save dataset
    output_path = raw_data_dir / "housing.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()