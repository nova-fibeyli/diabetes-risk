from pathlib import Path

import pandas as pd


DATASET_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]


def ensure_dataset(data_path: Path, dataset_url: str) -> Path:
    if data_path.exists():
        return data_path
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(dataset_url)
    df.to_csv(data_path, index=False)
    return data_path


def load_dataset(data_path: Path, target_column: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = [str(col).strip() for col in df.columns]
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")
    required_features = [col for col in DATASET_COLUMNS if col != "Outcome"]
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise KeyError(f"Dataset is missing required columns: {missing}")
    return df

