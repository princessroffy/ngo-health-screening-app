from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATASET_PATH = DATA_DIR / "health_risk_dataset.csv"
MODEL_PATH = MODEL_DIR / "health_risk_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

FEATURE_COLUMNS = [
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "blood_sugar",
    "cholesterol",
    "activity_level",
    "smoking",
    "family_history",
]


def build_synthetic_dataset(rows=900, seed=42):
    rng = np.random.default_rng(seed)

    bmi = rng.normal(27, 5.5, rows).clip(16, 45)
    systolic_bp = rng.normal(128, 22, rows).clip(85, 210)
    diastolic_bp = rng.normal(82, 14, rows).clip(50, 130)
    blood_sugar = rng.normal(118, 42, rows).clip(60, 330)
    cholesterol = rng.normal(195, 44, rows).clip(100, 340)
    activity_level = rng.choice([0, 1, 2], rows, p=[0.35, 0.45, 0.20])
    smoking = rng.choice([0, 1], rows, p=[0.78, 0.22])
    family_history = rng.choice([0, 1], rows, p=[0.70, 0.30])

    risk_points = (
        (bmi >= 30).astype(int) * 1.2
        + (systolic_bp >= 140).astype(int) * 1.7
        + (diastolic_bp >= 90).astype(int) * 1.2
        + (blood_sugar >= 126).astype(int) * 1.7
        + (cholesterol >= 240).astype(int) * 1.3
        + (activity_level == 0).astype(int) * 0.8
        + smoking * 1.0
        + family_history * 0.9
    )

    risk_level = np.where(
        risk_points >= 4.5,
        "High",
        np.where(risk_points >= 2.2, "Moderate", "Low"),
    )

    return pd.DataFrame(
        {
            "bmi": bmi.round(1),
            "systolic_bp": systolic_bp.round(0),
            "diastolic_bp": diastolic_bp.round(0),
            "blood_sugar": blood_sugar.round(0),
            "cholesterol": cholesterol.round(0),
            "activity_level": activity_level,
            "smoking": smoking,
            "family_history": family_history,
            "risk_level": risk_level,
        }
    )


def main():
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    dataset = build_synthetic_dataset()
    dataset.to_csv(DATASET_PATH, index=False)

    x = dataset[FEATURE_COLUMNS]
    y = dataset["risk_level"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train_scaled, y_train)

    accuracy = model.score(x_test_scaled, y_test)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Dataset saved to: {DATASET_PATH}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")
    print(f"Validation accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
