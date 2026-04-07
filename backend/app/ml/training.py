from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from app.ml.data import ensure_dataset, load_dataset


RANDOM_SEED = 42
MODEL_ARTIFACT_NAME = "diabetes_model.joblib"
MODEL_FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


@dataclass
class TrainingResult:
    artifact_path: Path
    metadata: dict


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
    return metrics


def _build_preprocessors(columns: list[str]) -> tuple[ColumnTransformer, ColumnTransformer]:
    num_cols = columns
    numeric_transform = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocess_onehot = ColumnTransformer(
        transformers=[("num", numeric_transform, num_cols)],
        remainder="drop",
    )
    preprocess_ordinal = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
        ],
        remainder="drop",
    )
    return preprocess_onehot, preprocess_ordinal


def _build_models(preprocess_onehot: ColumnTransformer, preprocess_ordinal: ColumnTransformer) -> dict[str, Pipeline]:
    base_models = {
        "LogisticRegression": LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=5000,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=RANDOM_SEED,
        ),
        "MLPClassifier": MLPClassifier(
            random_state=RANDOM_SEED,
            max_iter=500,
            hidden_layer_sizes=(128, 64),
            early_stopping=True,
            validation_fraction=0.1,
        ),
    }
    preprocessors = {
        "LogisticRegression": preprocess_onehot,
        "RandomForest": preprocess_onehot,
        "HistGradientBoosting": preprocess_ordinal,
        "MLPClassifier": preprocess_onehot,
    }
    return {
        name: Pipeline([("preprocess", preprocessors[name]), ("model", model)])
        for name, model in base_models.items()
    }


def train_and_save_model(model_dir: Path, data_dir: Path, dataset_url: str, target_column: str) -> TrainingResult:
    artifact_path = model_dir / MODEL_ARTIFACT_NAME
    data_path = data_dir / "diabetes.csv"
    ensure_dataset(data_path, dataset_url)
    df = load_dataset(data_path, target_column)

    X = df[MODEL_FEATURE_COLUMNS].copy()
    y = df[target_column].astype(int).values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    preprocess_onehot, preprocess_ordinal = _build_preprocessors(MODEL_FEATURE_COLUMNS)
    candidate_models = _build_models(preprocess_onehot, preprocess_ordinal)

    best_model_name = None
    best_model = None
    best_metrics = None
    best_score = -1.0
    model_cards: dict[str, dict[str, float]] = {}
    trained_models: dict[str, object] = {}

    for name, pipeline in candidate_models.items():
        pipeline.fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv="prefit")
        calibrated.fit(X_val, y_val)
        trained_models[name] = calibrated
        y_prob = calibrated.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_prob)
        model_cards[name] = metrics
        score = metrics["roc_auc"] + metrics["f1"]
        if score > best_score:
            best_score = score
            best_model_name = name
            best_model = calibrated
            best_metrics = metrics

    metadata = {
        "model_name": best_model_name,
        "trained_at": datetime.now(UTC).isoformat(),
        "target_column": target_column,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "metrics": best_metrics,
        "all_model_metrics": model_cards,
        "available_models": list(trained_models.keys()),
        "dataset_rows": int(df.shape[0]),
        "dataset_path": str(data_path),
    }
    joblib.dump({"model": best_model, "models": trained_models, "metadata": metadata}, artifact_path)
    return TrainingResult(artifact_path=artifact_path, metadata=metadata)
