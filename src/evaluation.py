from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict


def load_model(model_path: str | Path) -> Any:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def generate_oof_predictions(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    estimator = clone(model)

    return cross_val_predict(
        estimator,
        X,
        y,
        cv=cv,
        method="predict",
        n_jobs=-1,
    )


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def analyze_cohorts(
    raw_train: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    cohort_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for column in cohort_columns:
        for value in sorted(raw_train[column].dropna().unique(), key=str):
            mask = raw_train[column] == value
            metrics = calculate_metrics(
                y_true=np.asarray(y_true)[mask.to_numpy()],
                y_pred=np.asarray(y_pred)[mask.to_numpy()],
            )

            positives = int(np.asarray(y_true)[mask.to_numpy()].sum())
            cancellation_rate = positives / metrics["n"] if metrics["n"] else 0.0

            rows.append(
                {
                    "cohort_feature": column,
                    "cohort_value": value,
                    "n": metrics["n"],
                    "cancellation_rate": cancellation_rate,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tp": metrics["tp"],
                }
            )

    return pd.DataFrame(rows)


def save_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
