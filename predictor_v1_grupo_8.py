from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


MODEL_PATH = Path("models/final_model.joblib")
TARGET = "canceled_job"

DROP_COLUMNS = [
    "total_gpu_hours",
    "total_processes",
    "request_week",
    "queue_wait_time",
]


def _load_input(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()

    return pd.read_csv(data)


def _clean_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")


def predictor_v1_Grupo_8(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    """
    Parameters
    ----------
    data:
        Either a pandas DataFrame or a CSV path with the same input columns
        used during training. The target column may be present; if present,
        it is ignored.

    Returns
    -------
    pandas.DataFrame
        A dataframe with:
        - prediction: predicted class, 0 or 1.
        - probability_canceled_job: predicted probability for class 1,
          when the trained model supports predict_proba.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

    model: Any = joblib.load(MODEL_PATH)

    raw_df = _load_input(data)
    X = _clean_input(raw_df)

    predictions = model.predict(X)

    result = pd.DataFrame(
        {
            "prediction": predictions,
        }
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        result["probability_canceled_job"] = probabilities[:, 1]

    return result


if __name__ == "__main__":
    sample_path = Path("data/processed/test.csv")

    if not sample_path.exists():
        raise FileNotFoundError(f"Missing sample input: {sample_path}")

    predictions = predictor_v1_Grupo_8(sample_path)
    print(predictions.head().to_string(index=False))
