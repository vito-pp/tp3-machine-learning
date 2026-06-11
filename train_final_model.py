from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.cohort_smote_model import (
    CohortSmoteRandomForest,
    apply_smote_to_full_train,
)


TARGET = "canceled_job"
COHORT_COL = "client_type"
COHORT_VALUE = "internal_research"
RANDOM_STATE = 42

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
EXTRA_PATH = Path("data/raw/extra_client_type_internal_research.csv")
COMPARISON_PATH = Path("results/strategy_comparison_cv.csv")

FINAL_MODEL_PATH = Path("models/final_model.joblib")
FINAL_METADATA_PATH = Path("models/final_model_metadata.json")
FINAL_TEST_METRICS_PATH = Path("results/final_test_metrics.csv")
FINAL_TEST_PREDICTIONS_PATH = Path("results/final_test_predictions.csv")
FINAL_COHORT_METRICS_PATH = Path("results/final_test_cohort_metrics.csv")

DROP_COLUMNS = [
    "total_gpu_hours",
    "total_processes",
    "request_week",
    "queue_wait_time",
]


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")
    X = clean_features(df.drop(columns=[TARGET]))
    y = df[TARGET].astype(int)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class SimpleSmoteRandomForest:
    def __init__(self, X_reference: pd.DataFrame) -> None:
        self.preprocessor = build_preprocessor(X_reference)
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.smote_stats_: dict[str, int] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SimpleSmoteRandomForest":
        X_prepared = self.preprocessor.fit_transform(X)
        X_aug, y_aug = apply_smote_to_full_train(
            X_train=X_prepared,
            y_train=y.to_numpy(),
            random_state=RANDOM_STATE,
        )
        self.smote_stats_ = {
            "train_original_n": int(len(y)),
            "train_resampled_n": int(len(y_aug)),
            "synthetic_added_n": int(len(y_aug) - len(y)),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X_aug, y_aug)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.drop(columns=[TARGET], errors="ignore")
        return self.model.predict(self.preprocessor.transform(clean_features(X_clean)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.drop(columns=[TARGET], errors="ignore")
        return self.model.predict_proba(self.preprocessor.transform(clean_features(X_clean)))


def build_model_pipeline(X: pd.DataFrame, strategy: str) -> Any:
    if strategy == "smote_cohort":
        return CohortSmoteRandomForest()

    if strategy == "smote":
        return SimpleSmoteRandomForest(X)

    preprocessor = build_preprocessor(X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
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


def load_selected_strategy() -> str:
    if not COMPARISON_PATH.exists():
        print(f"Missing {COMPARISON_PATH}. Defaulting to 'smote'.")
        return "smote"

    comparison = pd.read_csv(COMPARISON_PATH)
    if "strategy" not in comparison.columns:
        raise ValueError(f"{COMPARISON_PATH} must contain a 'strategy' column.")

    return str(comparison.iloc[0]["strategy"])


def prepare_training_data(strategy: str) -> tuple[pd.DataFrame, pd.Series]:
    train_df = pd.read_csv(TRAIN_PATH)

    if strategy == "real_data":
        if not EXTRA_PATH.exists():
            raise FileNotFoundError(f"Missing extra dataset: {EXTRA_PATH}")
        extra_df = pd.read_csv(EXTRA_PATH)
        train_df = pd.concat([train_df, extra_df], axis=0, ignore_index=True)

    X_train, y_train = split_xy(train_df)
    return X_train, y_train


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TEST_PATH}")

    strategy = load_selected_strategy()
    print(f"Selected final strategy: {strategy}")

    X_train, y_train = prepare_training_data(strategy)
    final_model = build_model_pipeline(X_train, strategy=strategy)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_train, y_train)

    test_df = pd.read_csv(TEST_PATH)
    X_test, y_test = split_xy(test_df)

    y_pred = final_model.predict(X_test)

    global_metrics = calculate_metrics(y_test.to_numpy(), y_pred)
    global_metrics_df = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "scope": "global_test",
                **global_metrics,
            }
        ]
    )

    cohort_mask = test_df[COHORT_COL].eq(COHORT_VALUE).to_numpy()
    cohort_metrics = calculate_metrics(y_test.to_numpy()[cohort_mask], y_pred[cohort_mask])
    cohort_metrics_df = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "cohort_column": COHORT_COL,
                "cohort_value": COHORT_VALUE,
                **cohort_metrics,
            }
        ]
    )

    predictions_df = test_df.copy()
    predictions_df["y_true"] = y_test.to_numpy()
    predictions_df["y_pred"] = y_pred
    predictions_df["is_error"] = predictions_df["y_true"] != predictions_df["y_pred"]

    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_TEST_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, FINAL_MODEL_PATH)
    global_metrics_df.to_csv(FINAL_TEST_METRICS_PATH, index=False)
    cohort_metrics_df.to_csv(FINAL_COHORT_METRICS_PATH, index=False)
    predictions_df.to_csv(FINAL_TEST_PREDICTIONS_PATH, index=False)

    metadata = {
        "strategy": strategy,
        "target": TARGET,
        "cohort_column": COHORT_COL,
        "cohort_value": COHORT_VALUE,
        "model": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_leaf": 1,
            "class_weight": "balanced_subsample",
            "random_state": RANDOM_STATE,
        },
        "drop_columns": DROP_COLUMNS,
        "notes": [
            "This is the final model trained after strategy selection.",
            "The test set is used here once for final evaluation.",
            "The saved final model includes preprocessing and the classifier.",
            "If strategy is SMOTE, SMOTE is only used during fit, not during predict.",
        ],
    }
    if hasattr(final_model, "smote_stats_") and getattr(final_model, "smote_stats_", None):
        metadata["smote_stats"] = getattr(final_model, "smote_stats_")
    FINAL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nFinal test metrics:")
    print(global_metrics_df.to_string(index=False))

    print("\nFinal test cohort metrics:")
    print(cohort_metrics_df.to_string(index=False))

    print("\nSaved:")
    print(f"  {FINAL_MODEL_PATH}")
    print(f"  {FINAL_METADATA_PATH}")
    print(f"  {FINAL_TEST_METRICS_PATH}")
    print(f"  {FINAL_COHORT_METRICS_PATH}")
    print(f"  {FINAL_TEST_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
