from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError as exc:
    raise ImportError(
        "Missing dependency: imbalanced-learn. Install it with: "
        "python -m pip install imbalanced-learn"
    ) from exc


TARGET = "canceled_job"
COHORT_COL = "client_type"
COHORT_VALUE = "internal_research"
RANDOM_STATE = 42
N_SPLITS = 5

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
EXTRA_PATH = Path("data/raw/extra_client_type_internal_research.csv")

OUTPUT_COMPARISON_PATH = Path("results/strategy_comparison_cv.csv")
OUTPUT_OOF_PATH = Path("results/strategy_oof_predictions.csv")
OUTPUT_METADATA_PATH = Path("results/strategy_comparison_metadata.json")

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


def build_random_forest_pipeline(X: pd.DataFrame, use_smote: bool) -> Any:
    preprocessor = build_preprocessor(X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if use_smote:
        return ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", model),
            ]
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


def evaluate_strategy_cv(
    strategy_name: str,
    train_df: pd.DataFrame,
    extra_df: pd.DataFrame | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    X_all, y_all = split_xy(train_df)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y_pred_all = np.full(shape=len(train_df), fill_value=-1, dtype=int)
    fold_ids = np.full(shape=len(train_df), fill_value=-1, dtype=int)

    if strategy_name == "smote":
        base_estimator = build_random_forest_pipeline(X_all, use_smote=True)
    else:
        base_estimator = build_random_forest_pipeline(X_all, use_smote=False)

    for fold_id, (train_idx, val_idx) in enumerate(cv.split(X_all, y_all), start=1):
        X_fold_train = X_all.iloc[train_idx].copy()
        y_fold_train = y_all.iloc[train_idx].copy()
        X_fold_val = X_all.iloc[val_idx].copy()

        if strategy_name == "real_data":
            if extra_df is None:
                raise ValueError("The real_data strategy requires the extra dataset.")
            X_extra, y_extra = split_xy(extra_df)
            X_fold_train = pd.concat([X_fold_train, X_extra], axis=0, ignore_index=True)
            y_fold_train = pd.concat([y_fold_train, y_extra], axis=0, ignore_index=True)

        estimator = clone(base_estimator)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(X_fold_train, y_fold_train)

        y_pred = estimator.predict(X_fold_val)

        y_pred_all[val_idx] = y_pred
        fold_ids[val_idx] = fold_id

    if np.any(y_pred_all == -1):
        raise RuntimeError("Some rows did not receive an out-of-fold prediction.")

    y_true = y_all.to_numpy()
    global_metrics = calculate_metrics(y_true, y_pred_all)

    cohort_mask = train_df[COHORT_COL].eq(COHORT_VALUE).to_numpy()
    cohort_metrics = calculate_metrics(y_true[cohort_mask], y_pred_all[cohort_mask])

    row = {
        "strategy": strategy_name,
        "global_n": global_metrics["n"],
        "global_accuracy": global_metrics["accuracy"],
        "global_precision": global_metrics["precision"],
        "global_recall": global_metrics["recall"],
        "global_f1": global_metrics["f1"],
        "global_tn": global_metrics["tn"],
        "global_fp": global_metrics["fp"],
        "global_fn": global_metrics["fn"],
        "global_tp": global_metrics["tp"],
        "cohort_column": COHORT_COL,
        "cohort_value": COHORT_VALUE,
        "cohort_n": cohort_metrics["n"],
        "cohort_accuracy": cohort_metrics["accuracy"],
        "cohort_precision": cohort_metrics["precision"],
        "cohort_recall": cohort_metrics["recall"],
        "cohort_f1": cohort_metrics["f1"],
        "cohort_tn": cohort_metrics["tn"],
        "cohort_fp": cohort_metrics["fp"],
        "cohort_fn": cohort_metrics["fn"],
        "cohort_tp": cohort_metrics["tp"],
    }

    oof_df = train_df.copy()
    oof_df["strategy"] = strategy_name
    oof_df["fold"] = fold_ids
    oof_df["y_true"] = y_true
    oof_df["y_pred"] = y_pred_all
    oof_df["is_error"] = y_true != y_pred_all

    return row, oof_df


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)

    extra_df = None
    if EXTRA_PATH.exists():
        extra_df = pd.read_csv(EXTRA_PATH)

    rows = []
    oof_frames = []

    for strategy in ["baseline", "real_data", "smote"]:
        if strategy == "real_data" and extra_df is None:
            print(f"Skipping {strategy}: missing {EXTRA_PATH}")
            continue

        print(f"Evaluating strategy: {strategy}")
        row, oof_df = evaluate_strategy_cv(
            strategy_name=strategy,
            train_df=train_df,
            extra_df=extra_df,
        )
        rows.append(row)
        oof_frames.append(oof_df)

    comparison = (
        pd.DataFrame(rows)
        .sort_values(
            by=["global_f1", "cohort_f1"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    OUTPUT_COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUTPUT_COMPARISON_PATH, index=False)
    pd.concat(oof_frames, axis=0, ignore_index=True).to_csv(OUTPUT_OOF_PATH, index=False)

    metadata = {
        "target": TARGET,
        "cohort_column": COHORT_COL,
        "cohort_value": COHORT_VALUE,
        "selection_rule": "Best global_f1, with cohort_f1 as tie-breaker.",
        "recommended_strategy": comparison.loc[0, "strategy"],
        "notes": [
            "All strategies are compared with the same stratified CV splits over the original train set.",
            "SMOTE is applied inside each fold only to the sub-train split.",
            "The validation fold always contains real, non-synthetic data.",
            "The test set is not used in this comparison.",
        ],
    }
    OUTPUT_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nStrategy comparison:")
    print(comparison.to_string(index=False))

    print("\nRecommended strategy:")
    print(f"  {metadata['recommended_strategy']}")

    print("\nSaved:")
    print(f"  {OUTPUT_COMPARISON_PATH}")
    print(f"  {OUTPUT_OOF_PATH}")
    print(f"  {OUTPUT_METADATA_PATH}")


if __name__ == "__main__":
    main()
