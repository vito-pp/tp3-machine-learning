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

from src.cohort_smote_model import (
    apply_smote_to_full_train,
    apply_smote_to_problem_cohort,
    build_preprocessor as cohort_build_preprocessor,
    clean_features,
)


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
OUTPUT_SMOTE_COHORT_STATS_PATH = Path("results/smote_cohort_fold_stats.csv")
OUTPUT_SMOTE_COHORT_SUMMARY_PATH = Path("results/smote_cohort_summary.json")
OUTPUT_FOLD_METRICS_PATH = Path("results/strategy_fold_metrics.csv")

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
    del use_smote
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
            ("preprocessor", build_preprocessor(X)),
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
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_raw_all = clean_features(train_df.drop(columns=[TARGET]))
    X_all, y_all = split_xy(train_df)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y_pred_all = np.full(shape=len(train_df), fill_value=-1, dtype=int)
    fold_ids = np.full(shape=len(train_df), fill_value=-1, dtype=int)
    smote_fold_stats: list[dict[str, Any]] = []
    fold_metrics_rows: list[dict[str, Any]] = []

    base_estimator = build_random_forest_pipeline(X_all, use_smote=False)

    for fold_id, (train_idx, val_idx) in enumerate(cv.split(X_all, y_all), start=1):
        X_fold_train = X_all.iloc[train_idx].copy()
        y_fold_train = y_all.iloc[train_idx].copy()
        X_fold_val = X_all.iloc[val_idx].copy()
        X_fold_train_raw = X_raw_all.iloc[train_idx].copy()
        X_fold_val_raw = X_raw_all.iloc[val_idx].copy()

        if strategy_name == "real_data":
            if extra_df is None:
                raise ValueError("The real_data strategy requires the extra dataset.")
            X_extra, y_extra = split_xy(extra_df)
            X_fold_train = pd.concat([X_fold_train, X_extra], axis=0, ignore_index=True)
            y_fold_train = pd.concat([y_fold_train, y_extra], axis=0, ignore_index=True)

        if strategy_name == "smote_cohort":
            preprocessor = cohort_build_preprocessor(X_fold_train_raw)
            X_train_prepared = preprocessor.fit_transform(X_fold_train_raw)
            X_val_prepared = preprocessor.transform(X_fold_val_raw)
            cohort_mask_train = X_fold_train_raw[COHORT_COL].eq(COHORT_VALUE).to_numpy()
            X_aug, y_aug, smote_stats = apply_smote_to_problem_cohort(
                X_train=X_train_prepared,
                y_train=y_fold_train.to_numpy(),
                cohort_mask=cohort_mask_train,
                return_stats=True,
            )
            smote_fold_stats.append(
                {
                    "strategy": strategy_name,
                    "fold": fold_id,
                    "train_fold_n": int(len(y_fold_train)),
                    "cohort_fold_n": int(cohort_mask_train.sum()),
                    **smote_stats,
                    "fold_total_n_after_smote": int(len(y_aug)),
                }
            )

            estimator = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_aug, y_aug)
            y_train_pred = estimator.predict(X_aug)
            y_pred = estimator.predict(X_val_prepared)
        elif strategy_name == "smote":
            preprocessor = build_preprocessor(X_fold_train)
            X_train_prepared = preprocessor.fit_transform(X_fold_train)
            X_val_prepared = preprocessor.transform(X_fold_val)
            X_aug, y_aug = apply_smote_to_full_train(
                X_train=X_train_prepared,
                y_train=y_fold_train.to_numpy(),
                random_state=RANDOM_STATE,
            )

            estimator = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_aug, y_aug)
            y_train_pred = estimator.predict(X_aug)
            y_pred = estimator.predict(X_val_prepared)
        else:
            estimator = clone(base_estimator)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(X_fold_train, y_fold_train)

            y_train_pred = estimator.predict(X_fold_train)
            y_pred = estimator.predict(X_fold_val)

        if strategy_name in {"smote", "smote_cohort"}:
            y_train_true = y_aug
            train_n = int(len(y_aug))
        else:
            y_train_true = y_fold_train.to_numpy()
            train_n = int(len(y_fold_train))

        train_metrics = calculate_metrics(y_train_true, y_train_pred)
        val_metrics = calculate_metrics(y_all.iloc[val_idx].to_numpy(), y_pred)
        fold_metrics_rows.append(
            {
                "strategy": strategy_name,
                "fold": fold_id,
                "train_n": train_n,
                "val_n": int(len(val_idx)),
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
            }
        )

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

    smote_stats_df = pd.DataFrame(smote_fold_stats) if smote_fold_stats else pd.DataFrame()
    fold_metrics_df = pd.DataFrame(fold_metrics_rows)

    return row, oof_df, fold_metrics_df, smote_stats_df


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {TRAIN_PATH}")

    train_df = pd.read_csv(TRAIN_PATH)

    extra_df = None
    if EXTRA_PATH.exists():
        extra_df = pd.read_csv(EXTRA_PATH)

    rows = []
    oof_frames = []
    fold_metrics_frames = []
    smote_stats_frames = []

    for strategy in ["baseline", "real_data", "smote", "smote_cohort"]:
        if strategy == "real_data" and extra_df is None:
            print(f"Skipping {strategy}: missing {EXTRA_PATH}")
            continue

        print(f"Evaluating strategy: {strategy}")
        row, oof_df, fold_metrics_df, smote_stats_df = evaluate_strategy_cv(
            strategy_name=strategy,
            train_df=train_df,
            extra_df=extra_df,
        )
        rows.append(row)
        oof_frames.append(oof_df)
        if not fold_metrics_df.empty:
            fold_metrics_frames.append(fold_metrics_df)
        if not smote_stats_df.empty:
            smote_stats_frames.append(smote_stats_df)

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
    if fold_metrics_frames:
        pd.concat(fold_metrics_frames, axis=0, ignore_index=True).to_csv(OUTPUT_FOLD_METRICS_PATH, index=False)
    if smote_stats_frames:
        pd.concat(smote_stats_frames, axis=0, ignore_index=True).to_csv(OUTPUT_SMOTE_COHORT_STATS_PATH, index=False)

    if OUTPUT_SMOTE_COHORT_STATS_PATH.exists():
        smote_stats_df = pd.read_csv(OUTPUT_SMOTE_COHORT_STATS_PATH)
        smote_summary = {
            "strategy": "smote_cohort",
            "folds": int(len(smote_stats_df)),
            "train_fold_n_mean": float(smote_stats_df["train_fold_n"].mean()),
            "cohort_fold_n_mean": float(smote_stats_df["cohort_fold_n"].mean()),
            "cohort_original_n_mean": float(smote_stats_df["cohort_original_n"].mean()),
            "cohort_resampled_n_mean": float(smote_stats_df["cohort_resampled_n"].mean()),
            "cohort_synthetic_total": int(smote_stats_df["cohort_synthetic_n"].sum()),
            "cohort_synthetic_mean": float(smote_stats_df["cohort_synthetic_n"].mean()),
            "cohort_synthetic_min": int(smote_stats_df["cohort_synthetic_n"].min()),
            "cohort_synthetic_max": int(smote_stats_df["cohort_synthetic_n"].max()),
        }
        OUTPUT_SMOTE_COHORT_SUMMARY_PATH.write_text(
            json.dumps(smote_summary, indent=2),
            encoding="utf-8",
        )

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
    if OUTPUT_SMOTE_COHORT_STATS_PATH.exists():
        print(f"  {OUTPUT_SMOTE_COHORT_STATS_PATH}")
    if OUTPUT_SMOTE_COHORT_SUMMARY_PATH.exists():
        print(f"  {OUTPUT_SMOTE_COHORT_SUMMARY_PATH}")
    if OUTPUT_FOLD_METRICS_PATH.exists():
        print(f"  {OUTPUT_FOLD_METRICS_PATH}")


if __name__ == "__main__":
    main()
