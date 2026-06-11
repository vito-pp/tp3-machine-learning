from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


TARGET = "canceled_job"
COHORT_COL = "client_type"
COHORT_VALUE = "internal_research"
RANDOM_STATE = 42

DROP_COLUMNS = [
    "total_gpu_hours",
    "total_processes",
    "request_week",
    "queue_wait_time",
]


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors="ignore")


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


def apply_smote_to_problem_cohort(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cohort_mask: np.ndarray,
    return_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Apply SMOTE only to the problematic cohort.
    The rest of the training set is left untouched.
    """
    X_cohort = X_train[cohort_mask]
    y_cohort = y_train[cohort_mask]
    X_rest = X_train[~cohort_mask]
    y_rest = y_train[~cohort_mask]

    classes, counts = np.unique(y_cohort, return_counts=True)
    if len(classes) < 2 or counts.min() < 2:
        stats = {
            "cohort_original_n": int(len(y_cohort)),
            "cohort_original_minority_n": int(counts.min()) if len(counts) else 0,
            "cohort_original_majority_n": int(counts.max()) if len(counts) else 0,
            "cohort_resampled_n": int(len(y_cohort)),
            "cohort_synthetic_n": 0,
        }
        if return_stats:
            return X_train, y_train, stats
        return X_train, y_train

    smote = SMOTE(random_state=RANDOM_STATE)
    X_cohort_resampled, y_cohort_resampled = smote.fit_resample(X_cohort, y_cohort)

    X_aug = np.vstack([X_rest, X_cohort_resampled])
    y_aug = np.concatenate([y_rest, y_cohort_resampled])
    stats = {
        "cohort_original_n": int(len(y_cohort)),
        "cohort_original_minority_n": int(counts.min()),
        "cohort_original_majority_n": int(counts.max()),
        "cohort_resampled_n": int(len(y_cohort_resampled)),
        "cohort_synthetic_n": int(len(y_cohort_resampled) - len(y_cohort)),
    }
    if return_stats:
        return X_aug, y_aug, stats
    return X_aug, y_aug


def apply_smote_to_full_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


@dataclass
class CohortSmoteRandomForest:
    preprocessor: ColumnTransformer | None = None
    model: RandomForestClassifier | None = None
    smote_stats_: dict[str, int] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CohortSmoteRandomForest":
        X_clean = clean_features(X)
        self.preprocessor = build_preprocessor(X_clean)
        X_prepared = self.preprocessor.fit_transform(X_clean)
        y_array = y.to_numpy()
        cohort_mask = X[COHORT_COL].eq(COHORT_VALUE).to_numpy()
        X_aug, y_aug, self.smote_stats_ = apply_smote_to_problem_cohort(
            X_prepared,
            y_array,
            cohort_mask,
            return_stats=True,
        )

        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        self.model.fit(X_aug, y_aug)
        return self

    def _prepare_input(self, X: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None:
            raise RuntimeError("Model is not fitted yet.")

        X_clean = clean_features(X)
        if TARGET in X_clean.columns:
            X_clean = X_clean.drop(columns=[TARGET])
        return self.preprocessor.transform(X_clean)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(self._prepare_input(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(self._prepare_input(X))
