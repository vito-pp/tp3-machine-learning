from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold


@dataclass(frozen=True)
class BaselineSelection:
    model_name: str
    best_estimator: Any
    best_params: dict[str, Any]
    best_f1: float


def build_searches(
    random_state: int,
    n_splits: int,
) -> dict[str, GridSearchCV]:
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "accuracy": "accuracy",
    }

    logistic = GridSearchCV(
        estimator=LogisticRegression(
            max_iter=2000,
            random_state=random_state,
        ),
        param_grid={
            "C": [0.1, 1.0, 10.0],
            "class_weight": [None, "balanced"],
        },
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    random_forest = GridSearchCV(
        estimator=RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        param_grid={
            "max_depth": [None, 15],
            "min_samples_leaf": [1, 3],
            "class_weight": [None, "balanced_subsample"],
        },
        scoring=scoring,
        refit="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    return {
        "logistic_regression": logistic,
        "random_forest": random_forest,
    }


def train_and_select_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_splits: int,
) -> tuple[BaselineSelection, pd.DataFrame]:
    searches = build_searches(
        random_state=random_state,
        n_splits=n_splits,
    )

    rows: list[dict[str, Any]] = []
    selected: BaselineSelection | None = None

    for model_name, search in searches.items():
        search.fit(X_train, y_train)

        cv_results = pd.DataFrame(search.cv_results_)
        best_index = search.best_index_

        row = {
            "model": model_name,
            "best_params": str(search.best_params_),
            "mean_cv_f1": float(cv_results.loc[best_index, "mean_test_f1"]),
            "std_cv_f1": float(cv_results.loc[best_index, "std_test_f1"]),
            "mean_cv_recall": float(
                cv_results.loc[best_index, "mean_test_recall"]
            ),
            "mean_cv_precision": float(
                cv_results.loc[best_index, "mean_test_precision"]
            ),
            "mean_cv_accuracy": float(
                cv_results.loc[best_index, "mean_test_accuracy"]
            ),
        }
        rows.append(row)

        candidate = BaselineSelection(
            model_name=model_name,
            best_estimator=search.best_estimator_,
            best_params=search.best_params_,
            best_f1=float(search.best_score_),
        )

        if selected is None or candidate.best_f1 > selected.best_f1:
            selected = candidate

    if selected is None:
        raise RuntimeError("No baseline model was trained.")

    comparison = (
        pd.DataFrame(rows)
        .sort_values("mean_cv_f1", ascending=False)
        .reset_index(drop=True)
    )

    return selected, comparison


def save_baseline(
    selection: BaselineSelection,
    comparison: pd.DataFrame,
    model_path: str | Path,
    comparison_path: str | Path,
) -> None:
    model_path = Path(model_path)
    comparison_path = Path(comparison_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(selection.best_estimator, model_path)
    comparison.to_csv(comparison_path, index=False)
