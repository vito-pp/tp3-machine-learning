from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureGroups:
    numeric: list[str]
    categorical: list[str]


def identify_feature_groups(X: pd.DataFrame) -> FeatureGroups:
    categorical = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    numeric = X.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()

    classified = set(numeric) | set(categorical)
    unclassified = sorted(set(X.columns) - classified)

    if unclassified:
        raise ValueError(
            "Some columns could not be classified: "
            f"{unclassified}"
        )

    return FeatureGroups(
        numeric=numeric,
        categorical=categorical,
    )


def build_preprocessor(feature_groups: FeatureGroups) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, feature_groups.numeric),
            ("categorical", categorical_pipeline, feature_groups.categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    feature_groups = identify_feature_groups(X_train)
    preprocessor = build_preprocessor(feature_groups)

    train_array = preprocessor.fit_transform(X_train)
    test_array = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    train_transformed = pd.DataFrame(
        train_array,
        columns=feature_names,
        index=X_train.index,
    )

    test_transformed = pd.DataFrame(
        test_array,
        columns=feature_names,
        index=X_test.index,
    )

    return train_transformed, test_transformed, preprocessor


def save_preprocessor(
    preprocessor: ColumnTransformer,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
