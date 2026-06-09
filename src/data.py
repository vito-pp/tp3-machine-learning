from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)
    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    return dataframe


def validate_dataset(
    dataframe: pd.DataFrame,
    target_column: str,
    cohort_columns: list[str],
) -> None:
    required_columns = {target_column, *cohort_columns}
    missing_columns = sorted(required_columns - set(dataframe.columns))

    if missing_columns:
        raise ValueError(
            f"Required columns are missing from the dataset: {missing_columns}"
        )

    if dataframe[target_column].isna().any():
        raise ValueError(f"Target '{target_column}' contains missing values.")

    target_values = set(dataframe[target_column].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Target '{target_column}' must contain only 0 and 1. "
            f"Found: {sorted(target_values)}"
        )


def split_dataframe(
    dataframe: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        stratify=dataframe[target_column],
    )

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X = dataframe.drop(columns=[target_column]).copy()
    y = dataframe[target_column].copy()
    return X, y
