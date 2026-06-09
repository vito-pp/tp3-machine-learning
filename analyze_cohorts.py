import numpy as np
import pandas as pd

from src.data import load_config, load_dataset, split_features_target
from src.evaluation import (
    analyze_cohorts,
    calculate_metrics,
    generate_oof_predictions,
    load_model,
    save_dataframe,
)


def main() -> None:
    config = load_config()
    target = config["project"]["target"]

    raw_train = load_dataset(config["data"]["train_path"])
    transformed_train = load_dataset(
        config["data"]["train_transformed_path"]
    )

    X_train, y_train = split_features_target(
        transformed_train,
        target,
    )

    if len(raw_train) != len(transformed_train):
        raise ValueError(
            "Raw and transformed train datasets have different row counts."
        )

    model = load_model(config["baseline"]["model_path"])

    y_pred = generate_oof_predictions(
        model=model,
        X=X_train,
        y=y_train,
        n_splits=config["validation"]["n_splits"],
        random_state=config["project"]["random_state"],
    )

    global_metrics = pd.DataFrame([calculate_metrics(y_train, y_pred)])
    save_dataframe(
        global_metrics,
        config["evaluation"]["global_metrics_path"],
    )

    predictions = raw_train.copy()
    predictions["y_true"] = y_train.to_numpy()
    predictions["y_pred"] = y_pred
    predictions["is_error"] = (
        predictions["y_true"] != predictions["y_pred"]
    )
    save_dataframe(
        predictions,
        config["evaluation"]["predictions_path"],
    )

    cohort_metrics = analyze_cohorts(
        raw_train=raw_train,
        y_true=y_train,
        y_pred=y_pred,
        cohort_columns=config["cohorts"]["columns"],
    )
    save_dataframe(
        cohort_metrics,
        config["evaluation"]["cohort_metrics_path"],
    )

    false_positives = predictions[
        (predictions["y_true"] == 0)
        & (predictions["y_pred"] == 1)
    ]
    false_negatives = predictions[
        (predictions["y_true"] == 1)
        & (predictions["y_pred"] == 0)
    ]

    save_dataframe(
        false_positives,
        config["evaluation"]["false_positives_path"],
    )
    save_dataframe(
        false_negatives,
        config["evaluation"]["false_negatives_path"],
    )

    print("\nGlobal out-of-fold validation metrics:")
    print(global_metrics.to_string(index=False))

    print("\nWorst cohorts by F1 (minimum 30 rows):")
    print(
        cohort_metrics[cohort_metrics["n"] >= 30]
        .sort_values(["f1", "n"], ascending=[True, False])
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved:")
    for path in config["evaluation"].values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
