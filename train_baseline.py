from src.baseline import save_baseline, train_and_select_baseline
from src.data import load_config, load_dataset, split_features_target


def main() -> None:
    config = load_config()
    target = config["project"]["target"]

    train_df = load_dataset(
        config["data"]["train_transformed_path"]
    )

    X_train, y_train = split_features_target(
        train_df,
        target,
    )

    selection, comparison = train_and_select_baseline(
        X_train=X_train,
        y_train=y_train,
        random_state=config["project"]["random_state"],
        n_splits=config["validation"]["n_splits"],
    )

    save_baseline(
        selection=selection,
        comparison=comparison,
        model_path=config["baseline"]["model_path"],
        comparison_path=config["baseline"]["comparison_path"],
    )

    print("\nBaseline comparison:")
    print(comparison.to_string(index=False))

    print("\nSelected baseline:")
    print(f"  model: {selection.model_name}")
    print(f"  best params: {selection.best_params}")
    print(f"  mean CV F1: {selection.best_f1:.4f}")

    print("\nSaved:")
    print(f"  {config['baseline']['comparison_path']}")
    print(f"  {config['baseline']['model_path']}")


if __name__ == "__main__":
    main()
