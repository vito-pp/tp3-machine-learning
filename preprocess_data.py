from src.data import (
    load_config,
    load_dataset,
    save_dataframe,
    split_features_target,
)
from src.preprocessing import (
    save_preprocessor,
    transform_train_test,
)


def main() -> None:
    config = load_config()
    target = config["project"]["target"]

    train_df = load_dataset(config["data"]["train_path"])
    test_df = load_dataset(config["data"]["test_path"])

    X_train, y_train = split_features_target(train_df, target)
    X_test, y_test = split_features_target(test_df, target)

    train_transformed, test_transformed, preprocessor = transform_train_test(
        X_train=X_train,
        X_test=X_test,
    )

    train_transformed[target] = y_train.to_numpy()
    test_transformed[target] = y_test.to_numpy()

    save_dataframe(
        train_transformed,
        config["data"]["train_transformed_path"],
    )
    save_dataframe(
        test_transformed,
        config["data"]["test_transformed_path"],
    )
    save_preprocessor(
        preprocessor,
        config["preprocessing"]["transformer_path"],
    )

    print("Saved transformed datasets:")
    print(
        f"  train: {config['data']['train_transformed_path']} "
        f"-> {train_transformed.shape}"
    )
    print(
        f"  test:  {config['data']['test_transformed_path']} "
        f"-> {test_transformed.shape}"
    )
    print(
        f"Saved fitted preprocessor: "
        f"{config['preprocessing']['transformer_path']}"
    )


if __name__ == "__main__":
    main()
