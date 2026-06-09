from src.data import (
    load_config,
    load_dataset,
    save_dataframe,
    split_dataframe,
    validate_dataset,
)


def main() -> None:
    config = load_config()
    dataframe = load_dataset(config["data"]["raw_path"])

    validate_dataset(
        dataframe=dataframe,
        target_column=config["project"]["target"],
        cohort_columns=config["cohorts"]["columns"],
    )

    train_df, test_df = split_dataframe(
        dataframe=dataframe,
        target_column=config["project"]["target"],
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_state"],
    )

    save_dataframe(train_df, config["data"]["train_path"])
    save_dataframe(test_df, config["data"]["test_path"])

    print("Saved train/test split:")
    print(f"  train: {config['data']['train_path']} -> {train_df.shape}")
    print(f"  test:  {config['data']['test_path']} -> {test_df.shape}")


if __name__ == "__main__":
    main()
