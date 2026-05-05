from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.csv"
TARGET_COLUMN = "Class"


def load_modeling_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Role 1 train and validation artifacts for Role 3 modeling."""
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    for split_name, split_df in [
        ("data/processed/train.csv", train_df),
        ("data/processed/validation.csv", validation_df),
    ]:
        if TARGET_COLUMN not in split_df.columns:
            raise ValueError(f"Expected a '{TARGET_COLUMN}' column in {split_name}")

    return train_df, validation_df


def split_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=TARGET_COLUMN), df[TARGET_COLUMN]


def summarize_split(name: str, df: pd.DataFrame) -> str:
    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    non_fraud_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    fraud_rate = df[TARGET_COLUMN].mean()

    return (
        f"{name}: rows={len(df)}, features={df.shape[1] - 1}, "
        f"non-fraud={non_fraud_count}, fraud={fraud_count}, "
        f"fraud rate={fraud_rate:.6f} ({fraud_rate:.4%})"
    )


def main() -> None:
    train_df, validation_df = load_modeling_splits()
    x_train, y_train = split_features_and_target(train_df)
    x_validation, y_validation = split_features_and_target(validation_df)

    print("Role 3 modeling inputs")
    print(summarize_split("Train", train_df))
    print(summarize_split("Validation", validation_df))
    print(f"X_train shape: {x_train.shape}; y_train length: {len(y_train)}")
    print(
        f"X_validation shape: {x_validation.shape}; "
        f"y_validation length: {len(y_validation)}"
    )
    print("Test split intentionally unused for Role 3.")


if __name__ == "__main__":
    main()
