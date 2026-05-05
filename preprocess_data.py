from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_PATH = ROOT / "archive.zip"
DEFAULT_OUTPUT_PATH = ROOT / "clean_data.csv"
DEFAULT_SPLIT_DIR = ROOT / "data" / "processed"
EXPECTED_COLUMNS = {"Time", "Amount", "Class"}
DEFAULT_RANDOM_STATE = 42
DEFAULT_TRAIN_SIZE = 0.60
DEFAULT_VALIDATION_SIZE = 0.20
DEFAULT_TEST_SIZE = 0.20


def load_creditcard_archive(archive_path: Path) -> pd.DataFrame:
    """Read the Kaggle credit card CSV directly from archive.zip."""
    if not archive_path.exists():
        raise FileNotFoundError(f"Could not find data archive: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        csv_members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".csv")
        ]
        if len(csv_members) != 1:
            names = ", ".join(member.filename for member in csv_members) or "none"
            raise ValueError(f"Expected exactly one CSV in {archive_path}; found {names}")

        with archive.open(csv_members[0]) as csv_file:
            return pd.read_csv(csv_file)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = EXPECTED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing expected column(s): {missing}")

    if df.isna().sum().sum() > 0:
        raise ValueError("Input data contains null values; review before cleaning.")

    return df.drop_duplicates().reset_index(drop=True)


def stratified_train_validation_test_split(
    df: pd.DataFrame,
    train_size: float = DEFAULT_TRAIN_SIZE,
    validation_size: float = DEFAULT_VALIDATION_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_total = train_size + validation_size + test_size
    if abs(split_total - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test sizes must sum to 1.0")

    split_parts = {"train": [], "validation": [], "test": []}

    for _, class_df in df.groupby("Class", sort=True):
        shuffled = class_df.sample(frac=1, random_state=random_state)
        validation_count = round(len(shuffled) * validation_size)
        test_count = round(len(shuffled) * test_size)
        train_count = len(shuffled) - validation_count - test_count

        split_parts["train"].append(shuffled.iloc[:train_count])
        split_parts["validation"].append(
            shuffled.iloc[train_count : train_count + validation_count]
        )
        split_parts["test"].append(shuffled.iloc[train_count + validation_count :])

    train_df = pd.concat(split_parts["train"]).sample(frac=1, random_state=random_state)
    validation_df = pd.concat(split_parts["validation"]).sample(
        frac=1, random_state=random_state
    )
    test_df = pd.concat(split_parts["test"]).sample(frac=1, random_state=random_state)

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def summarize_split(name: str, split_df: pd.DataFrame) -> str:
    class_counts = split_df["Class"].value_counts().sort_index()
    non_fraud_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    fraud_rate = split_df["Class"].mean()

    return (
        f"{name}: rows={len(split_df)}, non-fraud={non_fraud_count}, "
        f"fraud={fraud_count}, fraud rate={fraud_rate:.6f} ({fraud_rate:.4%})"
    )


def summarize(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> str:
    class_counts = clean_df["Class"].value_counts().sort_index()
    non_fraud_count = int(class_counts.get(0, 0))
    fraud_count = int(class_counts.get(1, 0))
    fraud_rate = clean_df["Class"].mean()

    return "\n".join(
        [
            "Duplicate/null/class-balance checks",
            f"Raw rows: {len(raw_df)}",
            f"Null values: {int(raw_df.isna().sum().sum())}",
            f"Clean rows: {len(clean_df)}",
            f"Duplicate rows removed: {len(raw_df) - len(clean_df)}",
            f"Non-fraud rows: {non_fraud_count}",
            f"Fraud rows: {fraud_count}",
            f"Fraud rate: {fraud_rate:.6f} ({fraud_rate:.4%})",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess credit card fraud data directly from archive.zip."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE_PATH,
        help="Path to archive.zip containing creditcard.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the cleaned CSV should be written.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=DEFAULT_SPLIT_DIR,
        help="Directory where train/validation/test split CSVs should be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and clean the data, but do not write the output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = load_creditcard_archive(args.archive)
    clean_df = preprocess(raw_df)
    train_df, validation_df, test_df = stratified_train_validation_test_split(clean_df)

    print(summarize(raw_df, clean_df))
    print("Stratified train/validation/test split")
    print(summarize_split("Train", train_df))
    print(summarize_split("Validation", validation_df))
    print(summarize_split("Test", test_df))

    if args.dry_run:
        print("Dry run complete; no output file written.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(args.output, index=False)
    print(f"Wrote cleaned data to {args.output}")

    args.split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.split_dir / "train.csv", index=False)
    validation_df.to_csv(args.split_dir / "validation.csv", index=False)
    test_df.to_csv(args.split_dir / "test.csv", index=False)
    print(f"Wrote split data to {args.split_dir}")


if __name__ == "__main__":
    main()
