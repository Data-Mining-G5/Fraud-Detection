from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_ARCHIVE_PATH = ROOT / "archive.zip"
DEFAULT_OUTPUT_PATH = ROOT / "clean_data.csv"
EXPECTED_COLUMNS = {"Time", "Amount", "Class"}


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
        "--dry-run",
        action="store_true",
        help="Load and clean the data, but do not write the output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_df = load_creditcard_archive(args.archive)
    clean_df = preprocess(raw_df)

    print(summarize(raw_df, clean_df))

    if args.dry_run:
        print("Dry run complete; no output file written.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(args.output, index=False)
    print(f"Wrote cleaned data to {args.output}")


if __name__ == "__main__":
    main()
