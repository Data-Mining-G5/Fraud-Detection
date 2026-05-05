from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "clean_data.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.csv"
OUTPUT_DIR = ROOT / "role2_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column in clean_data.csv")
    return df


def load_model_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)

    for split_name, split_df in [
        ("data/processed/train.csv", train_df),
        ("data/processed/validation.csv", validation_df),
    ]:
        if "Class" not in split_df.columns:
            raise ValueError(f"Expected a 'Class' column in {split_name}")

    return train_df, validation_df


def save_class_balance_graph(df: pd.DataFrame) -> None:
    counts = df["Class"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    plt.bar(["Non-Fraud", "Fraud"], counts.values, color=["steelblue", "firebrick"])
    plt.title("Class Balance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_balance.png", dpi=200)
    plt.close()


def save_correlation_graph(correlations: pd.DataFrame) -> None:
    top_corr = correlations.head(10).sort_values("correlation_with_class")

    plt.figure(figsize=(8, 5))
    plt.barh(top_corr["feature"], top_corr["correlation_with_class"], color="teal")
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Top Correlations with Fraud")
    plt.xlabel("Correlation with Class")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_correlations.png", dpi=200)
    plt.close()


def get_correlations(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.corr(numeric_only=True)["Class"]
        .drop("Class")
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .rename_axis("feature")
        .reset_index(name="correlation_with_class")
    )


def print_eda(df: pd.DataFrame) -> None:
    counts = df["Class"].value_counts().sort_index()
    print("EDA")
    print(f"Rows: {len(df)}")
    print(f"Non-Fraud: {int(counts.get(0, 0))}")
    print(f"Fraud: {int(counts.get(1, 0))}")
    print(f"Fraud rate: {df['Class'].mean():.6f}")
    print()


def print_correlations(correlations: pd.DataFrame) -> None:
    print("Correlations With Class")
    for row in correlations.itertuples(index=False):
        print(f"{row.feature}: {row.correlation_with_class:.6f}")
    print()


def print_outliers(df: pd.DataFrame) -> None:
    baseline = df["Class"].mean()
    rows = []

    for feature in [col for col in df.columns if col != "Class"]:
        q1, q3 = df[feature].quantile([0.25, 0.75])
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        mask = (df[feature] < low) | (df[feature] > high)
        subset = df.loc[mask, "Class"]
        if len(subset) == 0:
            continue

        rows.append(
            {
                "feature": feature,
                "fraud_lift": subset.mean() / baseline if baseline > 0 else np.nan,
                "count": int(mask.sum()),
            }
        )

    outliers = pd.DataFrame(rows).sort_values("fraud_lift", ascending=False).head(10)

    print("Top Outlier Signals")
    for row in outliers.itertuples(index=False):
        print(f"{row.feature}: lift={row.fraud_lift:.2f} count={row.count}")
    print()


def print_indicators(df: pd.DataFrame) -> None:
    baseline = df["Class"].mean()
    rows = []

    for feature in [col for col in df.columns if col != "Class"]:
        series = df[feature]
        low_value = series.quantile(0.01)
        high_value = series.quantile(0.99)

        checks = [
            (f"{feature} <= {low_value:.6f}", series <= low_value),
            (f"{feature} >= {high_value:.6f}", series >= high_value),
        ]

        for rule, mask in checks:
            subset = df.loc[mask, "Class"]
            if len(subset) == 0:
                continue

            rows.append(
                {
                    "rule": rule,
                    "fraud_lift": subset.mean() / baseline if baseline > 0 else np.nan,
                    "count": int(mask.sum()),
                }
            )

    indicators = pd.DataFrame(rows).sort_values("fraud_lift", ascending=False).head(10)

    print("Top Indicator Rules")
    for row in indicators.itertuples(index=False):
        print(f"{row.rule}: lift={row.fraud_lift:.2f} count={row.count}")
    print()


def print_feature_importance_and_errors(
    train_df: pd.DataFrame, validation_df: pd.DataFrame
) -> None:
    x_train = train_df.drop(columns="Class")
    y_train = train_df["Class"]
    x_validation = validation_df.drop(columns="Class")
    y_validation = validation_df["Class"]

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(x_train, y_train)

    importance = pd.DataFrame(
        {
            "feature": x_train.columns,
            "importance": np.abs(model.coef_[0]),
        }
    ).sort_values("importance", ascending=False)

    probs = model.predict_proba(x_validation)[:, 1]
    preds = (probs >= 0.5).astype(int)

    results = x_validation.copy()
    results["actual"] = y_validation.to_numpy()
    results["predicted"] = preds
    results["error_type"] = np.select(
        [
            (results["actual"] == 1) & (results["predicted"] == 1),
            (results["actual"] == 1) & (results["predicted"] == 0),
            (results["actual"] == 0) & (results["predicted"] == 1),
        ],
        ["true_positive", "false_negative", "false_positive"],
        default="true_negative",
    )

    print("Top Feature Importance")
    for row in importance.head(10).itertuples(index=False):
        print(f"{row.feature}: {row.importance:.6f}")
    print()

    print("Error Counts")
    counts = results["error_type"].value_counts()
    for label in ["true_positive", "false_negative", "false_positive", "true_negative"]:
        print(f"{label}: {int(counts.get(label, 0))}")
    print()

    gap_table = (
        results[results["error_type"].isin(["false_negative", "true_positive"])]
        .groupby("error_type")[x_train.columns]
        .mean()
        .transpose()
    )
    if {"false_negative", "true_positive"}.issubset(gap_table.columns):
        gap_table["gap"] = (gap_table["false_negative"] - gap_table["true_positive"]).abs()
        gap_table = gap_table.sort_values("gap", ascending=False).head(10)

        print("Missed Fraud vs Caught Fraud Gaps")
        for feature, row in gap_table.iterrows():
            print(f"{feature}: {row['gap']:.6f}")
        print()


def main() -> None:
    df = load_data()
    train_df, validation_df = load_model_splits()
    correlations = get_correlations(df)
    save_class_balance_graph(df)
    save_correlation_graph(correlations)
    print_eda(df)
    print_correlations(correlations)
    print_outliers(df)
    print_indicators(df)
    print_feature_importance_and_errors(train_df, validation_df)


if __name__ == "__main__":
    main()
