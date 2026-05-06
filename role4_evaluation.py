from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
OUTPUT_DIR = ROOT / "role4_outputs"
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
THRESHOLD_GRID = [threshold / 100 for threshold in range(5, 100, 5)]
SELECTED_MODEL_NAME = "Random Forest"


def load_evaluation_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Role 1 processed splits for Role 4 final evaluation work."""
    train_df = pd.read_csv(TRAIN_PATH)
    validation_df = pd.read_csv(VALIDATION_PATH)
    test_df = pd.read_csv(TEST_PATH)

    for split_name, split_df in [
        ("data/processed/train.csv", train_df),
        ("data/processed/validation.csv", validation_df),
        ("data/processed/test.csv", test_df),
    ]:
        if TARGET_COLUMN not in split_df.columns:
            raise ValueError(f"Expected a '{TARGET_COLUMN}' column in {split_name}")

    return train_df, validation_df, test_df


def split_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=TARGET_COLUMN), df[TARGET_COLUMN]


def build_selected_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=50,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def calculate_threshold_metrics(
    y_true: pd.Series,
    fraud_probabilities,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (fraud_probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "accuracy": accuracy_score(y_true, predictions),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def tune_threshold_on_validation(
    y_validation: pd.Series,
    validation_probabilities,
) -> tuple[pd.DataFrame, pd.Series]:
    threshold_results = pd.DataFrame(
        [
            calculate_threshold_metrics(
                y_validation,
                validation_probabilities,
                threshold,
            )
            for threshold in THRESHOLD_GRID
        ]
    )
    best_index = threshold_results["f1"].idxmax()

    return threshold_results, threshold_results.loc[best_index]


def calculate_final_test_metrics(
    y_test: pd.Series,
    test_probabilities,
    threshold: float,
) -> dict[str, float | int]:
    threshold_metrics = calculate_threshold_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    return {
        "threshold": threshold_metrics["threshold"],
        "precision": threshold_metrics["precision"],
        "recall": threshold_metrics["recall"],
        "f1": threshold_metrics["f1"],
        "pr_auc": average_precision_score(y_test, test_probabilities),
        "accuracy": threshold_metrics["accuracy"],
        "tn": threshold_metrics["tn"],
        "fp": threshold_metrics["fp"],
        "fn": threshold_metrics["fn"],
        "tp": threshold_metrics["tp"],
    }


def build_final_summary(
    best_threshold_result: pd.Series,
    final_test_metrics: dict[str, float | int],
) -> str:
    return "\n".join(
        [
            "Role 4 Final Evaluation Summary",
            "",
            f"Selected model: {SELECTED_MODEL_NAME}",
            f"Selected threshold: {best_threshold_result['threshold']:.2f}",
            (
                "Validation reason: this threshold had the best validation F1 "
                f"score ({best_threshold_result['f1']:.6f}) across the simple "
                "threshold grid, balancing precision "
                f"({best_threshold_result['precision']:.6f}) and recall "
                f"({best_threshold_result['recall']:.6f}) for the rare fraud class."
            ),
            "",
            "Final held-out test metrics:",
            f"Precision: {final_test_metrics['precision']:.6f}",
            f"Recall: {final_test_metrics['recall']:.6f}",
            f"F1: {final_test_metrics['f1']:.6f}",
            f"PR-AUC: {final_test_metrics['pr_auc']:.6f}",
            (
                "Confusion matrix counts: "
                f"TN={int(final_test_metrics['tn'])}, "
                f"FP={int(final_test_metrics['fp'])}, "
                f"FN={int(final_test_metrics['fn'])}, "
                f"TP={int(final_test_metrics['tp'])}"
            ),
            "",
            "Limitations, challenges, and ethics notes:",
            (
                "- Fraud is extremely rare, so accuracy can look strong even when "
                "missed fraud remains costly."
            ),
            (
                "- False negatives may allow fraud through, while false positives "
                "can create customer friction; PCA/anonymized features also limit "
                "interpretability."
            ),
            (
                "- Lower thresholds can catch more fraud, but they usually increase "
                "false positives and manual review burden."
            ),
            (
                "- A real fraud system would need ongoing monitoring because fraud "
                "patterns can change over time."
            ),
            "",
        ]
    )


def save_role4_outputs(
    threshold_results: pd.DataFrame,
    best_threshold_result: pd.Series,
    final_test_metrics_df: pd.DataFrame,
    y_test: pd.Series,
    test_probabilities,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    threshold_results.to_csv(OUTPUT_DIR / "threshold_tuning.csv", index=False)
    final_test_metrics_df.to_csv(OUTPUT_DIR / "final_test_metrics.csv", index=False)

    selected_threshold = float(best_threshold_result["threshold"])
    test_predictions = pd.DataFrame(
        {
            "actual_class": y_test.to_numpy(),
            "fraud_probability": test_probabilities,
            "final_prediction": (test_probabilities >= selected_threshold).astype(int),
        }
    )
    test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    final_summary = build_final_summary(
        best_threshold_result,
        final_test_metrics_df.iloc[0].to_dict(),
    )
    (OUTPUT_DIR / "final_summary.txt").write_text(final_summary, encoding="utf-8")


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
    train_df, validation_df, test_df = load_evaluation_splits()
    x_train, y_train = split_features_and_target(train_df)
    x_validation, y_validation = split_features_and_target(validation_df)
    x_test, y_test = split_features_and_target(test_df)

    print("Role 4 evaluation inputs")
    print(summarize_split("Train", train_df))
    print(summarize_split("Validation", validation_df))
    print(summarize_split("Test", test_df))
    print(f"X_train shape: {x_train.shape}; y_train length: {len(y_train)}")
    print(
        f"X_validation shape: {x_validation.shape}; "
        f"y_validation length: {len(y_validation)}"
    )
    print(f"X_test shape: {x_test.shape}; y_test length: {len(y_test)}")

    model = build_selected_model()
    print("Training selected Role 3 model: Random Forest...")
    model.fit(x_train, y_train)
    print("Selected model trained.")

    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    threshold_results, best_threshold_result = tune_threshold_on_validation(
        y_validation,
        validation_probabilities,
    )

    print("Validation threshold tuning results")
    print(threshold_results.to_string(index=False, float_format="{:.6f}".format))
    print(
        "Selected validation threshold: "
        f"{best_threshold_result['threshold']:.2f} "
        f"(F1={best_threshold_result['f1']:.6f}, "
        f"precision={best_threshold_result['precision']:.6f}, "
        f"recall={best_threshold_result['recall']:.6f})"
    )

    selected_threshold = float(best_threshold_result["threshold"])
    test_probabilities = model.predict_proba(x_test)[:, 1]
    final_test_metrics = calculate_final_test_metrics(
        y_test,
        test_probabilities,
        selected_threshold,
    )
    final_test_metrics_df = pd.DataFrame([final_test_metrics])

    print("Final test metrics at selected validation threshold")
    print(
        final_test_metrics_df.to_string(
            index=False,
            float_format="{:.6f}".format,
        )
    )

    save_role4_outputs(
        threshold_results,
        best_threshold_result,
        final_test_metrics_df,
        y_test,
        test_probabilities,
    )
    print(f"Saved Role 4 outputs to {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
