from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
UNDERSAMPLE_NON_FRAUD_TO_FRAUD_RATIO = 10
OUTPUT_DIR = ROOT / "role3_outputs"
MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
VALIDATION_PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions.csv"
BEST_MODEL_SUMMARY_PATH = OUTPUT_DIR / "best_model_summary.txt"


class ModelSpec(NamedTuple):
    name: str
    model: object


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


def create_undersampled_training_set(
    train_df: pd.DataFrame,
    non_fraud_to_fraud_ratio: int = UNDERSAMPLE_NON_FRAUD_TO_FRAUD_RATIO,
) -> pd.DataFrame:
    fraud_df = train_df[train_df[TARGET_COLUMN] == 1]
    non_fraud_df = train_df[train_df[TARGET_COLUMN] == 0]

    if fraud_df.empty:
        raise ValueError("Cannot undersample training data because no fraud rows exist.")

    requested_non_fraud_count = len(fraud_df) * non_fraud_to_fraud_ratio
    sampled_non_fraud_count = min(requested_non_fraud_count, len(non_fraud_df))
    sampled_non_fraud_df = non_fraud_df.sample(
        n=sampled_non_fraud_count,
        random_state=RANDOM_STATE,
    )

    return (
        pd.concat([fraud_df, sampled_non_fraud_df], axis=0)
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "Logistic Regression",
            LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
        ),
        ModelSpec(
            "Logistic Regression (balanced)",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1_000,
                random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=50,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        ModelSpec(
            "Random Forest (balanced)",
            RandomForestClassifier(
                class_weight="balanced",
                n_estimators=50,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        ),
        ModelSpec(
            "Gradient Boosting",
            GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE),
        ),
    ]


def train_models(
    model_specs: list[ModelSpec],
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, object]:
    trained_models = {}

    for model_name, model in model_specs:
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        trained_models[model_name] = model

    return trained_models


def train_undersampled_logistic_regression(
    sampled_train_df: pd.DataFrame,
) -> object:
    x_sampled_train, y_sampled_train = split_features_and_target(sampled_train_df)
    sampled_model = LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE)

    print("Training Logistic Regression (undersampled 10:1)...")
    sampled_model.fit(x_sampled_train, y_sampled_train)
    return sampled_model


def compare_models_on_validation(
    trained_models: dict[str, object],
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    comparison_rows = []

    for model_name, model in trained_models.items():
        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        validation_predictions = (validation_probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_validation,
            validation_predictions,
            labels=[0, 1],
        ).ravel()

        comparison_rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "precision": precision_score(
                    y_validation,
                    validation_predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_validation,
                    validation_predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_validation,
                    validation_predictions,
                    zero_division=0,
                ),
                "pr_auc": average_precision_score(
                    y_validation,
                    validation_probabilities,
                ),
                "accuracy": accuracy_score(y_validation, validation_predictions),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )

    return pd.DataFrame(comparison_rows)


def normalize_model_name_for_column(model_name: str) -> str:
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in model_name
    )
    return "_".join(part for part in normalized.split("_") if part)


def build_validation_predictions(
    trained_models: dict[str, object],
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    prediction_df = pd.DataFrame(
        {"actual_class": y_validation.reset_index(drop=True)}
    )

    for model_name, model in trained_models.items():
        column_prefix = normalize_model_name_for_column(model_name)
        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        prediction_df[f"{column_prefix}_probability"] = validation_probabilities
        prediction_df[f"{column_prefix}_prediction"] = (
            validation_probabilities >= threshold
        ).astype(int)

    return prediction_df


def write_best_model_summary(model_comparison_df: pd.DataFrame) -> None:
    best_model = model_comparison_df.sort_values(
        by=["pr_auc", "f1", "recall"],
        ascending=False,
    ).iloc[0]

    lines = [
        "Role 3 validation best model summary",
        "",
        "Selection rule: highest validation PR-AUC, with F1 and recall as tie-breakers.",
        "Validation threshold: 0.5",
        "",
        f"Best model: {best_model['model']}",
        f"PR-AUC: {best_model['pr_auc']:.6f}",
        f"Precision: {best_model['precision']:.6f}",
        f"Recall: {best_model['recall']:.6f}",
        f"F1: {best_model['f1']:.6f}",
        f"Accuracy: {best_model['accuracy']:.6f}",
        (
            "Confusion matrix counts: "
            f"TN={int(best_model['tn'])}, FP={int(best_model['fp'])}, "
            f"FN={int(best_model['fn'])}, TP={int(best_model['tp'])}"
        ),
        "",
        "Role 4 can use model_comparison.csv for threshold tuning and final discussion.",
    ]
    BEST_MODEL_SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_role3_outputs(
    model_comparison_df: pd.DataFrame,
    trained_models: dict[str, object],
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    threshold: float = 0.5,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    model_comparison_df.to_csv(MODEL_COMPARISON_PATH, index=False)
    validation_predictions_df = build_validation_predictions(
        trained_models,
        x_validation,
        y_validation,
        threshold=threshold,
    )
    validation_predictions_df.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    write_best_model_summary(model_comparison_df)


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

    model_specs = build_model_specs()
    trained_models = train_models(model_specs, x_train, y_train)

    sampled_train_df = create_undersampled_training_set(train_df)
    print(summarize_split("Undersampled train 10:1", sampled_train_df))
    trained_models["Logistic Regression (undersampled 10:1)"] = (
        train_undersampled_logistic_regression(sampled_train_df)
    )

    print(f"Trained {len(trained_models)} Role 3 models.")

    model_comparison_df = compare_models_on_validation(
        trained_models,
        x_validation,
        y_validation,
    )
    print("Validation metrics at threshold 0.5")
    print(model_comparison_df.to_string(index=False))
    save_role3_outputs(
        model_comparison_df,
        trained_models,
        x_validation,
        y_validation,
    )
    print(f"Saved Role 3 outputs to {OUTPUT_DIR.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
