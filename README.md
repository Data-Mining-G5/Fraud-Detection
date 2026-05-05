# Fraud Detection

## Project Overview

`Fraud-Detection.ipynb` is the main project walkthrough and execution entry point. Run the notebook from top to bottom to understand and reproduce the project workflow.

Role-specific Python scripts contain the reusable implementation used by the notebook. This keeps the notebook readable for review while keeping the actual work easier to test, revise, and reuse.

## Python Setup

This project requires the dependencies listed in `requirements.txt` to run the notebook and the supporting scripts successfully.

From the project root, install them with:

```bash
python -m pip install -r requirements.txt
```

If `pip` is missing in your local Python install, bootstrap it first:

```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

Once the dependencies are installed, you can validate the environment with:

```bash
python preprocess_data.py --dry-run
```

## Project Roles

### Role 1: Data Preprocessing

Responsible: John Cha

Role 1 owns the shared preprocessing handoff for the rest of the project. The notebook runs this step by using `preprocess_data.py`, which:

- Load the Kaggle credit card fraud dataset from `archive.zip`.
- Run duplicate, null, and class-balance checks.
- Remove duplicate rows.
- Create stratified train, validation, and test splits.
- Scale `Time` and `Amount` with `StandardScaler` fit on the training split only.
- Save the cleaned dataset and split artifacts for downstream roles.

When the Role 1 notebook section is run, it writes:

- `clean_data.csv`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

Downstream roles should consume the generated CSVs instead of reading `archive.zip` or `creditcard.csv` directly. Use `clean_data.csv` for whole-dataset exploratory checks, and use the train/validation/test files in `data/processed/` for modeling and evaluation work.

For a quick no-write validation of the Role 1 script itself, run from the project root:

```bash
python preprocess_data.py --dry-run
```

### Role 2: Feature Analysis and EDA

Responsible: Clayton Simoneaux

Role 2 owns the exploratory feature analysis handoff for the modeling roles. The notebook runs this step by using `role2_analysis.py`, which:

- Load `clean_data.csv` for whole-dataset fraud vs non-fraud EDA.
- Load `data/processed/train.csv` and `data/processed/validation.csv` from Role 1 for validation-based analysis.
- Summarize class balance and the overall fraud rate.
- Save fraud vs non-fraud feature mean differences.
- Measure feature correlations with the `Class` target.
- Identify high-lift outlier signals and simple percentile-based indicator rules.
- Train an exploratory balanced logistic regression model on the Role 1 training split.
- Save feature-importance, validation error-count, and missed-fraud gap artifacts for review.

When the Role 2 notebook section is run, it writes:

- `role2_outputs/class_balance.png`
- `role2_outputs/top_correlations.png`
- `role2_outputs/fraud_group_summary.csv`
- `role2_outputs/correlations.csv`
- `role2_outputs/outlier_signals.csv`
- `role2_outputs/indicator_rules.csv`
- `role2_outputs/feature_importance.csv`
- `role2_outputs/validation_error_counts.csv`
- `role2_outputs/missed_vs_caught_fraud_gaps.csv`

Role 2 should be treated as exploratory analysis, not final model selection. The logistic regression in this step is used to inspect feature importance and validation errors.

To run the Role 2 script directly after Role 1 artifacts exist, run from the project root:

```bash
python role2_analysis.py
```

### Role 3: Model Building

Responsible: Jack Underhill

Role 3 owns validation-set model comparison before final test-set evaluation. The notebook runs this step by using `role3_modeling.py`, which:

- Load `data/processed/train.csv` and `data/processed/validation.csv` from Role 1.
- Train logistic regression, random forest, and gradient boosted tree models.
- Compare baseline models, class-weighted models, and one simple undersampled logistic regression experiment.
- Score each model on the validation split at the default `0.5` threshold.
- Save validation comparison outputs for Role 4 threshold tuning and final discussion.

When the Role 3 script is run, it writes these artifacts:

- `role3_outputs/model_comparison.csv`
- `role3_outputs/validation_predictions.csv`
- `role3_outputs/best_model_summary.txt`

To run the Role 3 script directly after Role 1 artifacts exist, run from the project root:

```bash
python role3_modeling.py
```

### Role 4: Evaluation and Final Integration

Responsible: Jack Underhill

Stub: threshold tuning, final evaluation tables, limitations/challenges/ethics discussion, report integration, and presentation prep.
