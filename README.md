# Fraud Detection

## Project Overview

`Fraud-Detection.ipynb` is the main project walkthrough and execution entry point. Run the notebook from top to bottom to understand and reproduce the project workflow.

Role-specific Python scripts contain the reusable implementation used by the notebook. This keeps the notebook readable for review while keeping the actual work easier to test, revise, and reuse.

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

Stub: fraud vs non-fraud EDA, correlation/outlier/indicator analysis, feature importance, and error analysis.

### Role 3: Model Building

Responsible: Jack Underhill

Stub: logistic regression baseline, random forest, gradient boosted trees, class weighting, and sampling experiments.

### Role 4: Evaluation and Final Integration

Responsible: Jack Underhill

Stub: threshold tuning, final evaluation tables, limitations/challenges/ethics discussion, report integration, and presentation prep.
