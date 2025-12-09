## Churn Prediction – Telco Customer Churn

End-to-end churn prediction project using the Telco Customer Churn dataset.  
The repo contains:

- Data loading and cleaning
- Feature engineering pipelines
- Model training and batch inference
- Data and model drift monitoring
- Tests, Dockerfile, and CI workflow ready for automation

---

## Project Structure

Key paths:

- `src/churn_prediction/data/load_data.py` – `TrainingDataLoader` for raw → cleaned data.
- `src/churn_prediction/features/build_features.py` – `FeatureBuilder` and `FeatureConfig`.
- `src/churn_prediction/models/train.py` – model training utilities.
- `src/churn_prediction/models/predict.py` – batch prediction utilities.
- `src/churn_prediction/pipelines/train_pipeline.py` – orchestrates training.
- `src/churn_prediction/pipelines/batch_inference.py` – batch scoring pipeline.
- `src/churn_prediction/drift/data_drift.py` – data drift detection (PSI, L1 distance).
- `src/churn_prediction/drift/model_drift.py` – model performance drift monitoring.
- `scripts/train_model.py` – CLI for training.
- `scripts/run_inference.py` – CLI for batch inference.
- `scripts/run_drift_checks.py` – CLI for drift checks.
- `tests/` – unit and integration tests.

Data layout (example files already included):

- `data/raw/` – raw input CSVs (e.g. `Telco-Customer-Churn.csv`).
- `data/interim/` – intermediate cleaned data.
- `data/processed/` – train/val/test splits and predictions.

Models and artifacts:

- `models/xgb_churn_model.joblib` – trained model.
- `models/feature_pipeline.joblib` – fitted feature pipeline.

---

## Installation

You can install using `pip` directly from the repo root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install .
```

This will install:

- Runtime dependencies (XGBoost, LightGBM, pandas, scikit-learn, etc.)
- Dev dependencies (`pytest`, `black`, `ruff`, etc.) for local development

---

## Running the Pipelines

Make sure the Telco dataset is available at `data/raw/Telco-Customer-Churn.csv`
(a sample is already included in this repo).

### 1. Train the Model

```bash
python -m scripts.train_model
```

This will:

- Load and clean data from `data/raw/Telco-Customer-Churn.csv`.
- Split into train/val/test.
- Fit the feature pipeline and XGBoost model.
- Store artifacts in `models/`.

### 2. Run Batch Inference

```bash
python -m scripts.run_inference
```

This will:

- Load and clean the configured dataset.
- Apply the saved feature pipeline and model.
- Write predictions to `data/processed/churn_predictions.csv`.

### 3. Data & Model Drift Checks

Data drift (feature distributions):

```bash
python -m scripts.run_drift_checks
```

This compares reference and current data based on PSI (numeric) and L1 distance (categorical)
and prints a simple drift report for each feature.

Model performance drift (over time):

```python
from churn_prediction.drift.model_drift import ModelDriftConfig, ModelPerformanceMonitor
```

See `scripts/run_drift_checks.py:log_model_drift_example` for an example of how to log metrics
and check for performance drift when you get labeled production batches.

---

## Testing

Run the test suite from the repo root:

```bash
pytest
```

Included tests:

- `tests/test_features.py` – basic checks for `FeatureBuilder`.
- `tests/test_models.py` – small synthetic end-to-end train → save → load → predict.
- `tests/test_drift.py` – verifies that drift detection flags shifted numeric features.

These tests are designed to be lightweight so they can run on every CI build.

---

## Docker

The `Dockerfile` builds a container image with Python 3.11, installs the package
and dev dependencies, and runs `pytest` by default.

Build the image:

```bash
docker build -t churn-prediction .
```

Run tests inside the container:

```bash
docker run --rm churn-prediction
```

Override the default command, for example to run training:

```bash
docker run --rm -v "$(pwd)/data:/app/data" churn-prediction \
    python -m scripts.train_model
```

Note: mount your own `data/` and `models/` directories as needed for your environment.

---

## CI/CD

The repo is set up for a GitHub Actions–based CI workflow:

- Install dependencies from `requirements-dev.txt`.
- Install the package (`pip install .`).
- Run `pytest`.

Once you add `.github/workflows/ci.yml` (see below), every push and pull request
to your main branch can automatically:

- Validate that the code builds.
- Run the test suite.
- Optionally run formatting/linting (if you add `black`, `ruff`, etc.).

Example minimal CI workflow (to put in `.github/workflows/ci.yml`):

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run tests
        run: pytest

```

You can extend this with additional jobs (build/publish Docker image, deploy model, etc.)
depending on your CI/CD platform and deployment target.

---

## Development Notes

- Source code lives in `src/churn_prediction`.
- Tests live in `tests/`.
- Data under `data/` is for local experiments only; do not commit sensitive data.
- Use `requirements-dev.txt` to keep local dev and CI environments aligned.

Future enhancements you may want to add:

- Pre-commit hooks for `black` and `ruff`.
- Additional tests for evaluation and metrics modules.
- Automated model retraining and promotion workflows. 
