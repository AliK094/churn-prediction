from pathlib import Path
import pandas as pd

from churn_prediction.drift.data_drift import DataDriftConfig, DataDriftDetector
from churn_prediction.drift.model_drift import ModelDriftConfig, ModelPerformanceMonitor

def main() -> None:
    # Example: compare training window vs latest batch
    train_sample_path = Path("data/processed/train_sample.parquet")
    latest_batch_path = Path("data/processed/latest_batch.parquet")

    df_ref = pd.read_parquet(train_sample_path)
    df_cur = pd.read_parquet(latest_batch_path)

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "PaymentMethod",
    ]

    cfg = DataDriftConfig(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        psi_threshold=0.2,
        cat_l1_threshold=0.2,
    )

    detector = DataDriftDetector(cfg)
    results = detector.detect_drift(df_ref, df_cur)

    for feat, res in results.items():
        status = "DRIFT" if res.drifted else "ok"
        print(f"{feat:20s} {res.metric_name}={res.metric:.3f}  -> {status}")

def log_model_drift_example():
    # Imagine you just got labels for batch "2025-12-01"
    batch_id = "2025-12-01"

    df = pd.read_parquet("data/processed/labeled_batch_2025-12-01.parquet")
    y_true = df["Churn"].values
    y_prob = df["churn_score"].values  # you created this in batch inference

    monitor = ModelPerformanceMonitor(ModelDriftConfig())
    metrics = monitor.log_performance(batch_id, y_true, y_prob, threshold=0.4)
    print("Logged metrics:", metrics)

    drift_info = monitor.check_drift(metric_name="pr_auc")
    print("Drift status:", drift_info)


if __name__ == "__main__":
    main()
