from pathlib import Path
from churn_prediction.data.load_data import DataConfig
from churn_prediction.pipelines.train_pipeline import  TrainingConfig, run_training_pipeline

if __name__ == "__main__":
    data_cfg = DataConfig(
        raw_data_path=Path("data/raw/Telco-Customer-Churn.csv"),
        target_col="Churn",
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        drop_cols=("customerID",),
    )

    train_cfg = TrainingConfig(data_config=data_cfg)
    run_training_pipeline(train_cfg)