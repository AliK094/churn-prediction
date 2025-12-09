# scripts/run_inference.py

from pathlib import Path
import json

from churn_prediction.data.load_data import DataConfig
from churn_prediction.models.predict import PredictionConfig
from churn_prediction.pipelines.batch_inference import (
    BatchInferenceConfig,
    run_batch_inference,
)

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]

    # load metadata to get threshold (optional but nice)
    meta_path = ROOT / "models" / "xgb_churn_model.meta.json"
    if meta_path.exists():
        with meta_path.open() as f:
            metadata = json.load(f)
        best_threshold = metadata.get("best_threshold", 0.5)
    else:
        best_threshold = 0.5

    data_cfg = DataConfig(
        raw_data_path=ROOT / "data" / "raw" / "Telco-Customer-Churn.csv",  # or new batch path
        target_col="Churn",     # if new data has no Churn, _basic_clean just ignores mapping
        test_size=0.2,          # not used in load_full()
        val_size=0.2,
        random_state=42,
        drop_cols=("customerID",),
    )

    pred_cfg = PredictionConfig(
        model_path=ROOT / "models" / "xgb_churn_model.joblib",
        feature_pipeline_path=ROOT / "models" / "feature_pipeline.joblib",
        threshold=best_threshold,
    )

    batch_cfg = BatchInferenceConfig(
        data_config=data_cfg,
        prediction_config=pred_cfg,
        output_path=ROOT / "data" / "processed" / "churn_predictions.csv",
    )

    run_batch_inference(batch_cfg)
