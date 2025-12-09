from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from churn_prediction.data.load_data import DataConfig, TrainingDataLoader
from churn_prediction.models.predict import PredictionConfig, ChurnPredictor


@dataclass
class BatchInferenceConfig:
    data_config: DataConfig
    prediction_config: PredictionConfig
    output_path: Path = Path("data/processed/churn_predictions.csv")


def run_batch_inference(config: BatchInferenceConfig) -> Dict[str, Any]:
    # 1. load & clean new data (no splitting)
    loader = TrainingDataLoader(config.data_config)
    df = loader.load_full()  # assumes same cleaning as training

    # 2. load model + feature pipeline
    predictor = ChurnPredictor.from_config(config.prediction_config)

    # 3. predict
    preds_df = predictor.predict_with_proba(df)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(config.output_path, index=False)

    print(f"Saved predictions to: {config.output_path}")

    return {
        "output_path": str(config.output_path),
        "n_rows": len(preds_df),
    }
