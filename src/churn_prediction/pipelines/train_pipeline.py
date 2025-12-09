from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import joblib
import json

from churn_prediction.data.load_data import DataConfig, TrainingDataLoader
from churn_prediction.features.build_features import (
    FeatureConfig,
    FeatureBuilder,
    DEFAULT_TELCO_FEATURE_CONFIG,
)
from churn_prediction.models.train import (
    XGBoostConfig,
    train_xgb,
    evaluate_binary_classifier,
)
from churn_prediction.models.train import find_best_threshold


@dataclass
class TrainingConfig:
    data_config: DataConfig
    feature_config: FeatureConfig = field(
        default_factory=lambda: DEFAULT_TELCO_FEATURE_CONFIG
    )
    xgb_config: XGBoostConfig = field(default_factory=XGBoostConfig)
    model_output_path: Path = Path("models/xgb_churn_model.joblib")
    feature_pipeline_output_path: Path = Path("models/feature_pipeline.joblib")


def run_training_pipeline(config: TrainingConfig) -> Dict[str, Any]:
    # 1. data
    loader = TrainingDataLoader(config.data_config)
    splits = loader.load_and_split()
    X_train, X_val, X_test = splits["X_train"], splits["X_val"], splits["X_test"]
    y_train, y_val, y_test = splits["y_train"], splits["y_val"], splits["y_test"]

    # 2. features
    fb = FeatureBuilder(config.feature_config)
    X_train_proc = fb.fit_transform(X_train)
    X_val_proc = fb.transform(X_val)
    X_test_proc = fb.transform(X_test)
    fb.save(config.feature_pipeline_output_path)

    # 3. model
    model = train_xgb(
        X_train_proc,
        y_train,
        X_val_proc,
        y_val,
        config.xgb_config,
    )

    val_metrics = evaluate_binary_classifier(model, X_val_proc, y_val)
    test_metrics = evaluate_binary_classifier(model, X_test_proc, y_test)

    best_threshold, best_f1 = find_best_threshold(model, X_val_proc, y_val)
    print(f"Best F1 on val: {best_f1:.3f} at threshold {best_threshold:.2f}")

    # 4. save
    config.model_output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.model_output_path)

    metadata = {
        "model_type": "xgboost_native",
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_threshold": float(best_threshold),
    }
    meta_path = config.model_output_path.with_suffix(".meta.json")
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_path": str(config.model_output_path),
        "feature_pipeline_path": str(config.feature_pipeline_output_path),
        "best_threshold": float(best_threshold),
        "metadata_path": str(meta_path),
    }
