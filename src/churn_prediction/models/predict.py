from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from churn_prediction.features.build_features import (
    FeatureConfig,
    FeatureBuilder,
    DEFAULT_TELCO_FEATURE_CONFIG,
)

PathLike = Union[str, Path]


@dataclass
class PredictionConfig:
    """
    Configuration for loading model + feature pipeline for inference.
    """
    model_path: Path
    feature_pipeline_path: Path
    feature_config: FeatureConfig = DEFAULT_TELCO_FEATURE_CONFIG
    threshold: float = 0.50


class ChurnPredictor:
    """
    Thin wrapper around the trained model + feature pipeline.

    Assumes:
      - Input df has already been cleaned by the same logic as training
      - Columns match the FeatureConfig (numeric + categorical features)
    """

    def __init__(
        self,
        model: XGBClassifier,
        feature_builder: FeatureBuilder,
        threshold: float = 0.5,
    ):
        self.model = model
        self.feature_builder = feature_builder
        self.threshold = threshold

    @classmethod
    def from_config(cls, cfg: PredictionConfig) -> "ChurnPredictor":
        """
        Load model + feature pipeline from disk.
        """
        model_path = Path(cfg.model_path)
        feat_path = Path(cfg.feature_pipeline_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature pipeline not found at: {feat_path}")

        model: XGBClassifier = joblib.load(model_path)
        feature_builder = FeatureBuilder.load(
            path=feat_path,
            config=cfg.feature_config,
        )

        return cls(model=model, feature_builder=feature_builder, threshold=cfg.threshold)

    # ------------------------------------------------------------------
    # Public prediction API
    # ------------------------------------------------------------------
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probabilities for a cleaned DataFrame.

        Returns shape (n_samples,) – probability of churn (class 1).
        """
        X_proc = self.feature_builder.transform(df)
        proba_2d = self.model.predict_proba(X_proc)
        return proba_2d[:, 1]

    def predict_label(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict churn labels (0/1) using a probability threshold.
        """
        th = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(df)
        return (proba >= th).astype(int)

    def predict_with_proba(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Convenience: return a DataFrame with probability and predicted label.
        """
        th = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(df)
        labels = (proba >= th).astype(int)

        return pd.DataFrame(
            {
                "churn_proba": proba,
                "churn_pred": labels,
            },
            index=df.index,
        )
