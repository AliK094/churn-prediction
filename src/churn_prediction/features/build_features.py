from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

PathLike = Union[str, Path]


@dataclass
class FeatureConfig:
    """
    Configuration for the churn feature pipeline.
    Assumes input data is already cleaned by the DataLoader.
    """
    id_cols: List[str]
    numeric_features: List[str]
    categorical_features: List[str]


DEFAULT_TELCO_FEATURE_CONFIG = FeatureConfig(
    id_cols=["customerID"],  # may already be dropped in TrainingDataLoader, that's fine
    numeric_features=[
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ],
    categorical_features=[
        "gender",
        "SeniorCitizen",      # we now assume this is already a clean column
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ],
)


class FeatureBuilder:
    """
    Build and apply a preprocessing pipeline for churn prediction.

    Assumes:
      - Target column is handled by the DataLoader / training code.
      - Telco-specific cleaning (TotalCharges, SeniorCitizen, etc.)
        is done upstream.

    Responsibilities:
      - Drop ID columns (if present)
      - Handle missing values
      - One-hot encode categorical features
      - Scale numeric features
      - Save / load the fitted pipeline
    """

    def __init__(self, config: FeatureConfig):
        self.config = config
        self._pipeline: Optional[ColumnTransformer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        X = self._prepare_features_df(df)
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError(
                "FeatureBuilder pipeline is not fitted. "
                "Call `fit` or `fit_transform` first, or use `load`."
            )

        X = self._prepare_features_df(df)
        return self._pipeline.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X = self._prepare_features_df(df)
        self._pipeline = self._build_pipeline()
        return self._pipeline.fit_transform(X)

    def save(self, path: PathLike) -> None:
        if self._pipeline is None:
            raise RuntimeError(
                "Cannot save FeatureBuilder: pipeline is not fitted yet."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: PathLike, config: FeatureConfig) -> "FeatureBuilder":
        path = Path(path)
        fb = cls(config=config)
        fb._pipeline = joblib.load(path)
        return fb

    def get_feature_names(self) -> List[str]:
        """
        Optional helper: get output feature names after transformation.
        Useful for drift, SHAP, etc.
        """
        if self._pipeline is None:
            raise RuntimeError("Pipeline not fitted; cannot get feature names.")
        return self._pipeline.get_feature_names_out().tolist()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_pipeline(self) -> ColumnTransformer:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.config.numeric_features),
                ("cat", categorical_transformer, self.config.categorical_features),
            ],
            remainder="drop",
        )

        return preprocessor

    def _prepare_features_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Subset the DataFrame to feature columns and drop IDs.

        Assumes the DataLoader already:
          - mapped the target to numeric (if present)
          - fixed weird values / dtypes
        """
        df = df.copy()

        # Drop ID columns if present
        cols_to_drop = [c for c in self.config.id_cols if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        feature_cols = self.config.numeric_features + self.config.categorical_features
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing expected feature columns: {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )

        return df[feature_cols]
