from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import logging
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """
    Configuration for loading and splitting the Telco churn dataset.
    """
    raw_data_path: Path
    target_col: str = "Churn"
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    drop_cols: Tuple[str, ...] = ("customerID",)

    def __post_init__(self) -> None:
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if not (0.0 <= self.val_size < 1.0):
            raise ValueError(f"val_size must be between 0 and 1, got {self.val_size}")
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(
                f"test_size + val_size must be < 1.0, got {self.test_size + self.val_size}"
            )
        if not isinstance(self.raw_data_path, Path):
            self.raw_data_path = Path(self.raw_data_path)


class TrainingDataLoader:
    """
    Load the Telco churn dataset, apply basic cleaning,
    and return stratified train/val/test splits.

    Intended for *training time* only.
    """

    def __init__(self, config: DataConfig):
        self.config = config

    # -------- Public API -------- #

    def load_full(self) -> pd.DataFrame:
        """
        Load and clean the full dataset (no splitting).
        Useful for analysis or batch inference (when labels may be missing).
        """
        df = self._load_raw()
        df = self._basic_clean(df)
        return df

    def load_and_split(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dict containing:
          - X_train, X_val, X_test
          - y_train, y_val, y_test
        For training/validation only (requires target column present).
        """
        df = self._load_raw()
        df = self._basic_clean(df)

        if self.config.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.config.target_col}' not found in data."
            )

        X = df.drop(columns=[self.config.target_col])
        y = df[self.config.target_col]

        # First: train + temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Second: split temp into val + test with the desired ratio
        val_size_rel = self.config.val_size / (1.0 - self.config.test_size)

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=1.0 - val_size_rel,
            random_state=self.config.random_state,
            stratify=y_temp,
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        }

    # -------- Internal helpers -------- #

    def _load_raw(self) -> pd.DataFrame:
        if not self.config.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw data file not found at: {self.config.raw_data_path}"
            )
        logger.info("Loading raw data from %s", self.config.raw_data_path)
        df = pd.read_csv(self.config.raw_data_path)
        return df

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Telco-specific basic cleaning / type casting.
        """
        df = df.copy()

        # Drop ID columns
        for col in self.config.drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
                logger.debug("Dropped column %s", col)
            else:
                logger.debug(
                    "Configured to drop column '%s' but it is not present.", col
                )

        # Standardize target column: map 'Yes'/'No' to 1/0
        if self.config.target_col in df.columns:
            df[self.config.target_col] = (
                df[self.config.target_col].map({"Yes": 1, "No": 0}).astype("int8")
            )

        # TotalCharges sometimes has spaces; convert to numeric and handle errors
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"].replace(" ", pd.NA),
                errors="coerce"
            )
            # Basic imputation: replace NaNs with 0
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

        # SeniorCitizen is 0/1 but often int64; we can cast to int8
        if "SeniorCitizen" in df.columns:
            df["SeniorCitizen"] = df["SeniorCitizen"].astype("int8")

        # Strip whitespace in string columns
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            df[col] = df[col].astype(str).str.strip()

        return df
