from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataDriftConfig:
    numeric_features: List[str]
    categorical_features: List[str]
    psi_threshold: float = 0.2      # > 0.2 often considered “moderate” drift
    cat_l1_threshold: float = 0.2   # sum |p_ref - p_cur| / 2


@dataclass
class FeatureDriftResult:
    feature: str
    metric: float
    drifted: bool
    metric_name: str


class DataDriftDetector:
    """
    Simple data drift detector using:
      - PSI for numeric features
      - L1 distance between category distributions for categorical features
    """

    def __init__(self, config: DataDriftConfig) -> None:
        self.config = config

    # ------------------------------
    # Public API
    # ------------------------------
    def detect_drift(
        self,
        df_ref: pd.DataFrame,
        df_cur: pd.DataFrame,
    ) -> Dict[str, FeatureDriftResult]:
        results: Dict[str, FeatureDriftResult] = {}

        # Numeric features: PSI
        for col in self.config.numeric_features:
            psi = self._psi(df_ref[col].values, df_cur[col].values)
            results[col] = FeatureDriftResult(
                feature=col,
                metric=psi,
                drifted=psi >= self.config.psi_threshold,
                metric_name="psi",
            )

        # Categorical features: L1 distance between distributions
        for col in self.config.categorical_features:
            distance = self._categorical_l1_distance(
                df_ref[col].astype(str), df_cur[col].astype(str)
            )
            results[col] = FeatureDriftResult(
                feature=col,
                metric=distance,
                drifted=distance >= self.config.cat_l1_threshold,
                metric_name="l1_distance",
            )

        return results

    # ------------------------------
    # Helpers
    # ------------------------------
    def _psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        buckets: int = 10,
    ) -> float:
        """
        Population Stability Index for a single numeric feature.
        """
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Use quantile-based bins from expected
        quantiles = np.linspace(0, 1, buckets + 1)
        bin_edges = np.quantile(expected, quantiles)

        # Make sure edges are strictly increasing
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) <= 2:
            return 0.0

        expected_counts, _ = np.histogram(expected, bins=bin_edges)
        actual_counts, _ = np.histogram(actual, bins=bin_edges)

        expected_perc = expected_counts / expected_counts.sum()
        actual_perc = actual_counts / actual_counts.sum()

        # Avoid zeros
        epsilon = 1e-6
        expected_perc = np.clip(expected_perc, epsilon, 1)
        actual_perc = np.clip(actual_perc, epsilon, 1)

        psi_values = (expected_perc - actual_perc) * np.log(expected_perc / actual_perc)

        return float(psi_values.sum())

    def _categorical_l1_distance(
        self,
        ref: pd.Series,
        cur: pd.Series,
    ) -> float:
        """
        L1 distance between two categorical distributions.
        0.0 = identical; 1.0 = completely different.
        """
        ref_counts = ref.value_counts(normalize=True)
        cur_counts = cur.value_counts(normalize=True)

        all_cats = sorted(set(ref_counts.index).union(set(cur_counts.index)))

        ref_probs = np.array([ref_counts.get(cat, 0.0) for cat in all_cats])
        cur_probs = np.array([cur_counts.get(cat, 0.0) for cat in all_cats])

        l1 = 0.5 * np.abs(ref_probs - cur_probs).sum()
        return float(l1)
