from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


@dataclass
class ModelDriftConfig:
    metrics_history_path: Path = Path("artifacts/drift/metrics_history.parquet")
    drop_threshold: float = 0.1  # 10% relative drop triggers drift


class ModelPerformanceMonitor:
    """
    Logs model performance metrics over time and checks for drift.

    Assumes:
      - you call `log_performance` periodically (e.g., monthly) with a labeled batch
      - metrics are stored in a history file
    """

    def __init__(self, config: ModelDriftConfig) -> None:
        self.config = config
        self.config.metrics_history_path.parent.mkdir(parents=True, exist_ok=True)

    def log_performance(
        self,
        batch_id: str,
        y_true,
        y_prob,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute metrics for the current batch and append them to the history file.
        """
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "batch_id": batch_id,
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "f1": float(f1_score(y_true, y_pred)),
            "threshold": threshold,
        }

        path = self.config.metrics_history_path
        if path.exists():
            df_hist = pd.read_parquet(path)
            df_hist = pd.concat([df_hist, pd.DataFrame([metrics])], ignore_index=True)
        else:
            df_hist = pd.DataFrame([metrics])

        df_hist.to_parquet(path, index=False)

        return metrics

    def check_drift(self, metric_name: str = "pr_auc") -> Dict[str, float | bool]:
        """
        Compare latest batch metric to the baseline (first batch).

        Returns:
            {
              "baseline": ...,
              "latest": ...,
              "relative_drop": ...,
              "drift": True/False,
            }
        """
        path = self.config.metrics_history_path
        if not path.exists():
            return {
                "baseline": None,
                "latest": None,
                "relative_drop": 0.0,
                "drift": False,
            }

        df_hist = pd.read_parquet(path)
        df_hist = df_hist.sort_values("batch_id")

        baseline = float(df_hist.iloc[0][metric_name])
        latest = float(df_hist.iloc[-1][metric_name])

        if baseline == 0:
            rel_drop = 0.0
        else:
            rel_drop = (baseline - latest) / baseline

        drift = rel_drop >= self.config.drop_threshold

        return {
            "baseline": baseline,
            "latest": latest,
            "relative_drop": rel_drop,
            "drift": drift,
        }
