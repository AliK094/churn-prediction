from dataclasses import dataclass
from typing import Dict

import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score


@dataclass
class XGBoostConfig:
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0
    n_jobs: int = -1
    early_stopping_rounds: int = 50


def train_xgb(
    X_train: np.ndarray,
    y_train,
    X_val: np.ndarray,
    y_val,
    cfg: XGBoostConfig,
) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        min_child_weight=cfg.min_child_weight,
        n_jobs=cfg.n_jobs,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        early_stopping_rounds=cfg.early_stopping_rounds
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return model


def evaluate_binary_classifier(
    model: XGBClassifier,
    X: np.ndarray,
    y,
) -> Dict[str, float]:
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
    }

def find_best_threshold(
    model,
    X_val: np.ndarray,
    y_val,
    metric=f1_score,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)

    proba = model.predict_proba(X_val)[:, 1]
    best_t = 0.5
    best_score = -1.0

    for t in thresholds:
        preds = (proba >= t).astype(int)
        score = metric(y_val, preds)
        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score
