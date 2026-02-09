from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


@dataclass
class MetricBundle:
    classification: Dict[str, float]
    regression: Dict[str, float]


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    mask = ~np.isnan(y_prob)
    if mask.sum() == 0:
        return metrics
    y_true = y_true[mask]
    y_prob = y_prob[mask]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics["auc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    metrics["logloss"] = log_loss(y_true, y_prob, labels=[0, 1])
    metrics["brier"] = brier_score_loss(y_true, y_prob)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if len(y_true) == 0:
        return metrics
    metrics["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["directional_accuracy"] = accuracy_score(y_true > 0, y_pred > 0)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else float("nan")
    metrics["correlation"] = float(corr)
    return metrics


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).dropna()
    df["bin"] = pd.cut(df["y_prob"], bins=bins, labels=False, include_lowest=True)
    grouped = df.groupby("bin", observed=True)
    prob_true = grouped["y_true"].mean().to_numpy()
    prob_pred = grouped["y_prob"].mean().to_numpy()
    return prob_true, prob_pred


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).dropna()
    if df.empty:
        return float("nan")
    df["bin"] = pd.cut(df["y_prob"], bins=bins, labels=False, include_lowest=True)
    total = len(df)
    ece = 0.0
    for _, chunk in df.groupby("bin", observed=True):
        weight = len(chunk) / total
        ece += weight * abs(chunk["y_true"].mean() - chunk["y_prob"].mean())
    return float(ece)
