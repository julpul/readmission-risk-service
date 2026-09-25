"""Metrics and threshold selection.

Pure functions: arrays in, numbers out. This module knows nothing about
models or pipelines, which makes it testable with hand-made arrays.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

EPS = 1e-12


def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_flagged": int(y_pred.sum()),
    }


def find_threshold(
    y_true,
    y_proba,
    criterion: str = "f1",
    min_precision: float = 0.2,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]

    if criterion == "f1":
        f1 = 2 * precision * recall / (precision + recall + EPS)
        return float(thresholds[f1.argmax()])

    if criterion == "recall_at_precision":
        mask = precision >= min_precision
        if not mask.any():
            raise ValueError(
                f"no threshold reaches precision >= {min_precision}; "
                f"best achievable is {precision.max():.3f}"
            )
        candidates = np.flatnonzero(mask)
        best = candidates[recall[mask].argmax()]
        return float(thresholds[best])

    raise ValueError(f"unknown criterion: {criterion!r}")
