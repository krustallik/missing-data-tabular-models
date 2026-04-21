"""Evaluation metrics used across the benchmark.

Keeps one canonical ``compute_metrics`` implementation so every model/phase
writes comparable numbers.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Return the five metrics in :data:`config.METRICS` as floats.

    ``roc_auc`` is NaN for multiclass targets (to avoid ambiguous averaging)
    and for cases where ``y_proba`` is not available.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    unique = np.unique(y_true)
    if y_proba is not None and len(unique) == 2:
        try:
            proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics
