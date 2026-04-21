"""Evaluation metrics used across the benchmark.

Keeps one canonical ``compute_metrics`` implementation so every model/phase
writes comparable numbers.

Metrics produced:

- ``accuracy``           - overall correct fraction (sensitive to class imbalance).
- ``f1``                 - weighted F1 (dominated by the majority class).
- ``precision``          - weighted precision.
- ``recall``             - weighted recall.
- ``roc_auc``            - binary ROC-AUC (NaN for multiclass or missing proba).
- ``balanced_accuracy``  - mean of per-class recall; robust to imbalance.
- ``f1_macro``           - unweighted mean of per-class F1; treats minority and
                           majority class equally.
- ``recall_class1``      - recall for the positive/minority class (class == 1),
                           i.e. how many true positives are actually caught.
- ``pr_auc``             - binary Precision-Recall AUC (average precision),
                           a more informative summary than ROC-AUC on highly
                           imbalanced datasets.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
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
    """Return the full metric dict used across the benchmark.

    ``roc_auc``, ``pr_auc`` and ``recall_class1`` are only defined for
    binary classification; they are set to NaN otherwise. Probability-based
    metrics additionally require ``y_proba``.
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    unique = np.unique(y_true)
    is_binary = len(unique) == 2

    if is_binary:
        try:
            metrics["recall_class1"] = float(
                recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            )
        except Exception:
            metrics["recall_class1"] = float("nan")
    else:
        metrics["recall_class1"] = float("nan")

    if y_proba is not None and is_binary:
        try:
            proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics["pr_auc"] = float(average_precision_score(y_true, proba))
        except Exception:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics
