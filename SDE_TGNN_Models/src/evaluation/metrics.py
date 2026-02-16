"""Detection metrics for intrusion detection evaluation.

Computes standard classification metrics including accuracy,
precision, recall, F1-score, and AUC-ROC, with support for
both binary and multi-class settings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = "weighted",
) -> Dict[str, float]:
    """Compute standard detection performance metrics.

    Args:
        y_true: Ground truth labels of shape (N,).
        y_pred: Predicted labels of shape (N,).
        y_proba: Predicted probabilities of shape (N, C) for AUC-ROC.
        average: Averaging strategy for multi-class ('weighted', 'macro', 'micro').

    Returns:
        Dictionary with keys 'accuracy', 'precision', 'recall', 'f1',
        'auc_roc', and 'specificity'.
    """
    metrics: Dict[str, float] = {}

    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))

    # AUC-ROC
    if y_proba is not None:
        try:
            num_classes = y_proba.shape[1] if y_proba.ndim == 2 else 2
            if num_classes == 2:
                # Binary case
                proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["auc_roc"] = float(roc_auc_score(y_true, proba))
            else:
                # Multi-class: one-vs-rest
                metrics["auc_roc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average=average)
                )
        except ValueError:
            metrics["auc_roc"] = 0.0
    else:
        metrics["auc_roc"] = 0.0

    # Specificity (TNR) - averaged across classes
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / max(tn + fp, 1)
        specificities.append(spec)
    metrics["specificity"] = float(np.mean(specificities))

    # False positive rate
    metrics["fpr"] = 1.0 - metrics["specificity"]

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, F1, and support.

    Args:
        y_true: Ground truth labels of shape (N,).
        y_pred: Predicted labels of shape (N,).
        class_names: Optional list of class name strings.

    Returns:
        Dictionary mapping class name -> metric dictionary.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0,
    )

    num_classes = len(precision)
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    elif len(class_names) < num_classes:
        class_names = class_names + [f"class_{i}" for i in range(len(class_names), num_classes)]

    result: Dict[str, Dict[str, float]] = {}
    for i in range(num_classes):
        result[class_names[i]] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return result


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Compute the confusion matrix.

    Args:
        y_true: Ground truth labels of shape (N,).
        y_pred: Predicted labels of shape (N,).
        normalize: Normalization mode ('true', 'pred', 'all', or None).

    Returns:
        Confusion matrix of shape (C, C).
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize == "true":
        cm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)
    elif normalize == "pred":
        cm = cm.astype(np.float64) / cm.sum(axis=0, keepdims=True).clip(min=1)
    elif normalize == "all":
        cm = cm.astype(np.float64) / cm.sum().clip(min=1)

    return cm


def compute_detection_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive detection evaluation report.

    Combines overall metrics, per-class metrics, and the confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Optional probability matrix for AUC-ROC.
        class_names: Optional class name labels.

    Returns:
        Dictionary with 'overall', 'per_class', 'confusion_matrix',
        and 'classification_report' (sklearn text report).
    """
    overall = compute_detection_metrics(y_true, y_pred, y_proba)
    per_class = compute_per_class_metrics(y_true, y_pred, class_names)
    cm = compute_confusion_matrix(y_true, y_pred)
    cm_normalized = compute_confusion_matrix(y_true, y_pred, normalize="true")

    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_normalized,
        "classification_report": report_text,
    }
