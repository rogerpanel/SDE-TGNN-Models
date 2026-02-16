"""Evaluation utilities for SDE-TGNN.

This subpackage provides:
- Detection metrics (accuracy, precision, recall, F1, AUC-ROC)
- Calibration metrics (ECE, Brier score, reliability diagrams)
- Adversarial robustness evaluation (PGD, FGSM, certified accuracy)
- Visualization tools (reliability diagrams, confusion matrices, etc.)
"""

from src.evaluation.metrics import (
    compute_detection_metrics,
    compute_per_class_metrics,
    compute_confusion_matrix,
)
from src.evaluation.calibration import (
    expected_calibration_error,
    brier_score,
    prediction_interval_coverage,
    reliability_diagram,
    temperature_scaling,
)
from src.evaluation.adversarial import (
    certified_accuracy,
    pgd_attack,
    fgsm_attack,
    evaluate_robustness,
)

__all__ = [
    "compute_detection_metrics",
    "compute_per_class_metrics",
    "compute_confusion_matrix",
    "expected_calibration_error",
    "brier_score",
    "prediction_interval_coverage",
    "reliability_diagram",
    "temperature_scaling",
    "certified_accuracy",
    "pgd_attack",
    "fgsm_attack",
    "evaluate_robustness",
]
