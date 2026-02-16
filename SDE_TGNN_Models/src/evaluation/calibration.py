"""Calibration metrics and post-hoc calibration methods.

Provides:
- Expected Calibration Error (ECE)
- Brier Score
- Prediction Interval Coverage Probability (PICP)
- Reliability Diagram data
- Temperature Scaling for post-hoc calibration
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15,
) -> Dict[str, float]:
    """Compute the Expected Calibration Error (ECE).

    ECE = sum_{b=1}^{B} (n_b / N) * |acc(b) - conf(b)|

    Args:
        confidences: Predicted confidence (max prob) of shape (N,).
        accuracies: Binary correctness indicators of shape (N,).
        num_bins: Number of uniform confidence bins.

    Returns:
        Dictionary with 'ece', 'mce' (maximum calibration error),
        'bin_confidences', 'bin_accuracies', and 'bin_counts'.
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []

    ece = 0.0
    mce = 0.0
    n = len(confidences)

    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == num_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        bin_count = mask.sum()
        bin_counts.append(int(bin_count))

        if bin_count > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            bin_confidences.append(float(bin_conf))
            bin_accuracies.append(float(bin_acc))

            gap = abs(bin_conf - bin_acc)
            ece += (bin_count / n) * gap
            mce = max(mce, gap)
        else:
            bin_confidences.append(0.0)
            bin_accuracies.append(0.0)

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


def brier_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """Compute the Brier score (multi-class generalization).

    BS = (1/N) * sum_{i=1}^{N} sum_{c=1}^{C} (p_ic - y_ic)^2

    Lower is better. Range: [0, 2].

    Args:
        y_true: Ground truth labels of shape (N,).
        y_proba: Predicted probabilities of shape (N, C).

    Returns:
        Brier score (scalar).
    """
    num_classes = y_proba.shape[1]
    # One-hot encode targets
    y_onehot = np.zeros_like(y_proba)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0

    bs = np.mean(np.sum((y_proba - y_onehot) ** 2, axis=1))
    return float(bs)


def prediction_interval_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    nominal_coverage: float = 0.95,
) -> Dict[str, float]:
    """Compute Prediction Interval Coverage Probability (PICP).

    For uncertainty-aware models, checks how often the true value
    falls within the predicted confidence intervals.

    Args:
        y_true: Ground truth values of shape (N,).
        y_lower: Lower bound predictions of shape (N,).
        y_upper: Upper bound predictions of shape (N,).
        nominal_coverage: Target coverage level.

    Returns:
        Dictionary with 'picp', 'mean_width', and 'coverage_gap'.
    """
    covered = ((y_true >= y_lower) & (y_true <= y_upper)).astype(float)
    picp = covered.mean()
    mean_width = (y_upper - y_lower).mean()

    return {
        "picp": float(picp),
        "mean_width": float(mean_width),
        "coverage_gap": float(picp - nominal_coverage),
    }


def reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 15,
) -> Dict[str, np.ndarray]:
    """Compute data for a reliability diagram.

    Args:
        confidences: Predicted confidence (max prob) of shape (N,).
        accuracies: Binary correctness of shape (N,).
        num_bins: Number of bins.

    Returns:
        Dictionary with 'bin_centers', 'bin_confidences',
        'bin_accuracies', 'bin_counts', suitable for plotting.
    """
    ece_result = expected_calibration_error(confidences, accuracies, num_bins)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])

    return {
        "bin_centers": bin_centers,
        "bin_confidences": np.array(ece_result["bin_confidences"]),
        "bin_accuracies": np.array(ece_result["bin_accuracies"]),
        "bin_counts": np.array(ece_result["bin_counts"]),
        "ece": ece_result["ece"],
        "mce": ece_result["mce"],
    }


class TemperatureScaling(nn.Module):
    """Learnable temperature parameter for post-hoc calibration.

    Learns a single scalar T such that calibrated probabilities are:
        p_cal = softmax(logits / T)

    Attributes:
        temperature: Learnable scalar parameter.
    """

    def __init__(self) -> None:
        """Initialize temperature scaling with T=1."""
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Raw logits of shape (N, C).

        Returns:
            Calibrated logits of shape (N, C).
        """
        return logits / self.temperature.clamp(min=0.01)


def temperature_scaling(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_iter: int = 100,
    lr: float = 0.01,
) -> TemperatureScaling:
    """Learn the optimal temperature from a validation set.

    Minimizes the negative log-likelihood on the validation set
    with respect to a single temperature parameter.

    Args:
        model: Trained model that produces logits.
        val_loader: Validation DataLoader.
        device: Compute device.
        max_iter: Maximum optimization iterations.
        lr: Learning rate for temperature optimization.

    Returns:
        Fitted TemperatureScaling module.
    """
    temp_model = TemperatureScaling().to(device)
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=lr, max_iter=max_iter)

    # Collect all logits and labels
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            logits, _ = model(features, return_uncertainty=False)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits_cat = torch.cat(all_logits, dim=0)
    all_labels_cat = torch.cat(all_labels, dim=0)

    # Optimize temperature
    nll_criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        calibrated_logits = temp_model(all_logits_cat)
        loss = nll_criterion(calibrated_logits, all_labels_cat)
        loss.backward()
        return loss

    optimizer.step(closure)

    optimal_temp = temp_model.temperature.item()
    logger_msg = f"Optimal temperature: {optimal_temp:.4f}"
    print(logger_msg)

    return temp_model
