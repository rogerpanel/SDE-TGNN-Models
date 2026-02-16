"""Loss functions for SDE-TGNN training.

Combines classification, ELBO, calibration, and KL-divergence
objectives into a unified loss for joint optimization.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ELBOLoss(nn.Module):
    """Evidence Lower Bound (ELBO) loss for variational inference.

    Combines the negative log-likelihood (reconstruction / classification
    term) with a KL divergence regularizer that encourages the SDE's
    latent distribution to remain close to a standard normal prior.

    ELBO = E_q[log p(y|z)] - beta * KL(q(z|x) || p(z))

    Attributes:
        beta: KL divergence weight (beta-VAE style).
        reduction: Reduction mode ('mean' or 'sum').
    """

    def __init__(
        self,
        beta: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize the ELBO loss.

        Args:
            beta: Weight for the KL divergence term.
            reduction: How to reduce the loss ('mean' or 'sum').
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        kl_divergence: torch.Tensor,
        log_variance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the ELBO loss.

        Args:
            logits: Model output logits of shape (N, C).
            targets: Ground truth labels of shape (N,).
            kl_divergence: KL divergence scalar from the Fokker-Planck solver.
            log_variance: Optional per-sample log variance from the
                aleatoric uncertainty head, shape (N, C).

        Returns:
            ELBO loss scalar.
        """
        # Negative log-likelihood (classification)
        if log_variance is not None:
            # Heteroscedastic loss: integrate over aleatoric noise
            # L = log(sigma) + 0.5 * (y - mu)^2 / sigma^2
            precision = torch.exp(-log_variance)  # (N, C)
            nll = F.cross_entropy(logits * precision, targets, reduction="none")
            nll = nll + 0.5 * log_variance.mean(dim=-1)
        else:
            nll = F.cross_entropy(logits, targets, reduction="none")

        if self.reduction == "mean":
            nll = nll.mean()
        else:
            nll = nll.sum()

        # ELBO = NLL + beta * KL
        loss = nll + self.beta * kl_divergence

        return loss


class CalibrationLoss(nn.Module):
    """Expected Calibration Error (ECE) as a differentiable loss.

    Uses a soft-binning approximation to make ECE differentiable,
    enabling joint optimization of calibration alongside classification.

    Attributes:
        num_bins: Number of confidence bins.
        temperature: Softness of the bin assignment (lower = harder).
    """

    def __init__(
        self,
        num_bins: int = 15,
        temperature: float = 1.0,
    ) -> None:
        """Initialize the calibration loss.

        Args:
            num_bins: Number of confidence bins for ECE computation.
            temperature: Temperature for soft bin assignment.
        """
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature

        # Bin boundaries
        boundaries = torch.linspace(0, 1, num_bins + 1)
        self.register_buffer("bin_lowers", boundaries[:-1])
        self.register_buffer("bin_uppers", boundaries[1:])
        self.register_buffer("bin_centers", (boundaries[:-1] + boundaries[1:]) / 2)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the differentiable ECE loss.

        Args:
            logits: Model logits of shape (N, C).
            targets: Ground truth labels of shape (N,).

        Returns:
            ECE loss scalar.
        """
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        accuracies = (predictions == targets).float()

        ece = torch.zeros(1, device=logits.device, dtype=logits.dtype)
        total_samples = float(len(targets))

        for lower, upper in zip(self.bin_lowers, self.bin_uppers):
            # Soft bin membership via sigmoid
            in_bin_lower = torch.sigmoid(
                (confidences - lower) / self.temperature
            )
            in_bin_upper = torch.sigmoid(
                (upper - confidences) / self.temperature
            )
            in_bin = in_bin_lower * in_bin_upper
            bin_weight = in_bin.sum()

            if bin_weight > 1e-8:
                # Weighted accuracy and confidence within the bin
                bin_accuracy = (accuracies * in_bin).sum() / bin_weight
                bin_confidence = (confidences * in_bin).sum() / bin_weight
                ece = ece + (bin_weight / total_samples) * (bin_confidence - bin_accuracy).abs()

        return ece.squeeze()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.

    FL(p) = -alpha * (1 - p)^gamma * log(p)

    Focuses learning on hard, misclassified examples by down-weighting
    easy negatives.

    Attributes:
        alpha: Class balancing weight.
        gamma: Focusing parameter.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize Focal Loss.

        Args:
            alpha: Balancing weight.
            gamma: Focusing parameter (higher = more focus on hard examples).
            reduction: Reduction mode.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the focal loss.

        Args:
            logits: Logits of shape (N, C).
            targets: Labels of shape (N,).

        Returns:
            Focal loss scalar.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """Combined training loss for SDE-TGNN.

    Aggregates:
    1. Classification loss (cross-entropy or focal)
    2. ELBO loss (NLL + KL)
    3. Calibration loss (differentiable ECE)
    4. KL divergence regularizer (from FP solver)

    The total loss is:
        L = w_cls * L_cls + w_elbo * L_elbo + w_cal * L_cal + w_kl * KL

    Attributes:
        weights: Dictionary of loss component weights.
    """

    def __init__(
        self,
        classification_weight: float = 1.0,
        elbo_weight: float = 0.1,
        calibration_weight: float = 0.05,
        kl_weight: float = 0.01,
        label_smoothing: float = 0.05,
        focal_gamma: float = 0.0,
        num_calibration_bins: int = 15,
    ) -> None:
        """Initialize the combined loss.

        Args:
            classification_weight: Weight for classification loss.
            elbo_weight: Weight for ELBO loss.
            calibration_weight: Weight for calibration loss.
            kl_weight: Weight for KL divergence.
            label_smoothing: Label smoothing factor for CE loss.
            focal_gamma: Focal loss gamma (0 = standard CE).
            num_calibration_bins: ECE bins.
        """
        super().__init__()
        self.weights = {
            "classification": classification_weight,
            "elbo": elbo_weight,
            "calibration": calibration_weight,
            "kl": kl_weight,
        }

        # Classification loss
        if focal_gamma > 0:
            self.cls_loss = FocalLoss(gamma=focal_gamma)
        else:
            self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # ELBO loss
        self.elbo_loss = ELBOLoss(beta=kl_weight / max(elbo_weight, 1e-8))

        # Calibration loss
        self.cal_loss = CalibrationLoss(num_bins=num_calibration_bins)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        uncertainty_dict: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined loss.

        Args:
            logits: Model logits of shape (N, C).
            targets: Ground truth labels of shape (N,).
            uncertainty_dict: Dictionary from the SDE-TGNN forward pass
                containing 'kl_divergence', 'log_variance', etc.

        Returns:
            Dictionary with individual loss components and 'total'.
        """
        if uncertainty_dict is None:
            uncertainty_dict = {}

        losses: Dict[str, torch.Tensor] = {}
        device = logits.device

        # 1. Classification loss
        cls = self.cls_loss(logits, targets)
        losses["classification"] = cls

        # 2. ELBO loss
        kl_div = uncertainty_dict.get(
            "kl_divergence",
            torch.tensor(0.0, device=device),
        )
        log_var = uncertainty_dict.get("log_variance", None)
        elbo = self.elbo_loss(logits, targets, kl_div, log_var)
        losses["elbo"] = elbo

        # 3. Calibration loss
        cal = self.cal_loss(logits, targets)
        losses["calibration"] = cal

        # 4. KL divergence
        losses["kl_divergence"] = kl_div

        # Total weighted loss
        total = (
            self.weights["classification"] * cls
            + self.weights["elbo"] * elbo
            + self.weights["calibration"] * cal
            + self.weights["kl"] * kl_div
        )
        losses["total"] = total

        return losses
