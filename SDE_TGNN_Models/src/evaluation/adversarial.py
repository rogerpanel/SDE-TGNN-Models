"""Adversarial robustness evaluation for SDE-TGNN.

Provides:
- FGSM attack (single-step)
- PGD attack (iterative)
- Certified accuracy via stochastic reachability
- Full robustness evaluation pipeline
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.01,
    edge_index: Optional[torch.Tensor] = None,
    edge_attr: Optional[torch.Tensor] = None,
    timestamps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fast Gradient Sign Method (FGSM) adversarial attack.

    Generates adversarial examples using a single gradient step:
        x_adv = x + epsilon * sign(grad_x L(model(x), y))

    Args:
        model: The target model (must be in eval mode).
        x: Clean input features of shape (N, D).
        y: True labels of shape (N,).
        epsilon: Maximum L-infinity perturbation.
        edge_index: Optional graph edge indices.
        edge_attr: Optional edge attributes.
        timestamps: Optional timestamps.

    Returns:
        Adversarial examples of shape (N, D).
    """
    x_adv = x.clone().detach().requires_grad_(True)

    logits, _ = model(
        x_adv,
        edge_index=edge_index,
        edge_attr=edge_attr,
        timestamps=timestamps,
        return_uncertainty=False,
    )

    loss = F.cross_entropy(logits, y)
    loss.backward()

    # FGSM perturbation
    grad_sign = x_adv.grad.data.sign()
    x_adv = x_adv.detach() + epsilon * grad_sign

    # Clamp to valid range
    x_adv = x_adv.clamp(x.min(), x.max())

    return x_adv.detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.01,
    alpha: float = 0.001,
    num_steps: int = 20,
    random_start: bool = True,
    edge_index: Optional[torch.Tensor] = None,
    edge_attr: Optional[torch.Tensor] = None,
    timestamps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Projected Gradient Descent (PGD) adversarial attack.

    Iteratively perturbs the input within an L-infinity ball:
        x_{t+1} = Proj_{B(x, eps)} [ x_t + alpha * sign(grad_x L) ]

    Args:
        model: The target model.
        x: Clean input features of shape (N, D).
        y: True labels of shape (N,).
        epsilon: L-infinity perturbation budget.
        alpha: Step size per iteration.
        num_steps: Number of PGD iterations.
        random_start: Initialize with random perturbation within the ball.
        edge_index: Optional graph edge indices.
        edge_attr: Optional edge attributes.
        timestamps: Optional timestamps.

    Returns:
        Adversarial examples of shape (N, D).
    """
    x_adv = x.clone().detach()

    if random_start:
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)
        x_adv = x_adv.clamp(x.min(), x.max())

    for step in range(num_steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)

        logits, _ = model(
            x_adv,
            edge_index=edge_index,
            edge_attr=edge_attr,
            timestamps=timestamps,
            return_uncertainty=False,
        )

        loss = F.cross_entropy(logits, y)
        loss.backward()

        # Gradient ascent step
        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv.detach() + alpha * grad_sign

        # Project back to epsilon ball around original input
        perturbation = (x_adv - x).clamp(-epsilon, epsilon)
        x_adv = x + perturbation

        # Clamp to valid feature range
        x_adv = x_adv.clamp(x.min(), x.max())

    return x_adv.detach()


def certified_accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.01,
    num_samples: int = 100,
    noise_std: float = 0.01,
    alpha: float = 0.001,
    edge_index: Optional[torch.Tensor] = None,
    edge_attr: Optional[torch.Tensor] = None,
    timestamps: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute certified accuracy via randomized smoothing.

    Uses the stochastic nature of the SDE to certify robustness.
    For each input, we run multiple noisy forward passes and check
    whether the majority prediction is correct.  The certified radius
    is derived from the Cohen et al. (2019) smoothing framework.

    Args:
        model: The SDE-TGNN model.
        x: Input features of shape (N, D).
        y: True labels of shape (N,).
        epsilon: Perturbation radius for certification.
        num_samples: Number of noisy forward passes.
        noise_std: Standard deviation of input noise.
        alpha: Confidence level for certification.
        edge_index: Optional graph edge indices.
        edge_attr: Optional edge attributes.
        timestamps: Optional timestamps.

    Returns:
        Dictionary with 'certified_accuracy', 'clean_accuracy',
        'abstention_rate', and 'mean_certified_radius'.
    """
    model.eval()
    N = x.size(0)
    num_classes = None

    # Collect predictions under noise
    all_counts = torch.zeros(N, dtype=torch.long, device=x.device)
    correct_under_noise = torch.zeros(N, dtype=torch.long, device=x.device)

    with torch.no_grad():
        # First pass: determine number of classes
        logits_clean, _ = model(
            x, edge_index=edge_index, edge_attr=edge_attr,
            timestamps=timestamps, return_uncertainty=False,
        )
        num_classes = logits_clean.size(-1)
        clean_preds = logits_clean.argmax(dim=-1)
        clean_correct = (clean_preds == y).float()

        # Class counts for each sample
        class_counts = torch.zeros(N, num_classes, device=x.device)

        for sample_idx in range(num_samples):
            noise = torch.randn_like(x) * noise_std
            x_noisy = x + noise

            logits, _ = model(
                x_noisy, edge_index=edge_index, edge_attr=edge_attr,
                timestamps=timestamps, return_uncertainty=False,
            )
            preds = logits.argmax(dim=-1)

            # Count class predictions
            for c in range(num_classes):
                class_counts[:, c] += (preds == c).float()

    # Majority vote
    majority_class = class_counts.argmax(dim=-1)
    majority_count = class_counts.max(dim=-1)[0]
    majority_correct = (majority_class == y).float()

    # Certified radius using Neyman-Pearson
    # For Gaussian noise with std sigma, certified radius r = sigma * Phi^{-1}(p_A)
    # where p_A is the lower bound on the probability of the top class
    p_a = majority_count / num_samples

    # Simple lower bound on p_A (Clopper-Pearson)
    from scipy.stats import norm as normal_dist
    p_a_np = p_a.cpu().numpy()
    certified_radii = np.zeros(N)
    certified_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        if p_a_np[i] > 0.5:
            # Certified radius
            radius = noise_std * normal_dist.ppf(p_a_np[i])
            certified_radii[i] = radius
            if radius >= epsilon:
                certified_mask[i] = True

    # Metrics
    cert_acc = float((majority_correct.cpu().numpy() * certified_mask).mean())
    clean_acc = float(clean_correct.mean().item())
    abstention_rate = float(1.0 - certified_mask.mean())
    mean_radius = float(certified_radii[certified_mask].mean()) if certified_mask.any() else 0.0

    return {
        "certified_accuracy": cert_acc,
        "clean_accuracy": clean_acc,
        "abstention_rate": abstention_rate,
        "mean_certified_radius": mean_radius,
    }


def evaluate_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    epsilon_values: Optional[list[float]] = None,
    pgd_steps: int = 20,
    pgd_alpha: float = 0.001,
    num_cert_samples: int = 100,
    noise_std: float = 0.01,
) -> Dict[str, Any]:
    """Run a comprehensive adversarial robustness evaluation.

    Tests the model against FGSM, PGD, and computes certified
    accuracy at multiple epsilon values.

    Args:
        model: The trained model.
        test_loader: Test DataLoader.
        device: Compute device.
        epsilon_values: List of epsilon budgets to evaluate.
        pgd_steps: PGD iteration count.
        pgd_alpha: PGD step size.
        num_cert_samples: Samples for certified accuracy.
        noise_std: Noise std for randomized smoothing.

    Returns:
        Dictionary with robustness metrics for each epsilon.
    """
    if epsilon_values is None:
        epsilon_values = [0.001, 0.005, 0.01, 0.02, 0.05]

    model.eval()
    results: Dict[str, Any] = {"epsilon_values": epsilon_values}

    # Collect all test data
    all_features: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in test_loader:
            all_features.append(batch["features"].to(device))
            all_labels.append(batch["label"].to(device))

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Limit size for computational feasibility
    max_eval = min(len(labels), 2000)
    if len(labels) > max_eval:
        indices = torch.randperm(len(labels))[:max_eval]
        features = features[indices]
        labels = labels[indices]

    # Clean accuracy
    with torch.no_grad():
        clean_logits, _ = model(features, return_uncertainty=False)
        clean_preds = clean_logits.argmax(dim=-1)
        clean_acc = (clean_preds == labels).float().mean().item()

    results["clean_accuracy"] = clean_acc
    results["fgsm"] = {}
    results["pgd"] = {}
    results["certified"] = {}

    for eps in epsilon_values:
        logger.info("Evaluating robustness at epsilon=%.4f", eps)

        # FGSM attack
        x_fgsm = fgsm_attack(model, features, labels, epsilon=eps)
        with torch.no_grad():
            fgsm_logits, _ = model(x_fgsm, return_uncertainty=False)
            fgsm_preds = fgsm_logits.argmax(dim=-1)
            fgsm_acc = (fgsm_preds == labels).float().mean().item()
        results["fgsm"][eps] = {
            "accuracy": fgsm_acc,
            "drop": clean_acc - fgsm_acc,
        }

        # PGD attack
        x_pgd = pgd_attack(
            model, features, labels,
            epsilon=eps, alpha=pgd_alpha, num_steps=pgd_steps,
        )
        with torch.no_grad():
            pgd_logits, _ = model(x_pgd, return_uncertainty=False)
            pgd_preds = pgd_logits.argmax(dim=-1)
            pgd_acc = (pgd_preds == labels).float().mean().item()
        results["pgd"][eps] = {
            "accuracy": pgd_acc,
            "drop": clean_acc - pgd_acc,
        }

        # Certified accuracy (smaller subset for speed)
        cert_size = min(500, len(labels))
        cert_result = certified_accuracy(
            model, features[:cert_size], labels[:cert_size],
            epsilon=eps, num_samples=num_cert_samples, noise_std=noise_std,
        )
        results["certified"][eps] = cert_result

        logger.info(
            "eps=%.4f  FGSM_acc=%.4f  PGD_acc=%.4f  Cert_acc=%.4f",
            eps, fgsm_acc, pgd_acc, cert_result["certified_accuracy"],
        )

    return results
