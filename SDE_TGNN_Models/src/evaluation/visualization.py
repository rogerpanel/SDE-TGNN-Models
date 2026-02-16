"""Visualization utilities for SDE-TGNN evaluation results.

Provides publication-quality plots for:
- Reliability diagrams
- F1-score comparisons across models/datasets
- Latency vs. throughput scatter plots
- Training convergence curves
- Confusion matrices
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Publication-quality defaults
plt.rcParams.update({
    "font.size": 12,
    "font.family": "serif",
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def plot_reliability_diagram(
    bin_confidences: np.ndarray,
    bin_accuracies: np.ndarray,
    bin_counts: np.ndarray,
    ece: float,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a reliability diagram showing calibration quality.

    The ideal model follows the diagonal (confidence = accuracy).
    Gaps above the diagonal indicate under-confidence; below indicate
    over-confidence.

    Args:
        bin_confidences: Average confidence per bin of shape (B,).
        bin_accuracies: Average accuracy per bin of shape (B,).
        bin_counts: Number of samples per bin of shape (B,).
        ece: Expected Calibration Error value.
        title: Plot title.
        save_path: Optional file path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    num_bins = len(bin_confidences)
    bin_width = 1.0 / num_bins
    bin_centers = np.linspace(bin_width / 2, 1 - bin_width / 2, num_bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]})

    # Top: Reliability diagram
    ax1.bar(
        bin_centers, bin_accuracies, width=bin_width * 0.9,
        alpha=0.7, color="#4C72B0", edgecolor="black", linewidth=0.5,
        label="Model",
    )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration")

    # Gap visualization
    for i in range(num_bins):
        if bin_counts[i] > 0:
            gap = bin_confidences[i] - bin_accuracies[i]
            color = "#E07070" if gap > 0 else "#70B070"
            ax1.bar(
                bin_centers[i], abs(gap), bottom=min(bin_confidences[i], bin_accuracies[i]),
                width=bin_width * 0.9, alpha=0.3, color=color, edgecolor="none",
            )

    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"{title}\nECE = {ece:.4f}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bottom: Sample distribution
    ax2.bar(
        bin_centers, bin_counts / max(bin_counts.sum(), 1),
        width=bin_width * 0.9, alpha=0.7, color="#DD8452",
        edgecolor="black", linewidth=0.5,
    )
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Fraction")
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_f1_comparison(
    model_names: List[str],
    dataset_names: List[str],
    f1_scores: np.ndarray,
    title: str = "F1-Score Comparison Across Models and Datasets",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a grouped bar chart comparing F1 scores across models and datasets.

    Args:
        model_names: List of model names (length M).
        dataset_names: List of dataset names (length D).
        f1_scores: Matrix of shape (M, D) with F1 scores.
        title: Plot title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    num_models = len(model_names)
    num_datasets = len(dataset_names)

    fig, ax = plt.subplots(figsize=(max(10, num_datasets * 1.5), 6))

    x = np.arange(num_datasets)
    bar_width = 0.8 / num_models

    colors = sns.color_palette("husl", num_models)

    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        offset = (i - num_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, f1_scores[i], bar_width * 0.9,
            label=model_name, color=color, edgecolor="black", linewidth=0.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, f1_scores[i]):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=90,
                )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_latency_throughput(
    model_names: List[str],
    latencies_ms: List[float],
    throughputs: List[float],
    f1_scores: Optional[List[float]] = None,
    title: str = "Latency vs. Throughput",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot latency vs. throughput scatter with optional F1 size coding.

    Args:
        model_names: List of model names.
        latencies_ms: Per-sample inference latency in milliseconds.
        throughputs: Samples processed per second.
        f1_scores: Optional F1 scores for marker sizing.
        title: Plot title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = sns.color_palette("husl", len(model_names))

    if f1_scores is not None:
        sizes = [max(50, f1 * 300) for f1 in f1_scores]
    else:
        sizes = [100] * len(model_names)

    for i, (name, lat, thr, sz, col) in enumerate(
        zip(model_names, latencies_ms, throughputs, sizes, colors)
    ):
        ax.scatter(lat, thr, s=sz, c=[col], alpha=0.8, edgecolors="black", linewidth=0.5, zorder=5)
        ax.annotate(
            name, (lat, thr),
            textcoords="offset points", xytext=(8, 5),
            fontsize=9, ha="left",
        )

    ax.set_xlabel("Latency (ms/sample)")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_training_convergence(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    title: str = "Training Convergence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot training and validation loss/accuracy curves.

    Args:
        train_losses: Training loss per epoch.
        val_losses: Validation loss per epoch.
        train_accs: Optional training accuracy per epoch.
        val_accs: Optional validation accuracy per epoch.
        title: Plot title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    has_acc = train_accs is not None and val_accs is not None
    ncols = 2 if has_acc else 1

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    axes[0].plot(epochs, train_losses, "b-", linewidth=1.5, label="Train Loss", alpha=0.8)
    axes[0].plot(epochs, val_losses, "r-", linewidth=1.5, label="Val Loss", alpha=0.8)

    # Mark best validation loss
    best_epoch = np.argmin(val_losses) + 1
    best_loss = min(val_losses)
    axes[0].axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5)
    axes[0].scatter([best_epoch], [best_loss], c="red", s=80, zorder=5, marker="*")
    axes[0].annotate(
        f"Best: {best_loss:.4f}\n(epoch {best_epoch})",
        (best_epoch, best_loss),
        textcoords="offset points", xytext=(10, 10),
        fontsize=9, color="red",
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Convergence")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if has_acc:
        axes[1].plot(epochs[:len(train_accs)], train_accs, "b-", linewidth=1.5, label="Train Acc", alpha=0.8)
        axes[1].plot(epochs[:len(val_accs)], val_accs, "r-", linewidth=1.5, label="Val Acc", alpha=0.8)

        best_acc_epoch = np.argmax(val_accs) + 1
        best_acc = max(val_accs)
        axes[1].axvline(x=best_acc_epoch, color="gray", linestyle="--", alpha=0.5)
        axes[1].scatter([best_acc_epoch], [best_acc], c="red", s=80, zorder=5, marker="*")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Convergence")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix of shape (C, C).
        class_names: List of class label strings.
        normalize: Whether to show normalized values.
        title: Plot title.
        cmap: Matplotlib colormap name.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm.astype(float) / row_sums
        fmt = ".2f"
        vmin, vmax = 0, 1
    else:
        cm_plot = cm
        fmt = "d"
        vmin, vmax = None, None

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    figsize = max(6, num_classes * 0.7)
    fig, ax = plt.subplots(figsize=(figsize, figsize))

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="gray",
        square=True,
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig


def plot_uncertainty_histogram(
    uncertainties: np.ndarray,
    correct_mask: np.ndarray,
    title: str = "Uncertainty Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot uncertainty histograms for correct vs. incorrect predictions.

    A well-calibrated uncertainty model should assign higher uncertainty
    to incorrect predictions.

    Args:
        uncertainties: Uncertainty values of shape (N,).
        correct_mask: Boolean mask (True = correct prediction).
        title: Plot title.
        save_path: Optional file path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    correct_unc = uncertainties[correct_mask]
    incorrect_unc = uncertainties[~correct_mask]

    bins = np.linspace(uncertainties.min(), uncertainties.max(), 50)

    ax.hist(
        correct_unc, bins=bins, alpha=0.6, color="#4C72B0",
        label=f"Correct (n={len(correct_unc)})", density=True,
    )
    ax.hist(
        incorrect_unc, bins=bins, alpha=0.6, color="#DD8452",
        label=f"Incorrect (n={len(incorrect_unc)})", density=True,
    )

    ax.axvline(correct_unc.mean(), color="#4C72B0", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(incorrect_unc.mean(), color="#DD8452", linestyle="--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Uncertainty")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)

    return fig
