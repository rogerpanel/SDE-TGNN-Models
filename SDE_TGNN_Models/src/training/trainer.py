"""Training loop for SDE-TGNN with early stopping, checkpointing, and logging.

Provides the ``SDETGNNTrainer`` class that orchestrates the full
training pipeline including learning rate scheduling, gradient
clipping, mixed-precision training, and TensorBoard logging.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import CombinedLoss

logger = logging.getLogger(__name__)


class SDETGNNTrainer:
    """Full training pipeline for the SDE-TGNN model.

    Handles epoch loops, validation, early stopping, model
    checkpointing, and optional TensorBoard logging.

    Attributes:
        model: The SDE-TGNN (or baseline) model.
        optimizer: PyTorch optimizer.
        scheduler: Learning rate scheduler.
        loss_fn: Combined loss function.
        config: Training configuration dictionary.
        device: Target device (CPU or CUDA).
        best_val_loss: Best validation loss observed so far.
        patience_counter: Epochs since last improvement.
        epoch: Current epoch index.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        loss_fn: CombinedLoss,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        output_dir: str = "outputs",
    ) -> None:
        """Initialize the trainer.

        Args:
            model: The neural network model.
            optimizer: Optimizer instance.
            scheduler: Optional LR scheduler.
            loss_fn: Combined loss function.
            config: Training hyperparameter dictionary.
            device: Compute device.
            output_dir: Directory for checkpoints and logs.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.output_dir = output_dir

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.epoch = 0
        self.global_step = 0
        self.train_history: list[Dict[str, float]] = []
        self.val_history: list[Dict[str, float]] = []

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

        # TensorBoard
        self.writer = None
        if config.get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))
            except ImportError:
                logger.warning("TensorBoard not available, disabling logging.")

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda") if self.device.type == "cuda" else None
        self.use_amp = self.device.type == "cuda"

        # Gradient clipping
        self.grad_clip = config.get("grad_clip", 1.0)

        logger.info(
            "Trainer initialized: device=%s, amp=%s, grad_clip=%.2f",
            self.device, self.use_amp, self.grad_clip,
        )

    def _forward_batch(
        self,
        batch: Dict[str, torch.Tensor],
        return_uncertainty: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run a single batch through the model.

        Args:
            batch: Dictionary from the DataLoader with keys 'features',
                'label', and optionally 'edge_index', 'edge_attr', 'timestamp'.
            return_uncertainty: Whether to request uncertainty from the model.

        Returns:
            Tuple of (logits, labels, uncertainty_dict).
        """
        features = batch["features"].to(self.device)
        labels = batch["label"].to(self.device)

        edge_index = batch.get("edge_index", None)
        edge_attr = batch.get("edge_attr", None)
        timestamps = batch.get("timestamp", None)

        if edge_index is not None:
            edge_index = edge_index.to(self.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(self.device)
        if timestamps is not None:
            timestamps = timestamps.to(self.device)

        logits, uncertainty_dict = self.model(
            features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            timestamps=timestamps,
            return_uncertainty=return_uncertainty,
        )

        return logits, labels, uncertainty_dict

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training DataLoader.

        Returns:
            Dictionary of averaged training metrics.
        """
        self.model.train()

        epoch_losses: Dict[str, list[float]] = {
            "total": [], "classification": [], "elbo": [],
            "calibration": [], "kl_divergence": [],
        }
        correct = 0
        total = 0
        start_time = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.epoch + 1} [Train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()

            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits, labels, uncertainty_dict = self._forward_batch(
                        batch, return_uncertainty=True,
                    )
                    losses = self.loss_fn(logits, labels, uncertainty_dict)

                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, labels, uncertainty_dict = self._forward_batch(
                    batch, return_uncertainty=True,
                )
                losses = self.loss_fn(logits, labels, uncertainty_dict)

                losses["total"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # Track metrics
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key].append(losses[key].item())

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "acc": f"{correct / max(total, 1):.4f}",
            })

            # TensorBoard batch logging
            if self.writer and batch_idx % self.config.get("log_interval", 10) == 0:
                self.writer.add_scalar("train/batch_loss", losses["total"].item(), self.global_step)

        elapsed = time.time() - start_time

        metrics = {
            key: sum(vals) / max(len(vals), 1)
            for key, vals in epoch_losses.items()
        }
        metrics["accuracy"] = correct / max(total, 1)
        metrics["epoch_time"] = elapsed

        # TensorBoard epoch logging
        if self.writer:
            for key, val in metrics.items():
                self.writer.add_scalar(f"train/{key}", val, self.epoch)
            self.writer.add_scalar(
                "train/lr", self.optimizer.param_groups[0]["lr"], self.epoch,
            )

        self.train_history.append(metrics)
        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on the validation set.

        Args:
            val_loader: Validation DataLoader.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()

        epoch_losses: Dict[str, list[float]] = {
            "total": [], "classification": [], "elbo": [],
            "calibration": [], "kl_divergence": [],
        }
        correct = 0
        total = 0

        pbar = tqdm(
            val_loader,
            desc=f"Epoch {self.epoch + 1} [Val]",
            leave=False,
        )

        for batch in pbar:
            logits, labels, uncertainty_dict = self._forward_batch(
                batch, return_uncertainty=True,
            )
            losses = self.loss_fn(logits, labels, uncertainty_dict)

            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key].append(losses[key].item())

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        metrics = {
            key: sum(vals) / max(len(vals), 1)
            for key, vals in epoch_losses.items()
        }
        metrics["accuracy"] = correct / max(total, 1)

        # TensorBoard
        if self.writer:
            for key, val in metrics.items():
                self.writer.add_scalar(f"val/{key}", val, self.epoch)

        self.val_history.append(metrics)
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping and checkpointing.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            num_epochs: Override for maximum epochs.

        Returns:
            Dictionary with training history and best metrics.
        """
        num_epochs = num_epochs or self.config.get("epochs", 100)
        patience = self.config.get("patience", 15)

        logger.info("Starting training for %d epochs", num_epochs)
        logger.info("Model parameters: %d", sum(p.numel() for p in self.model.parameters()))

        best_metrics: Dict[str, Any] = {}

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(
                "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  time=%.1fs",
                epoch + 1, num_epochs,
                train_metrics["total"],
                train_metrics["accuracy"],
                train_metrics["epoch_time"],
            )

            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(
                "Epoch %d/%d  val_loss=%.4f  val_acc=%.4f",
                epoch + 1, num_epochs,
                val_metrics["total"],
                val_metrics["accuracy"],
            )

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["total"])
                else:
                    self.scheduler.step()

            # Early stopping check
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                self.patience_counter = 0
                best_metrics = {
                    "epoch": epoch + 1,
                    "train": train_metrics,
                    "val": val_metrics,
                }

                # Save best model
                if self.config.get("save_best", True):
                    self.save_checkpoint(
                        os.path.join(self.output_dir, "checkpoints", "best_model.pt"),
                    )
                    logger.info("Saved best model (val_loss=%.4f)", self.best_val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

            # Periodic checkpoint
            save_every = self.config.get("save_every", 10)
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(
                    os.path.join(
                        self.output_dir, "checkpoints", f"epoch_{epoch + 1}.pt",
                    ),
                )

        # Close TensorBoard
        if self.writer:
            self.writer.close()

        return {
            "best": best_metrics,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save a model checkpoint.

        Args:
            path: File path for the checkpoint.
        """
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "train_history": self.train_history,
            "val_history": self.val_history,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint.

        Args:
            path: Path to the checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(
            "Loaded checkpoint from %s (epoch %d, val_loss=%.4f)",
            path, self.epoch, self.best_val_loss,
        )


def build_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> Optimizer:
    """Build an optimizer from the config dictionary.

    Args:
        model: The model whose parameters are to be optimized.
        config: Training configuration with keys 'lr', 'weight_decay'.

    Returns:
        Configured optimizer.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 2e-4),
        weight_decay=config.get("weight_decay", 1e-5),
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any],
) -> Optional[_LRScheduler]:
    """Build a learning rate scheduler from the config dictionary.

    Args:
        optimizer: The optimizer to schedule.
        config: Training configuration with key 'scheduler'.

    Returns:
        LR scheduler or None.
    """
    scheduler_name = config.get("scheduler", "cosine")
    epochs = config.get("epochs", 100)
    warmup = config.get("warmup_epochs", 5)

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs - warmup, eta_min=1e-7,
        )
    elif scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1,
        )
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )
    else:
        return None
