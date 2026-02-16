#!/usr/bin/env python3
"""Main training script for SDE-TGNN.

Usage:
    python scripts/train.py --dataset cic_ids2018 --config config/default_config.yaml
    python scripts/train.py --dataset all --output_dir outputs/multi_domain
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import DataPreprocessor
from src.data.dataset import SecurityFlowDataset, MultiDomainDataLoader
from src.models.sde_tgnn import SDETGNN
from src.training.trainer import SDETGNNTrainer, build_optimizer, build_scheduler
from src.training.losses import CombinedLoss
from src.evaluation.metrics import compute_detection_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the SDE-TGNN model for network intrusion detection.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=[
            "microsoft_cloud", "edge_iiot", "kubernetes_docker",
            "cic_ids2018", "cic_iot2023", "unsw_nb15", "all",
        ],
        help="Dataset to train on (default: all).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory containing raw datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints, logs, and results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', or 'cuda:0'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str) -> torch.device:
    """Resolve device string to a torch.device.

    Args:
        device_str: 'auto', 'cpu', or 'cuda:N'.

    Returns:
        Resolved torch.device.
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_dataset(
    dataset_name: str,
    preprocessor: DataPreprocessor,
) -> Dict[str, np.ndarray]:
    """Load and preprocess a single dataset.

    Args:
        dataset_name: Name of the dataset.
        preprocessor: DataPreprocessor instance.

    Returns:
        Dictionary with train/val/test splits.
    """
    loaders = {
        "microsoft_cloud": preprocessor.preprocess_microsoft_cloud,
        "edge_iiot": preprocessor.preprocess_edge_iiot,
        "kubernetes_docker": preprocessor.preprocess_kubernetes_docker,
        "cic_ids2018": preprocessor.preprocess_cic_ids2018,
        "cic_iot2023": preprocessor.preprocess_cic_iot2023,
        "unsw_nb15": preprocessor.preprocess_unsw_nb15,
    }
    return loaders[dataset_name]()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, "train.log")),
        ],
    )
    logger = logging.getLogger(__name__)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    config = load_config(args.config)
    logger.info("Configuration loaded from %s", args.config)

    # Set seed
    set_seed(args.seed)
    logger.info("Random seed set to %d", args.seed)

    # Device
    device = get_device(args.device)
    logger.info("Using device: %s", device)

    # Save config to output
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Preprocessing
    preprocessor = DataPreprocessor(
        data_root=args.data_root,
        normalize_method="standard" if config["data"]["normalize"] else "none",
        test_split=config["data"]["test_split"],
        val_split=config["data"]["val_split"],
        random_state=args.seed,
    )

    # Load datasets
    if args.dataset == "all":
        datasets = preprocessor.preprocess_all()
    else:
        data = load_dataset(args.dataset, preprocessor)
        datasets = {args.dataset: data}
        datasets = preprocessor.harmonize_features(datasets)

    if not datasets:
        logger.error("No datasets loaded. Check data_root: %s", args.data_root)
        sys.exit(1)

    logger.info("Loaded %d dataset(s): %s", len(datasets), list(datasets.keys()))

    # Determine dimensions
    first_dataset = next(iter(datasets.values()))
    input_dim = first_dataset["X_train"].shape[1]

    # Count unique classes across all datasets
    all_labels = np.concatenate([d["y_train"] for d in datasets.values()])
    num_classes = len(np.unique(all_labels))
    logger.info("Input dim: %d, Num classes: %d", input_dim, num_classes)

    # Build PyTorch datasets
    train_datasets = {}
    val_datasets = {}
    for domain_id, (name, splits) in enumerate(datasets.items()):
        train_datasets[name] = SecurityFlowDataset(
            splits["X_train"], splits["y_train"], domain_id=domain_id,
        )
        val_datasets[name] = SecurityFlowDataset(
            splits["X_val"], splits["y_val"], domain_id=domain_id,
        )

    # Create DataLoaders
    batch_size = config["training"]["batch_size"]
    multi_loader = MultiDomainDataLoader(
        train_datasets,
        batch_size=batch_size,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
    )
    train_loader = multi_loader.get_combined_loader(balanced=True)

    val_multi_loader = MultiDomainDataLoader(
        val_datasets,
        batch_size=batch_size,
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
    )
    val_loader = val_multi_loader.get_combined_loader(balanced=False)

    # Build model
    model_config = config["model"]
    sde_config = config["sde"]
    fp_config = config["fokker_planck"]

    model = SDETGNN(
        input_dim=input_dim,
        hidden_dim=model_config["hidden_dim"],
        state_dim=model_config["state_dim"],
        num_classes=num_classes,
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        num_scales=model_config["num_scales"],
        dropout=model_config["dropout"],
        sde_config={
            "solver": sde_config["solver"],
            "dt": sde_config["dt"],
            "noise_type": sde_config["noise_type"],
            "adjoint": sde_config["adjoint"],
            "integration_steps": sde_config["integration_steps"],
            "drift_layers": model_config["drift_layers"],
            "diffusion_layers": model_config["diffusion_layers"],
        },
        fokker_planck_config={
            "moment_order": fp_config["moment_order"],
            "gaussian_approx": fp_config["gaussian_approx"],
            "propagation_steps": fp_config.get("propagation_steps", 10),
            "regularization": fp_config.get("regularization", 1e-6),
        },
    )

    # Log parameter counts
    param_counts = model.get_num_parameters()
    logger.info("Model parameters: %s", json.dumps(param_counts, indent=2))

    # Loss function
    loss_weights = config["training"].get("loss_weights", {})
    loss_fn = CombinedLoss(
        classification_weight=loss_weights.get("classification", 1.0),
        elbo_weight=loss_weights.get("elbo", 0.1),
        calibration_weight=loss_weights.get("calibration", 0.05),
        kl_weight=loss_weights.get("kl_divergence", 0.01),
        label_smoothing=config["training"].get("label_smoothing", 0.05),
        num_calibration_bins=config["evaluation"].get("num_calibration_bins", 15),
    )

    # Optimizer and scheduler
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"])

    # Trainer
    trainer = SDETGNNTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config={**config["training"], **config.get("logging", {})},
        device=device,
        output_dir=args.output_dir,
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info("Resumed from %s", args.resume)

    # Train
    start_time = time.time()
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["training"]["epochs"],
    )
    total_time = time.time() - start_time
    logger.info("Training completed in %.1f seconds", total_time)

    # Quick evaluation on test set
    logger.info("Running test evaluation...")
    model.eval()

    for name, splits in datasets.items():
        test_ds = SecurityFlowDataset(splits["X_test"], splits["y_test"])
        test_loader = MultiDomainDataLoader(
            {name: test_ds}, batch_size=batch_size,
        ).get_loader(name, shuffle=False)

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(device)
                labels = batch["label"]

                logits, _ = model(features, return_uncertainty=False)
                probs = torch.softmax(logits, dim=-1)

                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy())

        y_pred = np.concatenate(all_preds)
        y_proba = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels)

        metrics = compute_detection_metrics(y_true, y_pred, y_proba)
        logger.info("Test results for %s: %s", name, json.dumps(metrics, indent=2))

        # Save per-dataset results
        results_path = os.path.join(args.output_dir, f"test_results_{name}.json")
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # Save overall training results
    summary = {
        "total_training_time": total_time,
        "best_epoch": results.get("best", {}).get("epoch", -1),
        "best_val_loss": results.get("best", {}).get("val", {}).get("total", -1),
        "best_val_accuracy": results.get("best", {}).get("val", {}).get("accuracy", -1),
        "num_parameters": param_counts,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("All results saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
