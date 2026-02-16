#!/usr/bin/env python3
"""Full evaluation script for a trained SDE-TGNN model.

Runs detection metrics, calibration analysis, adversarial robustness
evaluation, and generates publication-quality visualizations.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt \
        --dataset cic_ids2018 --output_dir evaluation_results
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import DataPreprocessor
from src.data.dataset import SecurityFlowDataset, MultiDomainDataLoader
from src.models.sde_tgnn import SDETGNN
from src.evaluation.metrics import compute_detection_metrics, compute_per_class_metrics, compute_confusion_matrix
from src.evaluation.calibration import (
    expected_calibration_error,
    brier_score,
    reliability_diagram,
    temperature_scaling,
)
from src.evaluation.adversarial import evaluate_robustness
from src.evaluation.visualization import (
    plot_reliability_diagram,
    plot_confusion_matrix,
    plot_uncertainty_histogram,
)


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SDE-TGNN model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML (uses checkpoint config if not specified).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=[
            "microsoft_cloud", "edge_iiot", "kubernetes_docker",
            "cic_ids2018", "cic_iot2023", "unsw_nb15", "all",
        ],
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device.",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="Run adversarial robustness evaluation.",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        default=True,
        help="Run calibration analysis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    # Setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    # Load checkpoint
    logger.info("Loading checkpoint: %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    elif "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config_path = os.path.join(os.path.dirname(args.checkpoint), "..", "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    # Preprocess data
    data_config = config.get("data", {})
    preprocessor = DataPreprocessor(
        data_root=args.data_root,
        normalize_method="standard" if data_config.get("normalize", True) else "none",
        test_split=data_config.get("test_split", 0.2),
        val_split=data_config.get("val_split", 0.1),
        random_state=args.seed,
    )

    if args.dataset == "all":
        datasets = preprocessor.preprocess_all()
    else:
        loaders = {
            "microsoft_cloud": preprocessor.preprocess_microsoft_cloud,
            "edge_iiot": preprocessor.preprocess_edge_iiot,
            "kubernetes_docker": preprocessor.preprocess_kubernetes_docker,
            "cic_ids2018": preprocessor.preprocess_cic_ids2018,
            "cic_iot2023": preprocessor.preprocess_cic_iot2023,
            "unsw_nb15": preprocessor.preprocess_unsw_nb15,
        }
        data = loaders[args.dataset]()
        datasets = {args.dataset: data}
        datasets = preprocessor.harmonize_features(datasets)

    if not datasets:
        logger.error("No datasets loaded.")
        sys.exit(1)

    # Determine dimensions
    first_dataset = next(iter(datasets.values()))
    input_dim = first_dataset["X_test"].shape[1]
    all_labels = np.concatenate([d["y_test"] for d in datasets.values()])
    num_classes = len(np.unique(all_labels))

    # Build model
    model_config = config.get("model", {})
    sde_config = config.get("sde", {})
    fp_config = config.get("fokker_planck", {})

    model = SDETGNN(
        input_dim=input_dim,
        hidden_dim=model_config.get("hidden_dim", 256),
        state_dim=model_config.get("state_dim", 64),
        num_classes=num_classes,
        num_layers=model_config.get("num_layers", 4),
        num_heads=model_config.get("num_heads", 8),
        num_scales=model_config.get("num_scales", 3),
        dropout=model_config.get("dropout", 0.1),
        sde_config={
            "solver": sde_config.get("solver", "euler_maruyama"),
            "dt": sde_config.get("dt", 0.01),
            "noise_type": sde_config.get("noise_type", "diagonal"),
            "adjoint": sde_config.get("adjoint", False),
            "integration_steps": sde_config.get("integration_steps", 20),
            "drift_layers": model_config.get("drift_layers", 3),
            "diffusion_layers": model_config.get("diffusion_layers", 2),
        },
        fokker_planck_config={
            "moment_order": fp_config.get("moment_order", 2),
            "gaussian_approx": fp_config.get("gaussian_approx", True),
            "propagation_steps": fp_config.get("propagation_steps", 10),
            "regularization": fp_config.get("regularization", 1e-6),
        },
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")

    # ---- Evaluation loop ----
    all_results: Dict[str, Any] = {}

    for name, splits in datasets.items():
        logger.info("=" * 60)
        logger.info("Evaluating on %s", name)
        logger.info("=" * 60)

        # Create test DataLoader
        test_ds = SecurityFlowDataset(splits["X_test"], splits["y_test"])
        batch_size = config.get("training", {}).get("batch_size", 256)
        test_loader = MultiDomainDataLoader(
            {name: test_ds}, batch_size=batch_size, num_workers=0,
        ).get_loader(name, shuffle=False)

        # Collect predictions
        all_preds = []
        all_probs = []
        all_labels_list = []
        all_uncertainties = []

        start_time = time.time()
        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(device)
                labels = batch["label"]

                logits, unc_dict = model(features, return_uncertainty=True)
                probs = torch.softmax(logits, dim=-1)

                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels_list.append(labels.numpy())

                if "predictive_entropy" in unc_dict:
                    all_uncertainties.append(unc_dict["predictive_entropy"].cpu().numpy())

        inference_time = time.time() - start_time

        y_pred = np.concatenate(all_preds)
        y_proba = np.concatenate(all_probs)
        y_true = np.concatenate(all_labels_list)
        uncertainties = np.concatenate(all_uncertainties) if all_uncertainties else None

        # 1. Detection metrics
        detection = compute_detection_metrics(y_true, y_pred, y_proba)
        logger.info("Detection metrics: %s", json.dumps(detection, indent=2))

        # 2. Per-class metrics
        label_encoder = preprocessor.label_encoders.get(name)
        class_names = list(label_encoder.classes_) if label_encoder else None
        per_class = compute_per_class_metrics(y_true, y_pred, class_names)

        # 3. Confusion matrix
        cm = compute_confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(
            cm, class_names=class_names,
            title=f"Confusion Matrix - {name}",
            save_path=os.path.join(args.output_dir, "figures", f"cm_{name}.png"),
        )

        # 4. Calibration
        calibration_results = {}
        if args.calibration:
            confidences = y_proba.max(axis=1)
            correct = (y_pred == y_true).astype(float)

            ece_result = expected_calibration_error(confidences, correct)
            bs = brier_score(y_true, y_proba)
            rel_diag = reliability_diagram(confidences, correct)

            calibration_results = {
                "ece": ece_result["ece"],
                "mce": ece_result["mce"],
                "brier_score": bs,
            }
            logger.info("Calibration: ECE=%.4f, MCE=%.4f, Brier=%.4f", ece_result["ece"], ece_result["mce"], bs)

            # Plot reliability diagram
            plot_reliability_diagram(
                bin_confidences=np.array(rel_diag["bin_confidences"]),
                bin_accuracies=np.array(rel_diag["bin_accuracies"]),
                bin_counts=np.array(rel_diag["bin_counts"]),
                ece=rel_diag["ece"],
                title=f"Reliability Diagram - {name}",
                save_path=os.path.join(args.output_dir, "figures", f"reliability_{name}.png"),
            )

        # 5. Uncertainty visualization
        if uncertainties is not None:
            correct_mask = y_pred == y_true
            plot_uncertainty_histogram(
                uncertainties, correct_mask,
                title=f"Uncertainty Distribution - {name}",
                save_path=os.path.join(args.output_dir, "figures", f"uncertainty_{name}.png"),
            )

        # 6. Adversarial robustness
        adversarial_results = {}
        if args.adversarial:
            logger.info("Running adversarial evaluation...")
            eval_config = config.get("evaluation", {})
            adversarial_results = evaluate_robustness(
                model=model,
                test_loader=test_loader,
                device=device,
                epsilon_values=[0.001, 0.005, 0.01, 0.02, 0.05],
                pgd_steps=eval_config.get("pgd_steps", 20),
                pgd_alpha=eval_config.get("pgd_alpha", 0.001),
                num_cert_samples=eval_config.get("mc_samples", 50),
                noise_std=eval_config.get("adversarial_eps", 0.01),
            )

        # 7. Latency
        latency_per_sample = inference_time / max(len(y_true), 1) * 1000  # ms
        throughput = len(y_true) / max(inference_time, 1e-6)

        # Compile results
        dataset_results = {
            "detection": detection,
            "per_class": per_class,
            "calibration": calibration_results,
            "adversarial": adversarial_results,
            "latency_ms": latency_per_sample,
            "throughput_samples_per_sec": throughput,
            "num_test_samples": len(y_true),
        }
        all_results[name] = dataset_results

        # Save per-dataset results
        results_path = os.path.join(args.output_dir, f"results_{name}.json")
        with open(results_path, "w") as f:
            json.dump(dataset_results, f, indent=2, default=str)
        logger.info("Results saved to %s", results_path)

    # Save aggregate results
    aggregate_path = os.path.join(args.output_dir, "aggregate_results.json")
    with open(aggregate_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("Evaluation complete. Results saved to %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
