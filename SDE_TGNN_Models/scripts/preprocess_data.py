#!/usr/bin/env python3
"""Data preprocessing script for SDE-TGNN.

Preprocesses raw datasets and optionally constructs temporal graphs.

Usage:
    python scripts/preprocess_data.py --data_root data/raw --output_dir data/processed
    python scripts/preprocess_data.py --dataset cic_ids2018 --build_graphs
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from typing import Any, Dict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import DataPreprocessor
from src.data.feature_engineering import GraphConstructor


def parse_args() -> argparse.Namespace:
    """Parse preprocessing arguments.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for SDE-TGNN.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/raw",
        help="Root directory containing raw CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save preprocessed data.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=[
            "microsoft_cloud", "edge_iiot", "kubernetes_docker",
            "cic_ids2018", "cic_iot2023", "unsw_nb15", "all",
        ],
        help="Which dataset to preprocess.",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="standard",
        choices=["standard", "minmax", "robust"],
        help="Normalization method.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Test set fraction.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation set fraction.",
    )
    parser.add_argument(
        "--build_graphs",
        action="store_true",
        help="Build temporal graphs from flow data.",
    )
    parser.add_argument(
        "--graph_k",
        type=int,
        default=10,
        help="k for k-NN graph construction.",
    )
    parser.add_argument(
        "--max_graph_nodes",
        type=int,
        default=10000,
        help="Maximum nodes per graph.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def save_splits(
    splits: Dict[str, np.ndarray],
    output_dir: str,
    dataset_name: str,
) -> None:
    """Save preprocessed data splits to disk as .npz files.

    Args:
        splits: Dictionary with X_train, X_val, X_test, y_train, y_val, y_test.
        output_dir: Output directory.
        dataset_name: Name of the dataset.
    """
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        X = splits[f"X_{split_name}"]
        y = splits[f"y_{split_name}"]
        path = os.path.join(dataset_dir, f"{split_name}.npz")
        np.savez_compressed(path, features=X, labels=y)

    logger = logging.getLogger(__name__)
    logger.info(
        "Saved %s: train=%d, val=%d, test=%d",
        dataset_name,
        len(splits["y_train"]),
        len(splits["y_val"]),
        len(splits["y_test"]),
    )


def build_graphs_for_dataset(
    splits: Dict[str, np.ndarray],
    output_dir: str,
    dataset_name: str,
    graph_k: int = 10,
    max_nodes: int = 10000,
) -> None:
    """Build k-NN temporal graphs for each split and save as pickle.

    Args:
        splits: Preprocessed data splits.
        output_dir: Output directory.
        dataset_name: Dataset identifier.
        graph_k: Number of nearest neighbours.
        max_nodes: Maximum nodes per graph.
    """
    logger = logging.getLogger(__name__)
    graph_constructor = GraphConstructor(k=graph_k, mode="knn", max_nodes=max_nodes)

    graph_dir = os.path.join(output_dir, dataset_name, "graphs")
    os.makedirs(graph_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        X = splits[f"X_{split_name}"]
        y = splits[f"y_{split_name}"]

        # Build graphs in chunks to manage memory
        chunk_size = max_nodes
        graphs = []
        num_chunks = max(1, len(X) // chunk_size)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(X))
            X_chunk = X[start:end]
            y_chunk = y[start:end]

            graph = graph_constructor.build_knn_graph(X_chunk, y_chunk)
            graphs.append(graph)

        # Save graphs
        graph_path = os.path.join(graph_dir, f"{split_name}_graphs.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(graphs, f)

        logger.info(
            "Built %d graphs for %s/%s (%d total nodes)",
            len(graphs), dataset_name, split_name, len(X),
        )


def main() -> None:
    """Main preprocessing entry point."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Starting preprocessing...")
    logger.info("Data root: %s", args.data_root)
    logger.info("Output: %s", args.output_dir)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        data_root=args.data_root,
        normalize_method=args.normalize,
        test_split=args.test_split,
        val_split=args.val_split,
        random_state=args.seed,
    )

    # Preprocess datasets
    start_time = time.time()

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
        logger.error("No datasets successfully preprocessed.")
        sys.exit(1)

    # Save preprocessed splits
    for name, splits in datasets.items():
        save_splits(splits, args.output_dir, name)

    # Build graphs if requested
    if args.build_graphs:
        logger.info("Building temporal graphs...")
        for name, splits in datasets.items():
            build_graphs_for_dataset(
                splits, args.output_dir, name,
                graph_k=args.graph_k,
                max_nodes=args.max_graph_nodes,
            )

    # Save label encoders and scalers
    metadata = {
        "label_encoders": {
            name: {
                "classes": list(le.classes_) if hasattr(le, "classes_") else []
            }
            for name, le in preprocessor.label_encoders.items()
            if hasattr(le, "classes_")
        },
        "datasets": list(datasets.keys()),
        "feature_dim": next(iter(datasets.values()))["X_train"].shape[1],
        "normalize_method": args.normalize,
    }

    import json
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    elapsed = time.time() - start_time
    logger.info("Preprocessing completed in %.1f seconds", elapsed)
    logger.info("Processed %d datasets", len(datasets))

    # Print summary
    for name, splits in datasets.items():
        logger.info(
            "%s: features=%d, classes=%d, train=%d, val=%d, test=%d",
            name,
            splits["X_train"].shape[1],
            len(np.unique(splits["y_train"])),
            len(splits["y_train"]),
            len(splits["y_val"]),
            len(splits["y_test"]),
        )


if __name__ == "__main__":
    main()
