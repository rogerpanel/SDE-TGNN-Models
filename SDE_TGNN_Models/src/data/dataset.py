"""PyTorch Dataset and DataLoader classes for SDE-TGNN.

Provides flow-level and temporal-graph-level dataset wrappers
along with a multi-domain DataLoader that samples across all
six intrusion-detection datasets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


# ======================================================================
# Flow-level dataset
# ======================================================================

class SecurityFlowDataset(Dataset):
    """Per-flow network traffic dataset for tabular classification.

    Each sample is an individual network flow represented as a fixed-size
    feature vector paired with an integer label.

    Attributes:
        features: Tensor of shape (N, D) with float32 features.
        labels: Tensor of shape (N,) with int64 class indices.
        timestamps: Optional tensor of shape (N,) with float64 timestamps.
        domain_id: Integer identifier for the source domain/dataset.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        domain_id: int = 0,
    ) -> None:
        """Initialize the flow dataset.

        Args:
            features: Feature matrix of shape (N, D).
            labels: Label vector of shape (N,).
            timestamps: Optional timestamp vector of shape (N,).
            domain_id: Integer identifier for the source domain.
        """
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.timestamps = (
            torch.as_tensor(timestamps, dtype=torch.float64)
            if timestamps is not None
            else torch.zeros(len(labels), dtype=torch.float64)
        )
        self.domain_id = domain_id

    def __len__(self) -> int:
        """Return the number of flows in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a single flow sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys 'features', 'label', 'timestamp',
            and 'domain_id'.
        """
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
            "timestamp": self.timestamps[idx],
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced training.

        Returns:
            Tensor of shape (num_classes,) with per-class weights.
        """
        counts = torch.bincount(self.labels)
        weights = 1.0 / counts.float().clamp(min=1.0)
        weights = weights / weights.sum() * len(counts)
        return weights

    def get_sampler(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler for class-balanced batching.

        Returns:
            PyTorch WeightedRandomSampler instance.
        """
        counts = torch.bincount(self.labels).float()
        class_weight = 1.0 / counts.clamp(min=1.0)
        sample_weights = class_weight[self.labels]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(self.labels),
            replacement=True,
        )


# ======================================================================
# Temporal graph dataset
# ======================================================================

class TemporalGraphDataset(Dataset):
    """Temporal graph dataset for graph-level classification.

    Each sample is a ``torch_geometric.data.Data`` object encoding a
    snapshot of the traffic graph within a temporal window.

    Attributes:
        graphs: List of ``Data`` objects.
        domain_id: Integer identifier for the source domain.
    """

    def __init__(
        self,
        graphs: List[Data],
        domain_id: int = 0,
    ) -> None:
        """Initialize the temporal graph dataset.

        Args:
            graphs: List of PyG Data objects.
            domain_id: Integer identifier for the source domain.
        """
        self.graphs = graphs
        self.domain_id = domain_id

    def __len__(self) -> int:
        """Return the number of graph snapshots."""
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        """Return a single graph snapshot.

        Args:
            idx: Graph index.

        Returns:
            PyG Data object augmented with a ``domain_id`` attribute.
        """
        data = self.graphs[idx]
        data.domain_id = torch.tensor(self.domain_id, dtype=torch.long)
        return data

    @staticmethod
    def collate_fn(batch: List[Data]) -> Batch:
        """Collate a list of Data objects into a Batch.

        Args:
            batch: List of PyG Data objects.

        Returns:
            Batched PyG Data object.
        """
        return Batch.from_data_list(batch)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights from graph labels.

        Returns:
            Tensor of per-class weights.
        """
        all_labels = torch.cat([g.y for g in self.graphs if hasattr(g, "y") and g.y is not None])
        counts = torch.bincount(all_labels)
        weights = 1.0 / counts.float().clamp(min=1.0)
        weights = weights / weights.sum() * len(counts)
        return weights


# ======================================================================
# Domain-cycling sampler
# ======================================================================

class DomainCycleSampler(Sampler[int]):
    """Cycles through domains, drawing equal-sized mini-batches from each.

    This ensures that every training batch contains an equal mix from
    all loaded domains, preventing any single large dataset from
    dominating the gradient signal.

    Args:
        domain_sizes: List of dataset lengths, one per domain.
        batch_size: Desired batch size (total across all domains).
    """

    def __init__(self, domain_sizes: List[int], batch_size: int) -> None:
        self.domain_sizes = domain_sizes
        self.batch_size = batch_size
        self.num_domains = len(domain_sizes)
        # Per-domain batch size (rounded down, remainder goes to last domain)
        self.per_domain = max(1, batch_size // self.num_domains)
        self._total = sum(domain_sizes)

    def __len__(self) -> int:
        """Return total number of samples across all domains."""
        return self._total

    def __iter__(self) -> Iterator[int]:
        """Yield indices interleaved across domains."""
        # Build per-domain shuffled index lists
        offset = 0
        domain_indices: List[List[int]] = []
        for size in self.domain_sizes:
            perm = torch.randperm(size).tolist()
            domain_indices.append([p + offset for p in perm])
            offset += size

        # Yield in round-robin batches
        pointers = [0] * self.num_domains
        while True:
            batch: List[int] = []
            exhausted = 0
            for d in range(self.num_domains):
                start = pointers[d]
                end = min(start + self.per_domain, len(domain_indices[d]))
                if start >= len(domain_indices[d]):
                    exhausted += 1
                    continue
                batch.extend(domain_indices[d][start:end])
                pointers[d] = end
            if exhausted == self.num_domains or len(batch) == 0:
                break
            yield from batch


# ======================================================================
# Multi-domain DataLoader
# ======================================================================

class MultiDomainDataLoader:
    """Unified DataLoader that wraps all six intrusion-detection datasets.

    Provides both individual per-dataset loaders and a combined loader
    that cycles through domains for multi-domain training.

    Attributes:
        datasets: Mapping of domain name -> SecurityFlowDataset.
        batch_size: Global batch size.
        num_workers: Number of DataLoader worker processes.
        pin_memory: Whether to pin host memory for GPU transfers.
    """

    def __init__(
        self,
        datasets: Dict[str, SecurityFlowDataset],
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        """Initialize the multi-domain DataLoader.

        Args:
            datasets: Mapping of domain name -> SecurityFlowDataset.
            batch_size: Global batch size.
            num_workers: Number of DataLoader workers.
            pin_memory: Pin memory for faster GPU transfer.
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_loader(
        self,
        domain: str,
        shuffle: bool = True,
        balanced: bool = False,
    ) -> DataLoader:
        """Return a DataLoader for a single domain.

        Args:
            domain: Dataset domain name.
            shuffle: Whether to shuffle samples each epoch.
            balanced: Use class-balanced sampling.

        Returns:
            PyTorch DataLoader.
        """
        ds = self.datasets[domain]
        sampler = ds.get_sampler() if balanced else None
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def get_combined_loader(self, balanced: bool = True) -> DataLoader:
        """Return a single DataLoader that interleaves all domains.

        Args:
            balanced: Use domain-cycling sampler for balance.

        Returns:
            DataLoader over the concatenated dataset with domain cycling.
        """
        all_datasets = list(self.datasets.values())
        combined = torch.utils.data.ConcatDataset(all_datasets)

        if balanced:
            domain_sizes = [len(ds) for ds in all_datasets]
            sampler: Optional[Sampler] = DomainCycleSampler(domain_sizes, self.batch_size)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            combined,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def get_graph_loader(
        self,
        graph_datasets: Dict[str, TemporalGraphDataset],
        shuffle: bool = True,
    ) -> DataLoader:
        """Return a DataLoader for temporal graph data.

        Args:
            graph_datasets: Mapping of domain name -> TemporalGraphDataset.
            shuffle: Whether to shuffle graphs each epoch.

        Returns:
            DataLoader using the PyG collate function.
        """
        all_graphs = list(graph_datasets.values())
        combined = torch.utils.data.ConcatDataset(all_graphs)
        return DataLoader(
            combined,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=TemporalGraphDataset.collate_fn,
            drop_last=False,
        )
