"""Feature engineering utilities for SDE-TGNN.

Provides:
- FeatureHarmonizer: Projects heterogeneous feature sets to a
  unified latent space via a learned linear mapping.
- GraphConstructor: Builds temporal k-NN or IP-based graphs from
  flow-level feature matrices.
- TemporalEncoder: Encodes continuous timestamps into learnable
  positional representations for the SDE integrator.
"""

from __future__ import annotations

import math
import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


# ======================================================================
# Feature harmonization
# ======================================================================

class FeatureHarmonizer(nn.Module):
    """Learnable projection from dataset-specific feature spaces to a
    shared latent representation.

    Each source domain has its own linear layer that maps its native
    feature dimension to the common ``target_dim``.  A shared LayerNorm
    and optional non-linearity are applied afterwards.

    Attributes:
        projections: ``nn.ModuleDict`` of per-domain linear layers.
        norm: Shared ``LayerNorm`` over the target dimension.
        target_dim: Output feature dimension shared across domains.
    """

    def __init__(
        self,
        source_dims: dict[str, int],
        target_dim: int = 256,
        activation: str = "gelu",
    ) -> None:
        """Initialize the feature harmonizer.

        Args:
            source_dims: Mapping of domain name -> native feature dimension.
            target_dim: Shared output dimension.
            activation: Non-linearity applied after projection.
        """
        super().__init__()
        self.target_dim = target_dim

        self.projections = nn.ModuleDict()
        for name, dim in source_dims.items():
            self.projections[name] = nn.Sequential(
                nn.Linear(dim, target_dim),
                nn.LayerNorm(target_dim),
                self._get_activation(activation),
                nn.Linear(target_dim, target_dim),
            )

        self.norm = nn.LayerNorm(target_dim)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Return an activation module by name.

        Args:
            name: Activation function identifier.

        Returns:
            PyTorch activation module.
        """
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.GELU())

    def forward(
        self,
        x: torch.Tensor,
        domain: str,
    ) -> torch.Tensor:
        """Project domain-specific features to the shared space.

        Args:
            x: Feature tensor of shape (N, D_domain).
            domain: Name of the source domain (must match a key in
                ``source_dims``).

        Returns:
            Projected tensor of shape (N, target_dim).
        """
        projected = self.projections[domain](x)
        return self.norm(projected)


# ======================================================================
# Graph construction
# ======================================================================

class GraphConstructor:
    """Build temporal graphs from flow-level feature matrices.

    Two strategies are supported:
    1. **k-NN**: Connect each node to its *k* nearest neighbours in
       feature space (Euclidean distance).
    2. **IP-based**: Connect flows that share source or destination
       IP addresses within a time window.

    Attributes:
        k: Number of nearest neighbours for k-NN graphs.
        mode: Graph construction mode ('knn' or 'ip').
        temporal_window: Maximum time gap (seconds) between connected flows.
        max_nodes: Safety cap on graph size.
    """

    def __init__(
        self,
        k: int = 10,
        mode: str = "knn",
        temporal_window: float = 60.0,
        max_nodes: int = 10000,
    ) -> None:
        """Initialize the graph constructor.

        Args:
            k: Number of nearest neighbours.
            mode: 'knn' or 'ip'.
            temporal_window: Maximum temporal gap in seconds.
            max_nodes: Maximum number of nodes per graph.
        """
        self.k = k
        self.mode = mode
        self.temporal_window = temporal_window
        self.max_nodes = max_nodes

    def build_knn_graph(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Data:
        """Construct a k-NN graph from a feature matrix.

        Args:
            features: Array of shape (N, D).
            labels: Array of shape (N,).
            timestamps: Optional array of shape (N,).

        Returns:
            ``torch_geometric.data.Data`` with node features, edge
            index, edge attributes, and labels.
        """
        n = min(len(features), self.max_nodes)
        if n < len(features):
            indices = np.random.choice(len(features), n, replace=False)
            indices.sort()
            features = features[indices]
            labels = labels[indices]
            if timestamps is not None:
                timestamps = timestamps[indices]

        # Pairwise Euclidean distances
        dists = cdist(features, features, metric="euclidean")

        # k-NN edges (excluding self-loops)
        src_list = []
        dst_list = []
        edge_weights = []

        for i in range(n):
            sorted_indices = np.argsort(dists[i])
            count = 0
            for j in sorted_indices:
                if j == i:
                    continue
                # Optional temporal filtering
                if timestamps is not None:
                    dt = abs(float(timestamps[i]) - float(timestamps[j]))
                    if dt > self.temporal_window:
                        continue
                src_list.append(i)
                dst_list.append(j)
                edge_weights.append(dists[i, j])
                count += 1
                if count >= self.k:
                    break

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        t = (
            torch.tensor(timestamps, dtype=torch.float64)
            if timestamps is not None
            else torch.zeros(n, dtype=torch.float64)
        )

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, timestamps=t)

    def build_ip_graph(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        src_ips: np.ndarray,
        dst_ips: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Data:
        """Construct a graph where flows sharing an IP are connected.

        Args:
            features: Array of shape (N, D).
            labels: Array of shape (N,).
            src_ips: Array of source IP hashes/integers.
            dst_ips: Array of destination IP hashes/integers.
            timestamps: Optional array of shape (N,).

        Returns:
            PyG Data object.
        """
        n = min(len(features), self.max_nodes)
        if n < len(features):
            indices = np.random.choice(len(features), n, replace=False)
            indices.sort()
            features = features[indices]
            labels = labels[indices]
            src_ips = src_ips[indices]
            dst_ips = dst_ips[indices]
            if timestamps is not None:
                timestamps = timestamps[indices]

        # Build adjacency via shared IP addresses
        ip_to_nodes: dict[int, list[int]] = {}
        for i in range(n):
            for ip_val in (int(src_ips[i]), int(dst_ips[i])):
                ip_to_nodes.setdefault(ip_val, []).append(i)

        src_list = []
        dst_list = []
        for nodes in ip_to_nodes.values():
            for a_idx in range(len(nodes)):
                for b_idx in range(a_idx + 1, len(nodes)):
                    a, b = nodes[a_idx], nodes[b_idx]
                    if timestamps is not None:
                        dt = abs(float(timestamps[a]) - float(timestamps[b]))
                        if dt > self.temporal_window:
                            continue
                    src_list.extend([a, b])
                    dst_list.extend([b, a])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long) if src_list else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.ones(edge_index.size(1), 1, dtype=torch.float32)

        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        t = (
            torch.tensor(timestamps, dtype=torch.float64)
            if timestamps is not None
            else torch.zeros(n, dtype=torch.float64)
        )

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, timestamps=t)

    def build_graph(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        src_ips: Optional[np.ndarray] = None,
        dst_ips: Optional[np.ndarray] = None,
    ) -> Data:
        """Dispatch to the appropriate graph construction method.

        Args:
            features: Array of shape (N, D).
            labels: Array of shape (N,).
            timestamps: Optional array of shape (N,).
            src_ips: Optional source IP array (required for 'ip' mode).
            dst_ips: Optional destination IP array (required for 'ip' mode).

        Returns:
            PyG Data object.
        """
        if self.mode == "ip" and src_ips is not None and dst_ips is not None:
            return self.build_ip_graph(features, labels, src_ips, dst_ips, timestamps)
        return self.build_knn_graph(features, labels, timestamps)


# ======================================================================
# Temporal encoding
# ======================================================================

class TemporalEncoder(nn.Module):
    """Sinusoidal + learnable temporal encoding for continuous timestamps.

    Follows the positional encoding formulation from *Attention Is All
    You Need* but operates on continuous time values rather than
    discrete positions.  A learnable linear layer is applied on top of
    the sinusoidal components.

    Attributes:
        dim: Encoding dimension (must be even).
        linear: Learnable affine projection over the sinusoidal basis.
    """

    def __init__(self, dim: int = 64) -> None:
        """Initialize the temporal encoder.

        Args:
            dim: Output encoding dimension (will be rounded to even).
        """
        super().__init__()
        self.dim = dim if dim % 2 == 0 else dim + 1

        # Frequency bands (log-spaced)
        half = self.dim // 2
        freq = torch.exp(torch.arange(0, half, dtype=torch.float32) * -(math.log(10000.0) / half))
        self.register_buffer("freq", freq)

        self.linear = nn.Linear(self.dim, self.dim)
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode continuous timestamps.

        Args:
            t: Tensor of shape (...,) containing timestamp values.

        Returns:
            Encoded tensor of shape (..., dim).
        """
        # Expand last dimension for broadcasting
        t_expanded = t.unsqueeze(-1).float()  # (..., 1)
        angles = t_expanded * self.freq  # (..., half)

        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (..., dim)
        encoding = self.linear(encoding)
        encoding = self.norm(encoding)

        return encoding

    def encode_delta(
        self,
        t_src: torch.Tensor,
        t_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the time difference between two timestamps.

        Useful for encoding temporal edge attributes in graphs.

        Args:
            t_src: Source timestamps of shape (E,).
            t_dst: Destination timestamps of shape (E,).

        Returns:
            Encoded time-delta tensor of shape (E, dim).
        """
        delta = (t_dst - t_src).abs()
        return self.forward(delta)
