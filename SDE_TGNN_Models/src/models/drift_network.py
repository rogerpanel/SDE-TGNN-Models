"""Deterministic drift network for the SDE-TGNN model.

The drift network computes f_theta(h, G, t), the deterministic component
of the stochastic differential equation governing the evolution of
hidden states over continuous time.  It combines graph-aware message
passing with an MLP conditioned on time.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GraphDriftMessagePassing(MessagePassing):
    """Graph-aware message passing layer for drift computation.

    Aggregates neighbour information using a learned linear
    transformation and degree-normalized summation.

    Attributes:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the message passing layer.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
        """
        super().__init__(aggr="mean", node_dim=0)
        self.lin_src = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_edge = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform message passing.

        Args:
            x: Node features of shape (N, in_channels).
            edge_index: Edge indices of shape (2, E).
            edge_attr: Optional edge features of shape (E, in_channels).

        Returns:
            Aggregated node features of shape (N, out_channels).
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Degree normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype).clamp(min=1.0)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Source transformation
        x_src = self.lin_src(x)

        out = self.propagate(edge_index, x=x_src, norm=norm)

        # Destination transformation and bias
        out = out + self.lin_dst(x) + self.bias

        return out

    def message(self, x_j: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Compute normalized messages.

        Args:
            x_j: Source node features of shape (E, out_channels).
            norm: Normalization coefficients of shape (E,).

        Returns:
            Normalized messages of shape (E, out_channels).
        """
        return norm.unsqueeze(-1) * x_j


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding with learnable projection.

    Maps a scalar time value to a vector representation using
    sinusoidal basis functions followed by a linear layer.

    Attributes:
        dim: Output embedding dimension.
    """

    def __init__(self, dim: int) -> None:
        """Initialize the time embedding.

        Args:
            dim: Output embedding dimension (must be even).
        """
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(torch.arange(half, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / half)),
        )
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a scalar or batch of time values.

        Args:
            t: Time tensor of shape (B,) or scalar.

        Returns:
            Embedding tensor of shape (B, dim) or (dim,).
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.float().unsqueeze(-1)  # (B, 1)
        angles = t * self.freqs  # (B, half)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, dim)
        return self.proj(emb)


class DriftNetwork(nn.Module):
    """Deterministic drift f_theta(h, G, t) for graph-structured data.

    The drift combines:
    1. Graph-aware message passing to capture structural information.
    2. Time conditioning via sinusoidal embeddings.
    3. A multi-layer MLP to produce the drift vector.

    The drift satisfies a global Lipschitz condition (via spectral
    normalization) to ensure well-posedness of the SDE.

    Attributes:
        state_dim: Dimension of the SDE state.
        hidden_dim: Hidden dimension for internal computations.
        num_layers: Number of MLP layers.
        graph_layers: List of graph message passing layers.
        time_embed: Time embedding module.
        mlp: Multi-layer perceptron for drift computation.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        num_graph_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the drift network.

        Args:
            state_dim: SDE state dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of MLP layers after graph aggregation.
            num_graph_layers: Number of graph message passing layers.
            dropout: Dropout probability.
        """
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = TimeEmbedding(state_dim)

        # Graph message passing layers
        self.graph_layers = nn.ModuleList()
        self.graph_norms = nn.ModuleList()
        for i in range(num_graph_layers):
            in_dim = state_dim if i == 0 else hidden_dim
            self.graph_layers.append(GraphDriftMessagePassing(in_dim, hidden_dim))
            self.graph_norms.append(nn.LayerNorm(hidden_dim))

        # MLP for drift computation
        mlp_layers = []
        in_dim = hidden_dim + state_dim  # graph output + time embedding
        for layer_idx in range(num_layers):
            out_dim = hidden_dim if layer_idx < num_layers - 1 else state_dim
            mlp_layers.append(nn.utils.spectral_norm(nn.Linear(in_dim, out_dim)))
            if layer_idx < num_layers - 1:
                mlp_layers.append(nn.LayerNorm(out_dim))
                mlp_layers.append(nn.GELU())
                mlp_layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*mlp_layers)

        # Final activation (tanh to bound the drift)
        self.output_activation = nn.Tanh()

    def forward(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the drift vector f_theta(h, G, t).

        Args:
            h: Current state tensor of shape (N, state_dim).
            t: Current time scalar or tensor.
            edge_index: Graph edge indices of shape (2, E).
            edge_attr: Optional edge attributes of shape (E, D_edge).

        Returns:
            Drift vector of shape (N, state_dim).
        """
        # Time conditioning
        if t.dim() == 0:
            t_emb = self.time_embed(t).expand(h.size(0), -1)  # (N, state_dim)
        else:
            t_emb = self.time_embed(t)  # (N, state_dim)

        # Graph message passing
        if edge_index is not None and edge_index.numel() > 0:
            g = h
            for graph_layer, norm in zip(self.graph_layers, self.graph_norms):
                g_new = graph_layer(g, edge_index, edge_attr)
                g_new = norm(g_new)
                g_new = F.gelu(g_new)
                # Residual connection if dimensions match
                if g.size(-1) == g_new.size(-1):
                    g = g + g_new
                else:
                    g = g_new
        else:
            # No graph structure: apply a simple linear transform
            g = h
            for graph_layer, norm in zip(self.graph_layers, self.graph_norms):
                # Create self-loop only edges
                n = h.size(0)
                self_loops = torch.stack([torch.arange(n, device=h.device)] * 2)
                g_new = graph_layer(g, self_loops)
                g_new = norm(g_new)
                g_new = F.gelu(g_new)
                if g.size(-1) == g_new.size(-1):
                    g = g + g_new
                else:
                    g = g_new

        # Concatenate graph features and time embedding
        combined = torch.cat([g, t_emb], dim=-1)  # (N, hidden_dim + state_dim)

        # MLP
        drift = self.mlp(combined)
        drift = self.output_activation(drift)

        return drift
