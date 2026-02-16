"""Multi-head temporal graph attention layers for SDE-TGNN.

Implements graph attention with continuous-time positional encodings,
allowing the model to capture both structural and temporal patterns
in network traffic graphs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class TemporalGraphAttention(MessagePassing):
    """Multi-head graph attention with temporal encoding.

    Computes attention weights:
        e_ij = LeakyReLU(a^T [Wh_i || Wh_j || te(t_i - t_j)])
        alpha_ij = softmax_j(e_ij)
        h'_i = || _{k=1}^{K} sigma(sum_j alpha_{ij}^k W^k h_j)

    where ``te`` is a continuous temporal encoding of the time
    difference between nodes *i* and *j*.

    Attributes:
        in_channels: Input feature dimension.
        out_channels: Per-head output dimension.
        heads: Number of attention heads.
        dropout: Dropout probability on attention coefficients.
        negative_slope: LeakyReLU negative slope.
        concat: If True, concatenate head outputs; otherwise average.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
        concat: bool = True,
        time_dim: int = 64,
        edge_dim: int = 1,
    ) -> None:
        """Initialize the temporal graph attention layer.

        Args:
            in_channels: Input node feature dimension.
            out_channels: Per-head output dimension.
            heads: Number of attention heads.
            dropout: Dropout probability.
            negative_slope: LeakyReLU slope for attention scores.
            concat: Concatenate heads (True) or average (False).
            time_dim: Temporal encoding dimension.
            edge_dim: Edge attribute dimension.
        """
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.concat = concat
        self.time_dim = time_dim

        # Linear projections for queries, keys, and values
        self.W_q = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention scoring vector
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_channels + time_dim))

        # Temporal encoding projection
        self.time_proj = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Edge attribute projection
        self.edge_proj = nn.Linear(edge_dim, heads * out_channels) if edge_dim > 0 else None

        # Output
        if concat:
            self.out_proj = nn.Linear(heads * out_channels, heads * out_channels)
        else:
            self.out_proj = nn.Linear(out_channels, out_channels)

        self.norm = nn.LayerNorm(heads * out_channels if concat else out_channels)

        # Bias
        self.bias = nn.Parameter(torch.empty(heads * out_channels if concat else out_channels))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters with Xavier uniform and zeros."""
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the temporal graph attention layer.

        Args:
            x: Node features of shape (N, in_channels).
            edge_index: Edge indices of shape (2, E).
            edge_attr: Optional edge attributes of shape (E, edge_dim).
            timestamps: Optional node timestamps of shape (N,).

        Returns:
            Updated node features of shape (N, heads * out_channels)
            if ``concat=True``, otherwise (N, out_channels).
        """
        # Compute queries, keys, values
        q = self.W_q(x).view(-1, self.heads, self.out_channels)  # (N, H, C)
        k = self.W_k(x).view(-1, self.heads, self.out_channels)  # (N, H, C)
        v = self.W_v(x).view(-1, self.heads, self.out_channels)  # (N, H, C)

        # Propagate messages
        out = self.propagate(
            edge_index, q=q, k=k, v=v,
            edge_attr=edge_attr, timestamps=timestamps,
            size=None,
        )

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        out = self.out_proj(out)
        out = self.norm(out)

        return out

    def message(
        self,
        q_i: torch.Tensor,
        k_j: torch.Tensor,
        v_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        timestamps_i: Optional[torch.Tensor],
        timestamps_j: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int],
    ) -> torch.Tensor:
        """Compute attention-weighted messages.

        Args:
            q_i: Query features for target nodes, shape (E, H, C).
            k_j: Key features for source nodes, shape (E, H, C).
            v_j: Value features for source nodes, shape (E, H, C).
            edge_attr: Edge attributes of shape (E, edge_dim) or None.
            timestamps_i: Target node timestamps of shape (E,) or None.
            timestamps_j: Source node timestamps of shape (E,) or None.
            index: Target node indices for softmax grouping.
            ptr: CSR pointer for softmax (optional).
            size_i: Number of target nodes.

        Returns:
            Weighted messages of shape (E, H, C).
        """
        # Temporal encoding
        if timestamps_i is not None and timestamps_j is not None:
            dt = (timestamps_i - timestamps_j).float().unsqueeze(-1)  # (E, 1)
            time_enc = self.time_proj(dt)  # (E, time_dim)
            time_enc = time_enc.unsqueeze(1).expand(-1, self.heads, -1)  # (E, H, time_dim)
        else:
            time_enc = torch.zeros(
                q_i.size(0), self.heads, self.time_dim,
                device=q_i.device, dtype=q_i.dtype,
            )

        # Concatenate query, key, and time encoding for attention
        alpha = torch.cat([q_i, k_j, time_enc], dim=-1)  # (E, H, 2C + time_dim)
        alpha = (alpha * self.att).sum(dim=-1)  # (E, H)
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Scaled attention
        alpha = alpha / math.sqrt(self.out_channels)

        # Softmax over neighbourhood
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted values
        msg = v_j * alpha.unsqueeze(-1)  # (E, H, C)

        # Add edge attribute contribution
        if edge_attr is not None and self.edge_proj is not None:
            edge_emb = self.edge_proj(edge_attr).view(-1, self.heads, self.out_channels)
            msg = msg + edge_emb

        return msg


class TemporalGATBlock(nn.Module):
    """Temporal graph attention block with residual connection and feed-forward.

    Wraps a ``TemporalGraphAttention`` layer with a residual connection,
    feed-forward network, dropout, and layer normalization.

    Attributes:
        attention: The core attention layer.
        ffn: Two-layer feed-forward network.
        norm1: Pre-attention LayerNorm.
        norm2: Pre-FFN LayerNorm.
        dropout: Dropout module.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        time_dim: int = 64,
        ffn_ratio: float = 4.0,
        edge_dim: int = 1,
    ) -> None:
        """Initialize the GAT block.

        Args:
            channels: Total feature dimension (must be divisible by heads).
            heads: Number of attention heads.
            dropout: Dropout probability.
            time_dim: Temporal encoding dimension.
            ffn_ratio: Feed-forward hidden dimension multiplier.
            edge_dim: Edge attribute dimension.
        """
        super().__init__()
        assert channels % heads == 0, "channels must be divisible by heads"

        per_head = channels // heads
        self.attention = TemporalGraphAttention(
            in_channels=channels,
            out_channels=per_head,
            heads=heads,
            dropout=dropout,
            concat=True,
            time_dim=time_dim,
            edge_dim=edge_dim,
        )

        ffn_hidden = int(channels * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, channels),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

        # Linear projection for input if needed
        self.input_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the GAT block.

        Args:
            x: Node features of shape (N, channels).
            edge_index: Edge indices of shape (2, E).
            edge_attr: Optional edge attributes of shape (E, edge_dim).
            timestamps: Optional node timestamps of shape (N,).

        Returns:
            Updated node features of shape (N, channels).
        """
        # Pre-norm attention with residual
        residual = x
        x_normed = self.norm1(x)
        attn_out = self.attention(x_normed, edge_index, edge_attr, timestamps)
        x = residual + self.dropout(attn_out)

        # Pre-norm FFN with residual
        residual = x
        x_normed = self.norm2(x)
        ffn_out = self.ffn(x_normed)
        x = residual + ffn_out

        return x
