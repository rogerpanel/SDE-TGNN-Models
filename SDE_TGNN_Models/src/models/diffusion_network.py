"""Learned diffusion network for the SDE-TGNN model.

The diffusion network computes sigma_phi(h, t), the stochastic
component of the SDE.  It outputs a positive semi-definite diagonal
diffusion matrix, ensuring the noise has valid statistical properties.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionNetwork(nn.Module):
    """Learned diffusion sigma_phi(h, t) for state-dependent noise.

    The network produces diagonal diffusion coefficients that are
    guaranteed positive via a softplus activation.  The diffusion
    can depend on both the current state and time, enabling the
    model to express heteroscedastic uncertainty.

    Architecture:
        [h; t_emb] -> MLP -> softplus -> sigma (diagonal)

    Attributes:
        state_dim: SDE state dimension.
        hidden_dim: MLP hidden dimension.
        num_layers: Number of MLP layers.
        min_sigma: Minimum diffusion coefficient (numerical stability).
        max_sigma: Maximum diffusion coefficient (prevents divergence).
        noise_type: Type of noise ('diagonal', 'scalar', 'general').
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        min_sigma: float = 1e-4,
        max_sigma: float = 5.0,
        noise_type: str = "diagonal",
        dropout: float = 0.1,
    ) -> None:
        """Initialize the diffusion network.

        Args:
            state_dim: SDE state dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of MLP layers.
            min_sigma: Lower clamp for diffusion coefficients.
            max_sigma: Upper clamp for diffusion coefficients.
            noise_type: 'diagonal', 'scalar', or 'general'.
            dropout: Dropout probability.
        """
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.noise_type = noise_type

        # Time embedding (matches drift network)
        half = state_dim // 2
        self.register_buffer(
            "freqs",
            torch.exp(
                torch.arange(half, dtype=torch.float32)
                * -(math.log(10000.0) / max(half, 1))
            ),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
        )

        # Determine output dimension
        if noise_type == "scalar":
            out_dim = 1
        elif noise_type == "general":
            out_dim = state_dim * state_dim
        else:  # diagonal
            out_dim = state_dim

        # MLP
        layers = []
        in_dim = state_dim * 2  # state + time embedding
        for i in range(num_layers):
            is_last = i == num_layers - 1
            layer_out = out_dim if is_last else hidden_dim
            layers.append(nn.Linear(in_dim, layer_out))
            if not is_last:
                layers.append(nn.LayerNorm(layer_out))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            in_dim = layer_out

        self.mlp = nn.Sequential(*layers)

        # Learnable scale and bias for the output
        self.output_scale = nn.Parameter(torch.ones(out_dim) * 0.1)
        self.output_bias = nn.Parameter(torch.zeros(out_dim))

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal time embedding.

        Args:
            t: Time tensor of shape () or (B,).

        Returns:
            Embedding of shape (B, state_dim) or (state_dim,).
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.float().unsqueeze(-1)  # (B, 1)
        angles = t * self.freqs  # (B, half)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        # Pad or truncate to state_dim
        if emb.size(-1) < self.state_dim:
            pad = torch.zeros(
                *emb.shape[:-1], self.state_dim - emb.size(-1),
                device=emb.device, dtype=emb.dtype,
            )
            emb = torch.cat([emb, pad], dim=-1)
        elif emb.size(-1) > self.state_dim:
            emb = emb[..., :self.state_dim]

        return self.time_proj(emb)

    def forward(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute diffusion coefficients sigma_phi(h, t).

        Args:
            h: Current state tensor of shape (N, state_dim).
            t: Current time scalar or tensor.

        Returns:
            Diffusion tensor. For diagonal noise: (N, state_dim).
            For scalar noise: (N, 1). For general: (N, state_dim, state_dim).
        """
        # Time embedding
        t_emb = self._time_embedding(t)
        if t_emb.dim() == 2 and t_emb.size(0) == 1:
            t_emb = t_emb.expand(h.size(0), -1)
        elif t_emb.dim() == 1:
            t_emb = t_emb.unsqueeze(0).expand(h.size(0), -1)

        # Concatenate state and time
        combined = torch.cat([h, t_emb], dim=-1)  # (N, 2 * state_dim)

        # MLP forward
        raw = self.mlp(combined)

        # Apply scale and bias
        raw = raw * self.output_scale + self.output_bias

        # Softplus for positivity + clamping for stability
        sigma = F.softplus(raw)
        sigma = sigma.clamp(min=self.min_sigma, max=self.max_sigma)

        if self.noise_type == "scalar":
            # Broadcast to state_dim
            sigma = sigma.expand(-1, self.state_dim)
        elif self.noise_type == "general":
            # Reshape to matrix and ensure positive semi-definiteness via LL^T
            L = sigma.view(-1, self.state_dim, self.state_dim)
            sigma = torch.bmm(L, L.transpose(1, 2))
            # Extract diagonal for the SDE solver (which expects diagonal noise)
            sigma = torch.diagonal(sigma, dim1=1, dim2=2)
            sigma = sigma.clamp(min=self.min_sigma, max=self.max_sigma)

        return sigma

    def get_diffusion_matrix(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Return the full diffusion matrix (for Fokker-Planck equations).

        For diagonal noise, this returns a batch of diagonal matrices.

        Args:
            h: State tensor of shape (N, state_dim).
            t: Time scalar or tensor.

        Returns:
            Diffusion matrix of shape (N, state_dim, state_dim).
        """
        sigma = self.forward(h, t)  # (N, state_dim)
        # Construct diagonal matrix
        return torch.diag_embed(sigma)  # (N, state_dim, state_dim)
