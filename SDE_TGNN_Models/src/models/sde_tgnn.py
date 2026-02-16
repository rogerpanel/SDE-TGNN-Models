"""Main SDE-TGNN model architecture.

Stochastic Differential Equation Temporal Graph Neural Network for
multi-domain network intrusion detection.  The model integrates:

1. Feature embedding via a learnable projection layer.
2. Multi-head graph attention for structural message passing.
3. Continuous-time SDE integration (drift + diffusion).
4. Fokker-Planck moment propagation for analytical uncertainty.
5. Multi-scale temporal fusion for capturing patterns at different
   time granularities.
6. Classification head with calibrated uncertainty estimates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.drift_network import DriftNetwork
from src.models.diffusion_network import DiffusionNetwork
from src.models.graph_attention import TemporalGATBlock
from src.models.fokker_planck import FokkerPlanckSolver
from src.models.sde_solver import get_sde_solver

logger = logging.getLogger(__name__)


class MultiScaleTemporalFusion(nn.Module):
    """Multi-scale temporal fusion module.

    Aggregates SDE-integrated features at multiple temporal
    resolutions and fuses them via attention-weighted summation.

    Attributes:
        state_dim: Dimension of SDE state vectors.
        num_scales: Number of temporal scales.
        scale_attention: Learnable attention weights over scales.
        scale_projections: Per-scale linear projections.
    """

    def __init__(self, state_dim: int, num_scales: int = 3) -> None:
        """Initialize multi-scale temporal fusion.

        Args:
            state_dim: State dimension.
            num_scales: Number of scales to fuse.
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_scales = num_scales

        # Per-scale projection heads
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, state_dim),
                nn.LayerNorm(state_dim),
                nn.GELU(),
            )
            for _ in range(num_scales)
        ])

        # Attention over scales
        self.scale_attention = nn.Sequential(
            nn.Linear(state_dim * num_scales, num_scales),
            nn.Softmax(dim=-1),
        )

        self.output_norm = nn.LayerNorm(state_dim)

    def forward(self, scale_features: list[torch.Tensor]) -> torch.Tensor:
        """Fuse features from multiple temporal scales.

        Args:
            scale_features: List of tensors, each of shape (N, state_dim),
                one per temporal scale.

        Returns:
            Fused feature tensor of shape (N, state_dim).
        """
        # Project each scale
        projected = []
        for i, (feat, proj) in enumerate(zip(scale_features, self.scale_projections)):
            projected.append(proj(feat))

        # Compute attention weights
        concatenated = torch.cat(projected, dim=-1)  # (N, num_scales * state_dim)
        attention_weights = self.scale_attention(concatenated)  # (N, num_scales)

        # Weighted sum
        stacked = torch.stack(projected, dim=1)  # (N, num_scales, state_dim)
        weights = attention_weights.unsqueeze(-1)  # (N, num_scales, 1)
        fused = (stacked * weights).sum(dim=1)  # (N, state_dim)

        return self.output_norm(fused)


class UncertaintyHead(nn.Module):
    """Classification head with uncertainty estimation.

    Produces both logits and uncertainty metrics (aleatoric and epistemic).

    Attributes:
        state_dim: Input state dimension.
        num_classes: Number of output classes.
    """

    def __init__(self, state_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        """Initialize the uncertainty-aware classification head.

        Args:
            state_dim: Input dimension.
            num_classes: Number of classes.
            dropout: Dropout rate for MC dropout uncertainty.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim, num_classes),
        )

        # Aleatoric uncertainty head (log variance per class)
        self.aleatoric_head = nn.Sequential(
            nn.Linear(state_dim, state_dim // 2),
            nn.GELU(),
            nn.Linear(state_dim // 2, num_classes),
        )

        # Temperature parameter for calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        h: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Produce logits and optional uncertainty estimates.

        Args:
            h: Input features of shape (N, state_dim).
            return_uncertainty: Whether to compute uncertainty.

        Returns:
            Tuple of (logits of shape (N, C), uncertainty dict).
        """
        logits = self.classifier(h)
        logits = logits / self.temperature.clamp(min=0.1)

        uncertainty = {}
        if return_uncertainty:
            # Aleatoric uncertainty
            log_var = self.aleatoric_head(h)
            aleatoric = torch.exp(log_var).mean(dim=-1)  # (N,)
            uncertainty["aleatoric"] = aleatoric
            uncertainty["log_variance"] = log_var

            # Predictive entropy as total uncertainty
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            uncertainty["predictive_entropy"] = entropy

            # Max probability as confidence
            uncertainty["confidence"] = probs.max(dim=-1)[0]

        return logits, uncertainty


class SDETGNN(nn.Module):
    """Stochastic Differential Equation Temporal Graph Neural Network.

    End-to-end model that processes temporal graph-structured network
    traffic data through:
    1. Feature embedding into a shared latent space.
    2. Multi-layer graph attention for spatial encoding.
    3. SDE-based continuous-time evolution for temporal dynamics.
    4. Fokker-Planck moment propagation for uncertainty quantification.
    5. Multi-scale fusion across temporal resolutions.
    6. Calibrated classification with uncertainty.

    Attributes:
        input_dim: Raw input feature dimension.
        hidden_dim: Graph attention hidden dimension.
        state_dim: SDE state dimension.
        num_classes: Number of output classes.
        num_layers: Number of graph attention layers.
        num_heads: Attention heads.
        num_scales: Temporal scales for fusion.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        state_dim: int = 64,
        num_classes: int = 10,
        num_layers: int = 4,
        num_heads: int = 8,
        num_scales: int = 3,
        dropout: float = 0.1,
        sde_config: Optional[Dict[str, Any]] = None,
        fokker_planck_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the SDE-TGNN model.

        Args:
            input_dim: Raw input feature dimension.
            hidden_dim: Hidden dimension for graph attention layers.
            state_dim: SDE state dimension.
            num_classes: Number of output classification classes.
            num_layers: Number of graph attention layers.
            num_heads: Number of attention heads.
            num_scales: Number of temporal scales for multi-scale fusion.
            dropout: Dropout probability.
            sde_config: Configuration dict for the SDE solver.
            fokker_planck_config: Configuration dict for the FP solver.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_scales = num_scales

        # Default configs
        if sde_config is None:
            sde_config = {
                "solver": "euler_maruyama",
                "dt": 0.01,
                "noise_type": "diagonal",
                "adjoint": False,
                "integration_steps": 20,
            }
        if fokker_planck_config is None:
            fokker_planck_config = {
                "moment_order": 2,
                "gaussian_approx": True,
                "propagation_steps": 10,
                "regularization": 1e-6,
            }

        # ---- 1. Feature Embedding ----
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # ---- 2. Graph Attention Layers ----
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                TemporalGATBlock(
                    channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    time_dim=state_dim,
                )
            )

        # ---- 3. State Projection (hidden_dim -> state_dim for SDE) ----
        self.state_projection = nn.Sequential(
            nn.Linear(hidden_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.GELU(),
        )

        # ---- 4. Drift and Diffusion Networks ----
        drift_layers = sde_config.get("drift_layers", 3)
        diffusion_layers = sde_config.get("diffusion_layers", 2)

        self.drift_network = DriftNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=drift_layers,
            dropout=dropout,
        )

        self.diffusion_network = DiffusionNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=diffusion_layers,
            noise_type=sde_config.get("noise_type", "diagonal"),
            dropout=dropout,
        )

        # ---- 5. SDE Solver ----
        self.sde_solver = get_sde_solver(
            solver_name=sde_config.get("solver", "euler_maruyama"),
            dt=sde_config.get("dt", 0.01),
            num_steps=sde_config.get("integration_steps", 20),
            adjoint=sde_config.get("adjoint", False),
        )

        # ---- 6. Fokker-Planck Solver ----
        self.fp_solver = FokkerPlanckSolver(
            state_dim=state_dim,
            moment_order=fokker_planck_config.get("moment_order", 2),
            gaussian_approx=fokker_planck_config.get("gaussian_approx", True),
            propagation_steps=fokker_planck_config.get("propagation_steps", 10),
            dt=sde_config.get("dt", 0.01),
            regularization=fokker_planck_config.get("regularization", 1e-6),
        )

        # ---- 7. Multi-Scale Temporal Fusion ----
        self.multi_scale_fusion = MultiScaleTemporalFusion(
            state_dim=state_dim,
            num_scales=num_scales,
        )

        # Scale-specific SDE step counts
        self.scale_steps = nn.ParameterList([
            nn.Parameter(torch.tensor(float(sde_config.get("integration_steps", 20) * (2 ** i))))
            for i in range(num_scales)
        ])

        # ---- 8. Classification Head ----
        self.classification_head = UncertaintyHead(
            state_dim=state_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # ---- 9. State Uprojection (for residual from GAT) ----
        self.state_residual_proj = nn.Linear(hidden_dim, state_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier/Kaiming strategies."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _run_sde_at_scale(
        self,
        h: torch.Tensor,
        scale_idx: int,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run SDE integration at a specific temporal scale.

        Different scales use different numbers of integration steps,
        capturing short-term vs. long-term dynamics.

        Args:
            h: SDE state of shape (N, state_dim).
            scale_idx: Index of the temporal scale.
            edge_index: Optional graph edges.
            edge_attr: Optional edge attributes.

        Returns:
            Integrated state of shape (N, state_dim).
        """
        # Define drift and diffusion closures
        def drift_fn(state, t, ei=None, ea=None):
            ei_use = ei if ei is not None else edge_index
            ea_use = ea if ea is not None else edge_attr
            return self.drift_network(state, t, ei_use, ea_use)

        def diffusion_fn(state, t):
            return self.diffusion_network(state, t)

        # Integrate
        result = self.sde_solver(
            h0=h,
            drift_fn=drift_fn,
            diffusion_fn=diffusion_fn,
            t_start=0.0,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return result["final_state"]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Full forward pass of the SDE-TGNN model.

        Processing pipeline:
        1. Embed raw features into the hidden space.
        2. Apply graph attention layers for spatial encoding.
        3. Project to SDE state space.
        4. Run SDE integration at multiple temporal scales.
        5. Propagate moments via Fokker-Planck (if uncertainty requested).
        6. Fuse multi-scale features.
        7. Classify with uncertainty estimation.

        Args:
            x: Input features of shape (N, input_dim).
            edge_index: Graph edge indices of shape (2, E), or None.
            edge_attr: Edge attributes of shape (E, D_edge), or None.
            timestamps: Node timestamps of shape (N,), or None.
            return_uncertainty: If True, compute full uncertainty dict.

        Returns:
            Tuple of:
                - logits: Classification logits of shape (N, num_classes).
                - uncertainty_dict: Dictionary of uncertainty metrics.
        """
        N = x.size(0)
        uncertainty_dict: Dict[str, torch.Tensor] = {}

        # ---- Step 1: Feature Embedding ----
        h = self.input_embedding(x)  # (N, hidden_dim)

        # ---- Step 2: Graph Attention Message Passing ----
        if edge_index is not None and edge_index.numel() > 0:
            for gat_layer in self.gat_layers:
                h = gat_layer(h, edge_index, edge_attr, timestamps)
        else:
            # For non-graph inputs, create self-loop graph
            self_loops = torch.stack([torch.arange(N, device=x.device)] * 2)
            for gat_layer in self.gat_layers:
                h = gat_layer(h, self_loops, None, timestamps)

        # Store GAT output for residual
        gat_output = h  # (N, hidden_dim)

        # ---- Step 3: SDE State Projection ----
        h_state = self.state_projection(h)  # (N, state_dim)

        # ---- Step 4: Multi-Scale SDE Integration ----
        scale_features = []
        for scale_idx in range(self.num_scales):
            h_scale = self._run_sde_at_scale(
                h_state, scale_idx, edge_index, edge_attr,
            )
            scale_features.append(h_scale)

        # ---- Step 5: Fokker-Planck Moment Propagation ----
        if return_uncertainty:
            # Initialize moments
            mu = h_state.detach()
            sigma = self.fp_solver.get_initial_covariance(N, x.device)

            def drift_fn_fp(state, t, ei=None, ea=None):
                ei_use = ei if ei is not None else edge_index
                ea_use = ea if ea is not None else edge_attr
                return self.drift_network(state, t, ei_use, ea_use)

            fp_result = self.fp_solver.propagate_moments(
                mu=mu,
                sigma=sigma,
                drift_fn=drift_fn_fp,
                diffusion_fn=self.diffusion_network,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

            uncertainty_dict["fp_mu"] = fp_result["mu"]
            uncertainty_dict["fp_sigma"] = fp_result["sigma"]
            uncertainty_dict["fp_variance"] = fp_result["variance"]

            # KL divergence from prior
            kl_div = self.fp_solver.compute_kl_divergence(
                fp_result["mu"], fp_result["sigma"],
            )
            uncertainty_dict["kl_divergence"] = kl_div

            # Predictive entropy
            entropy = self.fp_solver.compute_predictive_entropy(
                fp_result["mu"], fp_result["sigma"],
            )
            uncertainty_dict["state_entropy"] = entropy

            # Epistemic uncertainty: variance from FP propagation
            epistemic = fp_result["variance"].mean(dim=-1)
            uncertainty_dict["epistemic"] = epistemic

        # ---- Step 6: Multi-Scale Temporal Fusion ----
        h_fused = self.multi_scale_fusion(scale_features)  # (N, state_dim)

        # Add residual from GAT output
        h_fused = h_fused + self.state_residual_proj(gat_output)

        # ---- Step 7: Classification with Uncertainty ----
        logits, head_uncertainty = self.classification_head(
            h_fused, return_uncertainty=return_uncertainty,
        )
        uncertainty_dict.update(head_uncertainty)

        return logits, uncertainty_dict

    def get_num_parameters(self) -> Dict[str, int]:
        """Count parameters by component.

        Returns:
            Dictionary mapping component name to parameter count.
        """
        components = {
            "input_embedding": self.input_embedding,
            "gat_layers": self.gat_layers,
            "state_projection": self.state_projection,
            "drift_network": self.drift_network,
            "diffusion_network": self.diffusion_network,
            "fp_solver": self.fp_solver,
            "multi_scale_fusion": self.multi_scale_fusion,
            "classification_head": self.classification_head,
        }

        counts = {}
        total = 0
        for name, module in components.items():
            n = sum(p.numel() for p in module.parameters())
            counts[name] = n
            total += n
        counts["total"] = total

        return counts
