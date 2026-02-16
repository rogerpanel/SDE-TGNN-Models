"""Baseline models for comparative evaluation against SDE-TGNN.

Implements eight baselines spanning classical ML, deep learning,
graph-based, ODE-based, and uncertainty-aware categories:

1. RandomForestBaseline: Scikit-learn Random Forest classifier.
2. XGBoostBaseline: Gradient boosting via XGBoost (or sklearn fallback).
3. LSTMBaseline: Bidirectional LSTM for sequential flow data.
4. CNNBiLSTMBaseline: 1D CNN + BiLSTM hybrid.
5. GraphSAGEBaseline: Inductive graph neural network.
6. NeuralODEBaseline: Continuous-depth model via ODE integration.
7. MCDropoutBaseline: Monte Carlo dropout for epistemic uncertainty.
8. DeepEnsembleBaseline: Ensemble of independent classifiers.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


# ======================================================================
# Classical ML baselines (wrapped for consistent API)
# ======================================================================

class RandomForestBaseline:
    """Random Forest baseline using scikit-learn.

    Attributes:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        model: Fitted RandomForestClassifier.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        """Initialize the Random Forest baseline.

        Args:
            n_estimators: Number of decision trees.
            max_depth: Maximum depth of each tree.
            random_state: Random seed.
            n_jobs: Number of parallel jobs.
        """
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the Random Forest model.

        Args:
            X: Training features of shape (N, D).
            y: Training labels of shape (N,).
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Predicted labels of shape (N,).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Probability matrix of shape (N, C).
        """
        return self.model.predict_proba(X)


class XGBoostBaseline:
    """XGBoost gradient boosting baseline.

    Falls back to sklearn GradientBoostingClassifier if xgboost
    is not installed.

    Attributes:
        model: Fitted classifier.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """Initialize the XGBoost baseline.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate.
            random_state: Random seed.
        """
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric="mlogloss",
                tree_method="hist",
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model.

        Args:
            X: Training features of shape (N, D).
            y: Training labels of shape (N,).
        """
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Predicted labels of shape (N,).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Probability matrix of shape (N, C).
        """
        return self.model.predict_proba(X)


# ======================================================================
# Deep learning baselines
# ======================================================================

class LSTMBaseline(nn.Module):
    """Bidirectional LSTM baseline for sequential flow classification.

    Treats each flow's features as a 1-step sequence (or the features
    can be reshaped into a multi-step sequence when temporal ordering
    is available).

    Attributes:
        input_dim: Feature dimension.
        hidden_dim: LSTM hidden size.
        num_layers: Number of LSTM layers.
        num_classes: Output classes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the LSTM baseline.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: LSTM hidden dimension.
            num_layers: Number of LSTM layers.
            num_classes: Number of output classes.
            dropout: Dropout probability.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (B, D) or (B, T, D).

        Returns:
            Logits of shape (B, num_classes).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)

        x = self.input_proj(x)  # (B, T, hidden_dim)
        output, (h_n, _) = self.lstm(x)

        # Use final hidden states from both directions
        h_forward = h_n[-2]  # (B, hidden_dim)
        h_backward = h_n[-1]  # (B, hidden_dim)
        h = torch.cat([h_forward, h_backward], dim=-1)  # (B, 2*hidden_dim)

        return self.classifier(h)


class CNNBiLSTMBaseline(nn.Module):
    """1D CNN + Bidirectional LSTM hybrid baseline.

    The CNN extracts local patterns from feature subsequences, and the
    BiLSTM captures sequential dependencies.

    Attributes:
        input_dim: Feature dimension.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 10,
        num_filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the CNN-BiLSTM baseline.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: LSTM hidden dimension.
            num_classes: Number of classes.
            num_filters: Number of CNN filters.
            kernel_size: CNN kernel size.
            dropout: Dropout rate.
        """
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters),
            nn.GELU(),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=num_filters * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input of shape (B, D).

        Returns:
            Logits of shape (B, num_classes).
        """
        # Reshape for 1D CNN: (B, 1, D)
        x = x.unsqueeze(1)

        # CNN feature extraction
        cnn_out = self.cnn(x)  # (B, C, D)

        # Transpose for LSTM: (B, D, C)
        lstm_in = cnn_out.permute(0, 2, 1)

        # BiLSTM
        output, (h_n, _) = self.lstm(lstm_in)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        return self.classifier(h)


class GraphSAGEBaseline(nn.Module):
    """GraphSAGE baseline for graph-level classification.

    Uses the inductive GraphSAGE convolution with mean aggregation,
    followed by a global readout and MLP classifier.

    Attributes:
        input_dim: Node feature dimension.
        hidden_dim: Hidden dimension.
        num_classes: Output classes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 10,
        num_layers: int = 3,
        dropout: float = 0.3,
    ) -> None:
        """Initialize the GraphSAGE baseline.

        Args:
            input_dim: Node feature dimension.
            hidden_dim: Hidden layer dimension.
            num_classes: Number of classes.
            num_layers: Number of GraphSAGE layers.
            dropout: Dropout rate.
        """
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape (N, input_dim).
            edge_index: Edge indices of shape (2, E).
            batch: Batch assignment vector for graph-level readout.

        Returns:
            Logits. If ``batch`` is given, shape (num_graphs, num_classes);
            otherwise (N, num_classes) for node-level classification.
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)

        if batch is not None:
            x = global_mean_pool(x, batch)

        return self.classifier(x)


class NeuralODEBaseline(nn.Module):
    """Neural ODE baseline (deterministic continuous-depth model).

    Similar to SDE-TGNN but without the diffusion (stochastic) component.
    Uses a simple MLP-based ODE function integrated with fixed-step RK4.

    Attributes:
        input_dim: Feature dimension.
        state_dim: ODE state dimension.
        num_classes: Output classes.
    """

    def __init__(
        self,
        input_dim: int,
        state_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 10,
        num_steps: int = 20,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the Neural ODE baseline.

        Args:
            input_dim: Input feature dimension.
            state_dim: ODE state dimension.
            hidden_dim: Hidden dimension.
            num_classes: Number of classes.
            num_steps: Number of RK4 integration steps.
            dropout: Dropout rate.
        """
        super().__init__()
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # ODE function f(h, t)
        self.ode_fn = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh(),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Dropout(dropout),
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _ode_step_rk4(self, h: torch.Tensor, t: float) -> torch.Tensor:
        """One step of 4th-order Runge-Kutta integration.

        Args:
            h: Current state of shape (B, state_dim).
            t: Current time.

        Returns:
            Updated state of shape (B, state_dim).
        """
        dt = self.dt

        def f(state: torch.Tensor, time: float) -> torch.Tensor:
            t_tensor = torch.full((state.size(0), 1), time, device=state.device, dtype=state.dtype)
            return self.ode_fn(torch.cat([state, t_tensor], dim=-1))

        k1 = f(h, t)
        k2 = f(h + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(h + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(h + dt * k3, t + dt)

        return h + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (B, input_dim).

        Returns:
            Logits of shape (B, num_classes).
        """
        h = self.encoder(x)

        t = 0.0
        for _ in range(self.num_steps):
            h = self._ode_step_rk4(h, t)
            t += self.dt

        return self.classifier(h)


class MCDropoutBaseline(nn.Module):
    """Monte Carlo Dropout baseline for epistemic uncertainty.

    A standard MLP classifier where dropout remains active at test
    time.  Multiple stochastic forward passes produce a distribution
    of predictions from which uncertainty is estimated.

    Attributes:
        input_dim: Feature dimension.
        num_classes: Number of classes.
        mc_samples: Number of MC forward passes at test time.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_layers: int = 4,
        dropout: float = 0.2,
        mc_samples: int = 50,
    ) -> None:
        """Initialize the MC Dropout baseline.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer size.
            num_classes: Number of classes.
            num_layers: Number of hidden layers.
            dropout: Dropout probability (kept at test time).
            mc_samples: Number of MC samples for uncertainty.
        """
        super().__init__()
        self.mc_samples = mc_samples

        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim

        layers.append(nn.Linear(hidden_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single forward pass (dropout active if training).

        Args:
            x: Input of shape (B, input_dim).

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.network(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run multiple forward passes with active dropout for uncertainty.

        Args:
            x: Input of shape (B, input_dim).

        Returns:
            Tuple of (mean logits, uncertainty dict).
        """
        self.train()  # Keep dropout active
        all_logits = []

        for _ in range(self.mc_samples):
            logits = self.forward(x)
            all_logits.append(F.softmax(logits, dim=-1))

        stacked = torch.stack(all_logits, dim=0)  # (S, B, C)

        # Mean prediction
        mean_probs = stacked.mean(dim=0)  # (B, C)
        mean_logits = mean_probs.log()

        # Epistemic uncertainty: variance across MC samples
        variance = stacked.var(dim=0).sum(dim=-1)  # (B,)

        # Predictive entropy
        entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum(dim=-1)

        # Mutual information (BALD criterion)
        per_sample_entropy = -(stacked * (stacked + 1e-10).log()).sum(dim=-1)
        expected_entropy = per_sample_entropy.mean(dim=0)
        mutual_info = entropy - expected_entropy

        uncertainty = {
            "epistemic": variance,
            "predictive_entropy": entropy,
            "mutual_information": mutual_info,
            "all_probs": stacked,
        }

        return mean_logits, uncertainty


class DeepEnsembleBaseline(nn.Module):
    """Deep Ensemble baseline for uncertainty estimation.

    Trains M independent neural networks with different random
    initializations and aggregates their predictions.

    Attributes:
        num_members: Number of ensemble members.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 10,
        num_members: int = 5,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the Deep Ensemble baseline.

        Args:
            input_dim: Input dimension.
            hidden_dim: Hidden layer size.
            num_classes: Number of classes.
            num_members: Number of ensemble members.
            num_layers: Layers per member.
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_members = num_members
        self.num_classes = num_classes

        self.members = nn.ModuleList()
        for _ in range(num_members):
            layers: list[nn.Module] = []
            in_dim = input_dim
            for j in range(num_layers):
                out_dim = hidden_dim
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                in_dim = out_dim
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.members.append(nn.Sequential(*layers))

        self._initialize_differently()

    def _initialize_differently(self) -> None:
        """Initialize each member with a different random seed."""
        for idx, member in enumerate(self.members):
            torch.manual_seed(idx * 1337 + 42)
            for module in member.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, member_idx: int = 0) -> torch.Tensor:
        """Forward pass through a single ensemble member.

        Args:
            x: Input of shape (B, input_dim).
            member_idx: Index of the ensemble member.

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.members[member_idx](x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Aggregate predictions from all ensemble members.

        Args:
            x: Input of shape (B, input_dim).

        Returns:
            Tuple of (mean logits, uncertainty dict).
        """
        all_probs = []
        for member in self.members:
            logits = member(x)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs)

        stacked = torch.stack(all_probs, dim=0)  # (M, B, C)
        mean_probs = stacked.mean(dim=0)
        mean_logits = (mean_probs + 1e-10).log()

        # Ensemble variance (epistemic)
        variance = stacked.var(dim=0).sum(dim=-1)

        # Predictive entropy
        entropy = -(mean_probs * (mean_probs + 1e-10).log()).sum(dim=-1)

        # Jensen-Shannon divergence among members
        log_mean = (mean_probs + 1e-10).log().unsqueeze(0).expand_as(stacked)
        kl_per_member = (stacked * ((stacked + 1e-10).log() - log_mean)).sum(dim=-1)
        js_divergence = kl_per_member.mean(dim=0)

        uncertainty = {
            "epistemic": variance,
            "predictive_entropy": entropy,
            "js_divergence": js_divergence,
            "all_probs": stacked,
        }

        return mean_logits, uncertainty
