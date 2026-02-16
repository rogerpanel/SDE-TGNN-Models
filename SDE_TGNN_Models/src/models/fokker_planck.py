"""Fokker-Planck equation solver for analytical uncertainty propagation.

Instead of running expensive Monte Carlo sampling to estimate the
distribution of SDE trajectories, the Fokker-Planck approach
propagates the first two moments (mean and covariance) of the
distribution analytically under a Gaussian approximation.

This yields:
    d mu / dt = E[f(h, t)]
    d Sigma / dt = J_f Sigma + Sigma J_f^T + G G^T

where J_f is the Jacobian of the drift, and G is the diffusion matrix.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FokkerPlanckSolver(nn.Module):
    """Analytical uncertainty propagation via Fokker-Planck equations.

    Under the Gaussian approximation, the probability density of the
    SDE state is fully characterized by the mean vector mu(t) and
    covariance matrix Sigma(t).  This module propagates these moments
    forward in time using the drift and diffusion functions.

    Attributes:
        state_dim: SDE state dimension.
        moment_order: Maximum moment order (currently 1 or 2).
        gaussian_approx: Whether to use the Gaussian closure.
        propagation_steps: Number of time steps for integration.
        dt: Time step for moment propagation.
        regularization: Small constant for numerical stability of Sigma.
    """

    def __init__(
        self,
        state_dim: int,
        moment_order: int = 2,
        gaussian_approx: bool = True,
        propagation_steps: int = 10,
        dt: float = 0.01,
        regularization: float = 1e-6,
    ) -> None:
        """Initialize the Fokker-Planck solver.

        Args:
            state_dim: Dimension of the SDE state.
            moment_order: Order of moment propagation (1 = mean only, 2 = mean + cov).
            gaussian_approx: Use Gaussian closure for higher moments.
            propagation_steps: Number of integration steps.
            dt: Time step size.
            regularization: Added to diagonal of Sigma for stability.
        """
        super().__init__()
        self.state_dim = state_dim
        self.moment_order = moment_order
        self.gaussian_approx = gaussian_approx
        self.propagation_steps = propagation_steps
        self.dt = dt
        self.regularization = regularization

        # Learnable initial covariance scale
        self.log_init_sigma = nn.Parameter(torch.zeros(state_dim))

        # Correction network for higher-order moment closure
        if moment_order >= 2:
            self.correction_net = nn.Sequential(
                nn.Linear(state_dim * 2, state_dim),
                nn.GELU(),
                nn.Linear(state_dim, state_dim),
                nn.Tanh(),
            )

    def get_initial_covariance(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute the initial covariance matrix from learnable parameters.

        Args:
            batch_size: Number of samples in the batch.
            device: Target device.

        Returns:
            Diagonal covariance matrix of shape (batch_size, D, D).
        """
        sigma_diag = F.softplus(self.log_init_sigma) + self.regularization
        sigma = torch.diag(sigma_diag).unsqueeze(0).expand(batch_size, -1, -1)
        return sigma.to(device)

    def _compute_jacobian(
        self,
        drift_fn: Callable,
        mu: torch.Tensor,
        t: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the Jacobian of the drift w.r.t. state via forward-mode AD.

        Uses a column-by-column finite-difference approximation for
        efficiency when the state dimension is moderate.

        Args:
            drift_fn: Drift function f(h, t) or f(h, t, edge_index, edge_attr).
            mu: Mean state of shape (N, D).
            t: Current time tensor.
            edge_index: Optional graph edges.
            edge_attr: Optional edge attributes.

        Returns:
            Jacobian tensor of shape (N, D, D).
        """
        N, D = mu.shape
        eps = 1e-4

        # Base evaluation
        mu_detached = mu.detach().requires_grad_(True)
        if edge_index is not None:
            f_base = drift_fn(mu_detached, t, edge_index, edge_attr)
        else:
            f_base = drift_fn(mu_detached, t)

        # Compute Jacobian column by column via finite differences
        jacobian = torch.zeros(N, D, D, device=mu.device, dtype=mu.dtype)

        for j in range(D):
            perturbation = torch.zeros_like(mu)
            perturbation[:, j] = eps

            mu_plus = mu_detached.detach() + perturbation
            if edge_index is not None:
                f_plus = drift_fn(mu_plus, t, edge_index, edge_attr)
            else:
                f_plus = drift_fn(mu_plus, t)

            jacobian[:, :, j] = (f_plus - f_base.detach()) / eps

        return jacobian

    def _propagate_mean(
        self,
        mu: torch.Tensor,
        drift_fn: Callable,
        t: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Propagate the mean forward by one time step.

        d mu / dt = E[f(mu, t)]  (under Gaussian approximation, f evaluated at mean)

        Args:
            mu: Current mean of shape (N, D).
            drift_fn: Drift function.
            t: Current time.
            edge_index: Optional graph edges.
            edge_attr: Optional edge attributes.

        Returns:
            Updated mean of shape (N, D).
        """
        if edge_index is not None:
            f_mu = drift_fn(mu, t, edge_index, edge_attr)
        else:
            f_mu = drift_fn(mu, t)
        return mu + f_mu * self.dt

    def _propagate_covariance(
        self,
        sigma: torch.Tensor,
        jacobian: torch.Tensor,
        diffusion_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """Propagate the covariance matrix forward by one time step.

        d Sigma / dt = J_f @ Sigma + Sigma @ J_f^T + G @ G^T

        Args:
            sigma: Current covariance of shape (N, D, D).
            jacobian: Drift Jacobian of shape (N, D, D).
            diffusion_matrix: Diffusion matrix G of shape (N, D, D).

        Returns:
            Updated covariance of shape (N, D, D).
        """
        # J_f @ Sigma + Sigma @ J_f^T
        jf_sigma = torch.bmm(jacobian, sigma)
        sigma_jft = torch.bmm(sigma, jacobian.transpose(1, 2))

        # G @ G^T (diffusion-induced variance)
        ggt = torch.bmm(diffusion_matrix, diffusion_matrix.transpose(1, 2))

        # Time derivative
        d_sigma = jf_sigma + sigma_jft + ggt

        # Forward Euler update
        sigma_new = sigma + d_sigma * self.dt

        # Symmetrize
        sigma_new = 0.5 * (sigma_new + sigma_new.transpose(1, 2))

        # Add regularization for positive-definiteness
        reg = self.regularization * torch.eye(
            sigma_new.size(-1), device=sigma_new.device, dtype=sigma_new.dtype,
        ).unsqueeze(0)
        sigma_new = sigma_new + reg

        return sigma_new

    def propagate_moments(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        drift_fn: Callable,
        diffusion_fn: Callable,
        t_span: Optional[Tuple[float, float]] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Propagate mean and covariance through time.

        Args:
            mu: Initial mean of shape (N, D).
            sigma: Initial covariance of shape (N, D, D).
            drift_fn: Drift function.
            diffusion_fn: Diffusion function (returns (N, D) diagonal or
                has ``get_diffusion_matrix`` method returning (N, D, D)).
            t_span: Optional (t_start, t_end) tuple.
            edge_index: Optional graph edge indices.
            edge_attr: Optional edge attributes.

        Returns:
            Dictionary with:
                'mu': Final mean of shape (N, D).
                'sigma': Final covariance of shape (N, D, D).
                'mu_trajectory': Mean trajectory of shape (T, N, D).
                'sigma_trajectory': Covariance trajectory of shape (T, N, D, D).
                'variance': Diagonal variance of shape (N, D).
        """
        t_start = t_span[0] if t_span else 0.0
        t = t_start

        mu_traj = [mu]
        sigma_traj = [sigma]

        for step in range(self.propagation_steps):
            t_tensor = torch.tensor(t, device=mu.device, dtype=mu.dtype)

            # Compute Jacobian of drift at current mean
            jacobian = self._compute_jacobian(drift_fn, mu, t_tensor, edge_index, edge_attr)

            # Get diffusion matrix
            if hasattr(diffusion_fn, "get_diffusion_matrix"):
                G = diffusion_fn.get_diffusion_matrix(mu, t_tensor)
            else:
                g_diag = diffusion_fn(mu, t_tensor)
                G = torch.diag_embed(g_diag)

            # Propagate mean
            mu = self._propagate_mean(mu, drift_fn, t_tensor, edge_index, edge_attr)

            # Propagate covariance (only for moment_order >= 2)
            if self.moment_order >= 2:
                sigma = self._propagate_covariance(sigma, jacobian, G)

                # Apply learned correction
                mu_sigma_cat = torch.cat([mu, torch.diagonal(sigma, dim1=1, dim2=2)], dim=-1)
                correction = self.correction_net(mu_sigma_cat)
                sigma_correction = torch.diag_embed(correction * 0.01)
                sigma = sigma + sigma_correction

            mu_traj.append(mu)
            sigma_traj.append(sigma)
            t += self.dt

        # Extract diagonal variance
        variance = torch.diagonal(sigma, dim1=1, dim2=2)

        return {
            "mu": mu,
            "sigma": sigma,
            "mu_trajectory": torch.stack(mu_traj, dim=0),
            "sigma_trajectory": torch.stack(sigma_traj, dim=0),
            "variance": variance,
        }

    def compute_kl_divergence(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between the propagated Gaussian and a standard normal.

        KL(N(mu, Sigma) || N(0, I)) = 0.5 * (tr(Sigma) + mu^T mu - d - log|Sigma|)

        Args:
            mu: Mean of shape (N, D).
            sigma: Covariance of shape (N, D, D).

        Returns:
            KL divergence scalar (averaged over batch).
        """
        d = mu.size(-1)

        # Trace of covariance
        trace_sigma = torch.diagonal(sigma, dim1=1, dim2=2).sum(dim=-1)

        # mu^T mu
        mu_sq = (mu ** 2).sum(dim=-1)

        # Log determinant (using Cholesky for numerical stability)
        reg = self.regularization * torch.eye(d, device=sigma.device, dtype=sigma.dtype)
        sigma_reg = sigma + reg.unsqueeze(0)

        try:
            L = torch.linalg.cholesky(sigma_reg)
            log_det = 2.0 * torch.diagonal(L, dim1=1, dim2=2).log().sum(dim=-1)
        except torch.linalg.LinAlgError:
            # Fallback: use diagonal approximation
            log_det = torch.diagonal(sigma_reg, dim1=1, dim2=2).log().sum(dim=-1)

        kl = 0.5 * (trace_sigma + mu_sq - d - log_det)
        return kl.mean()

    def compute_predictive_entropy(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the differential entropy of the predictive Gaussian.

        H = 0.5 * d * (1 + log(2*pi)) + 0.5 * log|Sigma|

        Args:
            mu: Mean of shape (N, D).
            sigma: Covariance of shape (N, D, D).

        Returns:
            Entropy tensor of shape (N,).
        """
        import math
        d = mu.size(-1)

        reg = self.regularization * torch.eye(d, device=sigma.device, dtype=sigma.dtype)
        sigma_reg = sigma + reg.unsqueeze(0)

        try:
            L = torch.linalg.cholesky(sigma_reg)
            log_det = 2.0 * torch.diagonal(L, dim1=1, dim2=2).log().sum(dim=-1)
        except torch.linalg.LinAlgError:
            log_det = torch.diagonal(sigma_reg, dim1=1, dim2=2).log().sum(dim=-1)

        entropy = 0.5 * d * (1.0 + math.log(2.0 * math.pi)) + 0.5 * log_det
        return entropy
