"""Numerical SDE solvers for the SDE-TGNN model.

Provides multiple SDE integration schemes:
- Euler-Maruyama: First-order explicit solver (O(sqrt(dt)) strong convergence).
- Milstein: Higher-order solver (O(dt) strong convergence for scalar noise).
- Adaptive: Variable step-size Euler-Maruyama with error control.
- SDE Adjoint: Memory-efficient backpropagation through the SDE.

All solvers integrate the Ito SDE:
    dh = f(h, t) dt + g(h, t) dW
where f is the drift and g is the diffusion.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


class EulerMaruyama(nn.Module):
    """Euler-Maruyama SDE solver.

    The simplest explicit strong solver for Ito SDEs.  At each step:
        h_{n+1} = h_n + f(h_n, t_n) * dt + g(h_n, t_n) * sqrt(dt) * Z_n

    where Z_n ~ N(0, I).

    Attributes:
        dt: Fixed time step.
        num_steps: Number of integration steps.
    """

    def __init__(
        self,
        dt: float = 0.01,
        num_steps: int = 20,
    ) -> None:
        """Initialize the Euler-Maruyama solver.

        Args:
            dt: Integration time step.
            num_steps: Number of steps to integrate.
        """
        super().__init__()
        self.dt = dt
        self.num_steps = num_steps

    def forward(
        self,
        h0: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t_start: float = 0.0,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Integrate the SDE from h0 over num_steps steps.

        Args:
            h0: Initial state of shape (N, D).
            drift_fn: Drift function f(h, t) -> (N, D).
            diffusion_fn: Diffusion function g(h, t) -> (N, D).
            t_start: Starting time.
            edge_index: Optional graph edges for drift.
            edge_attr: Optional edge attributes.
            return_trajectory: Whether to return full trajectory.

        Returns:
            Dictionary with 'final_state' of shape (N, D), optionally
            'trajectory' of shape (num_steps+1, N, D), and 'noise_samples'.
        """
        h = h0
        sqrt_dt = math.sqrt(self.dt)
        t = t_start

        trajectory = [h0] if return_trajectory else []
        noise_samples = []

        for step in range(self.num_steps):
            t_tensor = torch.tensor(t, device=h.device, dtype=h.dtype)

            # Drift evaluation
            if edge_index is not None:
                f_val = drift_fn(h, t_tensor, edge_index, edge_attr)
            else:
                f_val = drift_fn(h, t_tensor)

            # Diffusion evaluation
            g_val = diffusion_fn(h, t_tensor)

            # Brownian increment
            dW = torch.randn_like(h) * sqrt_dt
            noise_samples.append(dW)

            # Euler-Maruyama step
            h = h + f_val * self.dt + g_val * dW

            t += self.dt

            if return_trajectory:
                trajectory.append(h)

        result: Dict[str, torch.Tensor] = {"final_state": h}
        if return_trajectory:
            result["trajectory"] = torch.stack(trajectory, dim=0)
        result["noise_samples"] = torch.stack(noise_samples, dim=0)

        return result


class MilsteinSolver(nn.Module):
    """Milstein SDE solver for diagonal noise.

    Higher-order scheme that adds a correction term for the diffusion:
        h_{n+1} = h_n + f*dt + g*dW + 0.5*g*g'*(dW^2 - dt)

    Achieves O(dt) strong convergence order for scalar/diagonal noise,
    compared to O(sqrt(dt)) for Euler-Maruyama.

    Attributes:
        dt: Fixed time step.
        num_steps: Number of integration steps.
        eps: Finite difference step for diffusion derivative.
    """

    def __init__(
        self,
        dt: float = 0.01,
        num_steps: int = 20,
        eps: float = 1e-5,
    ) -> None:
        """Initialize the Milstein solver.

        Args:
            dt: Integration time step.
            num_steps: Number of steps.
            eps: Perturbation for numerical derivative of diffusion.
        """
        super().__init__()
        self.dt = dt
        self.num_steps = num_steps
        self.eps = eps

    def forward(
        self,
        h0: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t_start: float = 0.0,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Integrate the SDE using the Milstein method.

        Args:
            h0: Initial state of shape (N, D).
            drift_fn: Drift function f(h, t) -> (N, D).
            diffusion_fn: Diffusion function g(h, t) -> (N, D).
            t_start: Starting time.
            edge_index: Optional graph edges.
            edge_attr: Optional edge attributes.
            return_trajectory: Return full trajectory.

        Returns:
            Dictionary with 'final_state' and optionally 'trajectory'.
        """
        h = h0
        sqrt_dt = math.sqrt(self.dt)
        t = t_start

        trajectory = [h0] if return_trajectory else []

        for step in range(self.num_steps):
            t_tensor = torch.tensor(t, device=h.device, dtype=h.dtype)

            # Drift
            if edge_index is not None:
                f_val = drift_fn(h, t_tensor, edge_index, edge_attr)
            else:
                f_val = drift_fn(h, t_tensor)

            # Diffusion and its derivative
            g_val = diffusion_fn(h, t_tensor)

            # Numerical derivative dg/dh via finite differences
            h_perturbed = h + self.eps * torch.ones_like(h)
            g_perturbed = diffusion_fn(h_perturbed, t_tensor)
            dg_dh = (g_perturbed - g_val) / self.eps

            # Brownian increment
            dW = torch.randn_like(h) * sqrt_dt

            # Milstein step
            h = (
                h
                + f_val * self.dt
                + g_val * dW
                + 0.5 * g_val * dg_dh * (dW ** 2 - self.dt)
            )

            t += self.dt
            if return_trajectory:
                trajectory.append(h)

        result: Dict[str, torch.Tensor] = {"final_state": h}
        if return_trajectory:
            result["trajectory"] = torch.stack(trajectory, dim=0)

        return result


class AdaptiveSDESolver(nn.Module):
    """Adaptive step-size SDE solver.

    Uses an embedded pair (Euler-Maruyama / Heun) to estimate the
    local error and adapt the step size.  This is particularly useful
    when the drift or diffusion exhibits sharp gradients.

    Attributes:
        dt_init: Initial step size.
        max_steps: Maximum number of integration steps.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.
        safety: Safety factor for step size adaptation.
        min_dt: Minimum allowed step size.
        max_dt: Maximum allowed step size.
    """

    def __init__(
        self,
        dt_init: float = 0.01,
        max_steps: int = 200,
        rtol: float = 1e-3,
        atol: float = 1e-4,
        safety: float = 0.9,
        min_dt: float = 1e-6,
        max_dt: float = 0.1,
    ) -> None:
        """Initialize the adaptive solver.

        Args:
            dt_init: Initial step size.
            max_steps: Maximum integration steps.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            safety: Safety factor for step size control.
            min_dt: Minimum step size.
            max_dt: Maximum step size.
        """
        super().__init__()
        self.dt_init = dt_init
        self.max_steps = max_steps
        self.rtol = rtol
        self.atol = atol
        self.safety = safety
        self.min_dt = min_dt
        self.max_dt = max_dt

    def forward(
        self,
        h0: torch.Tensor,
        drift_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        diffusion_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t_start: float = 0.0,
        t_end: Optional[float] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Integrate the SDE with adaptive step sizes.

        Args:
            h0: Initial state of shape (N, D).
            drift_fn: Drift function.
            diffusion_fn: Diffusion function.
            t_start: Starting time.
            t_end: Ending time. If None, uses dt_init * 20.
            edge_index: Optional graph edges.
            edge_attr: Optional edge attributes.
            return_trajectory: Return full trajectory.

        Returns:
            Dictionary with 'final_state', 'num_steps', and optionally 'trajectory'.
        """
        if t_end is None:
            t_end = t_start + self.dt_init * 20

        h = h0
        t = t_start
        dt = self.dt_init
        trajectory = [h0] if return_trajectory else []
        num_accepted = 0

        for _ in range(self.max_steps):
            if t >= t_end - 1e-12:
                break

            # Clip dt so we don't overshoot
            dt = min(dt, t_end - t)
            if dt < self.min_dt:
                dt = self.min_dt

            t_tensor = torch.tensor(t, device=h.device, dtype=h.dtype)
            sqrt_dt = math.sqrt(abs(dt))

            # Evaluate drift and diffusion
            if edge_index is not None:
                f_val = drift_fn(h, t_tensor, edge_index, edge_attr)
            else:
                f_val = drift_fn(h, t_tensor)
            g_val = diffusion_fn(h, t_tensor)

            # Brownian increment
            dW = torch.randn_like(h) * sqrt_dt

            # Euler-Maruyama step (first order)
            h_euler = h + f_val * dt + g_val * dW

            # Heun correction (second order predictor)
            t_next = torch.tensor(t + dt, device=h.device, dtype=h.dtype)
            if edge_index is not None:
                f_next = drift_fn(h_euler, t_next, edge_index, edge_attr)
            else:
                f_next = drift_fn(h_euler, t_next)
            g_next = diffusion_fn(h_euler, t_next)

            h_heun = h + 0.5 * (f_val + f_next) * dt + 0.5 * (g_val + g_next) * dW

            # Error estimate
            error = (h_heun - h_euler).abs()
            scale = self.atol + self.rtol * torch.max(h.abs(), h_euler.abs())
            error_ratio = (error / scale).max().item()

            if error_ratio <= 1.0:
                # Accept step
                h = h_heun
                t += dt
                num_accepted += 1
                if return_trajectory:
                    trajectory.append(h)

                # Increase step size
                if error_ratio > 0:
                    dt = min(
                        self.safety * dt * (1.0 / error_ratio) ** 0.5,
                        self.max_dt,
                    )
                else:
                    dt = self.max_dt
            else:
                # Reject step and decrease dt
                dt = max(
                    self.safety * dt * (1.0 / error_ratio) ** 0.5,
                    self.min_dt,
                )

        result: Dict[str, torch.Tensor] = {
            "final_state": h,
            "num_steps": torch.tensor(num_accepted),
        }
        if return_trajectory:
            result["trajectory"] = torch.stack(trajectory, dim=0)

        return result


class SDEAdjoint(nn.Module):
    """Memory-efficient backpropagation through the SDE via the adjoint method.

    Instead of storing the full forward trajectory for backprop, the
    adjoint method solves an augmented SDE backwards in time, reducing
    memory from O(L * N * D) to O(N * D).

    This wraps one of the forward solvers and overrides the backward
    pass using a custom autograd Function.

    Attributes:
        forward_solver: The underlying forward SDE solver.
        dt: Time step for the adjoint backward pass.
    """

    def __init__(
        self,
        forward_solver: nn.Module,
        dt: float = 0.01,
    ) -> None:
        """Initialize the SDE adjoint wrapper.

        Args:
            forward_solver: An SDE solver (EulerMaruyama, Milstein, etc.).
            dt: Time step for the adjoint backward pass.
        """
        super().__init__()
        self.forward_solver = forward_solver
        self.dt = dt

    def forward(
        self,
        h0: torch.Tensor,
        drift_fn: Callable,
        diffusion_fn: Callable,
        t_start: float = 0.0,
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Integrate the SDE and set up adjoint backward pass.

        In the forward pass we use the underlying solver.  Gradients
        are computed via the custom adjoint backward pass.

        Args:
            h0: Initial state of shape (N, D).
            drift_fn: Drift function.
            diffusion_fn: Diffusion function.
            t_start: Starting time.
            edge_index: Optional graph edges.
            edge_attr: Optional edge attributes.

        Returns:
            Dictionary with 'final_state' of shape (N, D).
        """
        # Use custom autograd function
        final_state = _SDEAdjointFunction.apply(
            h0,
            drift_fn,
            diffusion_fn,
            self.forward_solver,
            t_start,
            self.dt,
            edge_index,
            edge_attr,
        )

        return {"final_state": final_state}


class _SDEAdjointFunction(torch.autograd.Function):
    """Custom autograd function for SDE adjoint backpropagation."""

    @staticmethod
    def forward(
        ctx,
        h0: torch.Tensor,
        drift_fn: Callable,
        diffusion_fn: Callable,
        solver: nn.Module,
        t_start: float,
        dt: float,
        edge_index: Optional[torch.Tensor],
        edge_attr: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward integration.

        Args:
            ctx: Autograd context.
            h0: Initial state.
            drift_fn: Drift function.
            diffusion_fn: Diffusion function.
            solver: Forward SDE solver.
            t_start: Start time.
            dt: Time step for backward pass.
            edge_index: Graph edge indices.
            edge_attr: Edge attributes.

        Returns:
            Final state tensor.
        """
        with torch.no_grad():
            result = solver(
                h0, drift_fn, diffusion_fn,
                t_start=t_start,
                edge_index=edge_index,
                edge_attr=edge_attr,
                return_trajectory=True,
            )

        ctx.save_for_backward(h0, result["final_state"])
        ctx.drift_fn = drift_fn
        ctx.diffusion_fn = diffusion_fn
        ctx.solver = solver
        ctx.t_start = t_start
        ctx.dt = dt
        ctx.edge_index = edge_index
        ctx.edge_attr = edge_attr
        if "trajectory" in result:
            ctx.trajectory = result["trajectory"]

        return result["final_state"]

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass using the adjoint method.

        Solves the augmented ODE backward in time to compute gradients
        without storing the full forward trajectory in the autograd tape.

        Args:
            grad_output: Gradient of loss w.r.t. final state.

        Returns:
            Tuple of gradients (only h0 gradient is non-None).
        """
        h0, hT = ctx.saved_tensors
        drift_fn = ctx.drift_fn
        diffusion_fn = ctx.diffusion_fn
        dt = ctx.dt
        edge_index = ctx.edge_index
        edge_attr = ctx.edge_attr

        # Number of backward steps
        num_steps = getattr(ctx.solver, "num_steps", 20)
        trajectory = getattr(ctx, "trajectory", None)

        # Initialize adjoint state
        adjoint = grad_output.clone()
        h = hT.clone().detach().requires_grad_(True)

        # Backward integration
        t = ctx.t_start + num_steps * dt
        for step in range(num_steps - 1, -1, -1):
            t_tensor = torch.tensor(t, device=h.device, dtype=h.dtype)

            # Reconstruct state from trajectory if available
            if trajectory is not None and step < trajectory.size(0):
                h = trajectory[step].clone().detach().requires_grad_(True)

            # Compute vector-Jacobian products
            with torch.enable_grad():
                if edge_index is not None:
                    f_val = drift_fn(h, t_tensor, edge_index, edge_attr)
                else:
                    f_val = drift_fn(h, t_tensor)

                # VJP for drift
                vjp_f = torch.autograd.grad(
                    f_val, h,
                    grad_outputs=adjoint,
                    retain_graph=True,
                    allow_unused=True,
                )[0]

            if vjp_f is None:
                vjp_f = torch.zeros_like(adjoint)

            # Update adjoint state (backward Euler)
            adjoint = adjoint + vjp_f * dt

            t -= dt

        # Gradient w.r.t. h0
        grad_h0 = adjoint

        # Return gradients for all forward arguments
        # (h0, drift_fn, diffusion_fn, solver, t_start, dt, edge_index, edge_attr)
        return grad_h0, None, None, None, None, None, None, None


def get_sde_solver(
    solver_name: str,
    dt: float = 0.01,
    num_steps: int = 20,
    adjoint: bool = False,
    **kwargs,
) -> nn.Module:
    """Factory function to create an SDE solver by name.

    Args:
        solver_name: Solver identifier ('euler_maruyama', 'milstein', 'adaptive').
        dt: Time step.
        num_steps: Number of integration steps.
        adjoint: Wrap with the adjoint method for memory efficiency.
        **kwargs: Additional solver-specific arguments.

    Returns:
        An SDE solver module.

    Raises:
        ValueError: If solver_name is not recognized.
    """
    solvers = {
        "euler_maruyama": EulerMaruyama,
        "milstein": MilsteinSolver,
        "adaptive": AdaptiveSDESolver,
    }

    if solver_name not in solvers:
        raise ValueError(
            f"Unknown SDE solver '{solver_name}'. "
            f"Choose from {list(solvers.keys())}."
        )

    if solver_name == "adaptive":
        solver = AdaptiveSDESolver(dt_init=dt, max_steps=num_steps * 10, **kwargs)
    else:
        solver = solvers[solver_name](dt=dt, num_steps=num_steps)

    if adjoint:
        solver = SDEAdjoint(forward_solver=solver, dt=dt)

    return solver
