"""Differentiable losses in torch, matching the NumPy definitions in `norms.py`.

Each loss takes target times `t` of shape (B, T), predictions and targets of
shape (B, T, d), and returns a scalar averaged over the batch.

`pointwise_mse` is the baseline. `integral_lp` is the quadrature-weighted L^p
loss of notebook 01. On an evenly spaced grid the two agree up to the two
endpoint half-weights, order 1/T, so they differ only when target times are
irregular; `tests/test_losses.py` asserts both halves of that.

torch is an optional dependency (`requirements-ml.txt`). Importing this module
without it raises at import time rather than silently degrading.
"""

from __future__ import annotations

import torch

__all__ = [
    "trapezoid_weights",
    "pointwise_mse",
    "integral_lp",
    "sobolev_h1",
    "LOSSES",
    "get_loss",
]


def trapezoid_weights(t: torch.Tensor, normalise: bool = True) -> torch.Tensor:
    """Trapezoid quadrature weights for a batch of sorted time grids.

    w_i = (t_{i+1} - t_{i-1}) / 2 on the interior, and half the adjacent gap at
    each end, so that sum_i w_i g(t_i) approximates the integral of g over
    [t_0, t_{T-1}].

    Parameters
    ----------
    t : (B, T) strictly increasing along the last axis.
    normalise : divide by the horizon t_{T-1} - t_0, so weights sum to 1 and the
        result is comparable with a mean rather than growing with the horizon.

    Returns
    -------
    (B, T) weights.
    """
    if t.ndim != 2:
        raise ValueError(f"t must be (B, T), got shape {tuple(t.shape)}")
    dt = t[:, 1:] - t[:, :-1]
    if torch.any(dt <= 0):
        raise ValueError("t must be strictly increasing along the last axis")

    w = torch.zeros_like(t)
    w[:, :-1] = w[:, :-1] + dt / 2.0
    w[:, 1:] = w[:, 1:] + dt / 2.0
    if normalise:
        horizon = (t[:, -1] - t[:, 0]).unsqueeze(-1)
        w = w / horizon
    return w


def pointwise_mse(
    t: torch.Tensor, pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean squared error, ignoring `t`.

    Every observation carries weight 1/T whatever the spacing, which is the
    behaviour under study: on an irregular grid this over-weights densely
    sampled stretches.
    """
    return ((pred - target) ** 2).sum(-1).mean()


def integral_lp(
    t: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    p: float = 2.0,
    squared: bool = True,
) -> torch.Tensor:
    """Quadrature-weighted L^p loss between two sampled paths.

    Computes (sum_i w_i |pred_i - target_i|^p) with w the normalised trapezoid
    weights, where |.| is Euclidean across channels.

    Parameters
    ----------
    p : exponent, p >= 1.
    squared : return the weighted sum itself, rather than its 1/p-th root.
        The root is a norm and so scales linearly in the residual; the unrooted
        sum matches MSE's scaling at p = 2 and has a bounded gradient at zero.

    Notes
    -----
    At p = 2 with `squared=True` this equals `pointwise_mse` on an evenly spaced
    grid, up to the endpoint half-weights.
    """
    if p < 1:
        raise ValueError(f"need p >= 1, got {p}")
    w = trapezoid_weights(t)  # (B, T)
    mag = torch.linalg.vector_norm(pred - target, dim=-1)  # (B, T)
    if p == float("inf"):
        return mag.max(dim=-1).values.mean()
    total = (w * mag.clamp_min(1e-12) ** p).sum(-1)  # (B,)
    if not squared:
        total = total ** (1.0 / p)
    return total.mean()


def sobolev_h1(
    t: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    pred_derivative: torch.Tensor,
    target_derivative: torch.Tensor,
    rho: float | torch.Tensor,
) -> torch.Tensor:
    """Elapsed-time value error plus elapsed-time derivative error.

    This discrete quadrature approximates

        T^{-1} integral (|pred - target|^2
                         + rho |pred' - target'|^2) dt.

    It is valid when both paths have square-integrable time derivatives.
    Three-argument loss registry cannot supply derivatives, and diffusion paths
    do not have an H1 derivative in continuous time, so this function is kept
    outside that registry.
    """
    if pred_derivative.shape != pred.shape:
        raise ValueError(
            "pred_derivative must match pred shape "
            f"{tuple(pred.shape)}, got {tuple(pred_derivative.shape)}"
        )
    if target_derivative.shape != target.shape:
        raise ValueError(
            "target_derivative must match target shape "
            f"{tuple(target.shape)}, got {tuple(target_derivative.shape)}"
        )
    rho_tensor = torch.as_tensor(rho, dtype=pred.dtype, device=pred.device)
    if torch.any(rho_tensor < 0):
        raise ValueError(f"rho must be non-negative, got {rho}")
    return integral_lp(t, pred, target, p=2.0) + rho_tensor * integral_lp(
        t, pred_derivative, target_derivative, p=2.0
    )


LOSSES = {
    "mse": pointwise_mse,
    "integral_l1": lambda t, a, b: integral_lp(t, a, b, p=1.0),
    "integral_l2": lambda t, a, b: integral_lp(t, a, b, p=2.0),
    "integral_l4": lambda t, a, b: integral_lp(t, a, b, p=4.0),
    "integral_linf": lambda t, a, b: integral_lp(t, a, b, p=float("inf")),
}


def get_loss(name: str):
    """Look up a loss by the name used in a config file."""
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; have {sorted(LOSSES)}")
    return LOSSES[name]
