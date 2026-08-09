"""Synthetic path generators and irregular-sampling / missingness utilities.

The convention throughout the project: generate on a *fine* grid, treat that as
ground truth "continuous" path, then subsample it to produce what the model
sees. Keeping those two objects separate is what makes the robustness
experiments in the proposal well-defined.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "smooth_test_path",
    "ornstein_uhlenbeck",
    "brownian_motion",
    "subsample_irregular",
    "apply_missingness",
]


def smooth_test_path(n: int = 1025, T: float = 1.0, freq: float = 3.0):
    """A deterministic smooth path with known analytic L^p norms.

    x(t) = sin(2 pi freq t). Used for verifying quadrature convergence rates,
    since ||x||_{L^2([0,1])}^2 = 1/2 exactly when freq is an integer.
    """
    t = np.linspace(0.0, T, n)
    x = np.sin(2.0 * np.pi * freq * t)[:, None]
    return t, x


def brownian_motion(n: int = 1025, T: float = 1.0, d: int = 1, rng=None):
    """Standard d-dimensional Brownian motion on a uniform grid."""
    rng = np.random.default_rng(rng)
    t = np.linspace(0.0, T, n)
    dt = np.diff(t)
    dW = rng.normal(scale=np.sqrt(dt)[:, None], size=(n - 1, d))
    x = np.concatenate([np.zeros((1, d)), np.cumsum(dW, axis=0)], axis=0)
    return t, x


def ornstein_uhlenbeck(
    n: int = 1025,
    T: float = 1.0,
    theta: float = 2.0,
    sigma: float = 0.5,
    x0: float = 1.0,
    d: int = 1,
    rng=None,
):
    """OU process dx = -theta x dt + sigma dW, Euler-Maruyama on a fine grid."""
    rng = np.random.default_rng(rng)
    t = np.linspace(0.0, T, n)
    dt = t[1] - t[0]
    x = np.empty((n, d))
    x[0] = x0
    sqdt = np.sqrt(dt)
    for i in range(1, n):
        x[i] = x[i - 1] - theta * x[i - 1] * dt + sigma * sqdt * rng.normal(size=d)
    return t, x


def subsample_irregular(
    t: np.ndarray,
    x: np.ndarray,
    keep: float = 0.2,
    mode: str = "bernoulli",
    density_bias: float | None = None,
    rng=None,
):
    """Subsample a fine path onto an irregular grid.

    mode:
      "bernoulli": keep each interior point independently w.p. `keep`.
      "clustered": keep probability ramps across [0, T], controlled by
                     `density_bias` (0 = uniform, larger = more points late).
                     This is the setting where plain MSE and the integral norm
                     disagree most, so it is the interesting stress test.

    Endpoints are always retained so the horizon is unchanged.
    """
    rng = np.random.default_rng(rng)
    n = t.size
    if mode == "bernoulli":
        prob = np.full(n, keep)
    elif mode == "clustered":
        b = 3.0 if density_bias is None else float(density_bias)
        u = (t - t[0]) / (t[-1] - t[0])
        prob = keep * np.exp(b * (u - 0.5))
        prob = np.clip(prob, 0.0, 1.0)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    mask = rng.random(n) < prob
    mask[0] = mask[-1] = True
    return t[mask], x[mask]


def apply_missingness(x: np.ndarray, rate: float = 0.1, rng=None):
    """Insert NaNs channel-wise at random, simulating missing observations."""
    rng = np.random.default_rng(rng)
    x = np.array(x, dtype=float, copy=True)
    mask = rng.random(x.shape) < rate
    x[mask] = np.nan
    return x, mask
