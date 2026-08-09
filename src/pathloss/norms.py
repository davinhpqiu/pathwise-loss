"""Integral norms and quadrature on sampled paths.

Everything here treats a "path" as a pair (t, x) where

    t : (T,)        strictly increasing sample times
    x : (..., T, d) values, batched over leading dimensions

i.e. the *time axis is the second-to-last*, matching the (batch, time, channel)
convention used by torchcde / signatory. All functions are pure NumPy so they
can be unit-tested cheaply; the torch versions live in `losses.py`.

References
----------
Ramsay & Silverman (2005), Functional Data Analysis, ch. 3 to 5: basis
    representation and the roughness penalty.
Ferraty & Vieu (2006): semi-metrics on functional data; the discretised
    L2 semi-metric with quadrature weights w_j = t_j - t_{j-1}.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quadrature_weights",
    "integral_norm",
    "integral_distance",
    "pointwise_mse",
    "p_variation_dyadic",
    "integral_norm_callable",
    "integral_norm_gauss",
    "p_variation_exact",
]


# ---------------------------------------------------------------------------
# quadrature weights
# ---------------------------------------------------------------------------

def quadrature_weights(t: np.ndarray, rule: str = "trapezoid") -> np.ndarray:
    """Weights w with sum_i w_i g(t_i) ~= int g(t) dt.

    Parameters
    ----------
    t : (T,) strictly increasing sample times.
    rule : {"trapezoid", "riemann_left", "riemann_right", "simpson"}

    Notes
    -----
    "trapezoid" is the default because it is the only second-order rule that
    is valid verbatim on a *non-uniform* grid, which is the case this project
    cares about. On a uniform grid it reduces to (1/2, 1, ..., 1, 1/2) * dt.

    "simpson" requires an odd number of *uniformly spaced* points; it is
    included only as a high-order reference for convergence studies.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be 1-D")
    if t.size < 2:
        raise ValueError("need at least two sample times")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("t must be strictly increasing")

    if rule == "trapezoid":
        w = np.zeros_like(t)
        w[:-1] += dt / 2.0
        w[1:] += dt / 2.0
        return w
    if rule == "riemann_left":
        w = np.zeros_like(t)
        w[:-1] = dt
        return w
    if rule == "riemann_right":
        w = np.zeros_like(t)
        w[1:] = dt
        return w
    if rule == "simpson":
        n = t.size
        if n % 2 == 0:
            raise ValueError("Simpson needs an odd number of points")
        if not np.allclose(dt, dt[0]):
            raise ValueError("Simpson implemented for uniform grids only")
        h = dt[0]
        w = np.ones(n)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        return w * h / 3.0
    raise ValueError(f"unknown rule {rule!r}")


# ---------------------------------------------------------------------------
# L^p integral norms
# ---------------------------------------------------------------------------

def integral_norm(
    t: np.ndarray,
    x: np.ndarray,
    p: float = 2.0,
    rule: str = "trapezoid",
    channel_norm: str = "euclidean",
) -> np.ndarray:
    r"""Estimate ||x||_{L^p([0,T])} = ( \int_0^T |x(t)|^p dt )^{1/p}.

    Parameters
    ----------
    t : (T,) sample times.
    x : (..., T, d) sampled values.
    p : exponent, p >= 1. Use p = np.inf for the sup norm.
    channel_norm : how |.| collapses the d channels at each time.
        "euclidean" -> sqrt(sum_k x_k^2); "max" -> max_k |x_k|.

    Returns
    -------
    (...,) array of norms.

    Notes
    -----
    Numerically stabilised by factoring out M = max_t |x(t)| before raising to
    the p-th power, so large p does not overflow. See notebook 01 for the
    convergence study against analytic values.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.shape[-2] != t.size:
        raise ValueError(
            f"x has {x.shape[-2]} time steps but t has {t.size}"
        )

    if channel_norm == "euclidean":
        mag = np.sqrt(np.sum(x**2, axis=-1))          # (..., T)
    elif channel_norm == "max":
        mag = np.max(np.abs(x), axis=-1)
    else:
        raise ValueError(f"unknown channel_norm {channel_norm!r}")

    if np.isinf(p):
        return np.max(mag, axis=-1)

    w = quadrature_weights(t, rule=rule)               # (T,)
    scale = np.max(mag, axis=-1, keepdims=True)        # (..., 1)
    safe = np.where(scale > 0, scale, 1.0)
    integral = np.sum(w * (mag / safe) ** p, axis=-1)  # (...,)
    return safe[..., 0] * integral ** (1.0 / p)


def integral_distance(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: float = 2.0,
    rule: str = "trapezoid",
    normalise: bool = False,
) -> np.ndarray:
    r"""||x - y||_{L^p}. If `normalise`, divide by T^{1/p} so the result is
    comparable to a root-mean-square rather than growing with the horizon.
    """
    d = integral_norm(t, np.asarray(x) - np.asarray(y), p=p, rule=rule)
    if normalise and not np.isinf(p):
        horizon = float(t[-1] - t[0])
        d = d / horizon ** (1.0 / p)
    return d


def pointwise_mse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain MSE over the time axis: the baseline the project compares to.

    Deliberately ignores `t`: this is exactly the pathology under study. On a
    uniform grid this equals integral_distance(..., p=2, normalise=True)**2 up
    to the endpoint half-weights; on an irregular grid it does not, because it
    weights every sample equally and so over-weights dense regions.
    """
    diff = np.asarray(x) - np.asarray(y)
    return np.mean(np.sum(diff**2, axis=-1), axis=-1)


# ---------------------------------------------------------------------------
# p-variation
# ---------------------------------------------------------------------------

def p_variation_dyadic(x: np.ndarray, p: float = 2.0) -> float:
    r"""Lower bound on the p-variation of a sampled path via dyadic partitions.

        V_p(x) = ( sup_P sum_i |x_{t_{i+1}} - x_{t_i}|^p )^{1/p}

    The true supremum is over *all* partitions, which is combinatorial. This
    restricts to nested dyadic partitions and takes the max over levels, giving
    a cheap lower bound that is the standard diagnostic in the rough-path
    literature. Requires len(x) = 2^k + 1 for exact dyadic refinement; other
    lengths are handled by taking every 2^j-th index.

    Returns the max over dyadic levels of the level-wise p-variation sum,
    raised to 1/p.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    best = 0.0
    step = 1
    while step < n:
        idx = np.arange(0, n, step)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        incr = np.diff(x[idx], axis=0)
        total = float(np.sum(np.linalg.norm(incr, axis=-1) ** p))
        best = max(best, total)
        step *= 2
    return best ** (1.0 / p)


# ---------------------------------------------------------------------------
# higher-accuracy estimators, for when the path is a *function* not a sample
# ---------------------------------------------------------------------------

def integral_norm_callable(
    f,
    a: float = 0.0,
    b: float = 1.0,
    p: float = 2.0,
    **quad_kwargs,
) -> float:
    r"""||f||_{L^p([a,b])} by adaptive quadrature, for f given as a callable.

    Uses scipy.integrate.quad (adaptive Gauss-Kronrod, QUADPACK). Near machine
    precision on smooth f, and the right tool when you *can* evaluate f
    anywhere, i.e. for an analytic ground truth or a fitted spline/basis
    representation, as opposed to raw samples.

    This is the reference value that `integral_norm` is checked against in
    notebook 01. It is NOT usable inside a training loop: it evaluates f at
    points chosen adaptively, which a sampled dataset cannot supply.
    """
    from scipy.integrate import quad

    val, _err = quad(lambda s: abs(f(s)) ** p, a, b, **quad_kwargs)
    return val ** (1.0 / p)


def integral_norm_gauss(
    f,
    a: float = 0.0,
    b: float = 1.0,
    p: float = 2.0,
    n: int = 64,
) -> float:
    r"""||f||_{L^p([a,b])} by n-point Gauss-Legendre quadrature.

    Exact for polynomial integrands of degree <= 2n-1, and converges
    exponentially for analytic f. Relevant when *we control the sampling*:
    if the experiment may choose observation times, placing them at Gauss
    nodes gives far more accuracy per observation than a uniform grid.

    Worth contrasting with the irregular-sampling experiments: Gauss nodes are
    non-uniform *by design* and the correct weights are not (t_{i+1}-t_i)/2 --
    a concrete demonstration that "which points" and "which weights" are two
    separate choices.

    Caveat, verified in tests: the spectral accuracy needs the *integrand*
    |f|^p to be analytic, not just f. For odd p the absolute value introduces
    kinks wherever f changes sign, and the convergence drops to algebraic --
    n = 80 nodes then gives only ~1e-3 relative accuracy on |sin|. A fixed node
    set cannot adapt to a kink; `integral_norm_callable` (adaptive QUADPACK)
    can, which is why it, not this, is the reference implementation.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    # map [-1, 1] -> [a, b]
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    ts = mid + half * nodes
    vals = np.array([abs(f(s)) ** p for s in ts])
    return float((half * np.sum(weights * vals)) ** (1.0 / p))


def p_variation_exact(x: np.ndarray, p: float = 2.0) -> float:
    r"""Exact p-variation over all sub-partitions of the sample grid, by DP.

        V_p(x)^p = max over subsequences 0 = i_0 < ... < i_k = n-1
                   of  sum_j |x_{i_{j+1}} - x_{i_j}|^p

    The supremum over *all* partitions of [0,T] is not computable from samples;
    this computes the supremum over subsequences of the *observed* grid, which
    is the best any finite algorithm can do, and is a lower bound on the truth.

    Complexity O(n^2) in time and O(n) in memory via

        D[j] = max_{i < j} ( D[i] + |x_j - x_i|^p ),   D[0] = 0.

    Compare `p_variation_dyadic`, which restricts to nested dyadic partitions:
    O(n log n) but a strictly weaker lower bound. Use this one to check how
    much the dyadic shortcut costs you. For n <~ 4000 it is fast enough, and
    the gap is worth measuring once rather than assuming.

    Note (Ferrucci, Perree & Lyons 2026, Remark 2.2): for p = 1 the optimal
    split is a cumulative-sum problem and hence linear; for p > 1 the analogous
    task "is computationally harder, since even computing the ordinary
    p-variation requires an optimisation over partitions". This function is
    that optimisation, done exactly on the sampled grid.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n < 2:
        return 0.0

    best = np.full(n, -np.inf)
    best[0] = 0.0
    for j in range(1, n):
        incr = np.linalg.norm(x[j] - x[:j], axis=-1) ** p   # (j,)
        best[j] = np.max(best[:j] + incr)
    return float(best[-1] ** (1.0 / p))
