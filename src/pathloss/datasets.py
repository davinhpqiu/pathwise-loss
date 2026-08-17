"""Synthetic path-to-path datasets: context points in, target points out.

Task. A path is generated on a fine grid and treated as ground truth. Two
disjoint irregular subsamples are drawn from it: **context**, which the model
sees, and **target**, where the loss is evaluated. The model is asked for values
at the target times, so it has to interpolate rather than copy.

Target times are irregular by design. On an evenly spaced target grid the
trapezoid-weighted L^2 loss equals MSE up to two endpoint half-weights, order
1/N (notebook 01 section 3, logbook 2026-08-13), so a comparison of losses on an
even grid cannot show anything. Irregular target times are what makes the
weighting do work.

Every array has a fixed length per split, drawn without replacement, so no
padding or masking is needed. Times are sorted within each split.
"""

from __future__ import annotations

import numpy as np

from .paths import brownian_motion, ornstein_uhlenbeck, smooth_test_path

__all__ = ["GENERATORS", "sample_times", "make_dataset"]

GENERATORS = {
    "ornstein_uhlenbeck": ornstein_uhlenbeck,
    "brownian_motion": brownian_motion,
    "smooth_test_path": smooth_test_path,
}


def sample_times(
    n_fine: int,
    n_ctx: int,
    n_tgt: int,
    mode: str = "clustered",
    density_bias: float = 3.0,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw disjoint context and target index sets from range(n_fine).

    Parameters
    ----------
    mode : {"uniform", "clustered"}
        "uniform" draws indices with equal probability, giving an irregular grid
        with no systematic bias. "clustered" draws with probability ramping
        across the interval, so sample density varies with t; `density_bias` is
        the exponential rate, 0 recovering "uniform".

    Returns
    -------
    (idx_ctx, idx_tgt), each sorted, disjoint, and excluding neither endpoint
    from consideration.

    Notes
    -----
    Disjoint by construction: context is drawn first, target from what is left.
    Overlap would let a model score well by copying, which would make the loss
    comparison a test of memorisation.
    """
    rng = np.random.default_rng(rng)
    if n_ctx + n_tgt > n_fine:
        raise ValueError(f"n_ctx + n_tgt = {n_ctx + n_tgt} exceeds n_fine = {n_fine}")

    if mode == "uniform":
        prob = np.ones(n_fine)
    elif mode == "clustered":
        u = np.linspace(0.0, 1.0, n_fine)
        prob = np.exp(float(density_bias) * (u - 0.5))
    else:
        raise ValueError(f"unknown mode {mode!r}")
    prob = prob / prob.sum()

    idx_ctx = rng.choice(n_fine, size=n_ctx, replace=False, p=prob)
    left = np.setdiff1d(np.arange(n_fine), idx_ctx, assume_unique=False)
    p_left = prob[left] / prob[left].sum()
    idx_tgt = rng.choice(left, size=n_tgt, replace=False, p=p_left)
    return np.sort(idx_ctx), np.sort(idx_tgt)


def make_dataset(
    n_samples: int = 512,
    generator: str = "ornstein_uhlenbeck",
    n_fine: int = 1025,
    n_ctx: int = 64,
    n_tgt: int = 64,
    mode: str = "clustered",
    density_bias: float = 3.0,
    noise: float = 0.0,
    T: float = 1.0,
    d: int = 1,
    rng=None,
    **gen_kwargs,
) -> dict[str, np.ndarray]:
    """Build a dataset of (context, target, truth) triples.

    Parameters
    ----------
    noise : standard deviation of observation noise added to context values
        only. Targets stay clean, so the loss measures recovery of the path
        rather than of the noise.

    Returns
    -------
    dict with
        t_ctx  (n_samples, n_ctx)        x_ctx  (n_samples, n_ctx, d)
        t_tgt  (n_samples, n_tgt)        x_tgt  (n_samples, n_tgt, d)
        t_fine (n_fine,)                 x_fine (n_samples, n_fine, d)

    `x_fine` is the ground truth on the fine grid, kept for evaluation against
    the path rather than against the sampled targets.
    """
    rng = np.random.default_rng(rng)
    if generator not in GENERATORS:
        raise ValueError(f"unknown generator {generator!r}; have {sorted(GENERATORS)}")
    gen = GENERATORS[generator]

    if generator == "smooth_test_path":
        gen_kwargs.pop("d", None)

    t_fine = None
    x_fine = np.empty((n_samples, n_fine, d))
    t_ctx = np.empty((n_samples, n_ctx))
    x_ctx = np.empty((n_samples, n_ctx, d))
    t_tgt = np.empty((n_samples, n_tgt))
    x_tgt = np.empty((n_samples, n_tgt, d))

    for i in range(n_samples):
        if generator == "smooth_test_path":
            t, x = gen(n=n_fine, T=T, **gen_kwargs)
            x = np.repeat(x, d, axis=-1)
        else:
            t, x = gen(n=n_fine, T=T, d=d, rng=rng, **gen_kwargs)
        if t_fine is None:
            t_fine = t
        x_fine[i] = x

        i_ctx, i_tgt = sample_times(
            n_fine, n_ctx, n_tgt, mode=mode, density_bias=density_bias, rng=rng
        )
        t_ctx[i], x_ctx[i] = t[i_ctx], x[i_ctx]
        t_tgt[i], x_tgt[i] = t[i_tgt], x[i_tgt]

    if noise > 0:
        x_ctx = x_ctx + rng.normal(scale=noise, size=x_ctx.shape)

    return {
        "t_ctx": t_ctx,
        "x_ctx": x_ctx,
        "t_tgt": t_tgt,
        "x_tgt": x_tgt,
        "t_fine": t_fine,
        "x_fine": x_fine,
    }
