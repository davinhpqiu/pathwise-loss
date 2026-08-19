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

Every array has a fixed length per split and times are sorted. Missing context
channels are zero-filled and accompanied by an observation mask.
"""

from __future__ import annotations

import numpy as np

from .paths import brownian_motion, ornstein_uhlenbeck, smooth_test_path

__all__ = ["GENERATORS", "make_dataset", "sample_times"]

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
    context_mode: str | None = None,
    target_mode: str | None = None,
    context_density_bias: float | None = None,
    target_density_bias: float | None = None,
    context_rng=None,
    target_rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw disjoint context and target index sets from range(n_fine).

    Parameters
    ----------
    mode : {"uniform", "clustered"}
        Legacy common mode used when separate context and target modes are
        omitted.
    context_mode, target_mode : {"uniform", "clustered"}, optional
        Separate controls. This permits target sampling to vary while model
        input stays fixed.

    Returns
    -------
    (idx_ctx, idx_tgt), each sorted and disjoint. Target indices include both
    endpoints, so every target loss spans the same full interval.

    Notes
    -----
    Disjoint by construction: endpoints are reserved for target, context is
    drawn from the interior, then remaining target points from what is left.
    Overlap would let a model score well by copying, which would make the loss
    comparison a test of memorisation.
    """
    rng = np.random.default_rng(rng)
    if context_rng is None:
        context_rng = np.random.default_rng(rng.integers(0, 2**63, dtype=np.int64))
    else:
        context_rng = np.random.default_rng(context_rng)
    if target_rng is None:
        target_rng = np.random.default_rng(rng.integers(0, 2**63, dtype=np.int64))
    else:
        target_rng = np.random.default_rng(target_rng)
    if n_fine < 2:
        raise ValueError("n_fine must be at least 2")
    if n_tgt < 2:
        raise ValueError("n_tgt must be at least 2 to include both endpoints")
    if n_ctx + n_tgt > n_fine:
        raise ValueError(f"n_ctx + n_tgt = {n_ctx + n_tgt} exceeds n_fine = {n_fine}")

    context_mode = mode if context_mode is None else context_mode
    target_mode = mode if target_mode is None else target_mode
    context_density_bias = (
        density_bias if context_density_bias is None else context_density_bias
    )
    target_density_bias = (
        density_bias if target_density_bias is None else target_density_bias
    )

    def probabilities(selected_mode: str, selected_bias: float) -> np.ndarray:
        if selected_mode == "uniform":
            out = np.ones(n_fine)
        elif selected_mode == "clustered":
            u = np.linspace(0.0, 1.0, n_fine)
            out = np.exp(float(selected_bias) * (u - 0.5))
        else:
            raise ValueError(f"unknown sampling mode {selected_mode!r}")
        return out / out.sum()

    context_prob = probabilities(context_mode, context_density_bias)
    target_prob = probabilities(target_mode, target_density_bias)

    interior = np.arange(1, n_fine - 1)
    p_interior = context_prob[interior] / context_prob[interior].sum()
    idx_ctx = context_rng.choice(interior, size=n_ctx, replace=False, p=p_interior)
    left = np.setdiff1d(interior, idx_ctx, assume_unique=False)
    if n_tgt == 2:
        sampled_tgt = np.empty(0, dtype=int)
    else:
        p_left = target_prob[left] / target_prob[left].sum()
        sampled_tgt = target_rng.choice(left, size=n_tgt - 2, replace=False, p=p_left)
    idx_tgt = np.concatenate(([0, n_fine - 1], sampled_tgt))
    return np.sort(idx_ctx), np.sort(idx_tgt)


def make_dataset(
    n_samples: int = 512,
    generator: str = "ornstein_uhlenbeck",
    n_fine: int = 1025,
    n_ctx: int = 64,
    n_tgt: int = 64,
    mode: str = "clustered",
    density_bias: float = 3.0,
    context_mode: str | None = None,
    target_mode: str | None = None,
    context_density_bias: float | None = None,
    target_density_bias: float | None = None,
    noise: float = 0.0,
    missing_rate: float = 0.0,
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
    missing_rate : independent probability that a context channel is hidden.
        Hidden values are set to zero and identified by `m_ctx`.

    Returns
    -------
    dict with
        t_ctx  (n_samples, n_ctx)        x_ctx, m_ctx  (n_samples, n_ctx, d)
        t_tgt  (n_samples, n_tgt)        x_tgt  (n_samples, n_tgt, d)
        t_fine (n_fine,)                 x_fine (n_samples, n_fine, d)

    `x_fine` is the ground truth on the fine grid, kept for evaluation against
    the path rather than against the sampled targets.
    """
    rng = np.random.default_rng(rng)
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must lie in [0, 1], got {missing_rate}")
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

    seeds = rng.integers(0, 2**63, size=(n_samples, 3), dtype=np.int64)
    for i in range(n_samples):
        path_rng = np.random.default_rng(seeds[i, 0])
        if generator == "smooth_test_path":
            t, x = gen(n=n_fine, T=T, **gen_kwargs)
            x = np.repeat(x, d, axis=-1)
        else:
            t, x = gen(n=n_fine, T=T, d=d, rng=path_rng, **gen_kwargs)
        if t_fine is None:
            t_fine = t
        x_fine[i] = x

        i_ctx, i_tgt = sample_times(
            n_fine,
            n_ctx,
            n_tgt,
            mode=mode,
            density_bias=density_bias,
            context_mode=context_mode,
            target_mode=target_mode,
            context_density_bias=context_density_bias,
            target_density_bias=target_density_bias,
            context_rng=seeds[i, 1],
            target_rng=seeds[i, 2],
        )
        t_ctx[i], x_ctx[i] = t[i_ctx], x[i_ctx]
        t_tgt[i], x_tgt[i] = t[i_tgt], x[i_tgt]

    if noise > 0:
        x_ctx = x_ctx + rng.normal(scale=noise, size=x_ctx.shape)
    m_ctx = (rng.random(x_ctx.shape) >= missing_rate).astype(x_ctx.dtype)
    x_ctx = x_ctx * m_ctx

    return {
        "t_ctx": t_ctx,
        "x_ctx": x_ctx,
        "m_ctx": m_ctx,
        "t_tgt": t_tgt,
        "x_tgt": x_tgt,
        "t_fine": t_fine,
        "x_fine": x_fine,
    }
