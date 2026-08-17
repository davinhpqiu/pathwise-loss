"""Tests for the synthetic path-to-path dataset. NumPy only, no torch."""

from __future__ import annotations

import numpy as np
import pytest

from pathloss.datasets import make_dataset, sample_times

def test_context_and_target_are_disjoint():
    for seed in range(20):
        i_ctx, i_tgt = sample_times(200, 30, 30, rng=seed)
        assert np.intersect1d(i_ctx, i_tgt).size == 0


def test_sampled_times_are_sorted_and_in_range():
    d = make_dataset(n_samples=8, n_fine=257, n_ctx=32, n_tgt=32, rng=0)
    for key in ("t_ctx", "t_tgt"):
        assert np.all(np.diff(d[key], axis=-1) > 0), f"{key} not strictly increasing"
        assert d[key].min() >= 0.0 and d[key].max() <= 1.0


def test_shapes_and_finiteness():
    d = make_dataset(n_samples=6, n_fine=129, n_ctx=16, n_tgt=20, d=2, rng=1)
    assert d["t_ctx"].shape == (6, 16)
    assert d["x_ctx"].shape == (6, 16, 2)
    assert d["t_tgt"].shape == (6, 20)
    assert d["x_tgt"].shape == (6, 20, 2)
    assert d["x_fine"].shape == (6, 129, 2)
    assert all(np.isfinite(v).all() for v in d.values())


def test_clustered_mode_biases_sample_density():
    """`clustered` should put more target points late, `uniform` should not."""
    late_clustered, late_uniform = [], []
    for seed in range(30):
        c = make_dataset(1, n_fine=513, n_ctx=64, n_tgt=64, mode="clustered", rng=seed)
        u = make_dataset(1, n_fine=513, n_ctx=64, n_tgt=64, mode="uniform", rng=seed)
        late_clustered.append((c["t_tgt"] > 0.5).mean())
        late_uniform.append((u["t_tgt"] > 0.5).mean())
    assert np.mean(late_clustered) > 0.6
    assert abs(np.mean(late_uniform) - 0.5) < 0.05


def test_noise_hits_context_only():
    clean = make_dataset(4, n_fine=129, n_ctx=16, n_tgt=16, noise=0.0, rng=3)
    noisy = make_dataset(4, n_fine=129, n_ctx=16, n_tgt=16, noise=0.5, rng=3)
    assert not np.allclose(clean["x_ctx"], noisy["x_ctx"])
    assert np.allclose(clean["x_tgt"], noisy["x_tgt"])
    assert np.allclose(clean["x_fine"], noisy["x_fine"])


def test_requesting_too_many_points_raises():
    with pytest.raises(ValueError, match="exceeds n_fine"):
        sample_times(50, 30, 30)
