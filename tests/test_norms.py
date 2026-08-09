"""Correctness checks for norms.py.

These are cheap and should stay cheap: they run on every commit. The point is
that the *library* is trustworthy, so a surprising result in a notebook is a
result about the maths, not about a bug.
"""

import numpy as np
import pytest

from pathloss.norms import (
    integral_distance,
    integral_norm,
    integral_norm_callable,
    integral_norm_gauss,
    p_variation_dyadic,
    p_variation_exact,
    pointwise_mse,
    quadrature_weights,
)
from pathloss.paths import brownian_motion, smooth_test_path, subsample_irregular


# --- quadrature weights ----------------------------------------------------

def test_weights_sum_to_horizon():
    """Any consistent rule integrates the constant 1 exactly."""
    for t in (np.linspace(0, 3, 17), np.sort(np.random.default_rng(0).uniform(0, 3, 40))):
        t = np.unique(np.concatenate([[0.0], t, [3.0]]))
        for rule in ("trapezoid", "riemann_left", "riemann_right"):
            assert quadrature_weights(t, rule).sum() == pytest.approx(3.0)


def test_trapezoid_is_uniform_rule_on_uniform_grid():
    t = np.linspace(0, 1, 11)
    w = quadrature_weights(t, "trapezoid")
    h = 0.1
    assert w[0] == pytest.approx(h / 2)
    assert w[-1] == pytest.approx(h / 2)
    assert np.allclose(w[1:-1], h)


def test_rejects_bad_time_grids():
    with pytest.raises(ValueError):
        quadrature_weights(np.array([0.0, 1.0, 0.5]))       # not increasing
    with pytest.raises(ValueError):
        quadrature_weights(np.array([0.0]))                  # too short


# --- L^p norms against analytic values -------------------------------------

def test_l2_of_sine_matches_analytic():
    t, x = smooth_test_path(n=1025, T=1.0, freq=3.0)
    assert float(integral_norm(t, x, p=2)) == pytest.approx(1 / np.sqrt(2), rel=1e-9)


def test_sampled_matches_adaptive_quadrature():
    f = lambda s: np.sin(2 * np.pi * 3 * s)
    t, x = smooth_test_path(n=2049, T=1.0, freq=3.0)
    for p in (1.0, 2.0, 4.0):
        sampled = float(integral_norm(t, x, p=p))
        adaptive = integral_norm_callable(f, 0.0, 1.0, p=p)
        assert sampled == pytest.approx(adaptive, rel=1e-6)


def test_gauss_matches_when_integrand_is_smooth():
    """Gauss-Legendre is exponentially accurate on analytic integrands."""
    f = lambda s: np.sin(2 * np.pi * 3 * s)
    for p in (2.0, 4.0):                       # even p -> |f|^p = f^p is analytic
        adaptive = integral_norm_callable(f, 0.0, 1.0, p=p)
        gauss = integral_norm_gauss(f, 0.0, 1.0, p=p, n=80)
        assert gauss == pytest.approx(adaptive, rel=1e-8)


def test_gauss_degrades_on_kinked_integrand():
    """Odd p puts |.| kinks in the integrand and Gauss-Legendre loses its edge.

    Documented rather than worked around: it is the reason `integral_norm` uses
    trapezoid, and a warning against assuming a high-order rule is always
    better. QUADPACK survives because it subdivides adaptively around the
    kinks; a fixed node set cannot.
    """
    f = lambda s: np.sin(2 * np.pi * 3 * s)
    adaptive = integral_norm_callable(f, 0.0, 1.0, p=1.0)
    gauss = integral_norm_gauss(f, 0.0, 1.0, p=1.0, n=80)
    assert gauss != pytest.approx(adaptive, rel=1e-6)     # not spectrally accurate
    assert gauss == pytest.approx(adaptive, rel=1e-2)     # but not wrong, either


def test_sup_norm():
    t, x = smooth_test_path(n=4097, T=1.0, freq=1.0)
    assert float(integral_norm(t, x, p=np.inf)) == pytest.approx(1.0, abs=1e-6)


def test_lp_increasing_in_p_on_unit_interval():
    """On a probability space (|[0,1]| = 1), ||f||_p is nondecreasing in p."""
    t, x = smooth_test_path(n=2049, T=1.0, freq=2.0)
    vals = [float(integral_norm(t, x, p=p)) for p in (1, 2, 3, 4, 8)]
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


def test_no_overflow_at_large_p():
    t = np.linspace(0, 1, 501)
    x = (1e6 * np.ones_like(t))[:, None]
    assert np.isfinite(integral_norm(t, x, p=50.0))


def test_norm_is_zero_iff_zero_and_scales():
    t = np.linspace(0, 2, 257)
    x = np.cos(t)[:, None]
    assert float(integral_norm(t, np.zeros_like(x), p=2)) == pytest.approx(0.0)
    a, b = float(integral_norm(t, x, p=2)), float(integral_norm(t, 3 * x, p=2))
    assert b == pytest.approx(3 * a)


def test_triangle_inequality():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 129)
    x, y = rng.normal(size=(129, 1)), rng.normal(size=(129, 1))
    lhs = float(integral_norm(t, x + y, p=2))
    rhs = float(integral_norm(t, x, p=2)) + float(integral_norm(t, y, p=2))
    assert lhs <= rhs + 1e-12


def test_batching():
    t = np.linspace(0, 1, 65)
    x = np.stack([np.sin(2 * np.pi * k * t) for k in (1, 2, 3)])[..., None]  # (3, 65, 1)
    out = integral_norm(t, x, p=2)
    assert out.shape == (3,)
    assert np.allclose(out, 1 / np.sqrt(2), atol=1e-6)


# --- the claim the project rests on ---------------------------------------

def test_mse_and_integral_norm_agree_on_uniform_grid():
    """On a uniform grid they coincide (up to endpoint half-weights)."""
    rng = np.random.default_rng(2)
    t = np.linspace(0, 1, 2001)
    x = rng.normal(size=(2001, 1))
    y = x + 0.1 * rng.normal(size=(2001, 1))
    l2 = float(integral_distance(t, x, y, p=2, normalise=True))
    mse = float(pointwise_mse(x, y))
    assert l2**2 == pytest.approx(mse, rel=2e-3)


def test_integral_norm_is_stable_under_irregular_subsampling():
    """The whole motivation: the quadrature-weighted distance tracks the
    fine-grid value under clustered sampling, and plain MSE does not."""
    t_fine = np.linspace(0, 1, 8193)
    truth = np.sin(2 * np.pi * 2 * t_fine)
    bump = 0.8 * np.exp(-0.5 * ((t_fine - 0.125) / 0.03) ** 2)
    pred = truth + bump
    ref = float(integral_distance(t_fine, truth[:, None], pred[:, None], p=2, normalise=True))

    t_cl, idx = subsample_irregular(
        t_fine, np.arange(t_fine.size)[:, None], keep=0.05, mode="clustered", rng=5
    )
    i = idx[:, 0].astype(int)
    l2 = float(integral_distance(t_cl, truth[i, None], pred[i, None], p=2, normalise=True))
    mse = float(pointwise_mse(truth[i, None], pred[i, None]))

    assert l2 == pytest.approx(ref, rel=0.15)      # weighted: tracks the truth
    assert abs(mse - ref**2) > 0.15 * ref**2       # unweighted: does not


# --- p-variation -----------------------------------------------------------

def test_p_variation_exact_dominates_dyadic():
    _, W = brownian_motion(n=513, rng=3)
    for p in (1.0, 2.0, 3.0):
        assert p_variation_exact(W, p) >= p_variation_dyadic(W, p) - 1e-9


def test_p_variation_of_monotone_path_is_total_increment():
    """For a monotone path, V_1 = |x(T) - x(0)| exactly, for any partition."""
    x = np.linspace(0, 5, 200)[:, None]
    assert p_variation_exact(x, 1.0) == pytest.approx(5.0)
    assert p_variation_dyadic(x, 1.0) == pytest.approx(5.0)


def test_p_variation_decreasing_in_p():
    _, W = brownian_motion(n=257, rng=11)
    vals = [p_variation_exact(W, p) for p in (1.0, 1.5, 2.0, 3.0)]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def test_brownian_1_variation_grows_with_refinement():
    """BM is a.s. not of bounded variation: the estimate should keep growing."""
    _, W = brownian_motion(n=2**12 + 1, rng=13)
    coarse = p_variation_dyadic(W[::8], 1.0)
    fine = p_variation_dyadic(W, 1.0)
    assert fine > 1.8 * coarse
