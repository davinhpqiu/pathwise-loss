"""Correctness checks for norms.py. p-variation is in test_pvar.py.

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
    pointwise_mse,
    quadrature_weights,
    romberg_table,
)
from pathloss.paths import smooth_test_path


# --- quadrature weights ----------------------------------------------------

def test_weights_sum():
    """Any consistent rule integrates the constant 1 exactly."""
    for t in (np.linspace(0, 3, 17), np.sort(np.random.default_rng(0).uniform(0, 3, 40))):
        t = np.unique(np.concatenate([[0.0], t, [3.0]]))
        for rule in ("trapezoid", "riemann_left", "riemann_right"):
            assert quadrature_weights(t, rule).sum() == pytest.approx(3.0)


def test_trapezoid_weights_on_uniform_grid():
    t = np.linspace(0, 1, 11)
    w = quadrature_weights(t, "trapezoid")
    h = 0.1
    assert w[0] == pytest.approx(h / 2)
    assert w[-1] == pytest.approx(h / 2)
    assert np.allclose(w[1:-1], h)


def test_trapezoid_on_constants_and_lines():
    """Degree of exactness 1, with no assumption on node placement.

    This is the whole argument for trapezoid: a rule that cannot integrate a
    straight line is wrong even when the integrand is as simple as possible,
    and trapezoid gets lines right wherever the samples happen to fall.
    """
    rng = np.random.default_rng(4)
    grids = [
        np.linspace(0, 1, 120),
        np.linspace(0, 1, 120) ** 0.35,                 # clustered late
        np.unique(np.concatenate([[0.0], rng.uniform(0, 1, 200), [1.0]])),
    ]
    for t in grids:
        w = quadrature_weights(t, "trapezoid")
        assert np.sum(w * np.ones_like(t)) == pytest.approx(1.0)      # constants
        assert np.sum(w * t) == pytest.approx(0.5)                    # lines

    # The contrast: equal weights are exact on constants but not on lines, and
    # the failure is invisible on the symmetric grid.
    t = grids[1]
    w = np.full(t.size, 1.0 / t.size)
    assert np.sum(w * np.ones_like(t)) == pytest.approx(1.0)
    assert abs(np.sum(w * t) - 0.5) > 0.2


def test_convergence_rates_on_non_periodic_integrand():
    """Fitted log-log slopes against the analytic value, for f(t) = e^t."""
    exact = np.sqrt((np.exp(2) - 1) / 2)
    ns = np.array([2**k + 1 for k in range(6, 14)])
    slopes = {}
    for rule in ("riemann_left", "trapezoid"):
        errs = []
        for n in ns:
            tt = np.linspace(0.0, 1.0, n)
            errs.append(abs(float(integral_norm(tt, np.exp(tt)[:, None], p=2, rule=rule)) - exact))
        e = np.array(errs)
        m = e > 1e-14
        slopes[rule] = np.polyfit(np.log(ns[m]), np.log(e[m]), 1)[0]
    assert slopes["riemann_left"] == pytest.approx(-1.0, abs=0.05)
    assert slopes["trapezoid"] == pytest.approx(-2.0, abs=0.05)


def test_trapezoid_on_smooth_periodic_integrand():
    """Euler-Maclaurin: boundary terms cancel, so error is beyond all orders."""
    for n in (17, 33, 65):
        t = np.linspace(0.0, 1.0, n)
        x = np.sin(2 * np.pi * 3 * t)[:, None]
        assert abs(float(integral_norm(t, x, p=2)) - 1 / np.sqrt(2)) < 1e-13


def test_bad_time_grids():
    with pytest.raises(ValueError):
        quadrature_weights(np.array([0.0, 1.0, 0.5]))       # not increasing
    with pytest.raises(ValueError):
        quadrature_weights(np.array([0.0]))                  # too short


def test_first_romberg_column_equals_composite_simpson():
    t = np.linspace(0.0, 1.0, 17)
    values = np.exp(t)
    table = romberg_table(t, values)
    simpson = float(np.sum(quadrature_weights(t, "simpson") * values))
    assert table[-1][1] == pytest.approx(simpson, abs=1e-15)


def test_romberg_integrates_quartic_exactly():
    t = np.linspace(0.0, 1.0, 17)
    assert romberg_table(t, t**4)[-1][-1] == pytest.approx(1.0 / 5.0, abs=1e-14)


def test_romberg_rejects_non_nested_grid():
    with pytest.raises(ValueError, match=r"2\*\*m"):
        romberg_table(np.linspace(0.0, 1.0, 10), np.ones(10))


# --- L^p norms ------------------------------------------------------------

def test_sampled_vs_adaptive_quadrature():
    f = lambda s: np.sin(2 * np.pi * 3 * s)
    t, x = smooth_test_path(n=2049, T=1.0, freq=3.0)
    for p in (1.0, 2.0, 4.0):
        sampled = float(integral_norm(t, x, p=p))
        adaptive = integral_norm_callable(f, 0.0, 1.0, p=p)
        assert sampled == pytest.approx(adaptive, rel=1e-6)


def test_gauss_legendre_and_the_kink():
    """Why a higher-order rule is not automatically better.

    Gauss-Legendre is exponentially accurate on analytic integrands. For odd p
    the absolute value puts a kink at every zero of f, and the rate collapses.
    This is the claim in notebook 01 section 0.1 that g = |f|^p, not f, is what
    the quadrature sees.
    """
    f = lambda s: np.sin(2 * np.pi * 3 * s)
    for p in (2.0, 4.0):                       # even p: |f|^p = f^p is analytic
        adaptive = integral_norm_callable(f, 0.0, 1.0, p=p)
        assert integral_norm_gauss(f, 0.0, 1.0, p=p, n=80) == pytest.approx(adaptive, rel=1e-8)

    adaptive = integral_norm_callable(f, 0.0, 1.0, p=1.0)
    gauss = integral_norm_gauss(f, 0.0, 1.0, p=1.0, n=80)
    assert gauss != pytest.approx(adaptive, rel=1e-6)     # no longer spectral
    assert gauss == pytest.approx(adaptive, rel=1e-2)     # but not wrong either


def test_sup_norm():
    n = 4097
    t, x = smooth_test_path(n=n, T=1.0, freq=1.0)
    # The grid misses the peak by at most h/2, and near the peak
    # f = 1 - (2 pi)^2 d^2 / 2, so the shortfall is at most (2 pi h / 2)^2 / 2.
    h = 1.0 / (n - 1)
    bound = (2 * np.pi * h / 2) ** 2 / 2
    assert 1.0 - float(integral_norm(t, x, p=np.inf)) < bound


def test_lp_monotonicity_in_p():
    """On a probability space (|[0,1]| = 1), ||f||_p is nondecreasing in p."""
    t, x = smooth_test_path(n=2049, T=1.0, freq=2.0)
    vals = [float(integral_norm(t, x, p=p)) for p in (1, 2, 3, 4, 8)]
    assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


def test_large_p():
    t = np.linspace(0, 1, 501)
    x = (1e6 * np.ones_like(t))[:, None]
    assert np.isfinite(integral_norm(t, x, p=50.0))


def test_norm_zero_and_scaling():
    t = np.linspace(0, 2, 257)
    x = np.cos(t)[:, None]
    assert float(integral_norm(t, np.zeros_like(x), p=2)) == pytest.approx(0.0)
    a, b = float(integral_norm(t, x, p=2)), float(integral_norm(t, 3 * x, p=2))
    assert b == pytest.approx(3 * a)


def test_batching():
    t = np.linspace(0, 1, 65)
    x = np.stack([np.sin(2 * np.pi * k * t) for k in (1, 2, 3)])[..., None]  # (3, 65, 1)
    out = integral_norm(t, x, p=2)
    assert out.shape == (3,)
    assert np.allclose(out, 1 / np.sqrt(2), atol=1e-6)


# --- the claim the project rests on ---------------------------------------

def test_mse_vs_integral_norm_on_uniform_grid():
    """On a uniform grid they coincide (up to endpoint half-weights)."""
    rng = np.random.default_rng(2)
    t = np.linspace(0, 1, 2001)
    x = rng.normal(size=(2001, 1))
    y = x + 0.1 * rng.normal(size=(2001, 1))
    l2 = float(integral_distance(t, x, y, p=2, normalise=True))
    mse = float(pointwise_mse(x, y))
    # The two rules differ only in the endpoint half-weights, a relative
    # difference of order 1/n. Allow a factor 4 for the sampling of x, y.
    assert abs(l2**2 - mse) / mse < 4 / t.size


def test_limits_under_non_uniform_sampling():
    """Proposition 1 and 2 of notebook 01 section 3, checked at 2^20 points.

    rho(t) = (1/a) t^{1/a - 1} on [0,1], realised by the quantile grid t = u^a.
    With f(t) = 0.4 t both limits are closed-form:

        (1/T) int f^2      = 0.16 / 3
        int f^2 rho        = 0.16 * (1/a) / (1/a + 2)
    """
    a = 0.35
    f = lambda t: 0.4 * t
    lebesgue = 0.16 / 3
    density = 0.16 * (1 / a) / ((1 / a) + 2)

    n = 2**20 + 1
    t = np.linspace(0, 1, n) ** a                      # quantile grid for rho
    z = np.zeros((n, 1))
    quad = float(integral_distance(t, z, f(t)[:, None], p=2, normalise=True)) ** 2
    mse = float(np.mean(f(t) ** 2))

    assert quad == pytest.approx(lebesgue, rel=1e-4)   # Proposition 1
    assert mse == pytest.approx(density, rel=1e-4)     # Proposition 2
    assert abs(mse - lebesgue) > 0.7 * lebesgue        # and they do not agree

    # Left Riemann is first order and crude, but its weights are the spacings,
    # so it is consistent here and MSE is not. The divide is whether the weights
    # see the grid, not how accurate they are.
    w = quadrature_weights(t, "riemann_left")
    assert float(np.sum(w * f(t) ** 2)) == pytest.approx(lebesgue, rel=1e-4)
