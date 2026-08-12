"""Correctness checks for pvar.py.

The three implementations of the same supremum check each other, anchored by
`p_variation_brute`, which enumerates every subsequence and so has nothing in it
that could be subtly wrong. Anchoring matters: before this file existed, every
p-variation test compared my code against my other code, or against a monotone
path, where the maximising subsequence is the first candidate tried and a bug in
the search would not show.

Tolerances are exact equality up to floating point, since all three compute the
same finite sum in a different order.
"""

import numpy as np
import pytest

from pathloss.paths import brownian_motion
from pathloss.pvar import (
    p_variation_brute,
    p_variation_dyadic,
    p_variation_exact,
    p_variation_pruned,
)

PS = (1.0, 1.5, 2.0, 3.0)


def _l1(a, b):
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))


def _zigzag(n, amp=1.0, drift=0.0):
    """Alternating up/down steps: the case where every point earns its keep."""
    steps = amp * (-1.0) ** np.arange(n - 1) + drift
    return np.concatenate([[0.0], np.cumsum(steps)])[:, None]


# --- anchored against enumeration ------------------------------------------

@pytest.mark.parametrize("d", [1, 2, 3])
def test_exact_against_enumeration(d):
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(2, 13))
        x = rng.normal(size=(n, d))
        for p in PS:
            assert p_variation_exact(x, p) == pytest.approx(p_variation_brute(x, p))


@pytest.mark.parametrize("d", [1, 2, 3])
def test_pruned_against_enumeration(d):
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = int(rng.integers(2, 13))
        x = rng.normal(size=(n, d))
        for p in PS:
            assert p_variation_pruned(x, p) == pytest.approx(p_variation_brute(x, p))


def test_enumeration_against_hand_computation():
    """One case small enough to check on paper.

    Path 0 -> 3 -> 1 in R. Three candidate subsequences:
        {0,2}    : 1^p
        {0,1,2}  : 3^p + 2^p
    The second wins for every p >= 1, so V_p = (3^p + 2^p)^(1/p).
    """
    x = np.array([0.0, 3.0, 1.0])
    for p in PS:
        expect = (3.0**p + 2.0**p) ** (1.0 / p)
        assert p_variation_brute(x, p) == pytest.approx(expect)
        assert p_variation_exact(x, p) == pytest.approx(expect)
        assert p_variation_pruned(x, p) == pytest.approx(expect)


# --- pruned against exact, on inputs too large to enumerate ----------------

@pytest.mark.parametrize("d", [1, 2, 3])
def test_pruned_against_exact_on_random_paths(d):
    rng = np.random.default_rng(2)
    for _ in range(5):
        x = rng.normal(size=(300, d))
        for p in PS:
            assert p_variation_pruned(x, p) == pytest.approx(p_variation_exact(x, p))


def test_pruned_against_exact_on_brownian():
    for seed in (3, 5, 8):
        _, w = brownian_motion(n=513, rng=seed)
        for p in PS:
            assert p_variation_pruned(w, p) == pytest.approx(p_variation_exact(w, p))


def test_pruned_against_exact_on_monotone_path():
    """The pruning's worst case, where the triangle inequality is an equality.

    Along a monotone path d(a,c) + d(c,b) = d(a,b) exactly, so the block bound
    has no slack and nothing is skipped. Cost degrades to the exact algorithm's;
    the answer must not.
    """
    x = np.linspace(0.0, 5.0, 400)[:, None]
    for p in PS:
        assert p_variation_pruned(x, p) == pytest.approx(p_variation_exact(x, p))


def test_pruned_against_exact_on_zigzag():
    """The opposite extreme: every point is a reversal."""
    x = _zigzag(400)
    for p in PS:
        assert p_variation_pruned(x, p) == pytest.approx(p_variation_exact(x, p))


def test_pruned_against_exact_on_repeated_points():
    """Zero distances, so delta can equal a bound and ties must break the same way."""
    rng = np.random.default_rng(4)
    x = np.repeat(rng.normal(size=(40, 2)), 3, axis=0)
    for p in PS:
        assert p_variation_pruned(x, p) == pytest.approx(p_variation_exact(x, p))


@pytest.mark.parametrize("n", list(range(2, 40)))
def test_pruned_against_exact_at_every_short_length(n):
    """Sweeps every length, since the index arithmetic is where off-by-ones live.

    Block sizes, the flat radius array and the centre-outside guard all depend
    on the bit pattern of n-1, so lengths either side of a power of two exercise
    different branches.
    """
    rng = np.random.default_rng(100 + n)
    x = rng.normal(size=(n, 2))
    for p in (1.0, 2.0, 3.0):
        assert p_variation_pruned(x, p) == pytest.approx(p_variation_exact(x, p))


# --- a supplied metric -----------------------------------------------------

def test_custom_metric_agrees_across_implementations():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(11, 2))
    for p in PS:
        ref = p_variation_brute(x, p, dist=_l1)
        assert p_variation_exact(x, p, dist=_l1) == pytest.approx(ref)
        assert p_variation_pruned(x, p, dist=_l1) == pytest.approx(ref)


def test_custom_metric_differs_from_euclidean():
    """Guards against `dist` being accepted and then ignored."""
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert p_variation_pruned(x, 2.0, dist=_l1) == pytest.approx(2.0)
    assert p_variation_pruned(x, 2.0) == pytest.approx(np.sqrt(2.0))


# --- the maximising subsequence --------------------------------------------

@pytest.mark.parametrize("fn", [p_variation_exact, p_variation_pruned])
def test_reported_subsequence_achieves_the_reported_value(fn):
    """The links are bookkeeping alongside the value, so they can drift from it."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        x = rng.normal(size=(120, 2))
        for p in (1.0, 2.0, 3.0):
            value, pts = fn(x, p, return_points=True)
            assert pts[0] == 0 and pts[-1] == x.shape[0] - 1
            assert all(b > a for a, b in zip(pts, pts[1:]))
            achieved = sum(
                float(np.linalg.norm(x[b] - x[a])) ** p
                for a, b in zip(pts, pts[1:])
            ) ** (1.0 / p)
            assert achieved == pytest.approx(value)


# --- properties that hold for any correct implementation -------------------

def test_monotone_path_is_its_displacement():
    """No intermediate point helps: splitting one hop into two can only lose."""
    x = np.linspace(0.0, 5.0, 200)[:, None]
    for p in PS:
        assert p_variation_exact(x, p) == pytest.approx(5.0)
        assert p_variation_pruned(x, p) == pytest.approx(5.0)
        assert p_variation_dyadic(x, p) == pytest.approx(5.0)


def test_zigzag_at_p_equals_one_is_total_distance():
    """At p = 1 every point earns its keep, so the answer is the walked length."""
    n, amp = 51, 0.7
    x = _zigzag(n, amp=amp)
    assert p_variation_exact(x, 1.0) == pytest.approx(amp * (n - 1))


def test_decreasing_in_p():
    _, w = brownian_motion(n=257, rng=11)
    for fn in (p_variation_exact, p_variation_pruned):
        vals = [fn(w, p) for p in (1.0, 1.5, 2.0, 3.0)]
        assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))


def test_dyadic_is_a_weaker_lower_bound():
    _, w = brownian_motion(n=513, rng=3)
    for p in PS:
        assert p_variation_exact(w, p) >= p_variation_dyadic(w, p) - 1e-9


def test_brownian_1_variation_grows_under_refinement():
    """Brownian motion is a.s. of unbounded variation, so the estimate must climb."""
    _, w = brownian_motion(n=2**12 + 1, rng=13)
    assert p_variation_dyadic(w, 1.0) > 1.8 * p_variation_dyadic(w[::8], 1.0)


def test_translation_invariance_and_scaling():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(150, 2))
    for p in (1.0, 2.0):
        base = p_variation_pruned(x, p)
        assert p_variation_pruned(x + 17.0, p) == pytest.approx(base)
        assert p_variation_pruned(3.0 * x, p) == pytest.approx(3.0 * base)


def test_reparametrisation_invariance():
    """Repeating points inserts zero increments, which change no sum."""
    rng = np.random.default_rng(10)
    x = rng.normal(size=(60, 2))
    slowed = np.repeat(x, 2, axis=0)
    for p in (1.0, 2.0, 3.0):
        assert p_variation_pruned(slowed, p) == pytest.approx(p_variation_pruned(x, p))


# --- degenerate inputs -----------------------------------------------------

def test_single_point_and_single_step():
    for fn in (p_variation_brute, p_variation_exact, p_variation_pruned):
        assert fn(np.array([[1.0, 2.0]]), 2.0) == pytest.approx(0.0)
        assert fn(np.array([[0.0], [3.0]]), 2.0) == pytest.approx(3.0)


def test_constant_path_is_zero():
    x = np.full((50, 3), 2.5)
    for fn in (p_variation_exact, p_variation_pruned, p_variation_dyadic):
        assert fn(x, 2.0) == pytest.approx(0.0)


def test_one_dimensional_input_is_accepted():
    rng = np.random.default_rng(12)
    flat = rng.normal(size=30)
    for fn in (p_variation_exact, p_variation_pruned):
        assert fn(flat, 2.0) == pytest.approx(fn(flat[:, None], 2.0))


def test_rejects_p_below_one():
    x = np.linspace(0, 1, 10)
    for fn in (p_variation_brute, p_variation_exact, p_variation_pruned):
        with pytest.raises(ValueError):
            fn(x, 0.5)


def test_brute_force_refuses_long_inputs():
    with pytest.raises(ValueError):
        p_variation_brute(np.linspace(0, 1, 40), 2.0)
