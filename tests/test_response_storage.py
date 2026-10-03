"""Independent anchors for exact Euler minimax and integer storage search."""

import math
from decimal import Decimal, localcontext
from itertools import product

import numpy as np
import pytest
from scipy.linalg import expm

from pathloss.response_storage import (
    engel_scales,
    lie_storage_dimension,
    log_fpl_euler_bound,
    log_scalar_euler_error,
    minimum_intervals,
    optimize_euler_storage,
)


@pytest.mark.parametrize("order,intervals,scale", [(1, 3, 1), (4, 17, 2), (2, 1, 4)])
def test_scalar_error_matches_decimal_closed_form(order, intervals, scale):
    with localcontext() as context:
        context.prec = 90
        local = Decimal(scale) / intervals
        polynomial = sum(local**j / math.factorial(j) for j in range(order + 1))
        exact = Decimal(scale).exp() - polynomial**intervals
        assert log_scalar_euler_error(order, intervals, scale) == pytest.approx(
            float(exact.ln()), abs=2e-13
        )


def test_underflowed_tail_preserves_log_error():
    # Relative tail is 1+O(1e-10), anchored by its first omitted term.
    order, intervals, scale = 200, 10**100, 1.0
    expected = 1 - order * math.log(intervals) - math.lgamma(order + 2)
    assert log_scalar_euler_error(order, intervals, scale) == pytest.approx(expected)


@pytest.mark.parametrize("criterion", ["exact", "fpl"])
def test_enormous_interval_boundary_against_decimal_closed_form(criterion):
    scale, epsilon = 30.0, 1e-12
    intervals = minimum_intervals(1, scale, epsilon, criterion=criterion)
    assert intervals > 2**53
    with localcontext() as context:
        context.prec = 160
        z = Decimal.from_float(scale)
        threshold = Decimal.from_float(epsilon)

        def error(m):
            if criterion == "fpl":
                return z.exp() * z**2 / (2 * m)
            local = z / m
            return z.exp() - (Decimal(m) * (1 + local).ln()).exp()

        assert error(intervals) <= threshold
        assert error(intervals - 1) > threshold


def test_lie_dimensions_match_lyndon_word_enumeration():
    for channel, depth in ((2, 6), (3, 4)):
        count = 0
        for level in range(1, depth + 1):
            for word in product(range(channel), repeat=level):
                if all(word < word[shift:] + word[:shift] for shift in range(1, level)):
                    count += 1
        assert lie_storage_dimension(channel, depth) == count


@pytest.mark.parametrize("criterion", ["exact", "fpl"])
def test_integer_optimizer_matches_exhaustive_grid(criterion):
    error = log_scalar_euler_error if criterion == "exact" else log_fpl_euler_bound
    best, rows = optimize_euler_storage(2, 0.8, 0.04, criterion=criterion)
    feasible = [
        (m * lie_storage_dimension(2, n), n, m)
        for n in range(1, 11)
        for m in range(1, 101)
        if error(n, m, 0.8) <= math.log(0.04)
    ]
    assert (best.storage, best.depth, best.intervals) == min(feasible)
    assert lie_storage_dimension(2, rows[-1].depth + 1) > best.storage


def test_interval_search_is_minimal_and_exact_error_is_below_fpl():
    for order, scale in product((1, 2, 5, 15), (0.1, 1.0, 10.0)):
        intervals = minimum_intervals(order, scale, 1e-7)
        exact = log_scalar_euler_error(order, intervals, scale)
        bound = log_fpl_euler_bound(order, intervals, scale)
        assert exact <= bound + 1e-12
        assert exact <= math.log(1e-7)
        if intervals > 1:
            assert log_scalar_euler_error(order, intervals - 1, scale) > math.log(1e-7)


def test_noncommuting_piecewise_linear_response_obeys_exact_majorant():
    # Each segment's response is known exactly from its matrix exponential.
    rng = np.random.default_rng(812)
    for _ in range(15):
        matrices = rng.normal(size=(2, 3, 3))
        matrices /= np.linalg.norm(matrices)
        increments = rng.normal(size=(3, 2))
        lengths = np.linalg.norm(increments, axis=1)
        fine = approximate = np.eye(3)
        for increment in increments:
            generator = np.einsum("i,ijk->jk", increment, matrices)
            fine = expm(generator) @ fine
            approximate = (np.eye(3) + generator) @ approximate
        upper = math.exp(sum(lengths)) - math.prod(1 + length for length in lengths)
        assert np.linalg.norm(fine - approximate, ord=2) <= upper + 1e-12


@pytest.mark.parametrize("degree", range(2, 9))
def test_engel_positive_eigenvalue_witness(degree):
    _, gauge, eigenvalue = engel_scales(degree)
    a = math.sqrt((degree - 1) / degree)
    b = 1 / math.sqrt(degree)
    first = np.diag([a, -a]) / math.sqrt(2)
    second = b * np.array([[0, 1], [(-1) ** (degree - 1), 0]]) / math.sqrt(2)
    result = second
    for _ in range(degree - 1):
        result = first @ result - result @ first
    assert np.linalg.norm(result, ord="fro") == pytest.approx(gauge)
    assert np.linalg.eigvalsh(result)[-1] == pytest.approx(eigenvalue)
    assert np.linalg.norm(first) ** 2 + np.linalg.norm(second) ** 2 == pytest.approx(1)


def test_zero_and_invalid_inputs():
    assert log_scalar_euler_error(1, 1, 0) == -math.inf
    for order, intervals, scale in ((0, 1, 1), (1, 0, 1), (1, 1, -1)):
        with pytest.raises(ValueError):
            log_scalar_euler_error(order, intervals, scale)
    with pytest.raises(ValueError):
        optimize_euler_storage(1, 1, 0.01)
    with pytest.raises(ValueError):
        minimum_intervals(1, 1, 0.01, criterion="unknown")
