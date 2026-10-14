"""Independent anchors for robust planar log-ODE response bounds."""

import math
from decimal import Decimal, localcontext
from itertools import pairwise

import numpy as np
import pytest
from scipy.linalg import expm

from response_distance.code.response import piecewise_linear_signature
from response_distance.code.response_mixed import (
    RHO3,
    planar_bv_logode_data,
    planar_bv_logode_neighborhood,
    planar_cubic_attaining_tuple,
    planar_merge_bounds,
    planar_merge_separation_radius,
    planar_mixed_merge_data,
    planar_mixed_merge_neighborhood,
    quadratic_exponential_tail4,
)


def bracket(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.kron(left, right) - np.kron(right, left)


X, Y = np.eye(2)
Z = bracket(X, Y)
E3, F3 = bracket(X, Z), bracket(Y, Z)


def local_logs(path: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tensor logarithm from Chen signature, independent of BCH recursion."""
    _, s1, s2, s3 = piecewise_linear_signature(path, 3)
    return (
        s1,
        s2 - np.kron(s1, s1) / 2,
        s3 - (np.kron(s1, s2) + np.kron(s2, s1)) / 2 + np.kron(np.kron(s1, s1), s1) / 3,
    )


def multiply(left: list[np.ndarray], right: list[np.ndarray]) -> list[np.ndarray]:
    return [
        sum(
            (np.kron(left[index], right[level - index]) for index in range(level + 1)),
            np.zeros(2**level),
        )
        for level in range(len(left))
    ]


def tensor_exp(v: np.ndarray, u: np.ndarray, depth: int) -> list[np.ndarray]:
    """Direct finite tensor-algebra exponential using powers, not BCH."""
    empty = lambda: [np.zeros(2**level) for level in range(depth + 1)]
    log = empty()
    log[1], log[2] = v, u
    power = empty()
    power[0][0] = 1.0
    result = [value.copy() for value in power]
    for index in range(1, depth + 1):
        power = multiply(power, log)
        for level in range(depth + 1):
            result[level] += power[level] / math.factorial(index)
    return result


def decoder_series(
    path: np.ndarray, cuts: tuple[int, ...], depth: int
) -> list[np.ndarray]:
    result = [np.zeros(2**level) for level in range(depth + 1)]
    result[0][0] = 1.0
    for left, right in pairwise(cuts):
        v, u, _ = local_logs(path[left : right + 1])
        result = multiply(result, tensor_exp(v, u, depth))
    return result


def bv_matrix_error(
    path: np.ndarray, cuts: tuple[int, ...], matrices: np.ndarray, scale: float
) -> float:
    fine = np.eye(matrices.shape[1])
    for step in np.diff(path, axis=0):
        fine = fine @ expm(scale * np.einsum("i,ijk->jk", step, matrices))
    coarse = np.eye(matrices.shape[1])
    commutator = matrices[0] @ matrices[1] - matrices[1] @ matrices[0]
    for left, right in pairwise(cuts):
        v, u, _ = local_logs(path[left : right + 1])
        coarse = coarse @ expm(
            scale * np.einsum("i,ijk->jk", v, matrices) + scale**2 * u[1] * commutator
        )
    return float(np.linalg.norm(fine - coarse, 2))


@pytest.mark.parametrize(
    "a,b,t",
    [
        (2.0, 0.3, 1e-8),
        (0.0, 1.0, 1e-8),
        (1.0, 0.0, 0.1),
        (0.0, 0.0, 0.7),
        (2.0, 0.4, 0.9),
    ],
)
def test_positive_tail_matches_high_precision(a: float, b: float, t: float) -> None:
    with localcontext() as context:
        context.prec = 90
        da, db, dt = map(Decimal.from_float, (a, b, t))
        p, q = da * dt, db * dt**2
        exact = (p + q).exp() - 1 - p - (p**2 / 2 + q) - (p**3 / 6 + p * q)
    assert quadratic_exponential_tail4(a, b, t) == pytest.approx(
        float(exact), rel=1e-12, abs=0.0
    )


def test_bv_cubic_coefficients_match_tensor_logarithm() -> None:
    rng = np.random.default_rng(211)
    for _ in range(12):
        path = np.cumsum(rng.normal(size=(6, 2)), axis=0)
        path -= path[0]
        for cuts in ((0, 5), (0, 2, 5)):
            data = planar_bv_logode_data(path, cuts)
            expected = sum(
                (local_logs(path[l : r + 1])[2] for l, r in pairwise(cuts)),
                np.zeros(8),
            )
            assert data.cubic[0] * E3 + data.cubic[1] * F3 == pytest.approx(
                expected, abs=1e-12
            )


def test_two_segment_cubic_and_sharp_matrix_witness() -> None:
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    data = planar_bv_logode_data(path)
    assert data.cubic == pytest.approx((1 / 12, -1 / 12))
    assert data.leading_hs == pytest.approx(math.sqrt(3) / 6)
    matrices = planar_cubic_attaining_tuple(data.cubic)
    assert np.linalg.norm(matrices) == pytest.approx(1.0)
    z = matrices[0] @ matrices[1] - matrices[1] @ matrices[0]
    w = np.einsum("i,ijk->jk", data.cubic, matrices)
    assert np.linalg.norm(w @ z - z @ w, 2) == pytest.approx(data.leading_response)


@pytest.mark.parametrize("cuts", [(0, 4), (0, 2, 4)])
def test_partitioned_full_response_and_certificate_bounds(
    cuts: tuple[int, ...],
) -> None:
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]])
    data = planar_bv_logode_data(path, cuts)
    actual = piecewise_linear_signature(path, 8)
    decoded = decoder_series(path, cuts, 8)
    assert decoded[1] == pytest.approx(actual[1])
    assert decoded[2] == pytest.approx(actual[2])
    assert actual[3] - decoded[3] == pytest.approx(
        data.cubic[0] * E3 + data.cubic[1] * F3
    )
    radius = planar_merge_separation_radius(data)
    assert radius > 0.0
    witness = planar_cubic_attaining_tuple(data.cubic)
    rng = np.random.default_rng(212)
    for scale in (radius / 2, radius, 0.2):
        bounds = planar_merge_bounds(data, scale)
        partial_hs = sum(
            scale**level * np.linalg.norm(actual[level] - decoded[level])
            for level in range(3, 9)
        )
        assert bounds.certificate_lower <= partial_hs * (1 + 1e-12)
        assert partial_hs <= bounds.certificate_upper
        error = bv_matrix_error(path, cuts, witness, scale)
        assert bounds.response_lower <= error + 1e-14
        assert error <= bounds.response_upper + 1e-14
        for dimension in (2, 3):
            for _ in range(4):
                matrices = rng.normal(size=(2, dimension, dimension))
                matrices /= np.linalg.norm(matrices)
                assert (
                    bv_matrix_error(path, cuts, matrices, scale)
                    <= bounds.response_upper + 1e-14
                )
        if scale <= radius:
            assert bounds.response_upper <= (
                1 + RHO3
            ) / 2 * bounds.certificate_lower * (1 + 1e-12)
            assert bounds.response_upper < bounds.certificate_lower


def test_mixed_cubic_matches_direct_exponential_coefficients() -> None:
    left, right = np.array([1.0, 0.2]), np.array([-0.3, 0.7])
    alpha, beta = 0.4, -0.2
    data = planar_mixed_merge_data(left, right, alpha, beta)
    cross = np.linalg.det(np.stack([left, right]))
    fine = multiply(tensor_exp(left, alpha * Z, 4), tensor_exp(right, beta * Z, 4))
    coarse = tensor_exp(left + right, (alpha + beta + cross / 2) * Z, 4)
    assert fine[1] == pytest.approx(coarse[1])
    assert fine[2] == pytest.approx(coarse[2])
    assert fine[3] - coarse[3] == pytest.approx(data.cubic[0] * E3 + data.cubic[1] * F3)
    radius = planar_merge_separation_radius(data)
    bounds = planar_merge_bounds(data, radius)
    assert bounds.response_upper < bounds.certificate_lower


def test_degenerate_cubic_cases_are_excluded() -> None:
    straight = planar_bv_logode_data(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    balanced = planar_mixed_merge_data(X, Y, -1 / 6, -1 / 6)
    pure_area = planar_mixed_merge_data(np.zeros(2), np.zeros(2), 1.0, 2.0)
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    cancellation = planar_bv_logode_data(square, (0, 2, 4))
    for data in (straight, balanced, pure_area, cancellation):
        assert data.leading_hs == 0.0
        assert planar_merge_separation_radius(data) == 0.0


def test_cubic_cancellation_can_leave_nonzero_higher_order_response() -> None:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    cuts = (0, 2, 4)
    data = planar_bv_logode_data(square, cuts)
    assert data.leading_hs == 0.0
    assert planar_merge_separation_radius(data) == 0.0
    fine = piecewise_linear_signature(square, 4)
    coarse = decoder_series(square, cuts, 4)
    assert np.linalg.norm(fine[3] - coarse[3]) < 1e-14
    assert np.linalg.norm(fine[4] - coarse[4]) > 1e-3
    witness = np.array(
        [np.diag([0.5, -0.5]), np.array([[1.0, 1.0], [1.0, -1.0]]) / (2 * math.sqrt(2))]
    )
    assert np.linalg.norm(witness) == pytest.approx(1.0)
    assert bv_matrix_error(square, cuts, witness, 0.2) > 1e-7


def test_dilation_covariance_and_subdivision() -> None:
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    base = planar_bv_logode_data(path)
    radius = planar_merge_separation_radius(base)
    midpoint_path = np.array(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]]
    )
    assert planar_bv_logode_data(midpoint_path) == base
    for factor in (1e-8, 1e8):
        scaled = planar_bv_logode_data(factor * path)
        assert scaled.leading_hs == pytest.approx(factor**3 * base.leading_hs)
        assert planar_merge_separation_radius(scaled) == pytest.approx(radius / factor)
        actual, expected = (
            planar_merge_bounds(scaled, radius / factor),
            planar_merge_bounds(base, radius),
        )
        assert actual.response_upper == pytest.approx(expected.response_upper)


def test_bv_neighborhood_keeps_a_uniform_gap() -> None:
    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0], [2.0, 2.0]])
    cuts = (0, 2, 4)
    baseline = planar_bv_logode_data(path, cuts)
    eta, scale = planar_bv_logode_neighborhood(path, cuts)
    assert eta > 0 and scale > 0
    rng = np.random.default_rng(213)
    for _ in range(12):
        increments = rng.normal(size=(4, 2))
        increments *= eta / (2 * np.sum(np.linalg.norm(increments, axis=1)))
        perturbation = np.vstack([np.zeros(2), np.cumsum(increments, axis=0)])
        data = planar_bv_logode_data(path + perturbation, cuts)
        assert data.leading_hs >= baseline.leading_hs / 2
        bounds = planar_merge_bounds(data, scale)
        assert bounds.response_upper < bounds.certificate_lower


def test_mixed_neighborhood_keeps_a_uniform_gap() -> None:
    eta, scale = planar_mixed_merge_neighborhood(X, Y, 0.3, -0.2)
    baseline = planar_mixed_merge_data(X, Y, 0.3, -0.2)
    assert eta > 0 and scale > 0
    rng = np.random.default_rng(214)
    for _ in range(12):
        perturbation = rng.uniform(-eta / 4, eta / 4, 6)
        data = planar_mixed_merge_data(
            X + perturbation[:2],
            Y + perturbation[2:4],
            0.3 + perturbation[4],
            -0.2 + perturbation[5],
        )
        assert data.leading_hs >= baseline.leading_hs / 2
        bounds = planar_merge_bounds(data, scale)
        assert bounds.response_upper < bounds.certificate_lower


@pytest.mark.parametrize("partition", [(0, 0, 2), (1, 2), (0, 3), (0, True, 2)])
def test_invalid_partitions_are_rejected(partition: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="partition"):
        planar_bv_logode_data(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), partition)
