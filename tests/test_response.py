"""Anchors for level-wise response discrepancy.

Each check compares against something computed a different way: a closed form,
a brute-force search over a one-dimensional parameter, or an inequality proved
independently (Ferrucci, Perree and Lyons eq. 1.9, and the shift construction
verified in ``scripts/check_response_separation.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

import pathloss.response as response_module
from pathloss.response import (
    area_block_gram_matrix,
    area_merge_response_gauge_bounds,
    area_response_gauge,
    composed_signature_polynomial,
    composed_truncated_response_sup,
    coordinate_loop_logode_hs_certificate,
    coordinate_loop_logode_merge_sup,
    estimate_area_merge_response_gauges,
    hilbert_schmidt_bound,
    level_response_sup,
    logode_merge_level_hs_norm,
    multi_logode_merge_level_hs_norm,
    orthogonal_logode_hs_certificate,
    piecewise_linear_signature,
    rank_one_bound,
    response_ratio,
    shift_bound,
    tensor_spectral_norm,
    truncated_response_sup,
)


def _coordinate_area_block(channel: int, left: int, right: int) -> np.ndarray:
    block = np.zeros((channel, channel))
    block[left, right] = 1.0
    block[right, left] = -1.0
    return block


def test_area_response_gauge_is_top_skew_singular_value() -> None:
    area = np.array(
        [
            [0.0, 2.0, -0.3, 0.0],
            [-2.0, 0.0, 0.0, 0.4],
            [0.3, 0.0, 0.0, -1.1],
            [0.0, -0.4, 1.1, 0.0],
        ]
    )
    assert area_response_gauge(area) == pytest.approx(
        np.linalg.norm(area, ord=2) / np.sqrt(2.0)
    )


def test_area_chain_has_closed_form_response_gauge_bounds() -> None:
    for block_count in (2, 3, 5, 8):
        channel = block_count + 1
        blocks = np.stack(
            [
                _coordinate_area_block(channel, index, index + 1)
                for index in range(block_count)
            ]
        )
        gram = area_block_gram_matrix(blocks)
        assert gram == pytest.approx(2.0 * np.eye(block_count))
        growth, commutator = area_merge_response_gauge_bounds(blocks)
        path_radius = np.cos(np.pi / (block_count + 2.0))
        assert growth == pytest.approx(np.sqrt(2.0) * path_radius)
        assert commutator == pytest.approx(np.sqrt(2.0) * path_radius**2)


def test_area_helpers_reject_nonskew_coefficients() -> None:
    with pytest.raises(ValueError, match="skew-symmetric"):
        area_response_gauge(np.eye(2))
    with pytest.raises(ValueError, match="skew-symmetric"):
        area_response_gauge(1e-16 * np.eye(2))


def test_area_commutator_bound_preserves_small_partner() -> None:
    """Positive pair products must survive disparate block magnitudes."""
    small = 1e-18
    blocks = np.stack(
        [
            _coordinate_area_block(3, 0, 1),
            small * _coordinate_area_block(3, 1, 2),
        ]
    )
    _, bound = area_merge_response_gauge_bounds(blocks)
    assert bound == pytest.approx(small / math.sqrt(2), rel=1e-14, abs=0.0)
    h = np.diag([1.0, -1.0]) / math.sqrt(2)
    upper = np.array([[0.0, 1.0], [0.0, 0.0]])
    lower = upper.T
    a, b, c = upper / 2, h / math.sqrt(2), lower / 2
    x, y = a @ b - b @ a, small * (b @ c - c @ b)
    assert 0.0 < np.linalg.norm(x @ y - y @ x, 2) <= bound


@pytest.mark.parametrize(
    "objective",
    [
        response_module._area_growth_objective_and_gradient,
        response_module._area_commutator_objective_and_gradient,
    ],
)
def test_area_gauge_gradients_match_finite_differences(
    objective: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]],
) -> None:
    rng = np.random.default_rng(73)
    raw = rng.standard_normal((3, 3, 3))
    blocks = 0.5 * (raw - raw.transpose(0, 2, 1))
    response = rng.standard_normal((3, 2, 2))
    response /= np.linalg.norm(response)
    value, gradient = objective(response, blocks)
    assert value > 0.0
    epsilon = 1.0e-6
    finite_difference = np.zeros_like(response)
    for index in np.ndindex(response.shape):
        perturbation = np.zeros_like(response)
        perturbation[index] = epsilon
        plus = objective(response + perturbation, blocks)[0]
        minus = objective(response - perturbation, blocks)[0]
        finite_difference[index] = (plus - minus) / (2.0 * epsilon)
    assert gradient == pytest.approx(finite_difference, rel=3e-6, abs=3e-7)


def test_area_gauge_estimates_stay_below_spectral_bounds() -> None:
    blocks = np.stack(
        [
            _coordinate_area_block(4, 0, 1),
            _coordinate_area_block(4, 1, 2),
            _coordinate_area_block(4, 2, 3),
        ]
    )
    bounds = area_merge_response_gauge_bounds(blocks)
    estimates = estimate_area_merge_response_gauges(
        blocks, 2, restarts=8, steps=300, seed=79
    )
    assert estimates[0] <= bounds[0] * (1.0 + 1e-10)
    assert estimates[1] <= bounds[1] * (1.0 + 1e-10)
    assert estimates[0] > 0.8 * bounds[0]
    assert estimates[1] > 0.1 * bounds[1]


def _tensor_bracket(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Tensor-algebra commutator of flattened homogeneous tensors."""
    return np.outer(left, right).reshape(-1) - np.outer(right, left).reshape(-1)


def test_degree_six_identity_separates_response_dimensions() -> None:
    """Cayley--Hamilton kills [ad_x^3(y), [x,y]] at n=2; n=3 sees it."""
    x, y = np.eye(2)
    area = _tensor_bracket(x, y)
    engel4 = _tensor_bracket(x, _tensor_bracket(x, area))
    identity = _tensor_bracket(engel4, area).reshape((2,) * 6)
    rng = np.random.default_rng(211)
    for _ in range(12):
        matrices = rng.standard_normal((2, 2, 2))
        value, _ = response_module._polynomial_matrix_and_gradient(
            matrices, (identity,)
        )
        assert value == pytest.approx(np.zeros((2, 2)), abs=1e-11)

    matrices = np.zeros((2, 3, 3))
    matrices[0] = np.diag([1.0, 0.0, -2.0])
    matrices[1, 0, 1] = matrices[1, 1, 2] = 1.0
    value, _ = response_module._polynomial_matrix_and_gradient(matrices, (identity,))
    expected = np.zeros((3, 3))
    expected[0, 2] = -6.0
    assert np.array_equal(value, expected)
    matrices /= np.sqrt(7.0)
    value, _ = response_module._polynomial_matrix_and_gradient(matrices, (identity,))
    assert np.linalg.norm(value, 2) == pytest.approx(6.0 / 343.0)


@pytest.mark.parametrize("scale", [2.0**-26, 1.0, 2.0**26])
@pytest.mark.parametrize("correlation", [0.0, 1.0 - 1e-13])
def test_gram_merge_norm_preserves_small_positive_discrepancies(
    scale: float, correlation: float
) -> None:
    """D_2 = (U tensor V - V tensor U)/2 has a closed wedge norm."""
    expected = scale**2 * math.sqrt((1.0 - correlation) * (1.0 + correlation) / 2.0)
    value = logode_merge_level_hs_norm(scale, scale, correlation * scale**2, 2)
    assert value > 0.0
    assert value == pytest.approx(expected, rel=1e-12, abs=0.0)
    gram = scale**2 * np.array([[1.0, correlation], [correlation, 1.0]])
    assert multi_logode_merge_level_hs_norm(gram, 2) == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_small_indefinite_gram_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        multi_logode_merge_level_hs_norm(1e-16 * np.array([[1.0, 2.0], [2.0, 1.0]]), 2)
    with pytest.raises(ValueError, match="positive semidefiniteness"):
        logode_merge_level_hs_norm(1e-8, 1e-8, 2e-16, 2)


def test_orthogonal_certificate_controls_omitted_tail() -> None:
    """A loose stopping tolerance must still exceed an independent partial sum."""
    left, right = 0.3, 1.2
    reference = sum(
        math.sqrt(
            sum(
                math.comb(level, count)
                * (math.comb(level, count) - 1)
                * left ** (2 * count)
                * right ** (2 * (level - count))
                for count in range(1, level)
            )
        )
        / math.factorial(level)
        for level in range(2, 40)
    )
    value = orthogonal_logode_hs_certificate(left, right, tolerance=1e-4)
    assert reference <= value <= reference + 1e-4


def test_level_one_attains_hilbert_schmidt_bound() -> None:
    """At level one the map is linear in A, so the supremum is ``||D||_2``."""
    rng = np.random.default_rng(0)
    delta = rng.standard_normal(3)
    for response_dim in (1, 2, 4):
        value = level_response_sup(delta, 3, 1, response_dim, restarts=4, steps=200)
        assert value == pytest.approx(hilbert_schmidt_bound(delta), rel=1e-6)


def test_scalar_responses_match_brute_force_symmetric_form() -> None:
    """Response dimension one reduces to a symmetric form on the circle."""
    rng = np.random.default_rng(1)
    delta = rng.standard_normal(4)  # d = 2, level 2
    tensor = delta.reshape(2, 2)
    angles = np.linspace(0.0, 2.0 * np.pi, 20001)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    brute = np.max(np.abs(np.einsum("ti,ij,tj->t", directions, tensor, directions)))
    value = level_response_sup(delta, 2, 2, 1, restarts=8, steps=300)
    assert value == pytest.approx(brute, rel=1e-4)


def test_supremum_is_homogeneous_in_the_discrepancy() -> None:
    rng = np.random.default_rng(2)
    delta = rng.standard_normal(8)  # d = 2, level 3
    base = level_response_sup(delta, 2, 3, 3, restarts=6, steps=300, seed=3)
    scaled = level_response_sup(2.5 * delta, 2, 3, 3, restarts=6, steps=300, seed=3)
    assert scaled == pytest.approx(2.5 * base, rel=1e-6)


@pytest.mark.parametrize("level", [2, 3])
def test_supremum_lies_inside_the_bracket(level: int) -> None:
    rng = np.random.default_rng(4)
    delta = rng.standard_normal(2**level)
    value = level_response_sup(delta, 2, level, level + 1, restarts=10, steps=400)
    assert value <= hilbert_schmidt_bound(delta) * (1.0 + 1e-9)
    assert value >= shift_bound(delta, 2, level) - 1e-9
    assert value >= rank_one_bound(delta, 2, level, restarts=8, steps=300) - 1e-6


def test_level_two_lie_discrepancy_hits_one_half() -> None:
    """Closed form: an antisymmetric level-two difference gives ratio exactly 1/2.

    Such a difference is ``c`` times the bracket, so the objective is
    ``c ||[A_0, A_1]||_op``.  Its maximum subject to
    ``||A_0||_F^2 + ||A_1||_F^2 = 1`` is ``1/sqrt 2``, attained by the two
    explicit matrices in the proof, while ``||D||_2 = c sqrt 2``.
    """
    delta = np.array([0.0, 1.0, -1.0, 0.0])
    assert response_ratio(delta, 2, 2, 2, restarts=12, steps=600) == pytest.approx(
        0.5, rel=1e-6
    )
    assert response_ratio(delta, 2, 2, 1, restarts=6, steps=200) == pytest.approx(
        0.0, abs=1e-12
    )


def test_level_three_planar_lie_constant() -> None:
    """Two sharp commutator bounds give ``rho = 2 sqrt(2) / 9``."""
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    delta = _tensor_bracket(x, _tensor_bracket(x, y))
    expected = 2.0 * np.sqrt(2.0) / 9.0
    for response_dim in (2, 3):
        value = response_ratio(
            delta, 2, 3, response_dim, restarts=16, steps=600, seed=3
        )
        assert value == pytest.approx(expected, rel=2e-5)


def test_level_four_engel_constant() -> None:
    """Three sharp commutator bounds give ``rho = 3 sqrt(30) / 80``."""
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    delta = _tensor_bracket(x, _tensor_bracket(x, _tensor_bracket(x, y)))
    expected = 3.0 * np.sqrt(30.0) / 80.0
    for response_dim in (2, 3):
        value = response_ratio(
            delta, 2, 4, response_dim, restarts=20, steps=800, seed=4
        )
        assert value == pytest.approx(expected, rel=2e-5)


def test_level_four_mixed_lie_constant_at_response_dimension_two() -> None:
    """Exact ``sl(2)`` reduction gives the mixed degree-four radical."""
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    delta = _tensor_bracket(x, _tensor_bracket(y, _tensor_bracket(x, y)))
    expected = np.sqrt((411.0 + 41.0 * np.sqrt(41.0)) / 20480.0)
    value = response_ratio(delta, 2, 4, 2, restarts=30, steps=900, seed=11)
    assert value == pytest.approx(expected, rel=3e-5)


@pytest.mark.parametrize("channel", [2, 3])
def test_isotropic_symmetric_level_two_hits_inverse_root_dimension(
    channel: int,
) -> None:
    """Closed form: ``D = d^{-1/2} sum_i e_i (x) e_i`` gives ratio ``d^{-1/2}``.

    Objective is ``d^{-1/2} || sum_i A_i^2 ||_op``, bounded by
    ``d^{-1/2} sum_i ||A_i||_F^2 <= d^{-1/2}`` and attained by one rank-one
    projection.  Separates sharpness from symmetry alone, since a rank-one
    symmetric tensor gives ratio one.
    """
    delta = (np.eye(channel) / np.sqrt(channel)).reshape(-1)
    value = response_ratio(delta, channel, 2, 3, restarts=14, steps=600)
    assert value == pytest.approx(1.0 / np.sqrt(channel), rel=1e-5)


def test_spectral_norm_matches_brute_force_for_a_matrix() -> None:
    """At level two the injective norm is the largest singular value."""
    rng = np.random.default_rng(5)
    delta = rng.standard_normal(9)  # d = 3, level 2
    expected = np.linalg.svd(delta.reshape(3, 3), compute_uv=False)[0]
    assert tensor_spectral_norm(delta, 3, 2, restarts=6, steps=200) == pytest.approx(
        expected, rel=1e-8
    )


def test_signature_of_straight_line_is_tensor_powers_over_factorial() -> None:
    path = np.array([[0.0, 0.0], [1.0, -2.0]])
    levels = piecewise_linear_signature(path, 3)
    increment = np.array([1.0, -2.0])
    assert levels[1] == pytest.approx(increment)
    assert levels[2] == pytest.approx(np.outer(increment, increment).reshape(-1) / 2.0)
    third = np.einsum("i,j,k->ijk", increment, increment, increment) / 6.0
    assert levels[3] == pytest.approx(third.reshape(-1))


def test_closed_loop_has_zero_increment_and_signed_area() -> None:
    """Unit square traversed anticlockwise has level-two antisymmetric part 1/2."""
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    levels = piecewise_linear_signature(square, 2)
    assert levels[1] == pytest.approx(np.zeros(2), abs=1e-12)
    second = levels[2].reshape(2, 2)
    assert 0.5 * (second[0, 1] - second[1, 0]) == pytest.approx(1.0, rel=1e-12)


def test_ratio_is_zero_for_tree_like_pair() -> None:
    """Appending an out-and-back spur leaves every signature level unchanged."""
    base = np.array([[0.0, 0.0], [1.0, 0.3], [1.4, 1.1]])
    spur = np.array([[1.4, 1.1], [2.0, 1.6], [1.4, 1.1]])
    extended = np.vstack((base, spur[1:]))
    left = piecewise_linear_signature(base, 3)
    right = piecewise_linear_signature(extended, 3)
    for level in range(1, 4):
        delta = left[level] - right[level]
        assert np.max(np.abs(delta)) < 1e-12
        assert response_ratio(delta, 2, level, 3, restarts=2, steps=50) == 0.0


def test_common_response_can_attain_its_maximum_inside_coefficient_ball() -> None:
    """Scalar polynomial ``a^2 - a^4`` has maximum ``1/4`` inside ``[-1, 1]``."""
    deltas = (
        np.zeros(1),
        np.ones(1),
        np.zeros(1),
        -np.ones(1),
    )
    value = truncated_response_sup(deltas, 1, 1, restarts=12, steps=400, seed=17)
    assert value == pytest.approx(0.25, rel=1e-6)


def test_common_response_uses_one_coefficient_across_levels() -> None:
    """One-dimensional response reduces to a polynomial on a closed interval."""
    deltas = (np.array([0.4]), np.array([-0.7]), np.array([0.2]))
    grid = np.linspace(-1.0, 1.0, 200001)
    expected = np.max(np.abs(0.4 * grid - 0.7 * grid**2 + 0.2 * grid**3))
    value = truncated_response_sup(deltas, 1, 1, restarts=10, steps=400, seed=23)
    assert value == pytest.approx(expected, rel=1e-5)


def test_composed_response_matches_scalar_product_brute_force() -> None:
    """Depth-one interval factors give an explicit scalar product."""
    left = (np.array([[0.4], [-0.2]]),)
    right = (np.array([[0.1], [0.3]]),)
    grid = np.linspace(-1.0, 1.0, 200001)
    left_response = (1.0 + 0.4 * grid) * (1.0 - 0.2 * grid)
    right_response = (1.0 + 0.1 * grid) * (1.0 + 0.3 * grid)
    expected = np.max(np.abs(left_response - right_response))
    value = composed_truncated_response_sup(
        left, right, 1, 1, restarts=10, steps=400, seed=29
    )
    assert value == pytest.approx(expected, rel=1e-6)


def test_composed_response_compares_different_partition_sizes() -> None:
    """Two depth-one factors differ from one merged factor by their cross term."""
    fine = (np.array([[0.4], [-0.2]]),)
    coarse = (np.array([[0.2]]),)
    value = composed_truncated_response_sup(
        fine, coarse, 1, 1, restarts=6, steps=200, seed=37
    )
    assert value == pytest.approx(0.08, rel=1e-8)


def test_composed_response_is_zero_for_identical_interval_signatures() -> None:
    signatures = (
        np.array([[0.4, -0.2], [0.1, 0.3]]),
        np.arange(8, dtype=float).reshape(2, 4) / 10.0,
    )
    value = composed_truncated_response_sup(
        signatures, signatures, 2, 3, restarts=2, steps=20
    )
    assert value == 0.0


def test_composed_response_gradient_matches_finite_differences() -> None:
    """Chronological product gradient is checked independently coordinatewise."""
    rng = np.random.default_rng(41)
    response = rng.standard_normal((2, 2, 2))
    response *= 0.7 / np.linalg.norm(response)
    left_raw = (
        rng.standard_normal((2, 2)),
        rng.standard_normal((2, 4)),
    )
    right_raw = (
        rng.standard_normal((3, 2)),
        rng.standard_normal((3, 4)),
    )
    left = response_module._interval_tensors(left_raw, 2, 1.0)
    right = response_module._interval_tensors(right_raw, 2, 1.0)
    value, gradient = response_module._composed_objective_and_gradient(
        response, left, right
    )
    assert value > 0.0
    epsilon = 1.0e-6
    finite_difference = np.zeros_like(response)
    for index in np.ndindex(response.shape):
        perturbation = np.zeros_like(response)
        perturbation[index] = epsilon
        plus = response_module._composed_objective_and_gradient(
            response + perturbation, left, right
        )[0]
        minus = response_module._composed_objective_and_gradient(
            response - perturbation, left, right
        )[0]
        finite_difference[index] = (plus - minus) / (2.0 * epsilon)
    assert gradient == pytest.approx(finite_difference, rel=2e-6, abs=2e-7)


def test_composed_response_equals_its_expanded_tensor_polynomial() -> None:
    """Direct matrix products agree with independently expanded coefficients."""
    rng = np.random.default_rng(42)
    left = (
        rng.standard_normal((2, 2)),
        rng.standard_normal((2, 4)),
    )
    right = (
        rng.standard_normal((1, 2)),
        rng.standard_normal((1, 4)),
    )
    left_polynomial = composed_signature_polynomial(left, 2)
    right_polynomial = composed_signature_polynomial(right, 2)
    depth = max(len(left_polynomial), len(right_polynomial))
    deltas = tuple(
        (
            left_polynomial[level]
            if level < len(left_polynomial)
            else np.zeros(2 ** (level + 1))
        )
        - (
            right_polynomial[level]
            if level < len(right_polynomial)
            else np.zeros(2 ** (level + 1))
        )
        for level in range(depth)
    )
    direct = composed_truncated_response_sup(
        left, right, 2, 2, restarts=12, steps=500, seed=47
    )
    expanded = truncated_response_sup(deltas, 2, 2, restarts=12, steps=500, seed=47)
    assert direct == pytest.approx(expanded, rel=2e-7, abs=2e-8)


def test_composed_coordinate_plane_loop_order_constant() -> None:
    """Depth-two loop-order response is exactly ``1 / (2 sqrt(2))``."""
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    area_xy = _tensor_bracket(x, y)
    area_yz = _tensor_bracket(y, z)
    zeros = np.zeros((2, 3))
    left = (zeros, np.stack((area_xy, area_yz)))
    right = (zeros, np.stack((area_yz, area_xy)))
    value = composed_truncated_response_sup(
        left, right, 3, 2, restarts=20, steps=600, seed=43
    )
    assert value == pytest.approx(1.0 / (2.0 * np.sqrt(2.0)), rel=2e-6)


def test_coordinate_plane_loop_merge_has_exact_response_and_hs_gap() -> None:
    """Merging two loop intervals has response ``1/4`` and HS bound ``2``."""
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    area_xy = _tensor_bracket(x, y)
    area_yz = _tensor_bracket(y, z)
    fine = (np.zeros((2, 3)), np.stack((area_xy, area_yz)))
    coarse = (
        np.zeros((1, 3)),
        (area_xy + area_yz)[None, :],
    )
    value = composed_truncated_response_sup(
        fine, coarse, 3, 2, restarts=20, steps=600, seed=53
    )
    assert value == pytest.approx(0.25, rel=2e-6)

    fine_polynomial = composed_signature_polynomial(fine, 3)
    coarse_polynomial = composed_signature_polynomial(coarse, 3)
    delta_four = fine_polynomial[3]
    assert coarse_polynomial[0] == pytest.approx(fine_polynomial[0])
    assert coarse_polynomial[1] == pytest.approx(fine_polynomial[1])
    assert fine_polynomial[2] == pytest.approx(np.zeros(27))
    assert np.linalg.norm(delta_four) == pytest.approx(2.0)


def test_coordinate_loop_logode_gradient_matches_finite_differences() -> None:
    """Fréchet derivative and commutator pullback are checked independently."""
    rng = np.random.default_rng(59)
    response = rng.standard_normal((3, 2, 2))
    response *= 0.7 / np.linalg.norm(response)
    scale = 0.4
    value, gradient = response_module._coordinate_loop_logode_objective_and_gradient(
        response, scale
    )
    assert value > 0.0
    epsilon = 1.0e-6
    finite_difference = np.zeros_like(response)
    for index in np.ndindex(response.shape):
        perturbation = np.zeros_like(response)
        perturbation[index] = epsilon
        plus = response_module._coordinate_loop_logode_objective_and_gradient(
            response + perturbation, scale
        )[0]
        minus = response_module._coordinate_loop_logode_objective_and_gradient(
            response - perturbation, scale
        )[0]
        finite_difference[index] = (plus - minus) / (2.0 * epsilon)
    assert gradient == pytest.approx(finite_difference, rel=2e-6, abs=2e-8)


def test_coordinate_loop_logode_merge_has_factor_eight_tangent_gap() -> None:
    """The sharp commutator tuple attains the leading log-ODE merge constant."""
    h = np.array([[1.0, 0.0], [0.0, -1.0]]) / np.sqrt(2.0)
    n = np.array([[0.0, 1.0], [0.0, 0.0]])
    m = np.array([[0.0, 0.0], [1.0, 0.0]])
    response = np.stack((0.5 * h, n / np.sqrt(2.0), 0.5 * m))
    scale = 1.0e-3
    value = response_module._coordinate_loop_logode_objective_and_gradient(
        response, scale
    )[0]
    leading = scale**2 / (4.0 * np.sqrt(2.0))
    remainder = scale**3 * np.exp(scale) / 3.0
    assert value == pytest.approx(leading, abs=remainder)
    hs_leading = np.sqrt(2.0) * scale**2
    assert leading / hs_leading == pytest.approx(1.0 / 8.0)


def test_diagonal_logode_responses_are_blind_to_loop_merge() -> None:
    value = coordinate_loop_logode_merge_sup(
        1.0, 3, restarts=3, steps=20, seed=61, block_size=1
    )
    assert value == pytest.approx(0.0, abs=1e-12)


def test_coordinate_loop_logode_hs_certificate_has_closed_form_levels() -> None:
    """The full certificate starts with the proved degree-four and six terms."""
    scale = 1.0e-4
    value = coordinate_loop_logode_hs_certificate(scale)
    leading = np.sqrt(2.0) * scale**2
    degree_six = 2.0 * np.sqrt(6.0) * scale**3 / 3.0
    assert value == pytest.approx(leading + degree_six, rel=2e-8)
    assert coordinate_loop_logode_hs_certificate(0.0) == 0.0
    assert orthogonal_logode_hs_certificate(0.0, 2.0) == 0.0
    assert orthogonal_logode_hs_certificate(2.0, 0.0) == 0.0
    assert coordinate_loop_logode_hs_certificate(0.7) == pytest.approx(
        orthogonal_logode_hs_certificate(
            np.sqrt(2.0) * 0.7,
            np.sqrt(2.0) * 0.7,
        )
    )

    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    area_xy = _tensor_bracket(x, y)
    area_yz = _tensor_bracket(y, z)

    def tensor_power(value: np.ndarray, exponent: int) -> np.ndarray:
        result = np.ones(1)
        for _ in range(exponent):
            result = np.kron(result, value)
        return result

    for level in range(2, 6):
        fine = sum(
            (
                np.kron(
                    tensor_power(area_xy, left),
                    tensor_power(area_yz, level - left),
                )
                / (math.factorial(left) * math.factorial(level - left))
                for left in range(level + 1)
            ),
            start=np.zeros(9**level),
        )
        merged = tensor_power(area_xy + area_yz, level) / math.factorial(level)
        expected = (
            2.0 ** (level / 2.0)
            / math.factorial(level)
            * np.sqrt(math.comb(2 * level, level) - 2**level)
        )
        assert np.linalg.norm(fine - merged) == pytest.approx(expected)

    left_scale = 0.3
    right_scale = 0.2
    abstract_left = np.array([np.sqrt(2.0) * left_scale, 0.0])
    abstract_right = np.array([0.0, np.sqrt(2.0) * right_scale])
    direct_certificate = 0.0
    for level in range(2, 12):
        fine = sum(
            (
                np.kron(
                    tensor_power(abstract_left, left),
                    tensor_power(abstract_right, level - left),
                )
                / (math.factorial(left) * math.factorial(level - left))
                for left in range(level + 1)
            ),
            start=np.zeros(2**level),
        )
        merged = tensor_power(abstract_left + abstract_right, level) / math.factorial(
            level
        )
        direct_certificate += np.linalg.norm(fine - merged)
    exact_certificate = orthogonal_logode_hs_certificate(
        np.sqrt(2.0) * left_scale,
        np.sqrt(2.0) * right_scale,
    )
    assert direct_certificate == pytest.approx(exact_certificate, rel=2e-9)


def test_coordinate_loop_logode_finite_scale_bounds_separate_certificate() -> None:
    """Explicit tuple attains the analytic lower bound at unit scale."""
    root_two = np.sqrt(2.0)
    response = np.array(
        [
            [[0.25, 0.0], [-1.0 / (2.0 * root_two), -0.25]],
            [[0.0, 1.0 / root_two], [0.0, 0.0]],
            [[-0.25, 0.0], [-1.0 / (2.0 * root_two), 0.25]],
        ]
    )
    assert np.linalg.norm(response) == pytest.approx(1.0)
    scale = 1.0
    value = response_module._coordinate_loop_logode_objective_and_gradient(
        response, scale
    )[0]
    analytic_lower = (2.0 * np.expm1(scale / 2.0) - scale) / root_two
    lie_trotter_upper = scale**2 * np.exp(scale) / (4.0 * root_two)
    certificate = coordinate_loop_logode_hs_certificate(scale)
    assert value == pytest.approx(analytic_lower, rel=1e-12)
    assert analytic_lower < lie_trotter_upper < certificate

    for checked_scale in (0.01, 0.25, 2.25, 5.0):
        checked_upper = checked_scale**2 * np.exp(checked_scale) / (4.0 * root_two)
        assert checked_upper < coordinate_loop_logode_hs_certificate(checked_scale)


def test_nonorthogonal_logode_level_certificate_depends_only_on_gram_matrix() -> None:
    """The Gram formula agrees with direct tensor expansion."""
    left = np.array([0.7, -0.2])
    right = np.array([0.3, 0.6])

    def tensor_power(value: np.ndarray, exponent: int) -> np.ndarray:
        result = np.ones(1)
        for _ in range(exponent):
            result = np.kron(result, value)
        return result

    for level in range(1, 8):
        fine = sum(
            (
                np.kron(
                    tensor_power(left, left_count),
                    tensor_power(right, level - left_count),
                )
                / (math.factorial(left_count) * math.factorial(level - left_count))
                for left_count in range(level + 1)
            ),
            start=np.zeros(2**level),
        )
        merged = tensor_power(left + right, level) / math.factorial(level)
        expected = np.linalg.norm(fine - merged)
        actual = logode_merge_level_hs_norm(
            np.linalg.norm(left),
            np.linalg.norm(right),
            float(left @ right),
            level,
        )
        assert actual == pytest.approx(expected, rel=2e-12, abs=2e-14)

    parallel = 2.0 * left
    assert logode_merge_level_hs_norm(
        np.linalg.norm(left),
        np.linalg.norm(parallel),
        float(left @ parallel),
        4,
    ) == pytest.approx(0.0, abs=1e-14)


def test_multiblock_logode_level_certificate_depends_only_on_gram_matrix() -> None:
    """The multiblock Gram formula agrees with direct tensor expansion."""
    blocks = np.array(
        [
            [0.7, -0.2, 0.1],
            [0.3, 0.6, -0.4],
            [-0.1, 0.2, 0.8],
        ]
    )
    gram = blocks @ blocks.T

    def tensor_power(value: np.ndarray, exponent: int) -> np.ndarray:
        result = np.ones(1)
        for _ in range(exponent):
            result = np.kron(result, value)
        return result

    for level in range(1, 6):
        fine = sum(
            (
                np.kron(
                    np.kron(
                        tensor_power(blocks[0], first_count),
                        tensor_power(blocks[1], second_count),
                    ),
                    tensor_power(blocks[2], level - first_count - second_count),
                )
                / (
                    math.factorial(first_count)
                    * math.factorial(second_count)
                    * math.factorial(level - first_count - second_count)
                )
                for first_count in range(level + 1)
                for second_count in range(level - first_count + 1)
            ),
            start=np.zeros(3**level),
        )
        merged = tensor_power(np.sum(blocks, axis=0), level) / math.factorial(level)
        expected = np.linalg.norm(fine - merged)
        actual = multi_logode_merge_level_hs_norm(gram, level)
        assert actual == pytest.approx(expected, rel=2e-11, abs=2e-13)

    for level in range(1, 6):
        assert multi_logode_merge_level_hs_norm(gram[:2, :2], level) == pytest.approx(
            logode_merge_level_hs_norm(
                np.linalg.norm(blocks[0]),
                np.linalg.norm(blocks[1]),
                float(blocks[0] @ blocks[1]),
                level,
            ),
            rel=2e-12,
            abs=2e-14,
        )


def test_three_coordinate_loop_chain_clears_multiblock_threshold() -> None:
    """Exact coefficients dominate the analytic three-loop response bound."""
    path_radius = np.cos(np.pi / 5.0)
    response_growth = np.sqrt(2.0) * path_radius
    commutator_bound = np.sqrt(2.0) * path_radius**2
    gram = 2.0 * np.eye(3)
    for level in range(2, 9):
        coefficient = math.factorial(level) * multi_logode_merge_level_hs_norm(
            gram, level
        )
        threshold = (
            2.0 * coefficient / (level * (level - 1) * response_growth ** (level - 2))
        )
        assert commutator_bound < 2.0 / path_radius < threshold

    for scale in (0.01, 0.25, 1.0, 3.0):
        response_upper = scale**2 * np.exp(np.sqrt(2.0) * scale) / np.sqrt(2.0)
        certificate_lower = (
            np.expm1(np.sqrt(6.0) * scale) - np.sqrt(6.0) * scale
        ) / np.sqrt(3.0)
        assert response_upper < certificate_lower


def test_diagonal_responses_are_blind_to_planar_area() -> None:
    delta = np.array([0.0, 1.0, -1.0, 0.0])
    value = level_response_sup(delta, 2, 2, 3, restarts=4, steps=100, block_size=1)
    assert value == pytest.approx(0.0, abs=1e-12)


def test_two_by_two_blocks_attain_planar_area_constant() -> None:
    delta = np.array([0.0, 1.0, -1.0, 0.0])
    value = response_ratio(
        delta,
        2,
        2,
        4,
        restarts=16,
        steps=600,
        seed=31,
        block_size=2,
    )
    assert value == pytest.approx(0.5, rel=1e-5)
