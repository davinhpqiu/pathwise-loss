"""Finite response discrepancies and level-wise bounds.

Conventions follow Ferrucci, Perree and Lyons (arXiv:2607.26281, eq. 1.9), and
use three distinct norms: Hilbert-Schmidt on ``(R^d)^{otimes k}``, Frobenius on
response tuple ``A = (A_1, ..., A_d)`` of ``n x n`` matrices,

    |A| = (sum_i ||A_i||_F^2)^{1/2},

and induced Euclidean operator norm on the terminal matrix.

Using row states, ``dY = sum_i Y A_i dX_i`` has terminal operator
``Phi^A(X) = sum_k sum_{|w|=k} A_{w_1} ... A_{w_k} S^w(X)``.  Level ``k`` part of
response discrepancy between paths ``X`` and ``Xt`` is

    d_r^{(k)} = sup_{|A| <= r} || sum_{|w|=k} D^w A_{w_1} ... A_{w_k} ||_op,
    D = S^{(k)}(X) - S^{(k)}(Xt).

Both sides of the bracket scale as ``r^k``, so
``ratio = d_1^{(k)} / ||D||_2`` carries all information and lies in ``[0, 1]``:
upper bound is eq. (1.9), attained at ``k = 1``.  Two lower bounds are available:

- shift bound ``k^{-k/2} max_w |D^w|``, from nilpotent shift matrices isolating
  one word (verified in ``scripts/check_response_separation.py``);
- rank-one bound ``k^{-k/2} ||D||_sigma`` with ``||D||_sigma`` the injective
  (tensor spectral) norm, from shift matrices carrying rank-one blocks.

Word order is chronological and matches flattened level arrays: ``w_1`` varies
slowest. Transposing the tuple gives the column-state convention and preserves
the operator-norm supremum.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from decimal import Decimal, localcontext
from itertools import product

import numpy as np
from scipy.linalg import expm, expm_frechet
from scipy.special import logsumexp

from .brownian_messages import chen_reduce

__all__ = [
    "area_block_gram_matrix",
    "area_merge_response_gauge_bounds",
    "area_response_gauge",
    "composed_signature_polynomial",
    "composed_truncated_response_sup",
    "coordinate_loop_logode_hs_certificate",
    "coordinate_loop_logode_merge_sup",
    "estimate_area_merge_response_gauges",
    "hilbert_schmidt_bound",
    "level_response_sup",
    "logode_merge_level_hs_norm",
    "multi_logode_merge_level_hs_norm",
    "orthogonal_logode_hs_certificate",
    "piecewise_linear_signature",
    "rank_one_bound",
    "response_ratio",
    "shift_bound",
    "tensor_spectral_norm",
    "truncated_response_sup",
    "tuple_norm",
]


def _as_area_blocks(area_blocks: np.ndarray) -> np.ndarray:
    """Validate skew coefficient matrices for homogeneous degree-two blocks."""
    values = np.asarray(area_blocks, dtype=float)
    if values.ndim == 2:
        values = values[None, ...]
    if (
        values.ndim != 3
        or values.shape[0] < 1
        or values.shape[1] != values.shape[2]
        or values.shape[1] < 2
    ):
        raise ValueError("area_blocks must have shape (blocks, d, d) with d >= 2")
    if not np.all(np.isfinite(values)):
        raise ValueError("area_blocks must contain only finite values")
    scales = np.max(np.abs(values), axis=(1, 2), keepdims=True)
    normalised = np.divide(
        values, scales, out=np.zeros_like(values), where=scales > 0.0
    )
    if not np.allclose(
        normalised + normalised.transpose(0, 2, 1), 0.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("area coefficient matrices must be skew-symmetric")
    return 0.5 * (values - values.transpose(0, 2, 1))


def area_response_gauge(area: np.ndarray) -> float:
    """Exact unit-ball response gauge of a homogeneous degree-two area block.

    `area[i, j]` is the coefficient of `e_i tensor e_j`. For every
    response dimension at least two, both the operator-output and
    Frobenius-output gauges equal `||area||_op / sqrt(2)`.
    """
    values = np.asarray(area, dtype=float)
    if values.ndim != 2:
        raise ValueError("area must be one square skew-symmetric matrix")
    block = _as_area_blocks(values)[0]
    return float(np.linalg.norm(block, ord=2) / np.sqrt(2.0))


def area_block_gram_matrix(area_blocks: np.ndarray) -> np.ndarray:
    """Hilbert--Schmidt Gram matrix of degree-two area blocks."""
    blocks = _as_area_blocks(area_blocks)
    return np.einsum("sij,tij->st", blocks, blocks)


def area_merge_response_gauge_bounds(
    area_blocks: np.ndarray,
) -> tuple[float, float]:
    """Cheap upper bounds for multiblock response gauges `M` and `K`.

    For skew coefficient matrices `Omega_s`, form the nonnegative symmetric
    overlap matrix `W` with
    `W[i, j] = sum_s abs(Omega_s[i, j])`.  If `Lambda` is its largest
    eigenvalue and `sigma_s = ||Omega_s||_op`, the returned bounds are

    `M <= min(Lambda, sum_s sigma_s) / sqrt(2)`,

    `K <= min(Lambda**2 / 2, sum_{s<t} sigma_s sigma_t) / sqrt(2)`.
    """
    blocks = _as_area_blocks(area_blocks)
    overlap = np.sum(np.abs(blocks), axis=0)
    overlap_radius = max(0.0, float(np.linalg.eigvalsh(overlap)[-1]))
    block_radii = np.linalg.norm(blocks, ord=2, axis=(1, 2))
    radius_sum = float(np.sum(block_radii))
    pair_radius_sum = math.fsum(
        float(block_radii[left] * block_radii[right])
        for left in range(len(block_radii))
        for right in range(left + 1, len(block_radii))
    )
    root_two = np.sqrt(2.0)
    growth_bound = min(overlap_radius, radius_sum) / root_two
    commutator_bound = (
        min(
            overlap_radius**2 / 2.0,
            max(0.0, pair_radius_sum),
        )
        / root_two
    )
    return growth_bound, commutator_bound


def estimate_area_merge_response_gauges(
    area_blocks: np.ndarray,
    response_dim: int,
    *,
    restarts: int = 12,
    steps: int = 400,
    seed: int = 0,
    block_size: int | None = None,
) -> tuple[float, float]:
    """Nonconvex lower estimates of the aggregate area-block gauges `M,K`."""
    blocks = _as_area_blocks(area_blocks)
    channel = blocks.shape[1]
    growth = _maximise_response(
        lambda response: _area_growth_objective_and_gradient(response, blocks),
        channel,
        response_dim,
        restarts=restarts,
        steps=steps,
        seed=seed,
        sphere=True,
        block_size=block_size,
    )
    commutator = _maximise_response(
        lambda response: _area_commutator_objective_and_gradient(response, blocks),
        channel,
        response_dim,
        restarts=restarts,
        steps=steps,
        seed=seed + 1,
        sphere=True,
        block_size=block_size,
    )
    return growth, commutator


def piecewise_linear_signature(path: np.ndarray, depth: int) -> tuple[np.ndarray, ...]:
    """Signature levels ``0..depth`` of piecewise-linear interpolation of ``path``.

    ``path`` has shape ``(points, d)``.  Level ``k`` is returned flattened with
    ``w_1`` slowest.
    """
    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("path must have shape (points, d) with at least two points")
    if depth < 1:
        raise ValueError("depth must be positive")
    increments = np.diff(values, axis=0)
    segments = increments.shape[0]
    levels = [np.ones((segments, 1))]
    current = np.ones((segments, 1))
    factorial = 1.0
    for order in range(1, depth + 1):
        current = (current[:, :, None] * increments[:, None, :]).reshape(segments, -1)
        factorial *= order
        levels.append(current / factorial)
    return chen_reduce(levels)


def tuple_norm(response: np.ndarray) -> float:
    """Frobenius norm of a response tuple of shape ``(d, n, n)``."""
    return float(np.sqrt(np.sum(np.asarray(response) ** 2)))


def _as_tensor(delta: np.ndarray, channel: int, level: int) -> np.ndarray:
    values = np.asarray(delta, dtype=float).reshape(-1)
    if values.size != channel**level:
        raise ValueError(f"expected {channel**level} coordinates, got {values.size}")
    return values.reshape((channel,) * level)


def _word_product(response: np.ndarray, word: tuple[int, ...]) -> np.ndarray:
    out = response[word[0]]
    for letter in word[1:]:
        out = out @ response[letter]
    return out


def _polynomial_matrix_and_gradient(
    response: np.ndarray,
    tensors: tuple[np.ndarray, ...],
    *,
    seed: np.ndarray | None = None,
    include_identity: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a tensor polynomial and its reverse-mode response gradient."""
    channel, size = response.shape[0], response.shape[1]
    total = np.eye(size) if include_identity else np.zeros((size, size))
    grad = np.zeros_like(response)
    identity = np.eye(size)
    for tensor in tensors:
        level = tensor.ndim
        for word in product(range(channel), repeat=level):
            weight = tensor[word]
            if weight == 0.0:
                continue
            total += weight * _word_product(response, word)
            if seed is None:
                continue
            prefix = [identity]
            for letter in word[:-1]:
                prefix.append(prefix[-1] @ response[letter])
            suffix = [identity]
            for letter in reversed(word[1:]):
                suffix.append(response[letter] @ suffix[-1])
            suffix.reverse()
            for position, letter in enumerate(word):
                grad[letter] += weight * (
                    prefix[position].T @ seed @ suffix[position].T
                )
    return total, grad


def _operator_seed(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Operator norm and one subgradient with respect to ``matrix``."""
    left, singular, right = np.linalg.svd(matrix)
    value = float(singular[0])
    if value == 0.0:
        return 0.0, np.zeros_like(matrix)
    return value, np.outer(left[:, 0], right[0])


def _objective_and_gradient(
    response: np.ndarray, tensors: tuple[np.ndarray, ...]
) -> tuple[float, np.ndarray]:
    total, _ = _polynomial_matrix_and_gradient(response, tensors)
    value, seed = _operator_seed(total)
    if value == 0.0:
        return 0.0, np.zeros_like(response)
    _, grad = _polynomial_matrix_and_gradient(response, tensors, seed=seed)
    return value, grad


def _product(factors: list[np.ndarray], size: int) -> np.ndarray:
    out = np.eye(size)
    for factor in factors:
        out = out @ factor
    return out


def _product_seeds(factors: list[np.ndarray], seed: np.ndarray) -> list[np.ndarray]:
    """Reverse-mode seeds for chronological product ``F_1 ... F_m``."""
    size = seed.shape[0]
    prefix = [np.eye(size)]
    for factor in factors:
        prefix.append(prefix[-1] @ factor)
    suffix = [np.eye(size) for _ in range(len(factors) + 1)]
    for index in range(len(factors) - 1, -1, -1):
        suffix[index] = factors[index] @ suffix[index + 1]
    return [
        prefix[index].T @ seed @ suffix[index + 1].T for index in range(len(factors))
    ]


def _composed_objective_and_gradient(
    response: np.ndarray,
    left: tuple[tuple[np.ndarray, ...], ...],
    right: tuple[tuple[np.ndarray, ...], ...],
) -> tuple[float, np.ndarray]:
    size = response.shape[1]
    left_factors = [
        _polynomial_matrix_and_gradient(response, interval, include_identity=True)[0]
        for interval in left
    ]
    right_factors = [
        _polynomial_matrix_and_gradient(response, interval, include_identity=True)[0]
        for interval in right
    ]
    difference = _product(left_factors, size) - _product(right_factors, size)
    value, seed = _operator_seed(difference)
    if value == 0.0:
        return 0.0, np.zeros_like(response)
    grad = np.zeros_like(response)
    for interval, factor_seed in zip(
        left, _product_seeds(left_factors, seed), strict=True
    ):
        grad += _polynomial_matrix_and_gradient(
            response, interval, seed=factor_seed, include_identity=True
        )[1]
    for interval, factor_seed in zip(
        right, _product_seeds(right_factors, -seed), strict=True
    ):
        grad += _polynomial_matrix_and_gradient(
            response, interval, seed=factor_seed, include_identity=True
        )[1]
    return value, grad


def _expm_adjoint(matrix: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Adjoint Fréchet derivative of the matrix exponential."""
    return expm_frechet(matrix.T, seed, compute_expm=False)


def _commutator_gradients(
    left: np.ndarray, right: np.ndarray, seed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Reverse-mode gradients for ``[left, right]``."""
    left_grad = seed @ right.T - right.T @ seed
    right_grad = left.T @ seed - seed @ left.T
    return left_grad, right_grad


def _area_response_matrices(
    response: np.ndarray, area_blocks: np.ndarray
) -> np.ndarray:
    """Evaluate degree-two area blocks on one response tuple."""
    block_count = area_blocks.shape[0]
    size = response.shape[1]
    outputs = np.zeros((block_count, size, size))
    for block_index, area in enumerate(area_blocks):
        for left in range(area.shape[0]):
            for right in range(left + 1, area.shape[1]):
                coefficient = area[left, right]
                if coefficient != 0.0:
                    outputs[block_index] += coefficient * (
                        response[left] @ response[right]
                        - response[right] @ response[left]
                    )
    return outputs


def _area_response_pullback(
    response: np.ndarray,
    area_blocks: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    """Pull matrix seeds through all degree-two area responses."""
    gradient = np.zeros_like(response)
    for area, seed in zip(area_blocks, seeds, strict=True):
        for left in range(area.shape[0]):
            for right in range(left + 1, area.shape[1]):
                coefficient = area[left, right]
                if coefficient == 0.0:
                    continue
                left_grad, right_grad = _commutator_gradients(
                    response[left], response[right], coefficient * seed
                )
                gradient[left] += left_grad
                gradient[right] += right_grad
    return gradient


def _area_growth_objective_and_gradient(
    response: np.ndarray, area_blocks: np.ndarray
) -> tuple[float, np.ndarray]:
    """Sum of operator norms defining the multiblock growth gauge `M`."""
    outputs = _area_response_matrices(response, area_blocks)
    seeds = np.zeros_like(outputs)
    value = 0.0
    for index, output in enumerate(outputs):
        block_value, seeds[index] = _operator_seed(output)
        value += block_value
    return value, _area_response_pullback(response, area_blocks, seeds)


def _area_commutator_objective_and_gradient(
    response: np.ndarray, area_blocks: np.ndarray
) -> tuple[float, np.ndarray]:
    """Sum of pairwise commutator norms defining the gauge `K`."""
    outputs = _area_response_matrices(response, area_blocks)
    seeds = np.zeros_like(outputs)
    value = 0.0
    for left in range(outputs.shape[0]):
        for right in range(left + 1, outputs.shape[0]):
            commutator = outputs[left] @ outputs[right] - outputs[right] @ outputs[left]
            pair_value, pair_seed = _operator_seed(commutator)
            value += pair_value
            left_seed, right_seed = _commutator_gradients(
                outputs[left], outputs[right], pair_seed
            )
            seeds[left] += left_seed
            seeds[right] += right_seed
    return value, _area_response_pullback(response, area_blocks, seeds)


def _coordinate_loop_logode_objective_and_gradient(
    response: np.ndarray, scale: float
) -> tuple[float, np.ndarray]:
    """Log-ODE merge error for loops in planes ``(0, 1)`` and ``(1, 2)``."""
    first = scale * (response[0] @ response[1] - response[1] @ response[0])
    second = scale * (response[1] @ response[2] - response[2] @ response[1])
    first_exp = expm(first)
    second_exp = expm(second)
    merged_exp = expm(first + second)
    difference = first_exp @ second_exp - merged_exp
    value, seed = _operator_seed(difference)
    if value == 0.0:
        return 0.0, np.zeros_like(response)

    merged_seed = _expm_adjoint(first + second, seed)
    first_seed = _expm_adjoint(first, seed @ second_exp.T) - merged_seed
    second_seed = _expm_adjoint(second, first_exp.T @ seed) - merged_seed
    first_seed *= scale
    second_seed *= scale

    grad = np.zeros_like(response)
    grad[0], first_shared = _commutator_gradients(response[0], response[1], first_seed)
    second_shared, grad[2] = _commutator_gradients(
        response[1], response[2], second_seed
    )
    grad[1] = first_shared + second_shared
    return value, grad


def _structure_mask(response_dim: int, block_size: int | None) -> np.ndarray:
    if block_size is None:
        return np.ones((response_dim, response_dim))
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer or None")
    rows = np.arange(response_dim)
    return (rows[:, None] // block_size == rows[None, :] // block_size).astype(float)


def _maximise_response(
    objective_and_gradient: Callable[[np.ndarray], tuple[float, np.ndarray]],
    channel: int,
    response_dim: int,
    *,
    restarts: int,
    steps: int,
    seed: int,
    sphere: bool,
    block_size: int | None,
) -> float:
    """Projected ascent on a structured Frobenius ball or sphere."""
    if channel < 1 or response_dim < 1:
        raise ValueError("channel and response_dim must be positive")
    if restarts < 1 or steps < 1:
        raise ValueError("restarts and steps must be positive")
    mask = _structure_mask(response_dim, block_size)
    rng = np.random.default_rng(seed)
    best = 0.0
    for restart in range(restarts):
        response = rng.standard_normal((channel, response_dim, response_dim))
        response *= mask
        norm = tuple_norm(response)
        if norm == 0.0:
            continue
        radius = 1.0 if sphere or restart == 0 else float(rng.uniform(0.05, 1.0))
        response *= radius / norm
        rate = 1.0
        value, grad = objective_and_gradient(response)
        for _ in range(steps):
            candidate = response + rate * grad * mask
            norm = tuple_norm(candidate)
            if norm > 1.0 or sphere:
                if norm == 0.0:
                    break
                candidate /= norm
            trial, trial_grad = objective_and_gradient(candidate)
            if trial > value:
                response, value, grad = candidate, trial, trial_grad
                rate *= 1.3
            else:
                rate *= 0.5
                if rate < 1e-12:
                    break
        best = max(best, value)
    return best


def _scaled_tensors(
    deltas: tuple[np.ndarray, ...], channel: int, response_bound: float
) -> tuple[np.ndarray, ...]:
    if not np.isfinite(response_bound) or response_bound < 0.0:
        raise ValueError("response_bound must be finite and nonnegative")
    return tuple(
        response_bound**level * _as_tensor(delta, channel, level)
        for level, delta in enumerate(deltas, start=1)
    )


def _interval_tensors(
    signatures: tuple[np.ndarray, ...], channel: int, response_bound: float
) -> tuple[tuple[np.ndarray, ...], ...]:
    if not signatures:
        raise ValueError("at least one positive signature level is required")
    if not np.isfinite(response_bound) or response_bound < 0.0:
        raise ValueError("response_bound must be finite and nonnegative")
    arrays = tuple(np.asarray(level, dtype=float) for level in signatures)
    if any(level.ndim != 2 for level in arrays):
        raise ValueError(
            "each signature level must have shape (intervals, coordinates)"
        )
    intervals = arrays[0].shape[0]
    if intervals < 1 or any(level.shape[0] != intervals for level in arrays):
        raise ValueError("signature levels must share a nonempty interval axis")
    scaled = []
    for level, array in enumerate(arrays, start=1):
        if array.shape[1] != channel**level:
            raise ValueError(f"level {level} requires {channel**level} coordinates")
        scaled.append(response_bound**level * array)
    return tuple(
        tuple(
            scaled[level - 1][index].reshape((channel,) * level)
            for level in range(1, len(scaled) + 1)
        )
        for index in range(intervals)
    )


def composed_signature_polynomial(
    signatures: tuple[np.ndarray, ...], channel: int
) -> tuple[np.ndarray, ...]:
    """Coefficient levels of a product of truncated interval signatures.

    Input level ``k`` has shape ``(intervals, channel**k)``. Each interval is
    truncated at supplied depth, while returned product retains every generated
    degree through ``intervals * depth``. This can grow exponentially and is
    intended for small compression blocks and validation.
    """
    intervals = _interval_tensors(signatures, channel, 1.0)
    current = [np.ones(1)]
    for interval in intervals:
        factor = [np.ones(1), *(tensor.reshape(-1) for tensor in interval)]
        combined = [
            np.zeros(channel**level) for level in range(len(current) + len(factor) - 1)
        ]
        for left_level, left in enumerate(current):
            for right_level, right in enumerate(factor):
                combined[left_level + right_level] += np.outer(left, right).reshape(-1)
        current = combined
    return tuple(current[1:])


def level_response_sup(
    delta: np.ndarray,
    channel: int,
    level: int,
    response_dim: int,
    *,
    restarts: int = 12,
    steps: int = 400,
    seed: int = 0,
    block_size: int | None = None,
) -> float:
    """``sup_{|A| <= 1}`` of the level-``k`` response discrepancy.

    Projected gradient ascent on the unit Frobenius sphere, since the objective
    is homogeneous of degree ``level`` and so attains its maximum on the
    boundary.  Reported value is the best of ``restarts`` random starts.
    """
    tensor = _as_tensor(delta, channel, level)
    if not np.any(tensor):
        return 0.0
    tensors = (tensor,)
    return _maximise_response(
        lambda response: _objective_and_gradient(response, tensors),
        channel,
        response_dim,
        restarts=restarts,
        steps=steps,
        seed=seed,
        sphere=True,
        block_size=block_size,
    )


def truncated_response_sup(
    deltas: tuple[np.ndarray, ...],
    channel: int,
    response_dim: int,
    *,
    response_bound: float = 1.0,
    restarts: int = 12,
    steps: int = 400,
    seed: int = 0,
    block_size: int | None = None,
) -> float:
    """Common-coefficient response supremum through a finite signature depth.

    ``deltas[k - 1]`` is level ``k`` signature difference. One tuple ``A``
    acts at every level and satisfies ``|A| <= response_bound``. Projected
    ascent returns a lower estimate of the nonconvex supremum. ``block_size=1``
    restricts matrices to diagonal form; any larger value selects consecutive
    square blocks.
    """
    if not deltas:
        raise ValueError("at least one positive signature level is required")
    tensors = _scaled_tensors(deltas, channel, response_bound)
    if not any(np.any(tensor) for tensor in tensors) or response_bound == 0.0:
        return 0.0
    return _maximise_response(
        lambda response: _objective_and_gradient(response, tensors),
        channel,
        response_dim,
        restarts=restarts,
        steps=steps,
        seed=seed,
        sphere=False,
        block_size=block_size,
    )


def composed_truncated_response_sup(
    left: tuple[np.ndarray, ...],
    right: tuple[np.ndarray, ...],
    channel: int,
    response_dim: int,
    *,
    response_bound: float = 1.0,
    restarts: int = 12,
    steps: int = 400,
    seed: int = 0,
    block_size: int | None = None,
) -> float:
    """Supremum of composed interval responses through finite depth.

    Each entry is one positive signature level with shape
    ``(intervals, channel**level)``. On every interval the truncated response
    is ``I + sum_k R_{A,k}(S_k)``; interval matrices are multiplied in supplied
    order. One norm-bounded tuple ``A`` acts on every level and interval.
    """
    if len(left) != len(right):
        raise ValueError("left and right signatures must have equal depth")
    left_intervals = _interval_tensors(left, channel, response_bound)
    right_intervals = _interval_tensors(right, channel, response_bound)
    if response_bound == 0.0:
        return 0.0
    return _maximise_response(
        lambda response: _composed_objective_and_gradient(
            response, left_intervals, right_intervals
        ),
        channel,
        response_dim,
        restarts=restarts,
        steps=steps,
        seed=seed,
        sphere=False,
        block_size=block_size,
    )


def coordinate_loop_logode_merge_sup(
    scale: float,
    response_dim: int,
    *,
    restarts: int = 12,
    steps: int = 400,
    seed: int = 0,
    block_size: int | None = None,
) -> float:
    """Supremum of a two-loop log-ODE merge error.

    For a normalized three-matrix tuple ``A``, this maximizes

    ``||exp(t[A_0,A_1]) exp(t[A_1,A_2])
       - exp(t([A_0,A_1]+[A_1,A_2]))||_op``

    over the joint Frobenius unit ball, where ``t = scale``. For square loops
    of side ``s`` and coefficient bound ``r``, use ``scale = (s*r)**2``.
    Projected ascent returns a lower estimate of the nonconvex supremum.
    """
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale must be finite and nonnegative")
    if scale == 0.0:
        return 0.0
    return _maximise_response(
        lambda response: _coordinate_loop_logode_objective_and_gradient(
            response, scale
        ),
        3,
        response_dim,
        restarts=restarts,
        steps=steps,
        seed=seed,
        sphere=False,
        block_size=block_size,
    )


def coordinate_loop_logode_hs_certificate(
    scale: float,
    *,
    tolerance: float = 1.0e-15,
    max_terms: int = 200,
) -> float:
    """Full weighted Hilbert--Schmidt certificate for the two-loop merge.

    ``scale`` is ``t = (s*r)**2`` for loop side ``s`` and coefficient bound
    ``r``. The closed-form level ``2m`` contribution is

    ``t**m * 2**(m/2) / m! * sqrt(comb(2m,m) - 2**m)``.
    """
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("scale must be finite and nonnegative")
    block_norm = np.sqrt(2.0) * scale
    return orthogonal_logode_hs_certificate(
        block_norm,
        block_norm,
        tolerance=tolerance,
        max_terms=max_terms,
    )


def orthogonal_logode_hs_certificate(
    left_norm: float,
    right_norm: float,
    *,
    tolerance: float = 1.0e-15,
    max_terms: int = 200,
) -> float:
    """Tail-controlled evaluation of the full orthogonal merge certificate.

    ``left_norm`` and ``right_norm`` are the weighted Hilbert--Schmidt norms of
    the two blocks, including path amplitude and response-radius scaling.
    Includes an analytic upper bound for omitted terms. Floating-point rounding
    is not interval-certified.
    """
    for name, value in (("left_norm", left_norm), ("right_norm", right_norm)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_terms < 3:
        raise ValueError("max_terms must be at least three")
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    left_log = 2.0 * math.log(left_norm)
    right_log = 2.0 * math.log(right_norm)
    total = 0.0
    growth = left_norm + right_norm
    for level in range(2, max_terms + 1):
        summands = []
        for left_count in range(1, level):
            choices = math.comb(level, left_count)
            summands.append(
                math.log(choices)
                + math.log(choices - 1)
                + left_count * left_log
                + (level - left_count) * right_log
            )
        log_squared_sum = float(logsumexp(summands))
        log_term = 0.5 * log_squared_sum - math.lgamma(level + 1.0)
        term = math.exp(log_term)
        total += term
        if level + 2 > growth:
            # Each level norm is <= 2 * growth**level / level!. Bound the
            # remaining exponential tail by its geometric majorant.
            log_tail = (
                math.log(2.0)
                + (level + 1) * math.log(growth)
                - math.lgamma(level + 2)
                - math.log1p(-growth / (level + 2))
            )
            tail = math.exp(log_tail)
            if tail <= tolerance * max(1.0, total):
                return total + tail
    raise RuntimeError("Hilbert--Schmidt certificate series did not converge")


def logode_merge_level_hs_norm(
    left_norm: float,
    right_norm: float,
    inner_product: float,
    level: int,
) -> float:
    """Level norm for merging two possibly nonorthogonal homogeneous blocks.

    The value depends only on their two norms and inner product. ``level`` is
    the meta-degree in the two blocks; if the blocks have tensor degree ``q``,
    this is signature degree ``q * level``.
    """
    for name, value in (
        ("left_norm", left_norm),
        ("right_norm", right_norm),
        ("inner_product", inner_product),
    ):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if left_norm < 0.0 or right_norm < 0.0:
        raise ValueError("block norms must be nonnegative")
    if not isinstance(level, int) or level < 1:
        raise ValueError("level must be a positive integer")
    gram_limit = left_norm * right_norm
    gram_tolerance = 1.0e-12 * gram_limit
    if abs(inner_product) > gram_limit + gram_tolerance:
        raise ValueError("inner_product violates Gram positive semidefiniteness")
    if abs(inner_product) > gram_limit:
        inner_product = math.copysign(gram_limit, inner_product)
    if level == 1 or left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    if abs(inner_product) == gram_limit:
        return 0.0
    norm_scale = max(left_norm, right_norm)
    if min(left_norm, right_norm) / norm_scale < 1.0e-50:
        # Global normalisation can erase the smaller squared norm.
        with localcontext() as context:
            context.prec = 80
            left, right, inner = map(
                Decimal.from_float, (left_norm, right_norm, inner_product)
            )
            return _decimal_gram_merge_norm(
                [[left * left, inner], [inner, right * right]], level
            )
    normalised_inner = (inner_product / norm_scale) / norm_scale
    matrix = np.array(
        [
            [(left_norm / norm_scale) ** 2, normalised_inner],
            [normalised_inner, (right_norm / norm_scale) ** 2],
        ]
    )
    factor = multi_logode_merge_level_hs_norm(matrix, level)
    if factor == 0.0:
        return 0.0
    log_value = level * math.log(norm_scale) + math.log(factor)
    return (
        math.exp(log_value) if log_value <= math.log(np.finfo(float).max) else math.inf
    )


def _weak_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    """All weak compositions of ``total`` into ``parts`` ordered parts."""
    if parts == 1:
        return [(total,)]
    return [
        (first, *tail)
        for first in range(total + 1)
        for tail in _weak_compositions(total - first, parts - 1)
    ]


def _decimal_gram_merge_difference(
    matrix: np.ndarray, words: list[np.ndarray], weights: list[int], level: int
) -> float:
    """Recompute cancellation-prone Gram polynomials at higher precision."""
    with localcontext() as context:
        context.prec = 80
        gram = [[Decimal.from_float(float(value)) for value in row] for row in matrix]
        fine = Decimal(0)
        for left_word, left_weight in zip(words, weights, strict=True):
            for right_word, right_weight in zip(words, weights, strict=True):
                value = Decimal(left_weight * right_weight)
                for left, right in zip(left_word, right_word, strict=True):
                    value *= gram[left][right]
                fine += value
        merged_base = sum((sum(row, Decimal(0)) for row in gram), Decimal(0))
        return float(fine - merged_base**level)


def _decimal_gram_merge_norm(matrix: list[list[Decimal]], level: int) -> float:
    """Original-scale evaluation across disparate Gram exponents."""
    nonzero = [abs(value) for row in matrix for value in row if value]
    if not nonzero:
        return 0.0
    spread = max(value.adjusted() for value in nonzero) - min(
        value.adjusted() for value in nonzero
    )
    with localcontext() as context:
        context.prec = max(80, level * spread + 80)
        compositions = _weak_compositions(level, len(matrix))
        words = [
            tuple(index for index, count in enumerate(counts) for _ in range(count))
            for counts in compositions
        ]
        weights = [
            math.factorial(level)
            // math.prod(math.factorial(count) for count in counts)
            for counts in compositions
        ]
        fine = Decimal(0)
        for left, left_weight in zip(words, weights, strict=True):
            for right, right_weight in zip(words, weights, strict=True):
                term = Decimal(left_weight * right_weight)
                for i, j in zip(left, right, strict=True):
                    term *= matrix[i][j]
                fine += term
        merged = sum((sum(row, Decimal(0)) for row in matrix), Decimal(0)) ** level
        difference = fine - merged
        if difference < 0:
            raise ArithmeticError("computed negative squared certificate norm")
        return float(difference.sqrt() / Decimal(math.factorial(level)))


def multi_logode_merge_level_hs_norm(
    gram: np.ndarray,
    level: int,
) -> float:
    """Level norm for merging an ordered family of homogeneous blocks.

    ``gram[i, j]`` is the Hilbert--Schmidt inner product of blocks ``i`` and
    ``j``.  All blocks must lie in the same homogeneous tensor layer.  The
    returned norm compares the level-``level`` parts of the ordered product of
    tensor exponentials and the exponential of their sum.
    """
    matrix = np.asarray(gram, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not matrix.size:
        raise ValueError("gram must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("gram must contain only finite values")
    if not isinstance(level, int) or level < 1:
        raise ValueError("level must be a positive integer")
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        return 0.0
    original = matrix.copy()
    matrix = matrix / scale
    tolerance = 1.0e-12
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=tolerance):
        raise ValueError("gram must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("gram must be positive semidefinite")
    if level == 1 or matrix.shape[0] == 1:
        return 0.0
    nonzero = np.abs(original[original != 0.0])
    if float(np.min(nonzero)) / scale < 1.0e-100:
        # Check small blocks using their own budgets before original-scale
        # evaluation. A normalised tiny diagonal may already have underflowed.
        diagonal = np.diag(original)
        if np.any(diagonal < 0.0):
            raise ValueError("gram must be positive semidefinite")
        positive = diagonal > 0.0
        if np.any(original[~positive] != 0.0):
            raise ValueError("gram must be positive semidefinite")
        roots = np.sqrt(diagonal[positive])
        correlations = original[np.ix_(positive, positive)] / roots[:, None]
        correlations /= roots[None, :]
        if (
            not np.all(np.isfinite(correlations))
            or float(np.min(np.linalg.eigvalsh(correlations))) < -tolerance
        ):
            raise ValueError("gram must be positive semidefinite")
        decimal_matrix = [
            [
                (
                    Decimal.from_float(float(original[i, j]))
                    + Decimal.from_float(float(original[j, i]))
                )
                / 2
                for j in range(len(original))
            ]
            for i in range(len(original))
        ]
        return _decimal_gram_merge_norm(decimal_matrix, level)

    block_count = matrix.shape[0]
    compositions = _weak_compositions(level, block_count)
    words = [np.repeat(np.arange(block_count), counts) for counts in compositions]
    multinomials = [
        math.factorial(level) // math.prod(math.factorial(count) for count in counts)
        for counts in compositions
    ]
    fine_terms = []
    for left_word, left_weight in zip(words, multinomials, strict=True):
        for right_word, right_weight in zip(words, multinomials, strict=True):
            fine_terms.append(
                left_weight
                * right_weight
                * float(np.prod(matrix[left_word, right_word]))
            )
    fine_squared_scaled = math.fsum(fine_terms)
    merged_squared_scaled = float(np.sum(matrix)) ** level
    difference_squared_scaled = fine_squared_scaled - merged_squared_scaled
    roundoff = 1.0e-11 * max(1.0, abs(fine_squared_scaled), abs(merged_squared_scaled))
    if abs(difference_squared_scaled) <= 100.0 * roundoff:
        difference_squared_scaled = _decimal_gram_merge_difference(
            matrix, words, multinomials, level
        )
    if difference_squared_scaled < -roundoff:
        raise ArithmeticError("computed negative squared certificate norm")
    difference_squared_scaled = max(0.0, difference_squared_scaled)
    if difference_squared_scaled == 0.0:
        return 0.0
    log_value = (
        (level / 2) * math.log(scale)
        + 0.5 * math.log(difference_squared_scaled)
        - math.lgamma(level + 1)
    )
    return (
        math.exp(log_value) if log_value <= math.log(np.finfo(float).max) else math.inf
    )


def hilbert_schmidt_bound(delta: np.ndarray) -> float:
    """Upper bound ``||D||_2`` of eq. (1.9) at ``r = 1``."""
    return float(np.linalg.norm(np.asarray(delta, dtype=float).reshape(-1)))


def shift_bound(delta: np.ndarray, channel: int, level: int) -> float:
    """Lower bound ``k^{-k/2} max_w |D^w|`` at ``r = 1``."""
    tensor = _as_tensor(delta, channel, level)
    return float(level ** (-level / 2) * np.max(np.abs(tensor)))


def tensor_spectral_norm(
    delta: np.ndarray,
    channel: int,
    level: int,
    *,
    restarts: int = 24,
    steps: int = 500,
    seed: int = 0,
) -> float:
    """Injective norm ``max |<D, u_1 (x) ... (x) u_k>|`` over unit vectors.

    Alternating maximisation: with all but one factor fixed the problem is
    linear, so the optimal factor is the normalised contraction.
    """
    tensor = _as_tensor(delta, channel, level)
    if not np.any(tensor):
        return 0.0
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(restarts):
        factors = [rng.standard_normal(channel) for _ in range(level)]
        factors = [f / np.linalg.norm(f) for f in factors]
        previous = -np.inf
        for _ in range(steps):
            for axis in range(level):
                contracted = tensor
                for other in reversed(range(level)):
                    if other == axis:
                        continue
                    contracted = np.tensordot(
                        contracted, factors[other], axes=([other], [0])
                    )
                norm = np.linalg.norm(contracted)
                if norm == 0.0:
                    break
                factors[axis] = contracted / norm
            contracted = tensor
            for axis in reversed(range(level)):
                contracted = np.tensordot(contracted, factors[axis], axes=([axis], [0]))
            value = float(abs(contracted))
            if value - previous < 1e-13:
                previous = value
                break
            previous = value
        best = max(best, previous)
    return best


def rank_one_bound(
    delta: np.ndarray, channel: int, level: int, **kwargs: object
) -> float:
    """Lower bound ``k^{-k/2} ||D||_sigma`` at ``r = 1``."""
    spectral = tensor_spectral_norm(delta, channel, level, **kwargs)  # type: ignore[arg-type]
    return float(level ** (-level / 2) * spectral)


def response_ratio(
    delta: np.ndarray,
    channel: int,
    level: int,
    response_dim: int,
    **kwargs: object,
) -> float:
    """``d_1^{(k)} / ||D||_2``, zero when ``D`` vanishes."""
    denominator = hilbert_schmidt_bound(delta)
    if denominator == 0.0:
        return 0.0
    return (
        level_response_sup(delta, channel, level, response_dim, **kwargs) / denominator
    )  # type: ignore[arg-type]
