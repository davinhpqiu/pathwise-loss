"""Finite-scale planar degree-two log-ODE response bounds.

For a planar BV path X, compare its exact development with degree-two log-ODE
factors on any fixed partition. Under path dilation t, the first discrepancy
is t**3 times the sum of local third log-signatures. If nonzero, the ratio tends
to RHO3 = 2 sqrt(2)/9, uniformly across response dimensions n >= 2.

The same bounds apply to merging two log blocks v_i + alpha_i [e1,e2].
The coarse block includes the degree-two BCH cross term. A positive exponential
tail bounds all omitted degrees, providing a finite separation radius.

Proofs: response_distance/technical.md. Evaluations are floating point,
without interval-certified rounding. No response-system optimization is used.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.special import logsumexp

from .response_storage import log_scalar_euler_error

RHO3 = 2.0 * math.sqrt(2.0) / 9.0


@dataclass(frozen=True)
class PlanarMergeData:
    """Exact cubic coefficients and scalar majorants for one comparison.

    Leading Lie tensor is [w_1 e1 + w_2 e2, [e1,e2]]. Source homogeneous
    coefficient norms are majorized by exp(source_a*t + source_b*t**2);
    merged coefficients by exp(merged_a*t + merged_b*t**2).
    Degree-two blocks are planar area tensors, so their response growth uses
    half the HS budget. For a BV source, source_a is path length, source_b=0.
    """

    cubic: tuple[float, float]
    source_a: float
    source_b: float
    merged_a: float
    merged_b: float

    def __post_init__(self) -> None:
        if len(self.cubic) != 2 or not all(math.isfinite(x) for x in self.cubic):
            raise ValueError("cubic must contain two finite coefficients")
        for value in (self.source_a, self.source_b, self.merged_a, self.merged_b):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("majorant parameters must be finite and nonnegative")

    @property
    def leading_hs(self) -> float:
        return math.sqrt(6.0) * math.hypot(*self.cubic)

    @property
    def leading_response(self) -> float:
        return RHO3 * self.leading_hs


@dataclass(frozen=True)
class PlanarMergeBounds:
    response_lower: float
    response_upper: float
    certificate_lower: float
    certificate_upper: float
    response_remainder: float
    hs_remainder: float


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite planar vector")
    return vector


def _det(left: np.ndarray, right: np.ndarray) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _block_log_data(steps: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    increment = np.zeros(2)
    area = 0.0
    cubic = np.zeros(2)
    for step in steps:
        cross = _det(increment, step)
        cubic += -0.5 * area * step + cross * (increment - step) / 12.0
        area += cross / 2.0
        increment += step
    return increment, area, cubic


def planar_bv_logode_data(
    path: np.ndarray, partition: tuple[int, ...] | None = None
) -> PlanarMergeData:
    """Cubic log-signature and majorants for a piecewise-linear BV path.

    Degree-three BCH recursion computes log coefficients directly; it avoids
    subtracting large symmetric signature terms to recover a small Lie part.
    """
    values = np.asarray(path, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != 2:
        raise ValueError("path must have shape (points, 2), with at least two points")
    if not np.all(np.isfinite(values)):
        raise ValueError("path must contain only finite values")
    steps = np.diff(values, axis=0)
    cuts = (0, len(values) - 1) if partition is None else tuple(partition)
    if (
        len(cuts) < 2
        or cuts[0] != 0
        or cuts[-1] != len(values) - 1
        or any(not isinstance(index, int) or isinstance(index, bool) for index in cuts)
        or any(left >= right for left, right in pairwise(cuts))
    ):
        raise ValueError(
            "partition must be increasing integer vertex indices including both endpoints"
        )
    blocks = [_block_log_data(steps[left:right]) for left, right in pairwise(cuts)]
    cubic = tuple(
        math.fsum(float(block[2][index]) for block in blocks) for index in range(2)
    )
    return PlanarMergeData(
        cubic,
        math.fsum(float(np.linalg.norm(step)) for step in steps),
        0.0,
        math.fsum(float(np.linalg.norm(block[0])) for block in blocks),
        math.sqrt(2.0) * math.fsum(abs(block[1]) for block in blocks),
    )


def planar_mixed_merge_data(
    left: np.ndarray, right: np.ndarray, left_area: float, right_area: float
) -> PlanarMergeData:
    """Two mixed log blocks versus their correct degree-two BCH merge."""
    left, right = _vector(left, "left"), _vector(right, "right")
    if not math.isfinite(left_area) or not math.isfinite(right_area):
        raise ValueError("area coefficients must be finite")
    cross = _det(left, right)
    cubic = (right_area * left - left_area * right) / 2.0 + cross * (
        left - right
    ) / 12.0
    return PlanarMergeData(
        tuple(float(x) for x in cubic),
        float(np.linalg.norm(left) + np.linalg.norm(right)),
        math.sqrt(2.0) * (abs(left_area) + abs(right_area)),
        float(np.linalg.norm(left + right)),
        math.sqrt(2.0) * abs(left_area + right_area + cross / 2.0),
    )


def quadratic_exponential_tail4(a: float, b: float, scale: float) -> float:
    """Sum of degrees >=4 in exp(a*t + b*t**2), evaluated without subtraction.

    With p=a*t and q=b*t**2 the positive decomposition is
    (exp(p)-P_3(p)) + q*(exp(p)-P_1(p)) + exp(p)*(exp(q)-P_1(q)).
    """
    if not all(math.isfinite(x) and x >= 0.0 for x in (a, b, scale)):
        raise ValueError("tail parameters must be finite and nonnegative")
    if scale == 0.0:
        return 0.0
    p, q = a * scale, (b * scale) * scale
    if not math.isfinite(p + q):
        return math.inf
    logs = [
        log_scalar_euler_error(3, 1, p),
        math.log(q) + log_scalar_euler_error(1, 1, p) if q else -math.inf,
        p + log_scalar_euler_error(1, 1, q),
    ]
    value = float(logsumexp(logs))
    return math.exp(value) if value < math.log(np.finfo(float).max) else math.inf


def planar_merge_bounds(data: PlanarMergeData, scale: float) -> PlanarMergeBounds:
    """Analytic full-response and full-HS bounds at joint scale t=s*r."""
    response_tail = quadratic_exponential_tail4(
        data.source_a, data.source_b / 2, scale
    ) + quadratic_exponential_tail4(data.merged_a, data.merged_b / 2, scale)
    hs_tail = quadratic_exponential_tail4(
        data.source_a, data.source_b, scale
    ) + quadratic_exponential_tail4(data.merged_a, data.merged_b, scale)
    leading_hs = ((data.leading_hs * scale) * scale) * scale
    leading_response = RHO3 * leading_hs
    return PlanarMergeBounds(
        max(0.0, leading_response - response_tail),
        leading_response + response_tail,
        leading_hs,
        leading_hs + hs_tail,
        response_tail,
        hs_tail,
    )


def planar_merge_separation_radius(
    data: PlanarMergeData, *, max_scale: float | None = None
) -> float:
    """Radius guaranteeing response <= (1+RHO3)/2 * leading HS < full HS.

    Returns zero if the cubic term vanishes. The default comparison scale is
    reciprocal to the largest linear/quadratic growth scale, giving covariance
    under path dilation. This is a sufficient radius, not the largest one.
    """
    if max_scale is not None and (not math.isfinite(max_scale) or max_scale <= 0.0):
        raise ValueError("max_scale must be finite and positive")
    if data.leading_hs == 0.0:
        return 0.0
    growth = max(
        data.source_a,
        math.sqrt(data.source_b),
        data.merged_a,
        math.sqrt(data.merged_b),
    )
    if growth == 0.0:
        raise ValueError("nonzero cubic term requires nonzero growth majorants")
    reference = max_scale if max_scale is not None else 1.0 / growth
    remainder = planar_merge_bounds(data, reference).response_remainder
    if remainder == 0.0:
        return reference
    # R(t)/t**4 is increasing: every remaining coefficient is nonnegative.
    normalised_leading = ((data.leading_hs * reference) * reference) * reference
    return min(
        reference, reference * (1.0 - RHO3) * normalised_leading / (2.0 * remainder)
    )


def planar_bv_logode_neighborhood(
    path: np.ndarray, partition: tuple[int, ...] | None = None
) -> tuple[float, float]:
    """Variation-perturbation radius and uniform separation scale for a BV path.

    For anchored paths Y with ||X-Y||_1-var <= eta, both radii are valid.
    Signature differentiation gives ||log_3 S(X)-log_3 S(Y)|| <= 3 L**2 eta
    when both lengths are <= L. Returns (0,0) in the cubic-degenerate case.
    """
    data = planar_bv_logode_data(path, partition)
    if data.leading_hs == 0.0:
        return 0.0, 0.0
    length = data.source_a + 1.0
    eta = min(1.0, data.leading_hs / (6.0 * length**2))
    # All paths in this neighborhood have a>=0, a<=L, b<=L**2 and h>=h0/2.
    uniform = PlanarMergeData(
        (data.leading_hs / (2.0 * math.sqrt(6.0)), 0.0),
        length,
        0.0,
        length,
        length**2,
    )
    return eta, planar_merge_separation_radius(uniform, max_scale=1.0 / length)


def planar_mixed_merge_neighborhood(
    left: np.ndarray, right: np.ndarray, left_area: float, right_area: float
) -> tuple[float, float]:
    """Parameter-perturbation radius and uniform separation scale.

    Each increment changes by at most eta in Euclidean norm, and each area
    coefficient by at most eta in absolute value. Cubic-degenerate cases return
    (0,0). These are sufficient radii from the explicit Lipschitz proof.
    """
    data = planar_mixed_merge_data(left, right, left_area, right_area)
    if data.leading_hs == 0.0:
        return 0.0, 0.0
    v_bound = max(np.linalg.norm(left), np.linalg.norm(right)) + 1.0
    area_bound = max(abs(left_area), abs(right_area)) + 1.0
    cubic_floor = data.leading_hs / (2.0 * math.sqrt(6.0))
    lipschitz = area_bound + v_bound + v_bound**2 / 2.0
    eta = min(1.0, cubic_floor / lipschitz)
    uniform = PlanarMergeData(
        (cubic_floor, 0.0),
        2.0 * v_bound,
        2.0 * math.sqrt(2.0) * area_bound,
        2.0 * v_bound,
        math.sqrt(2.0) * (2.0 * area_bound + v_bound**2 / 2.0),
    )
    return eta, planar_merge_separation_radius(uniform)


def planar_cubic_attaining_tuple(cubic: tuple[float, float]) -> np.ndarray:
    """A unit Frobenius 2x2 tuple attaining the degree-three response gauge."""
    vector = _vector(np.asarray(cubic), "cubic")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("attainment requires a nonzero cubic direction")
    first = vector / norm
    second = np.array([-first[1], first[0]])
    diagonal = np.diag([1.0, -1.0]) / math.sqrt(3.0)
    nilpotent = np.array([[0.0, 1.0], [0.0, 0.0]]) / math.sqrt(3.0)
    return first[:, None, None] * diagonal + second[:, None, None] * nilpotent
