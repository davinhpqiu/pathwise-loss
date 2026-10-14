"""Integer storage optimization for scalar Euler errors and pure Lie paths.

The scalar error is the exact bounded-variation minimax error, with adaptive
equal-variation partitions. For pure Lie paths the same function supplies
response lower/upper bounds; it is not an exact response supremum there.
Proofs and experimental scope: response_distance/technical.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import cache

from scipy.special import gammainc


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _scale(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("scale must be finite and nonnegative")


def _log_poisson_tail(order: int, scale: float) -> float:
    """Log Pr(Poisson(scale) > order), preserving underflowed small tails."""
    tail = float(gammainc(order + 1, scale))
    if tail > 0:
        return math.log(tail)
    # Underflow implies scale is well below order; sum relative to first term.
    term = total = 1.0
    index = order + 1
    while True:
        index += 1
        term *= scale / index
        previous = total
        total += term
        if total == previous:
            break
    return (
        -scale
        + (order + 1) * math.log(scale)
        - math.lgamma(order + 2)
        + math.log(total)
    )


def log_scalar_euler_error(order: int, intervals: int, scale: float) -> float:
    """Log of exp(scale) - P_order(scale / intervals)**intervals.

    ``P_order(x) = sum_{j=0}^order x**j / j!``. Evaluation uses the Poisson
    survival probability and log arithmetic, avoiding cancellation/overflow.
    """
    _positive_integer(order, "order")
    _positive_integer(intervals, "intervals")
    _scale(scale)
    if scale == 0:
        return -math.inf
    log_intervals = math.log(intervals)
    log_local_scale = math.log(scale) - log_intervals
    local_scale = math.exp(log_local_scale) if log_local_scale > -745 else 0.0
    if local_scale == 0:
        # First omitted term is asymptotically exact at a subnormal scale.
        log_tail = (order + 1) * log_local_scale - math.lgamma(order + 2)
    else:
        log_tail = _log_poisson_tail(order, local_scale)
    if log_tail == 0:
        return scale
    if log_tail < -36:
        log_hazard = log_tail
    else:
        log_hazard = math.log(-math.log1p(-math.exp(log_tail)))
    log_accumulated_hazard = log_intervals + log_hazard
    if log_accumulated_hazard < -36:
        log_deficit = log_accumulated_hazard
    elif log_accumulated_hazard > 4:
        log_deficit = 0.0
    else:
        log_deficit = math.log(-math.expm1(-math.exp(log_accumulated_hazard)))
    return scale + log_deficit


def log_fpl_euler_bound(order: int, intervals: int, scale: float) -> float:
    """FPL Theorem 1.3 bound, in log arithmetic."""
    _positive_integer(order, "order")
    _positive_integer(intervals, "intervals")
    _scale(scale)
    if scale == 0:
        return -math.inf
    return (
        scale
        + (order + 1) * math.log(scale)
        - math.lgamma(order + 2)
        - order * math.log(intervals)
    )


@cache
def _mobius(number: int) -> int:
    sign = 1
    prime = 2
    while prime * prime <= number:
        if number % prime == 0:
            number //= prime
            sign = -sign
            if number % prime == 0:
                return 0
        prime += 1
    return -sign if number > 1 else sign


@cache
def lie_storage_dimension(channel: int, depth: int) -> int:
    """Exact cumulative free Lie dimension, by Witt's formula."""
    _positive_integer(channel, "channel")
    _positive_integer(depth, "depth")
    return sum(
        sum(
            _mobius(divisor) * channel ** (level // divisor)
            for divisor in range(1, level + 1)
            if level % divisor == 0
        )
        // level
        for level in range(1, depth + 1)
    )


@dataclass(frozen=True)
class StorageChoice:
    depth: int
    order: int
    intervals: int
    storage: int
    log_error: float


def _decimal_gap(order, intervals, scale, epsilon, criterion):
    """High precision log(error / epsilon) at an integer boundary."""
    with localcontext() as context:
        context.prec = max(80, int(math.log10(intervals)) + 45)
        z = Decimal.from_float(float(scale))
        m = Decimal(intervals)
        if criterion == "fpl":
            log_error = (
                z
                + (order + 1) * z.ln()
                - Decimal(math.factorial(order + 1)).ln()
                - order * m.ln()
            )
        else:
            x = z / m
            term = total = Decimal(1)
            index = order + 1
            while True:
                index += 1
                term *= x / index
                previous = total
                total += term
                if total == previous:
                    break
            log_tail = (
                -x
                + (order + 1) * x.ln()
                - Decimal(math.factorial(order + 1)).ln()
                + total.ln()
            )
            tail = log_tail.exp()
            if tail < Decimal("0.1"):
                # -log(1-a)/a = 1 + a/2 + a^2/3 + ... .
                power = ratio = Decimal(1)
                index = 1
                while True:
                    power *= tail
                    previous = ratio
                    ratio += power / (index + 1)
                    if ratio == previous:
                        break
                    index += 1
                log_hazard = log_tail + ratio.ln()
            else:
                log_hazard = (-(1 - tail).ln()).ln()
            log_accumulated = m.ln() + log_hazard
            accumulated = log_accumulated.exp()
            if accumulated < Decimal("0.1"):
                # (1-exp(-t))/t = 1 - t/2 + t^2/6 - ... .
                term = ratio = Decimal(1)
                index = 1
                while True:
                    term *= -accumulated / (index + 1)
                    previous = ratio
                    ratio += term
                    if ratio == previous:
                        break
                    index += 1
                log_deficit = log_accumulated + ratio.ln()
            else:
                log_deficit = (1 - (-accumulated).exp()).ln()
            log_error = z + log_deficit
        return log_error - Decimal.from_float(float(epsilon)).ln()


def _decimal_feasible(order, intervals, scale, epsilon, criterion):
    return _decimal_gap(order, intervals, scale, epsilon, criterion) <= 0


def _refine_interval_boundary(order, candidate, scale, epsilon, criterion):
    if candidate > 10**9:
        # Near the root, d log(error) / d log(m) is close to -order.
        # A log Newton step avoids scanning hundreds of digits of float
        # rounding uncertainty. Final neighboring-integer checks are exact
        # to the working decimal precision, regardless of this approximation.
        for _ in range(32):
            gap = _decimal_gap(order, candidate, scale, epsilon, criterion)
            with localcontext() as context:
                context.prec = max(80, int(math.log10(candidate)) + 45)
                updated = max(1, int((Decimal(candidate).ln() + gap / order).exp()))
            difference = abs(updated - candidate)
            candidate = updated
            if difference <= 2:
                break
        else:
            raise RuntimeError("high precision boundary refinement did not converge")

    def feasible(m):
        return _decimal_feasible(order, m, scale, epsilon, criterion)

    if feasible(candidate):
        right = candidate
        step = 1
        while right > step and feasible(right - step):
            step *= 2
        left = max(0, right - step)
    else:
        left = candidate
        step = 1
        while not feasible(left + step):
            step *= 2
        right = left + step
    while right - left > 1:
        middle = (left + right) // 2
        if feasible(middle):
            right = middle
        else:
            left = middle
    return right


def minimum_intervals(
    order: int, scale: float, epsilon: float, *, criterion: str = "exact"
) -> int:
    """Least feasible positive integer m, found by bracketing and bisection."""
    _positive_integer(order, "order")
    _scale(scale)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if criterion not in ("exact", "fpl"):
        raise ValueError("criterion must be 'exact' or 'fpl'")
    error = log_scalar_euler_error if criterion == "exact" else log_fpl_euler_bound
    threshold = math.log(epsilon)
    if euler_error_feasible(order, 1, scale, epsilon, criterion=criterion):
        return 1
    left, right = 1, 2
    while error(order, right, scale) > threshold:
        left, right = right, right * 2
    while right - left > 1:
        middle = (left + right) // 2
        if error(order, middle, scale) <= threshold:
            right = middle
        else:
            left = middle
    # A float boundary may span many integers when m is enormous. Resolve
    # close comparisons using sufficient decimal digits to distinguish m-1.
    near = min(
        abs(error(order, right, scale) - threshold),
        abs(error(order, right - 1, scale) - threshold),
    )
    if near < 1e-9:
        return _refine_interval_boundary(order, right, scale, epsilon, criterion)
    return right


def euler_error_feasible(order, intervals, scale, epsilon, *, criterion="exact"):
    """Check feasibility, resolving close boundaries with decimal arithmetic."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if criterion not in ("exact", "fpl"):
        raise ValueError("criterion must be 'exact' or 'fpl'")
    error = log_scalar_euler_error if criterion == "exact" else log_fpl_euler_bound
    gap = error(order, intervals, scale) - math.log(epsilon)
    if abs(gap) < 1e-9:
        return _decimal_feasible(order, intervals, scale, epsilon, criterion)
    return gap <= 0


def optimize_euler_storage(
    channel: int,
    scale: float,
    epsilon: float,
    *,
    degree: int = 1,
    criterion: str = "exact",
) -> tuple[StorageChoice, list[StorageChoice]]:
    """Global integer minimum of m Lambda_d(degree * order).

    For degree=1 and criterion='exact', this solves the BV minimax problem.
    For degree>1, callers must supply an effective pure-Lie error scale and
    state whether that scale is a lower bound, an upper bound, or HS size.
    Dense generic log-signature storage is counted in all cases. Orders are
    exhausted until even one interval costs more than the incumbent optimum.
    All visited rows are returned, including ties; ties prefer smaller depth.
    """
    _positive_integer(channel, "channel")
    if channel < 2:
        raise ValueError("channel must be at least two")
    _positive_integer(degree, "degree")
    rows: list[StorageChoice] = []
    best: StorageChoice | None = None
    order = 1
    error = log_scalar_euler_error if criterion == "exact" else log_fpl_euler_bound
    while True:
        depth = degree * order
        dimension = lie_storage_dimension(channel, depth)
        if best is not None and dimension > best.storage:
            break
        intervals = minimum_intervals(order, scale, epsilon, criterion=criterion)
        row = StorageChoice(
            depth,
            order,
            intervals,
            intervals * dimension,
            error(order, intervals, scale),
        )
        rows.append(row)
        if best is None or row.storage < best.storage:
            best = row
        order += 1
    assert best is not None
    return best, rows


def engel_scales(degree: int) -> tuple[float, float, float]:
    """HS size, response upper gauge, and attained positive eigenvalue.

    E_k=ad_{e1}^{k-1}(e2), k>=2. The response gauge is exact, but Euler
    tails at this scale are upper bounds. A symmetric 2x2 development attains
    a positive eigenvalue gamma/sqrt(2), giving a rigorous error lower bound.
    """
    _positive_integer(degree, "degree")
    if degree < 2:
        raise ValueError("degree must be at least two")
    hs = math.sqrt(math.comb(2 * degree - 2, degree - 1))
    log_gauge = 0.5 * (degree - 1) * math.log(
        2 * (degree - 1)
    ) - 0.5 * degree * math.log(degree)
    gauge = math.exp(log_gauge)
    return hs, gauge, gauge / math.sqrt(2)
