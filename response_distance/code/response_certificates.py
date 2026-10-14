"""Exact rational anchors for the response-distance sending note.

These certify the displayed planar corner and small storage comparison.
General study outputs remain floating-point evaluations.
"""

from dataclasses import dataclass
from fractions import Fraction
from math import factorial


def exp_rational_bounds(value: Fraction, order: int = 24) -> tuple[Fraction, Fraction]:
    """Enclose exp(value) by a Taylor sum and a geometric omitted-tail bound."""
    if not isinstance(value, Fraction) or value < 0:
        raise ValueError("value must be a nonnegative Fraction")
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    ratio = value / (order + 2)
    if ratio >= 1:
        raise ValueError("order is too small for the geometric tail bound")
    term = total = Fraction(1)
    for degree in range(1, order + 1):
        term *= value / degree
        total += term
    omitted = term * value / (order + 1)
    return total, total + omitted / (1 - ratio)


def _tail4_upper(a: Fraction, b: Fraction, scale: Fraction) -> Fraction:
    p, q = a * scale, b * scale**2
    _, upper = exp_rational_bounds(p + q)
    return upper - (1 + p + p**2 / 2 + q + p**3 / 6 + p * q)


@dataclass(frozen=True)
class CornerCertificate:
    scale: Fraction
    response_upper: Fraction
    certificate_lower: Fraction


def planar_corner_certificate(
    scale: Fraction = Fraction(1, 20),
) -> CornerCertificate:
    """Rational bounds for the path 0 -> e1 -> e1+e2, on one interval.

    Rational squared comparisons bound sqrt(2), sqrt(3) and sqrt(6).
    Positive tail coefficients permit monotone replacement of their parameters.
    """
    if not isinstance(scale, Fraction) or scale < 0:
        raise ValueError("scale must be a nonnegative Fraction")
    root2_upper = Fraction(707107, 500000)
    root3_lower = Fraction(34641, 20000)
    root6_upper = Fraction(244949, 100000)
    response = (
        root6_upper * scale**3 / 27
        + _tail4_upper(Fraction(2), Fraction(0), scale)
        + _tail4_upper(root2_upper, root2_upper / 4, scale)
    )
    return CornerCertificate(scale, response, root3_lower * scale**3 / 6)


def certify_bv_storage_example() -> dict[str, tuple[int, int, int]]:
    """Certify the d=2, r=1, epsilon=1/100 optima with exact arithmetic.

    The five cumulative Lie dimensions are 2, 3, 5, 8 and 14.
    Larger depths cost more even with one interval. Every candidate up to the
    asserted optimum cost is checked against a rational enclosure of exp(1).
    """
    e_lower, e_upper = exp_rational_bounds(Fraction(1))
    epsilon = Fraction(1, 100)
    dimensions = (2, 3, 5, 8, 14)
    limits = {"exact": 8, "fpl": 14}
    results = {}
    for criterion, budget in limits.items():
        feasible = []
        for depth, dimension in enumerate(dimensions, start=1):
            for intervals in range(1, budget // dimension + 1):
                if criterion == "exact":
                    polynomial = sum(
                        (
                            Fraction(1, intervals) ** j / factorial(j)
                            for j in range(depth + 1)
                        ),
                        Fraction(0),
                    )
                    lower = e_lower - polynomial**intervals
                    upper = e_upper - polynomial**intervals
                else:
                    denominator = factorial(depth + 1) * intervals**depth
                    lower, upper = e_lower / denominator, e_upper / denominator
                if upper <= epsilon:
                    feasible.append((depth, intervals, intervals * dimension))
                elif lower <= epsilon:
                    raise ArithmeticError(
                        "rational enclosure does not decide feasibility"
                    )
        minimum = min(row[2] for row in feasible)
        winners = [row for row in feasible if row[2] == minimum]
        if len(winners) != 1:
            raise ArithmeticError("storage example has an unexpected tie")
        results[criterion] = winners[0]
    return results
