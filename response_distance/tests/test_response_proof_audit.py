"""Exact symbolic anchors for the response-distance proof review.

SymPy is optional and is not added to the core installation requirements.
These checks verify elimination and coefficient identities; external proof
review remains separate.
"""

import pytest

sympy = pytest.importorskip("sympy")


def test_mixed_degree_four_elimination_and_admissible_stationary_point() -> None:
    t, r = sympy.symbols("t r")
    first = 8 * t * r**2 - (6 * t + 2) * r + 1
    numerator = (1 - 2 * t) * (1 - 2 * t * r) - 8 * r * t * (1 - t)
    squared = numerator**2 - 16 * r**2 * t * (1 - t)
    assert sympy.factor(sympy.resultant(first, squared, r)) == (
        16 * t * (2 * t - 1) ** 3 * (50 * t**2 - 53 * t + 4)
    )
    root = sympy.sqrt(41)
    tau, ratio, q = (53 - 7 * root) / 100, (21 - root) / 32, (21 + root) / 100
    assert sympy.simplify(first.subs({t: tau, r: ratio})) == 0
    assert sympy.simplify(q**2 - tau * (1 - tau)) == 0
    unsquared = (1 - 2 * t) * (1 - 2 * t * r) - 4 * r * (q + 2 * t * (1 - t))
    assert sympy.simplify(unsquared.subs({t: tau, r: ratio})) == 0
    other_ratio = sympy.simplify(1 / (8 * tau * ratio))
    assert (other_ratio - 1).is_positive
    value = sympy.simplify(
        (1 + 2 * q) * ratio * (1 - ratio) * (1 - 2 * tau * ratio) ** 2
    )
    assert sympy.simplify(value - (411 + 41 * root) / 2048) == 0
    assert (value - sympy.Rational(1, 4)).is_positive


@pytest.mark.parametrize("order", range(1, 13))
def test_truncated_exponential_log_concavity_identity(order: int) -> None:
    x = sympy.symbols("x")
    polynomial = sum(x**j / sympy.factorial(j) for j in range(order + 1))
    numerator = sympy.diff(polynomial, x) ** 2 - polynomial * sympy.diff(
        polynomial, x, 2
    )
    positive = (
        x ** (order - 1)
        / sympy.factorial(order)
        * sum((order - j) * x**j / sympy.factorial(j) for j in range(order))
    )
    assert sympy.expand(numerator - positive) == 0


def test_chain_threshold_ratio_is_increasing_at_every_later_degree() -> None:
    # For m=s+4, positivity proves sqrt(3)*(m-1)/(m+1)>1 for every s>=0.
    s = sympy.symbols("s", nonnegative=True)
    m = s + 4
    squared_gap = sympy.expand(3 * (m - 1) ** 2 - (m + 1) ** 2)
    assert squared_gap == 2 * s**2 + 8 * s + 2
    assert squared_gap.is_positive
