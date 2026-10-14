from fractions import Fraction

import pytest

from response_distance.code.response_certificates import (
    certify_bv_storage_example,
    exp_rational_bounds,
    planar_corner_certificate,
)


def test_exponential_bound_against_known_rational_enclosure() -> None:
    lower, upper = exp_rational_bounds(Fraction(1))
    assert Fraction(2718281828459045, 10**15) < lower < upper
    assert upper < Fraction(2718281828459046, 10**15)
    assert upper - lower < Fraction(1, 10**24)
    assert exp_rational_bounds(Fraction(0)) == (1, 1)


@pytest.mark.parametrize(
    "value,order", [(Fraction(-1), 24), (Fraction(30), 1), (Fraction(1), 0)]
)
def test_exponential_bound_rejects_invalid_budgets(value, order) -> None:
    with pytest.raises(ValueError):
        exp_rational_bounds(value, order)


def test_corner_certificate_has_exact_sqrt_and_tolerance_margins() -> None:
    assert Fraction(707107, 500000) ** 2 > 2
    assert Fraction(34641, 20000) ** 2 < 3
    assert Fraction(244949, 100000) ** 2 > 6
    bounds = planar_corner_certificate()
    assert bounds.response_upper < Fraction(1934, 10**8)
    assert Fraction(1934, 10**8) < Fraction(1, 50000)
    assert Fraction(1, 50000) < Fraction(3608, 10**8)
    assert Fraction(3608, 10**8) < bounds.certificate_lower
    assert planar_corner_certificate(Fraction(0)).response_upper == 0


def test_storage_example_global_minima_are_rationally_decided() -> None:
    assert certify_bv_storage_example() == {"exact": (4, 1, 8), "fpl": (5, 1, 14)}
