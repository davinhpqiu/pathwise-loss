"""Exact arithmetic anchors for the homogeneous blindness checker."""

import numpy as np
from scripts.check_response_separation import P, evaluate, identity_table, matmul_mod_p


def test_modular_product_handles_maximal_residues() -> None:
    left = np.full((4, 4), P - 1, dtype=np.int64)
    right = left.copy()
    # Every entry is 4 * (-1) * (-1) = 4 modulo P.
    assert np.array_equal(matmul_mod_p(left, right), np.full((4, 4), 4))


def test_modular_bracket_matches_python_integer_arithmetic() -> None:
    rng = np.random.default_rng(211)
    for dimension in (2, 3, 4):
        matrices = [
            rng.integers(0, P, (dimension, dimension), dtype=np.int64) for _ in range(2)
        ]
        left, right = (matrix.astype(object) for matrix in matrices)
        expected = np.asarray((left @ right - right @ left) % P, dtype=np.int64)
        actual = evaluate((0, 1), matrices)
        assert np.array_equal(actual, expected)
        assert int(np.trace(actual)) % P == 0


def test_planar_degree_six_identity_threshold() -> None:
    rows = identity_table(2, 2, 6, seed=211)
    # Full rank is a nonzero minor modulo P and excludes a real identity.
    assert all(rank == dimension for level, dimension, rank, _, _ in rows if level < 6)
    assert rows[-1][3] > 0
    assert all(rank <= cap for _, _, rank, _, cap in rows)


def test_three_letter_degree_five_identity() -> None:
    """The trace-pairing identity has an explicit integer n=3 witness."""

    def bracket(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return left @ right - right @ left

    rng = np.random.default_rng(212)
    for _ in range(12):
        a, b, c = rng.integers(-10, 11, (3, 2, 2))
        assert not np.any(bracket(bracket(bracket(a, b), bracket(a, c)), a))
    a = np.diag([1, 0, -2])
    b, c = np.zeros((2, 3, 3), dtype=int)
    b[0, 1], c[1, 2] = 1, 1
    expected = np.zeros((3, 3), dtype=int)
    expected[0, 2] = -6
    assert np.array_equal(bracket(bracket(bracket(a, b), bracket(a, c)), a), expected)
