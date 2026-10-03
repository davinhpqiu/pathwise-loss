"""Measure how far the Hilbert-Schmidt bound overestimates response discrepancy.

For a pair of paths and signature level ``k``, with ``D = S^{(k)}(X) - S^{(k)}(Xt)``,

    d_1^{(k)} = sup_{|A| <= 1} || sum_{|w|=k} D^w A_{w_1} ... A_{w_k} ||_op,

and eq. (1.9) of Ferrucci, Perree and Lyons gives ``d_1^{(k)} <= ||D||_2``.  This
script records ``ratio = d_1^{(k)} / ||D||_2`` against response dimension ``n``,
alongside two lower bounds at the same normalisation: shift bound
``k^{-k/2} max_w |D^w| / ||D||_2`` and rank-one bound
``k^{-k/2} ||D||_sigma / ||D||_2``.  All three are independent of ``r``, since
both sides scale as ``r^k``.

Path pairs cover cases the project already cares about: equal increment with
different area, sign-balanced area moved between blocks, a local perturbation,
independent Brownian paths.  Synthetic discrepancies then span symmetry types
inside level ``k``: one word, rank-one symmetric, symmetrised random, generic
random, and free Lie basis elements.

Output is a CSV plus a metadata JSON under ``results/runs/response_gap`` by
default; analysis lives in the accompanying notes, not in this file.

Run::

    python scripts/study_response_gap.py --out results/runs/response_gap
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

from pathloss.provenance import create_run_directory, run_metadata
from pathloss.response import (
    hilbert_schmidt_bound,
    level_response_sup,
    piecewise_linear_signature,
    rank_one_bound,
    shift_bound,
)


def _loop(centre: np.ndarray, width: float, height: float, sign: float) -> np.ndarray:
    """Closed rectangle traversed anticlockwise when ``sign`` is positive."""
    x, y = centre
    corners = np.array(
        [
            [x, y],
            [x + sign * width, y],
            [x + sign * width, y + height],
            [x, y + height],
            [x, y],
        ]
    )
    return corners


def pair_equal_increment_different_area() -> tuple[np.ndarray, np.ndarray]:
    """Two closed loops, zero increment, signed areas 1 and 2."""
    return _loop(np.zeros(2), 1.0, 1.0, 1.0), _loop(np.zeros(2), 2.0, 1.0, 1.0)


def pair_balanced_area_moved() -> tuple[np.ndarray, np.ndarray]:
    """Sign-balanced pair of loops, same total area, different placement.

    Level-two difference cancels; higher levels see placement.
    """
    line = np.array([[0.0, 0.0], [3.0, 0.0]])
    first = np.vstack(
        (
            line,
            _loop(np.array([3.0, 0.0]), 0.5, 0.5, 1.0)[1:],
            np.array([[4.0, 0.0]]),
            _loop(np.array([4.0, 0.0]), 0.5, 0.5, -1.0)[1:],
            np.array([[6.0, 0.0]]),
        )
    )
    second = np.vstack(
        (
            line,
            _loop(np.array([3.0, 0.0]), 0.5, 0.5, -1.0)[1:],
            np.array([[4.0, 0.0]]),
            _loop(np.array([4.0, 0.0]), 0.5, 0.5, 1.0)[1:],
            np.array([[6.0, 0.0]]),
        )
    )
    return first, second


def pair_local_perturbation(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """One interior vertex displaced on an otherwise identical path."""
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.standard_normal((17, 2)) / np.sqrt(16), axis=0)
    base = np.vstack((np.zeros((1, 2)), base))
    moved = base.copy()
    moved[8] += np.array([0.35, -0.2])
    return base, moved


def pair_independent_brownian(seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    steps = 64
    first = np.vstack(
        (
            np.zeros((1, 2)),
            np.cumsum(rng.standard_normal((steps, 2)) / np.sqrt(steps), axis=0),
        )
    )
    second = np.vstack(
        (
            np.zeros((1, 2)),
            np.cumsum(rng.standard_normal((steps, 2)) / np.sqrt(steps), axis=0),
        )
    )
    return first, second


PATH_PAIRS = {
    "equal_increment_different_area": pair_equal_increment_different_area,
    "balanced_area_moved": pair_balanced_area_moved,
    "local_perturbation": pair_local_perturbation,
    "independent_brownian": pair_independent_brownian,
}


def _bracket(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.outer(left, right).reshape(-1) - np.outer(right, left).reshape(-1)


def lie_basis(channel: int, level: int) -> list[np.ndarray]:
    """Left-normed brackets spanning the degree-``level`` free Lie component."""
    letters = [np.eye(channel)[i] for i in range(channel)]
    current: list[np.ndarray] = letters
    for _ in range(level - 1):
        nxt = [_bracket(letter, element) for letter in letters for element in current]
        current = [x for x in nxt if np.any(x)]
    kept: list[np.ndarray] = []
    matrix = np.zeros((0, channel**level))
    for candidate in current:
        trial = np.vstack((matrix, candidate))
        if np.linalg.matrix_rank(trial, tol=1e-10) > matrix.shape[0]:
            matrix, _ = trial, kept.append(candidate)
    return kept


def synthetic_deltas(channel: int, level: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Discrepancies specified directly, chosen to span symmetry types."""
    rng = np.random.default_rng(seed + level)
    size = channel**level
    out: dict[str, np.ndarray] = {}

    alternating = np.zeros(size)
    index = 0
    for position in range(level):
        index = index * channel + (position % channel)
    alternating[index] = 1.0
    out["alternating_word"] = alternating

    out["rank_one_symmetric"] = np.ones(size) / np.sqrt(size)

    if level == 2:
        # Isotropic symmetric tensor, ratio d^{-1/2} in closed form.
        out["isotropic_symmetric"] = (np.eye(channel) / np.sqrt(channel)).reshape(-1)

    tensor = rng.standard_normal((channel,) * level)
    symmetric = np.zeros_like(tensor)
    for perm in permutations(range(level)):
        symmetric += np.transpose(tensor, perm)
    out["symmetric_random"] = symmetric.reshape(-1)

    out["generic_random"] = rng.standard_normal(size)

    for index, element in enumerate(lie_basis(channel, level)):
        out[f"lie_basis_{index}"] = element
    coefficients = rng.standard_normal(len(lie_basis(channel, level)))
    out["lie_random"] = sum(
        c * b for c, b in zip(coefficients, lie_basis(channel, level))
    )
    return out


def rows_for_delta(
    name: str,
    delta: np.ndarray,
    channel: int,
    level: int,
    dims: tuple[int, ...],
    restarts: int,
    steps: int,
    seed: int,
) -> list[dict[str, object]]:
    scale = hilbert_schmidt_bound(delta)
    if scale == 0.0:
        return [
            {
                "pair": name,
                "level": level,
                "response_dim": dim,
                "hs_norm": 0.0,
                "ratio": 0.0,
                "shift_ratio": 0.0,
                "rank_one_ratio": 0.0,
            }
            for dim in dims
        ]
    shift = shift_bound(delta, channel, level) / scale
    rank_one = rank_one_bound(delta, channel, level) / scale
    out = []
    for dim in dims:
        value = level_response_sup(
            delta, channel, level, dim, restarts=restarts, steps=steps, seed=seed
        )
        out.append(
            {
                "pair": name,
                "level": level,
                "response_dim": dim,
                "hs_norm": scale,
                "ratio": value / scale,
                "shift_ratio": shift,
                "rank_one_ratio": rank_one,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("results/runs/response_gap"))
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    create_run_directory(args.out)

    channel = 2
    dims = (1, 2, 3, 5)
    rows: list[dict[str, object]] = []

    for name, builder in PATH_PAIRS.items():
        left_path, right_path = builder()
        left = piecewise_linear_signature(left_path, args.depth)
        right = piecewise_linear_signature(right_path, args.depth)
        for level in range(2, args.depth + 1):
            delta = np.asarray(left[level]) - np.asarray(right[level])
            rows.extend(
                rows_for_delta(
                    name,
                    delta,
                    channel,
                    level,
                    dims,
                    args.restarts,
                    args.steps,
                    args.seed,
                )
            )

    for level in range(2, args.depth + 1):
        for name, delta in synthetic_deltas(channel, level).items():
            rows.extend(
                rows_for_delta(
                    name,
                    delta,
                    channel,
                    level,
                    dims,
                    args.restarts,
                    args.steps,
                    args.seed,
                )
            )

    csv_path = args.out / "level_ratios.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = run_metadata(
        "none",
        {},
        study="response gap",
        channel=channel,
        depth=args.depth,
        response_dims=list(dims),
        restarts=args.restarts,
        steps=args.steps,
        seed=args.seed,
        convention="Hilbert-Schmidt on levels, Frobenius on response tuple, operator norm on terminal response",
        command=[sys.executable, *sys.argv],
        source_sha256={
            str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parents[1] / "src/pathloss/response.py",
            )
        },
        rows=len(rows),
    )
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {csv_path} with {len(rows)} rows")


if __name__ == "__main__":
    main()
