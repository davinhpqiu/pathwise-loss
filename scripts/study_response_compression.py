"""Response-aware merge study for two coordinate-plane loops.

Original design: docs/response_distance/archive/2026-10-14_findings.md.
Current evidence index: docs/response_distance/findings.md.
Every optimized value is a nonconvex lower estimate; Hilbert--Schmidt quantity
is a proved upper bound for the expanded response polynomial.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np

from pathloss.response import (
    composed_signature_polynomial,
    composed_truncated_response_sup,
    piecewise_linear_signature,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(increments: np.ndarray) -> np.ndarray:
    origin = np.zeros((1, increments.shape[1]))
    return np.concatenate((origin, np.cumsum(increments, axis=0)), axis=0)


def _square_loop(first: int, second: int, channel: int, side: float) -> np.ndarray:
    increments = np.zeros((4, channel))
    increments[0, first] = side
    increments[1, second] = side
    increments[2, first] = -side
    increments[3, second] = -side
    return increments


def _interval_signatures(
    path: np.ndarray, boundaries: tuple[int, ...], depth: int
) -> tuple[np.ndarray, ...]:
    levels: list[list[np.ndarray]] = [[] for _ in range(depth)]
    for start, stop in pairwise(boundaries):
        signature = piecewise_linear_signature(path[start : stop + 1], depth)
        for level in range(1, depth + 1):
            levels[level - 1].append(signature[level])
    return tuple(np.stack(level, axis=0) for level in levels)


def _polynomial_difference_bound(
    fine: tuple[np.ndarray, ...],
    coarse: tuple[np.ndarray, ...],
    channel: int,
    response_bound: float,
) -> tuple[float, tuple[int, ...]]:
    fine_polynomial = composed_signature_polynomial(fine, channel)
    coarse_polynomial = composed_signature_polynomial(coarse, channel)
    depth = max(len(fine_polynomial), len(coarse_polynomial))
    bound = 0.0
    nonzero = []
    for index in range(depth):
        level = index + 1
        left = (
            fine_polynomial[index]
            if index < len(fine_polynomial)
            else np.zeros(channel**level)
        )
        right = (
            coarse_polynomial[index]
            if index < len(coarse_polynomial)
            else np.zeros(channel**level)
        )
        delta = left - right
        norm = float(np.linalg.norm(delta))
        bound += response_bound**level * norm
        if norm > 1.0e-12:
            nonzero.append(level)
    return bound, tuple(nonzero)


def _structures() -> tuple[tuple[str, int, int | None], ...]:
    return (
        ("dense_n2", 2, None),
        ("dense_n4", 4, None),
        ("diagonal_n4", 4, 1),
        ("block2_n4", 4, 2),
    )


def _run(args: argparse.Namespace) -> list[dict[str, object]]:
    channel = 3
    depth = 2
    coordinates_per_interval = sum(channel**level for level in range(1, depth + 1))
    rows: list[dict[str, object]] = []
    for side in args.scales:
        xy = _square_loop(0, 1, channel, side)
        yz = _square_loop(1, 2, channel, side)
        path = _path(np.concatenate((xy, yz), axis=0))
        fine = _interval_signatures(path, (0, 4, 8), depth)
        global_signature = piecewise_linear_signature(path, depth)
        coarse = tuple(level[None, :] for level in global_signature[1:])
        hs_bound, nonzero_levels = _polynomial_difference_bound(
            fine, coarse, channel, args.response_bound
        )
        for structure, response_dim, block_size in _structures():
            started = time.perf_counter()
            response = composed_truncated_response_sup(
                fine,
                coarse,
                channel,
                response_dim,
                response_bound=args.response_bound,
                restarts=args.restarts,
                steps=args.steps,
                seed=args.seed,
                block_size=block_size,
            )
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "side": side,
                    "structure": structure,
                    "response_dim": response_dim,
                    "block_size": "" if block_size is None else block_size,
                    "response_bound": args.response_bound,
                    "fine_intervals": 2,
                    "coarse_intervals": 1,
                    "fine_coordinates": 2 * coordinates_per_interval,
                    "coarse_coordinates": coordinates_per_interval,
                    "nonzero_error_levels": ";".join(map(str, nonzero_levels)),
                    "hs_bound": hs_bound,
                    "response_error": response,
                    "response_to_bound": response / hs_bound if hs_bound else 0.0,
                    "seconds": elapsed,
                }
            )
    return rows


def main() -> None:
    from pathloss.provenance import create_run_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--scales", type=float, nargs="+", default=(0.25, 0.5, 1.0, 1.5)
    )
    parser.add_argument("--response-bound", type=float, default=1.0)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    create_run_directory(args.out)
    if any(side <= 0.0 for side in args.scales):
        raise ValueError("scales must be positive")

    rows = _run(args)
    with (args.out / "response_compression.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "command": ["python", *sys.argv],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "depth": 2,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "response_bound": args.response_bound,
        "restarts": args.restarts,
        "scales": args.scales,
        "seed": args.seed,
        "source_sha256": {
            "scripts/study_response_compression.py": _sha256(Path(__file__)),
            "src/pathloss/response.py": _sha256(
                Path(__file__).parents[1] / "src/pathloss/response.py"
            ),
        },
        "status": "nonconvex response lower estimates; proved HS upper bound",
        "steps": args.steps,
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
