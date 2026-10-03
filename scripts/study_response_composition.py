"""Low-compute study of common and composed finite responses.

Design is recorded in ``docs/response_distance/findings.md`` before execution.
Suprema are nonconvex lower estimates. The script writes every optimization
budget and elapsed time beside its result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np

from pathloss.response import (
    composed_truncated_response_sup,
    level_response_sup,
    piecewise_linear_signature,
    truncated_response_sup,
)


@dataclass(frozen=True)
class Case:
    name: str
    left: np.ndarray
    right: np.ndarray
    boundaries: tuple[int, ...]


def _path(increments: np.ndarray) -> np.ndarray:
    origin = np.zeros((1, increments.shape[1]))
    return np.concatenate((origin, np.cumsum(increments, axis=0)), axis=0)


def _square_loop(first: int, second: int, channel: int) -> np.ndarray:
    increments = np.zeros((4, channel))
    increments[0, first] = 1.0
    increments[1, second] = 1.0
    increments[2, first] = -1.0
    increments[3, second] = -1.0
    return increments


def _cases() -> tuple[Case, ...]:
    planar_left = _path(np.array([[1.0, 0.0], [0.0, 1.0]]))
    planar_right = _path(np.array([[0.0, 1.0], [1.0, 0.0]]))

    xy = _square_loop(0, 1, 3)
    yz = _square_loop(1, 2, 3)
    loops_left = _path(np.concatenate((xy, yz), axis=0))
    loops_right = _path(np.concatenate((yz, xy), axis=0))
    return (
        Case("planar_vertex_order", planar_left, planar_right, (0, 1, 2)),
        Case("three_dimensional_loop_order", loops_left, loops_right, (0, 4, 8)),
    )


def _interval_signatures(
    path: np.ndarray, boundaries: tuple[int, ...], depth: int
) -> tuple[np.ndarray, ...]:
    levels: list[list[np.ndarray]] = [[] for _ in range(depth)]
    for start, stop in pairwise(boundaries):
        signature = piecewise_linear_signature(path[start : stop + 1], depth)
        for level in range(1, depth + 1):
            levels[level - 1].append(signature[level])
    return tuple(np.stack(level, axis=0) for level in levels)


def _structures() -> tuple[tuple[str, int, int | None], ...]:
    return (
        ("dense_n2", 2, None),
        ("dense_n4", 4, None),
        ("diagonal_n4", 4, 1),
        ("block2_n4", 4, 2),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in _cases():
        channel = case.left.shape[1]
        left_global = piecewise_linear_signature(case.left, args.depth)[1:]
        right_global = piecewise_linear_signature(case.right, args.depth)[1:]
        deltas = tuple(
            left - right for left, right in zip(left_global, right_global, strict=True)
        )
        left_local = _interval_signatures(case.left, case.boundaries, args.depth)
        right_local = _interval_signatures(case.right, case.boundaries, args.depth)
        hs_bound = sum(
            args.response_bound**level * float(np.linalg.norm(delta))
            for level, delta in enumerate(deltas, start=1)
        )
        for structure, response_dim, block_size in _structures():
            started = time.perf_counter()
            levelwise = sum(
                args.response_bound**level
                * level_response_sup(
                    delta,
                    channel,
                    level,
                    response_dim,
                    restarts=args.restarts,
                    steps=args.steps,
                    seed=args.seed + level,
                    block_size=block_size,
                )
                for level, delta in enumerate(deltas, start=1)
            )
            levelwise_seconds = time.perf_counter() - started

            started = time.perf_counter()
            common = truncated_response_sup(
                deltas,
                channel,
                response_dim,
                response_bound=args.response_bound,
                restarts=args.restarts,
                steps=args.steps,
                seed=args.seed,
                block_size=block_size,
            )
            common_seconds = time.perf_counter() - started

            started = time.perf_counter()
            composed = composed_truncated_response_sup(
                left_local,
                right_local,
                channel,
                response_dim,
                response_bound=args.response_bound,
                restarts=args.restarts,
                steps=args.steps,
                seed=args.seed,
                block_size=block_size,
            )
            composed_seconds = time.perf_counter() - started
            rows.append(
                {
                    "case": case.name,
                    "driver_dim": channel,
                    "depth": args.depth,
                    "structure": structure,
                    "response_dim": response_dim,
                    "block_size": "" if block_size is None else block_size,
                    "response_bound": args.response_bound,
                    "hs_bound": hs_bound,
                    "levelwise_sum": levelwise,
                    "common_global": common,
                    "composed": composed,
                    "levelwise_seconds": levelwise_seconds,
                    "common_seconds": common_seconds,
                    "composed_seconds": composed_seconds,
                }
            )
    return rows


def main() -> None:
    from pathloss.provenance import create_run_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--response-bound", type=float, default=1.0)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if args.depth < 1:
        raise ValueError("depth must be positive")
    create_run_directory(args.out)

    rows = _run(args)
    result_path = args.out / "response_composition.csv"
    with result_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "command": ["python", *sys.argv],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "depth": args.depth,
        "response_bound": args.response_bound,
        "restarts": args.restarts,
        "steps": args.steps,
        "seed": args.seed,
        "source_sha256": {
            "scripts/study_response_composition.py": _sha256(Path(__file__)),
            "src/pathloss/response.py": _sha256(
                Path(__file__).parents[1] / "src/pathloss/response.py"
            ),
        },
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "status": "nonconvex lower estimates",
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
