"""Log-ODE merge study for two coordinate-plane loops.

The fine representation uses ``exp(C_xy) exp(C_yz)`` and the merged depth-two
log-signature uses ``exp(C_xy + C_yz)``. Every optimized value is a nonconvex
lower estimate. Tangent and finite-scale bounds and the exact Hilbert--Schmidt
certificate are proved in ``docs/response_distance/proofs.md`` §§8.5--8.6.
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
from pathlib import Path

import numpy as np

from pathloss.response import (
    coordinate_loop_logode_hs_certificate,
    coordinate_loop_logode_merge_sup,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _structures() -> tuple[tuple[str, int, int | None], ...]:
    return (
        ("dense_n2", 2, None),
        ("dense_n4", 4, None),
        ("diagonal_n4", 4, 1),
        ("block2_n4", 4, 2),
    )


def _run(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side in args.scales:
        scale = (side * args.response_bound) ** 2
        tangent_response = scale**2 / (4.0 * np.sqrt(2.0))
        tangent_hs = np.sqrt(2.0) * scale**2
        uniform_remainder = scale**3 * np.exp(scale) / 3.0
        analytic_lower = (2.0 * np.expm1(scale / 2.0) - scale) / np.sqrt(2.0)
        lie_trotter_upper = tangent_response * np.exp(scale)
        hs_certificate = coordinate_loop_logode_hs_certificate(scale)
        for structure, response_dim, block_size in _structures():
            supports_noncommutative_response = block_size is None or block_size >= 2
            structure_lower = (
                analytic_lower if supports_noncommutative_response else 0.0
            )
            structure_upper = (
                lie_trotter_upper if supports_noncommutative_response else 0.0
            )
            started = time.perf_counter()
            response = coordinate_loop_logode_merge_sup(
                scale,
                response_dim,
                restarts=args.restarts,
                steps=args.steps,
                seed=args.seed,
                block_size=block_size,
            )
            elapsed = time.perf_counter() - started
            response_lower = max(response, structure_lower)
            rows.append(
                {
                    "side": side,
                    "scale_t": scale,
                    "structure": structure,
                    "response_dim": response_dim,
                    "block_size": "" if block_size is None else block_size,
                    "response_bound": args.response_bound,
                    "fine_intervals": 2,
                    "coarse_intervals": 1,
                    "fine_logsignature_coordinates": 12,
                    "coarse_logsignature_coordinates": 6,
                    "response_error_lower": response_lower,
                    "optimized_response": response,
                    "analytic_response_lower": structure_lower,
                    "lie_trotter_response_upper": structure_upper,
                    "full_hs_certificate": hs_certificate,
                    "tangent_response": tangent_response,
                    "tangent_hs_certificate": tangent_hs,
                    "tangent_response_to_hs": 0.125,
                    "uniform_remainder_bound": uniform_remainder,
                    "lower_to_tangent": (
                        response_lower / tangent_response if tangent_response else 0.0
                    ),
                    "upper_to_hs_certificate": (
                        structure_upper / hs_certificate if hs_certificate else 0.0
                    ),
                    "seconds": elapsed,
                }
            )
    return rows


def main() -> None:
    from pathloss.provenance import create_run_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=(0.1, 0.25, 0.5, 1.0, 1.5),
    )
    parser.add_argument("--response-bound", type=float, default=1.0)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    create_run_directory(args.out)
    if any(side <= 0.0 for side in args.scales):
        raise ValueError("scales must be positive")
    if args.response_bound <= 0.0:
        raise ValueError("response-bound must be positive")

    rows = _run(args)
    with (args.out / "response_logode_merge.csv").open("w", newline="") as handle:
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
        "representation": "truncated log-signature with log-ODE factors",
        "response_bound": args.response_bound,
        "restarts": args.restarts,
        "scales": args.scales,
        "seed": args.seed,
        "source_sha256": {
            "scripts/study_response_logode_merge.py": _sha256(Path(__file__)),
            "src/pathloss/response.py": _sha256(
                Path(__file__).parents[1] / "src/pathloss/response.py"
            ),
        },
        "status": (
            "nonconvex optimized lower estimates plus analytic lower bound; "
            "proved Lie-Trotter upper bound; tail-controlled numerical evaluation of exact HS series"
        ),
        "steps": args.steps,
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
