"""Deterministic anchors for planar BV and mixed-block log-ODE separation.

Theory and pre-run design: notebooks/08_response_distance.ipynb;
local working copy: docs/response_distance/mixed_blocks.md.
Reports analytic bounds plus explicit matrix witnesses, without ascent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import expm

from pathloss.provenance import create_run_directory
from pathloss.response import piecewise_linear_signature
from pathloss.response_mixed import (
    RHO3,
    planar_bv_logode_data,
    planar_bv_logode_neighborhood,
    planar_cubic_attaining_tuple,
    planar_merge_bounds,
    planar_merge_separation_radius,
    planar_mixed_merge_data,
    planar_mixed_merge_neighborhood,
)


def _cases() -> list[dict]:
    return [
        {"name": "bv_corner", "path": [[0, 0], [1, 0], [1, 1]], "cuts": [0, 2]},
        {
            "name": "bv_two_mixed_intervals",
            "path": [[0, 0], [0.8, 0.1], [1, 0.7], [0.3, 1], [1.2, 1.5]],
            "cuts": [0, 2, 4],
        },
        {
            "name": "bv_three_corners",
            "path": [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2], [3, 2], [3, 3]],
            "cuts": [0, 2, 4, 6],
        },
        {
            "name": "bv_cubic_cancellation",
            "path": [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
            "cuts": [0, 2, 4],
        },
        {"name": "bv_straight", "path": [[0, 0], [1, 0], [2, 0]], "cuts": [0, 2]},
        {
            "name": "mixed_generic",
            "left": [1, 0],
            "right": [0, 1],
            "left_area": 0.3,
            "right_area": -0.2,
        },
        {
            "name": "mixed_balanced",
            "left": [1, 0],
            "right": [0, 1],
            "left_area": -1 / 6,
            "right_area": -1 / 6,
        },
        {
            "name": "mixed_pure_area",
            "left": [0, 0],
            "right": [0, 0],
            "left_area": 1,
            "right_area": 2,
        },
    ]


def _witness(case: dict, matrices: np.ndarray, scale: float) -> float:
    commutator = matrices[0] @ matrices[1] - matrices[1] @ matrices[0]
    evaluate = lambda v: np.einsum("i,ijk->jk", v, matrices)
    if "path" in case:
        path, cuts = np.asarray(case["path"], dtype=float), tuple(case["cuts"])
        fine = np.eye(2)
        for step in np.diff(path, axis=0):
            fine = fine @ expm(scale * evaluate(step))
        coarse = np.eye(2)
        for left, right in pairwise(cuts):
            _, first, second = piecewise_linear_signature(path[left : right + 1], 2)
            area = (second[1] - second[2]) / 2
            coarse = coarse @ expm(
                scale * evaluate(first) + scale**2 * area * commutator
            )
    else:
        left, right = np.asarray(case["left"]), np.asarray(case["right"])
        alpha, beta = case["left_area"], case["right_area"]
        cross = float(left[0] * right[1] - left[1] * right[0])
        fine = expm(scale * evaluate(left) + scale**2 * alpha * commutator) @ expm(
            scale * evaluate(right) + scale**2 * beta * commutator
        )
        coarse = expm(
            scale * evaluate(left + right)
            + scale**2 * (alpha + beta + cross / 2) * commutator
        )
    return float(np.linalg.norm(fine - coarse, 2))


def _source_provenance(root: Path) -> dict:
    sources = [
        "scripts/study_response_mixed.py",
        "src/pathloss/response_mixed.py",
        "src/pathloss/response.py",
        "src/pathloss/response_storage.py",
        "src/pathloss/provenance.py",
        "tests/test_response_mixed.py",
        "notebooks/08_response_distance.ipynb",
    ]
    optional = "docs/response_distance/mixed_blocks.md"
    present = (root / optional).is_file()
    if present:
        sources.append(optional)
    return {
        "source_sha256": {
            source: hashlib.sha256((root / source).read_bytes()).hexdigest()
            for source in sources
        },
        "unavailable_optional_sources": [] if present else [optional],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    create_run_directory(args.out)
    cases = _cases()
    rows = []
    for case in cases:
        if "path" in case:
            path, cuts = np.asarray(case["path"], dtype=float), tuple(case["cuts"])
            data = planar_bv_logode_data(path, cuts)
            eta, neighborhood_scale = planar_bv_logode_neighborhood(path, cuts)
        else:
            parameters = (
                np.asarray(case["left"]),
                np.asarray(case["right"]),
                case["left_area"],
                case["right_area"],
            )
            data = planar_mixed_merge_data(*parameters)
            eta, neighborhood_scale = planar_mixed_merge_neighborhood(*parameters)
        radius = planar_merge_separation_radius(data)
        matrices = planar_cubic_attaining_tuple(
            data.cubic if data.leading_hs else (1.0, 0.0)
        )
        reference = radius or 1.0 / max(
            data.source_a,
            np.sqrt(data.source_b),
            data.merged_a,
            np.sqrt(data.merged_b),
            1.0,
        )
        for factor in (0.05, 0.25, 0.5, 1.0, 2.0, 5.0):
            scale = factor * reference
            bounds = planar_merge_bounds(data, scale)
            witness = _witness(case, matrices, scale)
            inside = radius > 0 and factor <= 1.0
            separated = bounds.response_upper < bounds.certificate_lower
            if witness > bounds.response_upper + 1e-13:
                raise AssertionError(f"witness exceeds response bound: {case['name']}")
            if witness + 1e-13 < bounds.response_lower:
                raise AssertionError(
                    f"attaining witness below response lower bound: {case['name']}"
                )
            if inside and not separated:
                raise AssertionError(f"certified gap failed: {case['name']}")
            rows.append(
                {
                    "case": case["name"],
                    "scale": scale,
                    "radius_factor": factor,
                    "cubic_degenerate": data.leading_hs == 0.0,
                    "leading_hs_coefficient": data.leading_hs,
                    "tangent_ratio": RHO3 if data.leading_hs else None,
                    "separation_radius": radius,
                    "inside_radius": inside,
                    "perturbation_radius": eta,
                    "uniform_neighborhood_scale": neighborhood_scale,
                    "matrix_witness_error": witness,
                    "analytic_gap": separated,
                    **asdict(bounds),
                }
            )
    with (args.out / "mixed_response_bounds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    root = Path(__file__).resolve().parents[1]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "status": "analytic envelopes and explicit n=2 matrix witnesses; supremum uncomputed; numerical rounding",
        "cases": cases,
        "rows": len(rows),
        "inside_radius_cases": sum(row["inside_radius"] for row in rows),
        "inside_radius_failures": sum(
            row["inside_radius"] and not row["analytic_gap"] for row in rows
        ),
        **_source_provenance(root),
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {len(rows)} deterministic checks to {args.out}")


if __name__ == "__main__":
    main()
