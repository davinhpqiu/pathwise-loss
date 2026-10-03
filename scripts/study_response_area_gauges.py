"""Measure conservatism of spectral response-gauge bounds for area blocks.

Each random degree-two block is represented by a skew coefficient matrix and
normalised to Hilbert--Schmidt norm `sqrt(2)`.  The script compares the
certified spectral upper bounds of `area_merge_response_gauge_bounds` with
nonconvex optimized lower estimates of the defining gauges `M` and `K`.
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
    area_block_gram_matrix,
    area_merge_response_gauge_bounds,
    estimate_area_merge_response_gauges,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalised_random_blocks(
    rng: np.random.Generator, block_count: int, channel: int
) -> np.ndarray:
    raw = rng.standard_normal((block_count, channel, channel))
    blocks = 0.5 * (raw - raw.transpose(0, 2, 1))
    norms = np.linalg.norm(blocks, axis=(1, 2))
    return blocks * (np.sqrt(2.0) / norms)[:, None, None]


def _coordinate_chain(block_count: int) -> np.ndarray:
    channel = block_count + 1
    blocks = np.zeros((block_count, channel, channel))
    for index in range(block_count):
        blocks[index, index, index + 1] = 1.0
        blocks[index, index + 1, index] = -1.0
    return blocks


def _max_abs_correlation(gram: np.ndarray) -> float:
    norms = np.sqrt(np.diag(gram))
    correlation = gram / np.outer(norms, norms)
    off_diagonal = correlation - np.eye(gram.shape[0])
    return float(np.max(np.abs(off_diagonal)))


def _row(
    blocks: np.ndarray,
    *,
    case: str,
    sample: int,
    response_dim: int,
    restarts: int,
    steps: int,
    seed: int,
) -> dict[str, object]:
    gram = area_block_gram_matrix(blocks)
    growth_bound, commutator_bound = area_merge_response_gauge_bounds(blocks)
    started = time.perf_counter()
    growth_estimate, commutator_estimate = estimate_area_merge_response_gauges(
        blocks,
        response_dim,
        restarts=restarts,
        steps=steps,
        seed=seed,
    )
    elapsed = time.perf_counter() - started
    return {
        "case": case,
        "sample": sample,
        "channel": blocks.shape[1],
        "block_count": blocks.shape[0],
        "response_dim": response_dim,
        "growth_bound": growth_bound,
        "growth_estimate": growth_estimate,
        "growth_estimate_to_bound": growth_estimate / growth_bound,
        "commutator_bound": commutator_bound,
        "commutator_estimate": commutator_estimate,
        "commutator_estimate_to_bound": (
            commutator_estimate / commutator_bound if commutator_bound else 0.0
        ),
        "max_abs_block_correlation": _max_abs_correlation(gram),
        "seconds": elapsed,
    }


def _run(args: argparse.Namespace) -> list[dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    for block_count in args.block_counts:
        chain = _coordinate_chain(block_count)
        for response_dim in args.response_dims:
            rows.append(
                _row(
                    chain,
                    case="coordinate_chain",
                    sample=0,
                    response_dim=response_dim,
                    restarts=args.restarts,
                    steps=args.steps,
                    seed=args.seed + block_count + response_dim,
                )
            )
    for channel in args.channels:
        for block_count in args.block_counts:
            for sample in range(args.samples):
                blocks = _normalised_random_blocks(rng, block_count, channel)
                for response_dim in args.response_dims:
                    rows.append(
                        _row(
                            blocks,
                            case="random",
                            sample=sample,
                            response_dim=response_dim,
                            restarts=args.restarts,
                            steps=args.steps,
                            seed=(
                                args.seed
                                + 10000 * channel
                                + 100 * block_count
                                + 10 * sample
                                + response_dim
                            ),
                        )
                    )
    return rows


def main() -> None:
    from pathloss.provenance import create_run_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--channels", type=int, nargs="+", default=(3, 4))
    parser.add_argument("--block-counts", type=int, nargs="+", default=(2, 3, 4))
    parser.add_argument("--response-dims", type=int, nargs="+", default=(2,))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()
    create_run_directory(args.out)
    if min(*args.channels, *args.block_counts, *args.response_dims) < 2:
        raise ValueError("channels, block counts and response dimensions must be >= 2")
    if args.samples < 1:
        raise ValueError("samples must be positive")

    rows = _run(args)
    with (args.out / "response_area_gauges.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "block_counts": args.block_counts,
        "channels": args.channels,
        "command": ["python", *sys.argv],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "response_dims": args.response_dims,
        "restarts": args.restarts,
        "samples": args.samples,
        "seed": args.seed,
        "source_sha256": {
            "scripts/study_response_area_gauges.py": _sha256(Path(__file__)),
            "src/pathloss/response.py": _sha256(
                Path(__file__).parents[1] / "src/pathloss/response.py"
            ),
        },
        "status": ("proved spectral upper bounds; nonconvex optimized lower estimates"),
        "steps": args.steps,
    }
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
