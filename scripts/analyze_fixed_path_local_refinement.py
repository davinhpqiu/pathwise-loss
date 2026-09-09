#!/usr/bin/env python
"""Compare Experiment A's hundred-block local-signature fits with controls.

The script never trains a model. It reloads the six completed refinement fits
and paired MSE, J2, H1, global-signature and ten-block local-signature fits,
evaluates both local partitions on every saved path, and writes ``comparison.csv``
plus ``summary.json`` for notebook 05.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from pathloss.fixed_path_study import FixedPathRun, find_completed_run
from pathloss.signatures import anchored_coordinate_mean_signature_loss

CAPACITIES = ("restricted", "expressive")
SEEDS = (0, 1, 2)
CONTROL_LOSSES = ("mse", "j2", "h1", "sig_global", "sig_local")
COMMON_METRICS = ("mse", "j2", "h1", "linf", "sig_global", "local_j2")


def _run(capacity: str, seed: int, loss: str) -> FixedPathRun:
    return FixedPathRun(capacity, seed, "uniform", loss, 0.001, 10000)


def _score_paths(run_dir: Path) -> tuple[float, float]:
    arrays = np.load(run_dir / "paths.npz")
    time = torch.from_numpy(arrays["time"])
    prediction = torch.from_numpy(arrays["prediction"])
    target = torch.from_numpy(arrays["target"])
    coarse = anchored_coordinate_mean_signature_loss(
        time,
        prediction,
        target,
        depth=2,
        intervals=10,
        output_scale=1.0,
    )
    fine = anchored_coordinate_mean_signature_loss(
        time,
        prediction,
        target,
        depth=2,
        intervals=100,
        output_scale=1.0,
        homogeneity_reference_intervals=10,
    )
    return float(coarse), float(fine)


def _read_row(run: FixedPathRun, meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    stored_metrics = json.loads((meta_path.parent / "metrics.json").read_text())
    coarse, fine = _score_paths(meta_path.parent)
    return {
        "capacity": run.capacity,
        "seed": run.seed,
        "training_loss": run.loss,
        "fingerprint": meta["initial_fingerprint"],
        "run_dir": str(meta_path.parent),
        **{name: float(stored_metrics[name]) for name in COMMON_METRICS},
        "sig_local_10": coarse,
        "sig_local_100_scaled": fine,
    }


def collect_rows(refinement_root: Path, baseline_roots: list[Path]) -> list[dict]:
    rows = []
    for capacity in CAPACITIES:
        for seed in SEEDS:
            for loss in CONTROL_LOSSES:
                run = _run(capacity, seed, loss)
                meta_path = find_completed_run(run, baseline_roots)
                if meta_path is None:
                    raise FileNotFoundError(f"missing control run: {run}")
                rows.append(_read_row(run, meta_path))
            run = _run(capacity, seed, "sig_local_fine")
            meta_path = find_completed_run(run, [refinement_root])
            if meta_path is None:
                raise FileNotFoundError(f"missing refinement run: {run}")
            rows.append(_read_row(run, meta_path))
    return rows


def summarize(rows: list[dict]) -> dict:
    evaluation_metrics = (*COMMON_METRICS, "sig_local_10", "sig_local_100_scaled")
    groups = {}
    all_fingerprints_match = True
    for capacity in CAPACITIES:
        for seed in SEEDS:
            group = [
                row
                for row in rows
                if row["capacity"] == capacity and row["seed"] == seed
            ]
            by_loss = {row["training_loss"]: row for row in group}
            fingerprints_match = len({row["fingerprint"] for row in group}) == 1
            all_fingerprints_match &= fingerprints_match
            groups[f"{capacity}/seed{seed}"] = {
                "fingerprints_match": fingerprints_match,
                "fine_over_local10": {
                    metric: (
                        by_loss["sig_local_fine"][metric] / by_loss["sig_local"][metric]
                    )
                    for metric in evaluation_metrics
                },
                "winner": {
                    metric: min(group, key=lambda row: row[metric])["training_loss"]
                    for metric in evaluation_metrics
                },
            }

    return {
        "complete": len(rows) == 36,
        "runs_compared": len(rows),
        "new_refinement_runs": sum(
            row["training_loss"] == "sig_local_fine" for row in rows
        ),
        "all_paired_fingerprints_match": all_fingerprints_match,
        "fine_beats_local10_count": {
            metric: sum(
                details["fine_over_local10"][metric] < 1.0
                for details in groups.values()
            )
            for metric in evaluation_metrics
        },
        "fine_best_of_all_losses_count": {
            metric: sum(
                details["winner"][metric] == "sig_local_fine"
                for details in groups.values()
            )
            for metric in evaluation_metrics
        },
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refinement-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = collect_rows(args.refinement_root, args.baseline_root)
    summary = summarize(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
