#!/usr/bin/env python
"""Evaluate quadrature and RK4 resolution of saved Experiment A fits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pathloss.fixed_path import (
    LatentNeuralODE,
    fixed_path_quadrature,
    fixed_target,
    fixed_target_derivative,
    h1_balance,
)


def evaluate_run(run: Path, n_points: int, max_steps: tuple[float, ...]) -> dict:
    if n_points < 3 or (n_points - 1) & (n_points - 2):
        raise ValueError("n_points must equal 2**m + 1")
    if len(max_steps) != 3 or any(step <= 0 for step in max_steps):
        raise ValueError("supply three positive solver steps from coarse to fine")
    if not max_steps[0] > max_steps[1] > max_steps[2]:
        raise ValueError("solver steps must be strictly decreasing")

    meta = json.loads((run / "meta.json").read_text())
    cfg = meta["fit_config"]
    state = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    time = torch.linspace(0.0, 1.0, n_points, dtype=torch.float64)
    target = fixed_target(time)
    target_derivative = fixed_target_derivative(time)
    rho = h1_balance(time, target)
    estimates = []
    for max_step in max_steps:
        model = LatentNeuralODE(
            hidden=cfg["hidden"],
            width=cfg["width"],
            n_fourier=cfg["n_fourier"],
            max_step=max_step,
        ).double()
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            prediction, prediction_derivative = model.forward_with_derivative(time)
        estimates.append(
            {
                "max_step": max_step,
                **fixed_path_quadrature(
                    time,
                    prediction,
                    target,
                    prediction_derivative,
                    target_derivative,
                    rho,
                ),
            }
        )

    resolution = {}
    for metric in ("j2", "h1"):
        values = [row[f"{metric}_simpson"] for row in estimates]
        coarse_change = abs(values[0] - values[1])
        fine_change = abs(values[1] - values[2])
        ratio = coarse_change / fine_change if fine_change > 0.0 else None
        richardson_applicable = ratio is not None and 8.0 <= ratio <= 32.0
        solver_error = fine_change / 15.0 if richardson_applicable else fine_change
        resolution[metric] = {
            "quadrature_difference": abs(
                estimates[-1][f"{metric}_simpson"]
                - estimates[-1][f"{metric}_romberg"]
            ),
            "coarse_solver_change": coarse_change,
            "fine_solver_change": fine_change,
            "rk4_change_ratio": ratio,
            "richardson_applicable": richardson_applicable,
            "solver_error_estimate": solver_error,
        }
        resolution[metric]["numerical_error_estimate"] = (
            resolution[metric]["quadrature_difference"]
            + resolution[metric]["solver_error_estimate"]
        )

    return {
        "run": str(run),
        "fit_config": cfg,
        "n_points": n_points,
        "estimates": estimates,
        "resolution": resolution,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--n-points", type=int, default=1025)
    parser.add_argument(
        "--solver-max-steps",
        type=float,
        nargs=3,
        default=(1.0 / 512.0, 1.0 / 1024.0, 1.0 / 2048.0),
    )
    args = parser.parse_args()

    if not args.run and not args.root:
        parser.error("supply at least one --run or --root")
    runs = list(args.run)
    for root in args.root:
        runs.extend(path.parent for path in root.glob("*/seed*/*/*/meta.json"))
    runs = sorted(set(runs))
    for index, run in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] evaluate {run}", flush=True)
        report = evaluate_run(run, args.n_points, tuple(args.solver_max_steps))
        (run / "resolution.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report["resolution"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
