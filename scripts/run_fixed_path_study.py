#!/usr/bin/env python
"""Run one fixed-path experiment stage locally, one fit at a time."""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

from pathloss.fixed_path_study import (
    FixedPathRun,
    configured_runs,
    find_completed_run,
)
from pathloss.provenance import load_config


STAGE_CASES = {
    "primary": (
        ("uniform", "mse"),
        ("clustered", "mse"),
        ("uniform", "j2"),
        ("clustered", "j2"),
    ),
    "h1": (("uniform", "h1"),),
    "signature": (
        ("uniform", "sig_global"),
        ("uniform", "sig_local"),
    ),
    "signature_pilot": (
        ("uniform", "mse"),
        ("uniform", "j2"),
        ("uniform", "sig_global"),
        ("uniform", "sig_local"),
    ),
    "configured": (),
}


def selected_values(available: list, selected) -> list:
    if selected is None:
        return list(available)
    if selected not in available:
        raise ValueError(f"selected value {selected!r} is absent from configuration")
    return [selected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_CASES), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--capacity", choices=("restricted", "expressive"))
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.stage == "configured":
        jobs = list(configured_runs(config))
        if args.seed is not None:
            jobs = [run for run in jobs if run.seed == args.seed]
        if args.capacity is not None:
            jobs = [run for run in jobs if run.capacity == args.capacity]
    else:
        seeds = selected_values(config["train"]["seeds"], args.seed)
        capacities = selected_values(list(config["capacities"]), args.capacity)
        jobs = [
            FixedPathRun(
                capacity=capacity,
                seed=seed,
                condition=condition,
                loss=loss,
                lr=float(config["train"].get("lr", 1.0e-3)),
                updates=int(config["train"]["updates"]),
            )
            for seed, capacity, (condition, loss) in product(
                seeds, capacities, STAGE_CASES[args.stage]
            )
        ]
    if any(run.loss in {"sig_global", "sig_local"} for run in jobs) and not config.get(
        "signature", {}
    ).get("audit_accepted", False):
        raise SystemExit(
            "signature stage is locked: review signature_audit.json and set "
            "signature.audit_accepted=true"
        )

    if args.task_id is not None:
        if args.seed is not None or args.capacity is not None:
            parser.error("--task-id cannot be combined with --seed or --capacity")
        if args.task_id < 0 or args.task_id >= len(jobs):
            parser.error(f"--task-id must lie in [0, {len(jobs) - 1}]")
        indexed_jobs = [(args.task_id, jobs[args.task_id])]
    else:
        indexed_jobs = list(enumerate(jobs))

    reuse_roots = [Path(path) for path in config.get("study", {}).get("reuse_roots", [])]
    for zero_index, run in indexed_jobs:
        run_dir = args.out / run.relative_directory
        current_meta = run_dir / "meta.json"
        if current_meta.exists() and not args.rerun_completed:
            completed = find_completed_run(run, [args.out])
            if completed is None:
                raise RuntimeError(
                    f"existing result at {run_dir} has a different run identity"
                )
            print(
                f"[{zero_index + 1}/{len(jobs)}] skip completed in output root: "
                f"seed={run.seed}, capacity={run.capacity}, "
                f"condition={run.condition}, loss={run.loss}",
                flush=True,
            )
            continue
        reused = None if args.rerun_completed else find_completed_run(run, reuse_roots)
        if reused is not None:
            print(
                f"[{zero_index + 1}/{len(jobs)}] reuse {reused.parent}: "
                f"seed={run.seed}, capacity={run.capacity}, "
                f"condition={run.condition}, loss={run.loss}",
                flush=True,
            )
            continue
        print(
            f"[{zero_index + 1}/{len(jobs)}] start: "
            f"seed={run.seed}, capacity={run.capacity}, "
            f"condition={run.condition}, loss={run.loss}",
            flush=True,
        )
        command = [
            sys.executable,
            "scripts/run_fixed_path.py",
            "--config",
            str(args.config),
            "--out",
            str(run_dir),
            "--seed",
            str(run.seed),
            "--capacity",
            run.capacity,
            "--condition",
            run.condition,
            "--loss",
            run.loss,
        ]
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
