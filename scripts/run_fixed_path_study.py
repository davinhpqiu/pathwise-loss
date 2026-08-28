#!/usr/bin/env python
"""Run one fixed-path experiment stage locally, one fit at a time."""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

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
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seeds = selected_values(config["train"]["seeds"], args.seed)
    capacities = selected_values(list(config["capacities"]), args.capacity)
    if args.stage in {"signature", "signature_pilot"} and not config.get(
        "signature", {}
    ).get("audit_accepted", False):
        raise SystemExit(
            "signature stage is locked: review signature_audit.json and set "
            "signature.audit_accepted=true"
        )

    jobs = list(product(seeds, capacities, STAGE_CASES[args.stage]))
    for index, (seed, capacity, case) in enumerate(jobs, start=1):
        condition, loss = case
        run_dir = args.out / capacity / f"seed{seed}" / condition / loss
        if (run_dir / "meta.json").exists() and not args.rerun_completed:
            print(
                f"[{index}/{len(jobs)}] skip completed: "
                f"seed={seed}, capacity={capacity}, condition={condition}, loss={loss}",
                flush=True,
            )
            continue
        print(
            f"[{index}/{len(jobs)}] start: "
            f"seed={seed}, capacity={capacity}, condition={condition}, loss={loss}",
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
            str(seed),
            "--capacity",
            capacity,
            "--condition",
            condition,
            "--loss",
            loss,
        ]
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
