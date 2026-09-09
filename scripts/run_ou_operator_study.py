#!/usr/bin/env python
"""Run one Brownian-to-OU experiment stage locally, one fit at a time."""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

from pathloss.provenance import load_config

STAGES = {
    "primary": (
        ("uniform", "mse"),
        ("clustered", "mse"),
        ("uniform", "j2"),
        ("clustered", "j2"),
    ),
    "signature": (("uniform", "sig_global"), ("uniform", "sig_local")),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "signature" and not config["signature"].get(
        "audit_accepted", False
    ):
        raise SystemExit("signature stage is locked until audit_accepted is true")
    seeds = config["train"]["seeds"] if args.seed is None else [args.seed]
    if any(seed not in config["train"]["seeds"] for seed in seeds):
        raise SystemExit("selected seed is absent from configuration")

    jobs = list(product(seeds, STAGES[args.stage]))
    for index, (seed, case) in enumerate(jobs, start=1):
        condition, loss = case
        run_dir = args.out / f"seed{seed}" / condition / loss
        if (run_dir / "meta.json").exists() and not args.rerun_completed:
            print(
                f"[{index}/{len(jobs)}] skip completed: seed={seed}, condition={condition}, loss={loss}",
                flush=True,
            )
            continue
        print(
            f"[{index}/{len(jobs)}] start: seed={seed}, condition={condition}, loss={loss}",
            flush=True,
        )
        command = [
            sys.executable,
            "scripts/run_ou_operator.py",
            "--config",
            str(args.config),
            "--out",
            str(run_dir),
            "--seed",
            str(seed),
            "--condition",
            condition,
            "--loss",
            loss,
        ]
        if args.evaluate_test:
            command.append("--evaluate-test")
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
