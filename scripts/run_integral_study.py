#!/usr/bin/env python
"""Run one or all jobs in the core integral-loss study.

The job grid crosses seed, architecture, target mechanism, and loss. Passing
``--index`` runs one flattened job, which is the interface used by the ARC
array script. Omitting it runs every job sequentially.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def jobs_from_config(cfg: dict) -> list[dict]:
    jobs = []
    for seed in cfg["seeds"]:
        for model in cfg["models"]:
            for mechanism in cfg["data"]["target_mechanisms"]:
                for loss in cfg["losses"]:
                    jobs.append(
                        {
                            "seed": int(seed),
                            "model": model,
                            "mechanism": mechanism,
                            "loss": loss,
                        }
                    )
    return jobs


def job_name(job: dict) -> str:
    return (
        f"{job['model']['name']}_{job['mechanism']['name']}_"
        f"{job['loss']}_seed{job['seed']}"
    )


def run_job(cfg: dict, job: dict, root: Path) -> None:
    from pathloss.models import build_model
    from pathloss.train import TrainConfig, train

    data = cfg["data"]
    training = cfg["train"]
    context = data["context_sampling"]
    mechanism = job["mechanism"]
    model_spec = job["model"]
    model_kwargs = {
        key: value
        for key, value in model_spec.items()
        if key in {"hidden", "layers", "width", "n_fourier", "step_size"}
    }
    train_cfg = TrainConfig(
        seed=job["seed"],
        device=training.get("device", "cpu"),
        generator=data["generator"],
        n_train=data["n_train"],
        n_val=data["n_val"],
        n_test=data["n_test"],
        n_fine=data["n_fine"],
        n_ctx=data["n_ctx"],
        n_tgt=data["n_tgt"],
        context_mode=context["mode"],
        target_mode=mechanism["mode"],
        context_density_bias=context.get("density_bias", 0.0),
        target_density_bias=mechanism.get("density_bias", 0.0),
        noise=data.get("noise", 0.0),
        missing_rate=data.get("missing_rate", 0.0),
        d=data.get("d", 1),
        model=model_spec["kind"],
        model_kwargs=model_kwargs,
        loss=job["loss"],
        epochs=training["epochs"],
        batch_size=training["batch_size"],
        lr=training["lr"],
    )

    destination = root / job_name(job)
    destination.mkdir(parents=True, exist_ok=True)
    parameter_probe = build_model(model_spec["kind"], d=train_cfg.d, **model_kwargs)
    parameters = sum(parameter.numel() for parameter in parameter_probe.parameters())
    del parameter_probe

    meta = {
        "study": cfg["name"],
        "job": job,
        "train_config": train_cfg.__dict__,
        "parameters": parameters,
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "host": platform.node(),
        "started": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "meta.json").write_text(json.dumps(meta, indent=2))
    result = train(train_cfg)
    (destination / "history.json").write_text(json.dumps(result["history"], indent=2))
    (destination / "validation.json").write_text(json.dumps(result["final"], indent=2))
    (destination / "test.json").write_text(json.dumps(result["test"], indent=2))
    meta["finished"] = datetime.now(timezone.utc).isoformat()
    meta["validation"] = result["final"]
    meta["test"] = result["test"]
    (destination / "meta.json").write_text(json.dumps(meta, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    jobs = jobs_from_config(cfg)
    if args.list:
        for index, job in enumerate(jobs):
            print(index, job_name(job))
        return 0
    selected = jobs if args.index is None else [jobs[args.index]]
    if args.seed is not None:
        selected = [job for job in selected if job["seed"] == args.seed]
    for job in selected:
        print(f"running {job_name(job)}")
        run_job(cfg, job, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
