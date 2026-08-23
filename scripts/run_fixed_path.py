#!/usr/bin/env python
"""Run one paired fixed-path Neural ODE fit or its Fourier adequacy check."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise SystemExit("pip install pyyaml")
    return yaml.safe_load(path.read_text())


def save_plot(path: Path, arrays: dict[str, np.ndarray], title: str) -> None:
    matplotlib_cache = path.parent / ".matplotlib"
    font_cache = path.parent / ".cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    font_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache))
    import matplotlib.pyplot as plt

    t = arrays["time"]
    target = arrays["target"]
    prediction = arrays["prediction"]
    residual = np.linalg.norm(prediction - target, axis=-1)
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(target[:, 0], target[:, 1], color="black", label="target")
    axes[0, 0].plot(prediction[:, 0], prediction[:, 1], color="tab:blue", label="fit")
    axes[0, 0].set_title("Output path")
    axes[0, 0].axis("equal")
    axes[0, 0].legend()
    for channel in range(2):
        axes[0, 1].plot(t, target[:, channel], color="black", alpha=0.7)
        axes[0, 1].plot(t, prediction[:, channel], label=f"fit channel {channel + 1}")
    axes[0, 1].axvspan(0.15, 0.35, color="tab:orange", alpha=0.15)
    axes[0, 1].set_title("Coordinates")
    axes[0, 1].legend()
    axes[1, 0].plot(t, residual, color="tab:red")
    axes[1, 0].axvspan(0.15, 0.35, color="tab:orange", alpha=0.15)
    axes[1, 0].set_title("Residual magnitude")
    axes[1, 0].set_xlabel("time")
    for channel in range(2):
        axes[1, 1].plot(
            t,
            arrays["target_derivative"][:, channel],
            color="black",
            alpha=0.7,
        )
        axes[1, 1].plot(
            t,
            arrays["prediction_derivative"][:, channel],
            label=f"fit derivative {channel + 1}",
        )
    axes[1, 1].set_title("Time derivatives")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].legend()
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def fit_config(
    config: dict,
    *,
    seed: int,
    capacity: str,
    condition: str,
    loss: str,
    n_fourier: int | None = None,
    updates: int | None = None,
):
    from pathloss.fixed_path import FixedPathTrainConfig

    capacity_config = config["capacities"][capacity]
    data = config["data"]
    model = config["model"]
    train = config["train"]
    return FixedPathTrainConfig(
        seed=seed,
        device=train.get("device", "cpu"),
        condition=condition,
        loss=loss,
        n_target=data.get("n_target", 64),
        n_fine=data.get("n_fine", 513),
        hidden=capacity_config["hidden"],
        width=capacity_config["width"],
        n_fourier=model["n_fourier"] if n_fourier is None else n_fourier,
        max_step=model.get("max_step", 1.0 / 512.0),
        updates=train["updates"] if updates is None else updates,
        lr=train.get("lr", 1.0e-3),
    )


def write_result(out: Path, result: dict, meta: dict) -> None:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    arrays = {name: value.numpy() for name, value in result["paths"].items()}
    np.savez_compressed(out / "paths.npz", **arrays)
    torch.save(result["model"].state_dict(), out / "model.pt")
    (out / "history.json").write_text(json.dumps(result["history"], indent=2))
    (out / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    meta.update(
        {
            "fit_config": asdict(result["config"]),
            "initial_fingerprint": result["initial_fingerprint"],
            "rho": result["rho"],
            "metrics": result["metrics"],
            "finished": datetime.now(timezone.utc).isoformat(),
        }
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    title = (
        f"{result['config'].loss}, {result['config'].condition}, "
        f"H={result['config'].hidden}, seed={result['config'].seed}"
    )
    save_plot(out / "fit.png", arrays, title)


def existing_metrics(out: Path, cfg) -> dict[str, float] | None:
    """Reuse a completed numerical fit after interruption during plotting."""
    meta_path = out / "meta.json"
    metrics_path = out / "metrics.json"
    if not meta_path.exists() or not metrics_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("fit_config") != asdict(cfg):
        raise RuntimeError(f"existing result at {out} has a different configuration")
    return json.loads(metrics_path.read_text())


def base_meta(config_path: Path, config: dict) -> dict:
    return {
        "config": str(config_path),
        "config_contents": config,
        "git_sha": git_sha(),
        "started": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "host": platform.node(),
    }


def run_adequacy(config_path: Path, config: dict, out: Path) -> int:
    from pathloss.fixed_path import train_fixed_path

    adequacy = config["adequacy"]
    seed = adequacy["seed"]
    capacity = adequacy.get("capacity", "expressive")
    updates = adequacy.get("updates", config["train"]["updates"])
    raw_cfg = fit_config(
        config,
        seed=seed,
        capacity=capacity,
        condition="uniform",
        loss="mse",
        n_fourier=0,
        updates=updates,
    )
    fourier_cfg = fit_config(
        config,
        seed=seed,
        capacity=capacity,
        condition="uniform",
        loss="mse",
        updates=updates,
    )
    raw_out = out / "raw_time"
    fourier_out = out / "fourier_time"
    raw_metrics = existing_metrics(raw_out, raw_cfg)
    if raw_metrics is None:
        raw_meta = base_meta(config_path, config)
        print("adequacy: raw scalar time", flush=True)
        raw = train_fixed_path(raw_cfg, verbose=True)
        write_result(raw_out, raw, raw_meta)
        raw_metrics = raw["metrics"]
    else:
        print("adequacy: reusing completed raw-time fit", flush=True)

    fourier_metrics = existing_metrics(fourier_out, fourier_cfg)
    if fourier_metrics is None:
        fourier_meta = base_meta(config_path, config)
        print("adequacy: Fourier time features", flush=True)
        fourier = train_fixed_path(fourier_cfg, verbose=True)
        write_result(fourier_out, fourier, fourier_meta)
        fourier_metrics = fourier["metrics"]
    else:
        print("adequacy: reusing completed Fourier-time fit", flush=True)

    threshold = float(adequacy["max_dense_mse"])
    verdict = {
        "raw_dense_mse": raw_metrics["mse"],
        "fourier_dense_mse": fourier_metrics["mse"],
        "max_dense_mse": threshold,
        "fourier_improves": fourier_metrics["mse"] < raw_metrics["mse"],
        "expressive_fit_adequate": fourier_metrics["mse"] <= threshold,
    }
    verdict["passed"] = (
        verdict["fourier_improves"] and verdict["expressive_fit_adequate"]
    )
    (out / "adequacy.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    return 0 if verdict["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--capacity", choices=("restricted", "expressive"))
    parser.add_argument("--condition", choices=("uniform", "clustered"))
    parser.add_argument("--loss", choices=("mse", "j2", "h1"))
    parser.add_argument("--adequacy", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.adequacy:
        return run_adequacy(args.config, config, args.out)
    missing = [
        name
        for name in ("seed", "capacity", "condition", "loss")
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("run mode requires " + ", ".join(f"--{name}" for name in missing))
    if args.loss == "h1" and args.condition != "uniform":
        parser.error("h1 is restricted to the uniform smooth-path comparison")

    from pathloss.fixed_path import train_fixed_path

    cfg = fit_config(
        config,
        seed=args.seed,
        capacity=args.capacity,
        condition=args.condition,
        loss=args.loss,
    )
    meta = base_meta(args.config, config)
    result = train_fixed_path(cfg, verbose=True)
    write_result(args.out, result, meta)
    print(json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
