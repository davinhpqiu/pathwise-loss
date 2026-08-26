#!/usr/bin/env python
"""Run one Brownian-to-OU Neural CDE fit or its implementation gates."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pathloss.provenance import load_config, run_metadata, utc_now


def make_train_config(
    config: dict,
    *,
    seed: int,
    condition: str,
    loss: str,
):
    from pathloss.operator import OUOperatorTrainConfig

    data = config["data"]
    model = config["model"]
    train = config["train"]
    signature = config["signature"]
    split_seeds = data["split_seeds"]
    return OUOperatorTrainConfig(
        seed=seed,
        device=train.get("device", "cpu"),
        condition=condition,
        loss=loss,
        n_train=data["n_train"],
        n_val=data["n_val"],
        n_test=data.get("n_test", 0),
        n_steps=data["n_steps"],
        n_target=data["n_target"],
        T=data.get("T", 1.0),
        lambd=data.get("lambda", 2.0),
        sigma=data.get("sigma", 0.5),
        y0=data.get("y0", 0.0),
        train_data_seed=split_seeds["train"],
        val_data_seed=split_seeds["val"],
        test_data_seed=split_seeds["test"],
        hidden=model.get("hidden", 16),
        width=model.get("width", 64),
        epochs=train.get("epochs", 500),
        batch_size=train.get("batch_size", 64),
        lr=train.get("lr", 1.0e-3),
        signature_global_depth=signature.get("global_depth", 4),
        signature_local_depth=signature.get("local_depth", 2),
        signature_local_intervals=signature.get("local_intervals", 5),
    )


def peak_memory_mb(device: str) -> float:
    if device.startswith("cuda"):
        import torch

        return float(torch.cuda.max_memory_allocated(device) / 1024**2)
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / 1024**2
    return value / 1024.0


def save_paths(path: Path, paths: dict) -> None:
    arrays = {name: value.numpy() for name, value in paths.items()}
    np.savez_compressed(path, **arrays)


def save_plot(path: Path, paths: dict, title: str) -> None:
    if paths["driver"].shape[0] == 0:
        return
    cache = path.parent / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    count = min(4, paths["driver"].shape[0])
    figure, axes = plt.subplots(count, 2, figsize=(11, 2.8 * count), squeeze=False)
    time_grid = paths["time"].numpy()
    for index in range(count):
        axes[index, 0].plot(time_grid, paths["driver"][index, :, 0].numpy())
        axes[index, 0].set_ylabel(f"path {index}")
        axes[index, 0].set_title("Brownian driver" if index == 0 else "")
        axes[index, 1].plot(
            time_grid,
            paths["target"][index, :, 0].numpy(),
            color="black",
            label="OU target",
        )
        axes[index, 1].plot(
            time_grid,
            paths["prediction"][index, :, 0].numpy(),
            label="prediction",
        )
        axes[index, 1].set_title("Response" if index == 0 else "")
        axes[index, 1].legend()
    for axis in axes[-1]:
        axis.set_xlabel("time")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_acceptance(config_path: Path, config: dict, out: Path) -> int:
    from pathloss.operator import run_ou_acceptance

    acceptance = config["acceptance"]
    cfg = make_train_config(
        config,
        seed=acceptance.get("seed", 991),
        condition="uniform",
        loss="mse",
    )
    report = run_metadata(config_path, config)
    started = time.perf_counter()
    result = run_ou_acceptance(
        cfg,
        overfit_epochs=acceptance.get("overfit_epochs", 300),
        overfit_max_ratio=acceptance.get("overfit_max_ratio", 0.1),
        overfit_max_loss=acceptance.get("overfit_max_loss", 1.0e-3),
        heldout_epochs=acceptance.get("heldout_epochs", 120),
        heldout_max_ratio=acceptance.get("heldout_max_ratio", 0.8),
        verbose=True,
    )
    report.update(result)
    report["runtime_seconds"] = time.perf_counter() - started
    report["peak_memory_mb"] = peak_memory_mb(cfg.device)
    report["finished"] = utc_now()
    (out / "acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


def run_signature_audit(config_path: Path, config: dict, out: Path) -> int:
    from pathloss.operator import ou_signature_gradient_audit

    cfg = make_train_config(
        config,
        seed=config["signature"].get("audit_seed", 991),
        condition="uniform",
        loss="sig_global",
    )
    report = run_metadata(config_path, config)
    report["fit_config"] = asdict(cfg)
    report["audit"] = ou_signature_gradient_audit(cfg)
    report["finished"] = utc_now()
    (out / "signature_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["audit"], indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--condition", choices=("uniform", "clustered"))
    parser.add_argument("--loss", choices=("mse", "j2", "sig_global", "sig_local"))
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--signature-audit", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.acceptance:
        return run_acceptance(args.config, config, args.out)
    if args.signature_audit:
        return run_signature_audit(args.config, config, args.out)
    missing = [
        name for name in ("seed", "condition", "loss") if getattr(args, name) is None
    ]
    if missing:
        parser.error("fit mode requires " + ", ".join(f"--{name}" for name in missing))
    if args.loss in {"sig_global", "sig_local"} and not config["signature"].get(
        "audit_accepted", False
    ):
        parser.error(
            "review signature_audit.json and set signature.audit_accepted=true "
            "before OU signature training"
        )

    import torch

    from pathloss.operator import train_ou_operator

    cfg = make_train_config(
        config,
        seed=args.seed,
        condition=args.condition,
        loss=args.loss,
    )
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(cfg.device)
    meta = run_metadata(args.config, config, fit_config=asdict(cfg))
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    started = time.perf_counter()
    result = train_ou_operator(
        cfg,
        include_test=args.evaluate_test,
        verbose=True,
    )
    runtime = time.perf_counter() - started

    torch.save(result["model"].state_dict(), args.out / "model.pt")
    (args.out / "history.json").write_text(json.dumps(result["history"], indent=2))
    (args.out / "validation.json").write_text(
        json.dumps(result["validation_metrics"], indent=2)
    )
    save_paths(args.out / "validation_paths.npz", result["validation_paths"])
    displayed_paths = result["validation_paths"]
    if "test_metrics" in result:
        (args.out / "test.json").write_text(
            json.dumps(result["test_metrics"], indent=2)
        )
        save_paths(args.out / "test_paths.npz", result["test_paths"])
        displayed_paths = result["test_paths"]
    save_plot(
        args.out / "fit.png",
        displayed_paths,
        f"{args.loss}, {args.condition}, seed={args.seed}",
    )

    meta.update(
        {
            "initial_fingerprint": result["initial_fingerprint"],
            "data_fingerprint": result["data_fingerprint"],
            "target_time_fingerprint": result["target_time_fingerprint"],
            "order_fingerprint": result["order_fingerprint"],
            "signature_output_scale": result["signature_output_scale"],
            "untrained_validation": result["untrained_validation"],
            "validation": result["validation_metrics"],
            "test": result.get("test_metrics"),
            "runtime_seconds": runtime,
            "peak_memory_mb": peak_memory_mb(cfg.device),
            "finished": utc_now(),
        }
    )
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        json.dumps(
            {
                "validation": result["validation_metrics"],
                "test": result.get("test_metrics"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
