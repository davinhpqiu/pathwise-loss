#!/usr/bin/env python
"""Entry point for a single experiment. Config-driven, no hardcoded settings.

    python scripts/run_experiment.py --config configs/baseline_mse.yaml --out results/

Deliberately thin: it resolves a config into calls on `pathloss`, records
provenance, and writes results. All maths lives in `src/pathloss/`.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_config(path: Path) -> dict:
    text = path.read_text()
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        raise SystemExit("pip install pyyaml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    # Provenance first, so even a crashed run is attributable.
    meta = {
        "config": str(args.config),
        "config_contents": cfg,
        "git_sha": git_sha(),
        "started": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "host": platform.node(),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    from pathloss.train import TrainConfig, train

    data, model_cfg, loss_cfg, tr = (
        cfg.get("data", {}),
        cfg.get("model", {}),
        cfg.get("loss", {}),
        cfg.get("train", {}),
    )
    tcfg = TrainConfig(
        seed=cfg.get("seed", 0),
        device=tr.get("device", "cpu"),
        generator=data.get("generator", "ornstein_uhlenbeck"),
        n_train=data.get("n_train", 512),
        n_val=data.get("n_val", 128),
        n_fine=data.get("n_fine", 513),
        n_ctx=data.get("n_ctx", 64),
        n_tgt=data.get("n_tgt", 64),
        mode=data.get("sampling", {}).get("mode", "clustered"),
        density_bias=data.get("sampling", {}).get("density_bias", 3.0),
        noise=data.get("noise", 0.0),
        d=data.get("d", 1),
        model=model_cfg.get("kind", "gru_query"),
        model_kwargs={
            k: v for k, v in model_cfg.items() if k in {"hidden", "layers", "width"}
        },
        loss=loss_cfg.get("kind", "mse"),
        epochs=tr.get("epochs", 200),
        batch_size=tr.get("batch_size", 64),
        lr=tr.get("lr", 1.0e-3),
    )

    out = train(tcfg)

    (args.out / "history.json").write_text(json.dumps(out["history"], indent=2))
    (args.out / "final.json").write_text(json.dumps(out["final"], indent=2))
    meta["finished"] = datetime.now(timezone.utc).isoformat()
    meta["final"] = out["final"]
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{cfg['name']} -> {args.out}")
    for k, v in out["final"].items():
        print(f"  {k:14s} {v:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
