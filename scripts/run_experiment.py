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

    # ---------------------------------------------------------------- TODO
    # 1. generate data      -> pathloss.paths
    # 2. build model        -> pathloss.models   (week 2)
    # 3. build loss         -> pathloss.losses   (week 3)
    # 4. train / evaluate   -> report every metric in cfg["eval"]["metrics"],
    #                          with bootstrap CIs, alongside sampling density
    print(f"[stub] would run {cfg['name']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
