"""Run provenance: git state, configuration loading, environment record.

Every runner in `scripts/` writes a `meta.json` before doing work, so a crashed
run stays attributable. Keys are fixed here so runs from different scripts stay
comparable.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["git_sha", "load_config", "run_metadata", "utc_now"]


def git_sha() -> str:
    """Short commit hash, or ``"unknown"`` outside a working repository.

    Exception handling is deliberately broad: provenance recording must never
    be the reason a training run fails.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def utc_now() -> str:
    """Timezone-aware UTC timestamp in ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str | Path) -> dict:
    """Read one YAML configuration file."""
    try:
        import yaml
    except ImportError:
        raise SystemExit("pip install pyyaml")
    return yaml.safe_load(Path(path).read_text())


def run_metadata(
    config_path: str | Path, config: dict, **extra: Any
) -> dict:
    """Environment and configuration record written at start of a run.

    `extra` merges additional keys, letting a script record study or job fields
    without redefining shared ones.
    """
    meta = {
        "config": str(config_path),
        "config_contents": config,
        "git_sha": git_sha(),
        "started": utc_now(),
        "python": sys.version.split()[0],
        "host": platform.node(),
    }
    meta.update(extra)
    return meta
