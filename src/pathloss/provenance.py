"""Run provenance: git state, configuration loading, environment record.

Helpers record shared environment keys and reserve fresh output directories.
Individual runners control when their metadata is written.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["create_run_directory", "git_sha", "load_config", "run_metadata", "utc_now"]


def create_run_directory(path: str | Path) -> Path:
    """Reserve a fresh output directory, preserving every existing run."""
    output = Path(path)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError(
            f"output already exists; choose a new run directory: {output}"
        ) from error
    return output


def git_sha() -> str:
    """Short commit hash, or ``"unknown"`` outside a working repository.

    Missing Git or a failed Git command must not prevent a run.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
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


def run_metadata(config_path: str | Path, config: dict, **extra: Any) -> dict:
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
