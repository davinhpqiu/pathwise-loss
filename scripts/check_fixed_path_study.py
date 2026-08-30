#!/usr/bin/env python
"""Report exact completion of one configured fixed-path study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathloss.fixed_path_study import (
    completion_report,
    configured_runs,
    load_run_registry,
)
from pathloss.provenance import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    expected = configured_runs(config)
    roots = [args.out]
    roots.extend(Path(path) for path in config.get("study", {}).get("reuse_roots", []))
    registry = load_run_registry(roots)
    report = completion_report(expected, registry)
    report["result_roots"] = [str(root) for root in roots]
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

