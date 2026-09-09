#!/usr/bin/env python
"""Check Brownian message cache and condition completion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathloss.brownian_messages import message_conditions
from pathloss.provenance import load_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("pilot", "main", "confirmation", "four-class"), required=True
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if args.stage == "four-class":
        expected = [args.out / "main" / "four_class" / name / "result.json" for name in config["message"]["templates"]]
    else:
        message = dict(config["message"])
        if args.stage == "confirmation":
            message["amplitudes"] = config["confirmation"].get("amplitudes", [])
        expected = [
            args.out / args.stage / "conditions" / condition.slug / "result.json"
            for condition in message_conditions(message)
        ]
    missing = [str(path) for path in expected if not path.exists()]
    invalid = []
    for path in expected:
        if path.exists():
            try:
                value = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as error:
                invalid.append({"path": str(path), "error": str(error)})
                continue
            if args.stage != "four-class" and "paired" not in value:
                invalid.append({"path": str(path), "error": "paired result absent"})
            if args.stage == "four-class" and "detection" not in value:
                invalid.append({"path": str(path), "error": "detection result absent"})
    report = {
        "stage": args.stage,
        "expected": len(expected),
        "completed": len(expected) - len(missing) - len(invalid),
        "complete": not missing and not invalid,
        "missing": missing,
        "invalid": invalid,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
