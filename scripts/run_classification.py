#!/usr/bin/env python
"""Classify labelled paths by 1-nearest neighbour under a fixed distance.

No trained model: only representation and distance change, so accuracy
differences are attributable to the distance itself. Notebook 04 reads the
output and reports raw, training-channel and per-series preprocessing
separately.

    python scripts/run_classification.py --config configs/classification_basicmotions.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathloss.provenance import git_sha, load_config, utc_now

from pathloss.classification import (
    classify_1nn,
    load_ts_split,
    preprocess_classification,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    destination = args.out or Path(cfg["output"])
    destination.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_ts_split(cfg["data"]["train"])
    test_x, test_y = load_ts_split(cfg["data"]["test"])
    train_x, test_x, normalisation = preprocess_classification(
        train_x,
        test_x,
        method=cfg["data"].get("normalisation", "none"),
    )

    results = []
    for specification in cfg["classifier"]["distances"]:
        specification = dict(specification)
        name = specification.pop("name")
        results.append(
            classify_1nn(
                train_x,
                train_y,
                test_x,
                test_y,
                distance=name,
                n_jobs=cfg["classifier"].get("n_jobs", 1),
                **specification,
            )
        )

    output = {
        "name": cfg["name"],
        "completed": utc_now(),
        "git_sha": git_sha(),
        "data": {
            "train": cfg["data"]["train"],
            "test": cfg["data"]["test"],
            "train_shape": list(train_x.shape),
            "test_shape": list(test_x.shape),
            "classes": sorted(np_unique_strings(train_y)),
            "normalisation": normalisation,
        },
        "results": results,
    }
    (destination / "results.json").write_text(json.dumps(output, indent=2))
    for result in results:
        score = result["scores"]
        print(
            f"{result['distance']:10s} accuracy={score['accuracy']:.3f} "
            f"balanced={score['balanced_accuracy']:.3f}"
        )
    return 0


def np_unique_strings(values) -> list[str]:
    return [str(value) for value in sorted(set(values.tolist()))]


if __name__ == "__main__":
    raise SystemExit(main())
