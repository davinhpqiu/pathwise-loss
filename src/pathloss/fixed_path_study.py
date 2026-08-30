"""Run identities and completion checks for fixed-path Experiment A.

One run is identified by every field that changes its fitted parameters. Result
directories omit learning rate and update budget, so metadata supplies those
fields before results from several roots are combined.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

__all__ = [
    "FixedPathRun",
    "configured_runs",
    "read_run",
    "load_run_registry",
    "find_completed_run",
    "completion_report",
    "numerical_resolution",
]


@dataclass(frozen=True, order=True)
class FixedPathRun:
    """Exact identity of one fitted-path optimization run."""

    capacity: str
    seed: int
    condition: str
    loss: str
    lr: float
    updates: int

    @property
    def relative_directory(self) -> Path:
        return Path(self.capacity) / f"seed{self.seed}" / self.condition / self.loss

    def to_dict(self) -> dict:
        return asdict(self)


def configured_runs(config: dict) -> tuple[FixedPathRun, ...]:
    """Expand a configured study into its complete ordered run matrix."""
    try:
        seeds = config["train"]["seeds"]
        capacities = tuple(config["capacities"])
        cases = config["study"]["cases"]
        lr = float(config["train"]["lr"])
        updates = int(config["train"]["updates"])
    except KeyError as error:
        raise ValueError(f"fixed-path study configuration lacks {error.args[0]!r}")

    runs = []
    for seed in seeds:
        for capacity in capacities:
            for case in cases:
                condition = case["condition"]
                loss = case["loss"]
                if loss == "h1" and condition != "uniform":
                    raise ValueError("h1 study cases must use uniform observations")
                if loss in {"sig_global", "sig_local"} and condition != "uniform":
                    raise ValueError("signature study cases must use uniform observations")
                runs.append(
                    FixedPathRun(
                        capacity=capacity,
                        seed=int(seed),
                        condition=condition,
                        loss=loss,
                        lr=lr,
                        updates=updates,
                    )
                )
    if len(runs) != len(set(runs)):
        raise ValueError("fixed-path study configuration contains duplicate runs")
    return tuple(runs)


def read_run(root: Path, meta_path: Path) -> FixedPathRun:
    """Read and cross-check one standard-layout result directory."""
    root = Path(root)
    meta_path = Path(meta_path)
    try:
        relative = meta_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{meta_path} is outside result root {root}") from error
    if len(relative.parts) != 5 or relative.name != "meta.json":
        raise ValueError(f"unexpected fixed-path result layout: {relative}")
    capacity, seed_part, condition, loss, _ = relative.parts
    if not seed_part.startswith("seed"):
        raise ValueError(f"unexpected seed directory {seed_part!r}")
    try:
        path_seed = int(seed_part[4:])
    except ValueError as error:
        raise ValueError(f"unexpected seed directory {seed_part!r}") from error

    meta = json.loads(meta_path.read_text())
    cfg = meta["fit_config"]
    run = FixedPathRun(
        capacity=capacity,
        seed=int(cfg["seed"]),
        condition=cfg["condition"],
        loss=cfg["loss"],
        lr=float(cfg["lr"]),
        updates=int(cfg["updates"]),
    )
    path_fields = (capacity, path_seed, condition, loss)
    meta_fields = (run.capacity, run.seed, run.condition, run.loss)
    if path_fields != meta_fields:
        raise ValueError(
            f"result path and metadata disagree at {meta_path}: "
            f"path={path_fields}, metadata={meta_fields}"
        )
    return run


def load_run_registry(roots: Iterable[Path]) -> dict[FixedPathRun, Path]:
    """Index completed runs across roots and reject ambiguous duplicates."""
    registry: dict[FixedPathRun, Path] = {}
    for root in map(Path, roots):
        if not root.exists():
            continue
        for meta_path in sorted(root.glob("*/seed*/*/*/meta.json")):
            run = read_run(root, meta_path)
            if run in registry:
                raise ValueError(
                    f"duplicate exact run in {registry[run].parent} and {meta_path.parent}"
                )
            registry[run] = meta_path
    return registry


def find_completed_run(run: FixedPathRun, roots: Iterable[Path]) -> Path | None:
    """Return metadata for an exact completed run, ignoring other budgets."""
    found = []
    for root in map(Path, roots):
        meta_path = root / run.relative_directory / "meta.json"
        if meta_path.exists() and read_run(root, meta_path) == run:
            found.append(meta_path)
    if len(found) > 1:
        raise ValueError(
            "duplicate exact run in " + " and ".join(str(path.parent) for path in found)
        )
    return found[0] if found else None


def completion_report(
    expected: Iterable[FixedPathRun], registry: dict[FixedPathRun, Path]
) -> dict:
    """Compare exact expected identities with completed metadata records."""
    expected_set = set(expected)
    completed_set = expected_set.intersection(registry)
    missing = sorted(expected_set - completed_set)
    unexpected = sorted(set(registry) - expected_set)
    return {
        "expected": len(expected_set),
        "completed": len(completed_set),
        "complete": not missing,
        "missing": [run.to_dict() for run in missing],
        "unexpected": [run.to_dict() for run in unexpected],
    }


def numerical_resolution(
    value_a: float,
    value_b: float,
    error_a: float,
    error_b: float,
) -> dict[str, float | bool]:
    """Determine whether two metric estimates separate beyond numerical error."""
    if error_a < 0.0 or error_b < 0.0:
        raise ValueError("numerical error estimates must be non-negative")
    difference = abs(float(value_a) - float(value_b))
    uncertainty = float(error_a) + float(error_b)
    return {
        "difference": difference,
        "combined_numerical_error": uncertainty,
        "resolution_margin": difference - uncertainty,
        "resolved": difference > uncertainty,
    }
