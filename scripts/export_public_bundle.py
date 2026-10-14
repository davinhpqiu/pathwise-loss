#!/usr/bin/env python
"""Write the frozen data bundle behind the public summary notebook.

Public folder ``public/pathwise_losses`` reproduces the presentation's figures
and headline numbers without torch, training, or the 240 MB of run outputs. This
script is its only data source: it reads the same runs that
``scripts/build_presentation_figures.py`` and notebooks 05 to 07 read, and
writes small arrays and tables plus a provenance record with source hashes.

Run selection mirrors the presentation script, imported directly so the two
cannot drift. Fixed-path metric tables use the closeout run matrix exactly as
notebook 05 builds it.

Observation times for Experiment A come from ``pathloss.fixed_path``, which
imports torch. Without torch they are recorded as pending and the notebook
omits that one panel; rerun this script in the project environment to complete
the bundle.

Run::

    python scripts/export_public_bundle.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import build_presentation_figures as deck  # noqa: E402
from pathloss.fixed_path_study import configured_runs, load_run_registry  # noqa: E402
from pathloss.provenance import git_sha, load_config  # noqa: E402

RUNS = REPO / "results" / "runs"
DEFAULT_OUT = REPO / "public" / "pathwise_losses" / "data"
CHECKPOINT_UPDATES = (100, 500, 1000, 2500, 10000)
SIGNATURE_LOSSES = ("mse", "j2", "sig_global", "sig_local")
CAPACITIES = ("restricted", "expressive")
METRICS = ("mse", "j2", "h1", "linf", "local_j2", "sig_global", "sig_local")
MESSAGES = ("area", "combined", "increment")
TEMPLATES = ("net", "balanced")


class Sources:
    """Collect every file read, with its hash, for the provenance record."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    def add(self, path: Path) -> Path:
        path = Path(path)
        self.files[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def export_fixed_path(out: Path, sources: Sources) -> dict:
    """Arrays for the target, residual, checkpoint and refinement figures."""
    arrays: dict[str, np.ndarray] = {}

    # Clustered residual pair, verified against caption claims by the deck script.
    deck.verify_fixed_path("mse")
    deck.verify_fixed_path("j2")
    for loss in ("mse", "j2"):
        stored = np.load(sources.add(deck.fixed_path_dir(loss) / "paths.npz"))
        arrays["time"] = stored["time"]
        arrays["target"] = stored["target"]
        arrays[f"clustered_prediction_{loss}"] = stored["prediction"]

    # Learning trajectories, seed 0, uniform, four original losses.
    for capacity in CAPACITIES:
        for loss in SIGNATURE_LOSSES:
            directory = RUNS / "neural_ode_fixed_path_signature_10k" / capacity / "seed0" / "uniform" / loss
            stored = np.load(sources.add(directory / "checkpoint_paths.npz"))
            for update in CHECKPOINT_UPDATES:
                arrays[f"checkpoint_{capacity}_{loss}_{update:05d}"] = (
                    stored[f"prediction_updates_{update:05d}"].astype(np.float32)
                )

    # Ten against hundred local blocks, restricted, seeds 0 to 2.
    for seed in range(3):
        coarse_dir = None
        for root in ("neural_ode_fixed_path_signature_10k", "neural_ode_fixed_path_closeout_10k"):
            candidate = RUNS / root / deck.REFINEMENT_CAPACITY / f"seed{seed}" / "uniform" / "sig_local"
            if (candidate / "paths.npz").exists():
                coarse_dir = candidate
                break
        fine_dir = (RUNS / "neural_ode_fixed_path_local_fine_10k" / deck.REFINEMENT_CAPACITY
                    / f"seed{seed}" / "uniform" / "sig_local_fine")
        coarse = np.load(sources.add(coarse_dir / "paths.npz"))
        fine = np.load(sources.add(fine_dir / "paths.npz"))
        arrays[f"refinement_seed{seed}_blocks10"] = coarse["prediction"].astype(np.float32)
        arrays[f"refinement_seed{seed}_blocks100"] = fine["prediction"].astype(np.float32)

    status = {"observation_times": "included"}
    try:
        from pathloss.fixed_path import observation_times  # noqa: PLC0415
        for condition in ("uniform", "clustered"):
            arrays[f"observation_times_{condition}"] = observation_times(64, condition).numpy()
    except ModuleNotFoundError as error:
        status["observation_times"] = f"pending: {error}"

    np.savez_compressed(out / "fixed_path.npz", **arrays)
    return status


def export_fixed_path_tables(out: Path, sources: Sources) -> None:
    """Final metrics and checkpoint metrics for the closeout run matrix."""
    config_path = REPO / "configs" / "neural_ode_fixed_path_closeout_10k.yaml"
    expected = configured_runs(load_config(sources.add(config_path)))
    registry = load_run_registry([
        RUNS / "neural_ode_fixed_path_closeout_10k",
        RUNS / "neural_ode_fixed_path_signature_10k",
        RUNS / "neural_ode_fixed_path_budget_10k",
    ])
    final_rows, checkpoint_rows = [], []
    for run in expected:
        if run not in registry:
            raise FileNotFoundError(f"closeout run missing: {run}")
        meta_path = registry[run]
        meta = json.loads(sources.add(meta_path).read_text())
        identity = {k: run.to_dict()[k] for k in ("capacity", "seed", "condition", "loss")}
        # Two budget-study runs predate signature metrics; blank cells read as NaN.
        final_rows.append({**identity, **{m: meta["metrics"].get(m, "") for m in METRICS}})
        if run.condition == "uniform":
            checkpoints = json.loads(sources.add(meta_path.parent / "checkpoints.json").read_text())
            for row in checkpoints:
                checkpoint_rows.append({
                    **{k: identity[k] for k in ("capacity", "seed", "loss")},
                    "update": row["updates_completed"],
                    **{m: row["metrics"].get(m, "") for m in METRICS},
                })
    _write_csv(out / "fixed_path_final_metrics.csv", final_rows)
    _write_csv(out / "fixed_path_checkpoint_metrics.csv", checkpoint_rows)

    comparison = sources.add(RUNS / "neural_ode_fixed_path_local_fine_10k" / "analysis" / "comparison.csv")
    with comparison.open() as handle:
        rows = [
            {k: v for k, v in row.items() if k not in ("fingerprint", "run_dir")}
            for row in csv.DictReader(handle)
        ]
    _write_csv(out / "refinement_metrics.csv", rows)


def export_ou(out: Path, sources: Sources) -> None:
    """Example held-out response, signature histories, and held-out metrics."""
    directory = deck.ou_dir()
    config = json.loads(sources.add(directory / "meta.json").read_text())["fit_config"]
    for key in ("seed", "condition", "loss"):
        if config[key] != deck.OU_RUN[key]:
            raise ValueError(f"{directory}: {key} is {config[key]!r}")
    stored = np.load(sources.add(directory / "test_paths.npz"))
    index = deck.OU_RUN["test_path_index"]
    arrays = {
        "time": stored["time"],
        "target": stored["target"][index, :, 0],
        "prediction": stored["prediction"][index, :, 0],
    }
    for loss in ("sig_global", "sig_local"):
        for seed in range(3):
            path = RUNS / "neural_cde_brownian_ou" / f"seed{seed}" / "uniform" / loss / "history.json"
            history = json.loads(sources.add(path).read_text())
            arrays[f"history_{loss}_seed{seed}_epoch"] = np.array([r["epoch"] for r in history])
            arrays[f"history_{loss}_seed{seed}_train_loss"] = np.array([r["train_loss"] for r in history])
    np.savez_compressed(out / "ou_operator.npz", **arrays)

    rows = []
    for path in sorted((RUNS / "neural_cde_brownian_ou").glob("seed*/*/*/test.json")):
        seed, condition, loss = path.parts[-4], path.parts[-3], path.parts[-2]
        metrics = json.loads(sources.add(path).read_text())
        rows.append({"seed": int(seed.removeprefix("seed")), "condition": condition,
                     "loss": loss, **metrics})
    _write_csv(out / "ou_test_metrics.csv", rows)


def export_brownian(out: Path, sources: Sources) -> None:
    """Paired sensitivity and detection curves, plus detection thresholds."""
    rows = []
    for template in TEMPLATES:
        for message in MESSAGES:
            for amplitude in deck.AMPLITUDES:
                path = sources.add(deck._amplitude_directory(template, message, amplitude) / "result.json")
                report = json.loads(path.read_text())
                for discrepancy, entry in report["paired"].items():
                    rows.append({"template": template, "message": message, "amplitude": amplitude,
                                 "section": "paired", "discrepancy": discrepancy,
                                 "value": entry["normalized_median"], "lower95": ""})
                for discrepancy, entry in report["detection"].items():
                    rows.append({"template": template, "message": message, "amplitude": amplitude,
                                 "section": "detection", "discrepancy": discrepancy,
                                 "value": entry["balanced_accuracy"],
                                 "lower95": entry["balanced_accuracy_interval"][0]})
    _write_csv(out / "brownian_messages.csv", rows)

    threshold_rows = []
    for ensemble in ("main", "confirmation"):
        summary = json.loads(sources.add(RUNS / "brownian_message" / ensemble / "summary.json").read_text())
        for entry in summary["transitions"]:
            threshold_rows.append({"ensemble": ensemble, **{
                k: entry[k] for k in ("template", "message_type", "representation",
                                      "amplitude", "preceding_amplitude")}})
    _write_csv(out / "brownian_thresholds.csv", threshold_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    sources = Sources()
    status = export_fixed_path(out, sources)
    export_fixed_path_tables(out, sources)
    export_ou(out, sources)
    export_brownian(out, sources)

    outputs = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir()) if p.name != "PROVENANCE.json"
    }
    provenance = {
        "exported": datetime.now(timezone.utc).isoformat(),
        "repository_git_sha": git_sha(),
        "exporter": "scripts/export_public_bundle.py",
        "status": status,
        "outputs_sha256": outputs,
        "sources_sha256": dict(sorted(sources.files.items())),
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"wrote {len(outputs) + 1} files, {total / 1e6:.2f} MB, to {out}")
    print("status:", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
