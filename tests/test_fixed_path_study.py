"""Exact run bookkeeping for Experiment A closeout."""

from __future__ import annotations

import json

import pytest

from pathloss.fixed_path_study import (
    FixedPathRun,
    completion_report,
    configured_runs,
    find_completed_run,
    load_run_registry,
    numerical_resolution,
)


def closeout_config() -> dict:
    return {
        "capacities": {"restricted": {}, "expressive": {}},
        "train": {"seeds": [0, 1, 2], "updates": 10000, "lr": 0.001},
        "study": {
            "cases": [
                {"condition": "uniform", "loss": "mse"},
                {"condition": "clustered", "loss": "mse"},
                {"condition": "uniform", "loss": "j2"},
                {"condition": "clustered", "loss": "j2"},
                {"condition": "uniform", "loss": "h1"},
                {"condition": "uniform", "loss": "sig_global"},
                {"condition": "uniform", "loss": "sig_local"},
            ]
        },
    }


def write_meta(root, run: FixedPathRun):
    path = root / run.relative_directory / "meta.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "fit_config": {
                    "seed": run.seed,
                    "condition": run.condition,
                    "loss": run.loss,
                    "lr": run.lr,
                    "updates": run.updates,
                }
            }
        )
    )
    return path


def test_closeout_matrix_has_exact_revised_counts():
    runs = configured_runs(closeout_config())
    counts = {loss: sum(run.loss == loss for run in runs) for loss in {
        "mse", "j2", "h1", "sig_global", "sig_local"
    }}
    assert len(runs) == 42
    assert counts == {
        "mse": 12,
        "j2": 12,
        "h1": 6,
        "sig_global": 6,
        "sig_local": 6,
    }


def test_completion_uses_learning_rate_and_budget(tmp_path):
    expected = FixedPathRun("restricted", 0, "uniform", "mse", 0.001, 10000)
    wrong_budget = FixedPathRun("restricted", 0, "uniform", "mse", 0.001, 5000)
    old_root = tmp_path / "old"
    write_meta(old_root, wrong_budget)
    registry = load_run_registry([old_root])
    report = completion_report([expected], registry)
    assert report["complete"] is False
    assert report["completed"] == 0
    assert report["missing"] == [expected.to_dict()]


def test_find_completed_run_reuses_exact_metadata(tmp_path):
    run = FixedPathRun("expressive", 2, "uniform", "sig_local", 0.001, 10000)
    root = tmp_path / "pilot"
    path = write_meta(root, run)
    assert find_completed_run(run, [root]) == path


def test_duplicate_exact_runs_are_rejected(tmp_path):
    run = FixedPathRun("expressive", 0, "clustered", "j2", 0.001, 10000)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_meta(first, run)
    write_meta(second, run)
    with pytest.raises(ValueError, match="duplicate exact run"):
        load_run_registry([first, second])


def test_path_and_metadata_disagreement_is_rejected(tmp_path):
    root = tmp_path / "root"
    run = FixedPathRun("restricted", 0, "uniform", "mse", 0.001, 10000)
    path = write_meta(root, run)
    meta = json.loads(path.read_text())
    meta["fit_config"]["loss"] = "j2"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="path and metadata disagree"):
        load_run_registry([root])


def test_numerical_resolution_uses_combined_error_intervals():
    separated = numerical_resolution(1.0, 1.2, 0.02, 0.03)
    assert separated == {
        "difference": pytest.approx(0.2),
        "combined_numerical_error": pytest.approx(0.05),
        "resolution_margin": pytest.approx(0.15),
        "resolved": True,
    }
    assert numerical_resolution(1.0, 1.04, 0.02, 0.03)["resolved"] is False
    with pytest.raises(ValueError, match="non-negative"):
        numerical_resolution(1.0, 1.2, -0.01, 0.03)
