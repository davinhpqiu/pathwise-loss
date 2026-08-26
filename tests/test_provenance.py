"""Provenance helpers shared by every runner in `scripts/`.

Anchors are external to the module: `git_sha` against a separate `git`
invocation, `load_config` against a literal dict, `started` against
`datetime.fromisoformat`.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

import pytest

from pathloss.provenance import git_sha, load_config, run_metadata, utc_now


def test_git_sha_matches_independent_git_call():
    try:
        expected = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        pytest.skip("not run from a git working tree")
    assert git_sha() == expected


def test_git_sha_reports_unknown_without_git(monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "check_output", fail)
    assert git_sha() == "unknown"


def test_utc_now_is_timezone_aware_iso8601():
    parsed = datetime.fromisoformat(utc_now())
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("name: demo\nseed: 3\nlosses:\n  - mse\n  - j2\n")
    assert load_config(path) == {"name": "demo", "seed": 3, "losses": ["mse", "j2"]}


def test_run_metadata_carries_required_keys(tmp_path):
    config = {"name": "demo", "seed": 0}
    meta = run_metadata(tmp_path / "c.yaml", config)
    assert set(meta) == {
        "config",
        "config_contents",
        "git_sha",
        "started",
        "python",
        "host",
    }
    assert meta["config_contents"] == config
    assert datetime.fromisoformat(meta["started"]).tzinfo is not None


def test_run_metadata_extra_merges_without_dropping_shared_keys(tmp_path):
    meta = run_metadata(tmp_path / "c.yaml", {}, study="integral", job={"seed": 1})
    assert meta["study"] == "integral"
    assert meta["job"] == {"seed": 1}
    assert "git_sha" in meta and "host" in meta
