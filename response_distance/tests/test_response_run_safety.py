"""Response studies must preserve existing output directories."""

import hashlib
import importlib
import sys
from pathlib import Path

import pytest

from pathloss.provenance import create_run_directory


def test_create_run_directory_reserves_new_nested_path(tmp_path: Path) -> None:
    output = tmp_path / "study" / "run"
    assert create_run_directory(output) == output
    assert output.is_dir()
    with pytest.raises(ValueError, match="output already exists"):
        create_run_directory(output)


@pytest.mark.parametrize(
    "script",
    [
        "check_response_separation",
        "study_response_gap",
        "study_response_composition",
        "study_response_compression",
        "study_response_logode_merge",
        "study_response_area_gauges",
        "study_response_storage",
        "study_response_mixed",
    ],
)
def test_response_runner_preserves_existing_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "metadata.json"
    sentinel.write_text("retained snapshot\n")
    module = importlib.import_module(f"response_distance.scripts.{script}")
    monkeypatch.setattr(sys, "argv", [f"{script}.py", "--out", str(output)])
    with pytest.raises((ValueError, FileExistsError)):
        module.main()
    assert sentinel.read_text() == "retained snapshot\n"
    assert list(output.iterdir()) == [sentinel]


@pytest.mark.parametrize(
    "script,required,optional",
    [
        (
            "study_response_mixed",
            [
                "response_distance/scripts/study_response_mixed.py",
                "response_distance/code/response_mixed.py",
                "response_distance/code/response.py",
                "response_distance/code/response_storage.py",
                "src/pathloss/provenance.py",
                "response_distance/tests/test_response_mixed.py",
                "response_distance/notebooks/examples.ipynb",
            ],
            "response_distance/technical.md",
        ),
        (
            "study_response_storage",
            [
                "response_distance/scripts/study_response_storage.py",
                "response_distance/code/response_storage.py",
                "response_distance/tests/test_response_storage.py",
            ],
            "response_distance/technical.md",
        ),
    ],
)
def test_runner_provenance_works_without_ignored_working_notes(
    tmp_path: Path, script: str, required: list[str], optional: str
) -> None:
    module = importlib.import_module(f"response_distance.scripts.{script}")
    digest = hashlib.sha256(b"fixture\n").hexdigest()
    for name in required:
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"fixture\n")
    snapshot = module._source_provenance(tmp_path)
    assert snapshot["source_sha256"] == dict.fromkeys(required, digest)
    assert snapshot["unavailable_optional_sources"] == [optional]
    note = tmp_path / optional
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_bytes(b"fixture\n")
    snapshot = module._source_provenance(tmp_path)
    assert snapshot["source_sha256"][optional] == digest
    assert snapshot["unavailable_optional_sources"] == []
