"""Keep the separate research branch navigable and reproducible after moves."""

import json
import re
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from response_distance.scripts.build_response_note import archive_members

ROOT = Path(__file__).resolve().parents[2]
BRANCH = ROOT / "response_distance"


def test_branch_has_only_three_active_markdown_documents() -> None:
    assert {p.name for p in BRANCH.glob("*.md")} == {
        "README.md",
        "technical.md",
        "literature.md",
    }
    for name in (
        "code",
        "scripts",
        "tests",
        "notebooks",
        "results",
        "output",
        "archive",
    ):
        assert (BRANCH / name).is_dir()


def test_current_branch_file_links_resolve() -> None:
    for source in BRANCH.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", source.read_text()):
            if target.startswith(("https:", "http:", "#")):
                continue
            path = target.split("#", 1)[0]
            assert (source.parent / path).is_file(), (source.name, target)


def test_previous_documents_remain_recoverable() -> None:
    path = BRANCH / "archive/response_distance_2026-10-14_before_reorganization.zip"
    with ZipFile(path) as bundle:
        names = set(bundle.namelist())
        assert bundle.testzip() is None
    assert {
        "docs/response_distance/proofs.md",
        "docs/response_distance/mathematics.md",
        "docs/response_distance/storage.md",
        "docs/response_distance/literature.md",
        "docs/response_distance/audit_2026-10-14_independent.md",
        "notebooks/08_response_distance.ipynb",
        "notebooks/09_response_distance_review.ipynb",
    } <= names


def test_archive_inputs_are_current_and_do_not_include_the_output_zip() -> None:
    names = archive_members(ROOT)
    assert len(names) == len(set(names))
    assert "response_distance/note.tex" in names
    assert "response_distance/technical.md" in names
    assert "response_distance/output/response_distance_source.zip" not in names
    assert all((ROOT / name).is_file() for name in names)


def test_note_scientific_notation_uses_explicit_multiplication() -> None:
    source = (BRANCH / "note.tex").read_text()
    assert r"1.934\times10^{-5}<2\times10^{-5}<3.608\times10^{-5}" in source
    assert not re.search(r"[0-9](?:\\,|\s+)10\^", source)


@pytest.mark.parametrize(
    "relative", [".", "response_distance", "response_distance/notebooks"]
)
def test_examples_run_from_each_documented_directory(relative: str) -> None:
    notebook = json.loads((BRANCH / "notebooks/examples.ipynb").read_text())
    code = "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT / relative,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
