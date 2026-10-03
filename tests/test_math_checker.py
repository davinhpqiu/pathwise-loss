"""Rendering failures must fail the verification command."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "text, expected", [("$x^2$", 0), (r"$\undefinedAuditMacro$", 1)]
)
def test_math_checker_exit_status(tmp_path: Path, text: str, expected: int) -> None:
    if shutil.which("node") is None or not (ROOT / "node_modules/katex").exists():
        pytest.skip("Node and the existing KaTeX installation are required")
    (tmp_path / "README.md").write_text(text)
    result = subprocess.run(
        ["node", str(ROOT / "scripts/check_math.js"), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected
    assert f"{expected} failed" in result.stdout
