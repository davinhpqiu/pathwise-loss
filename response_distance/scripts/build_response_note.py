"""Build the self-contained response note and a curated reproducibility archive.

Requires an installed pdflatex for PDF export. --prepare-only refreshes the
standalone source without compiling. All storage rows are rechecked; references
come exclusively from response_distance/references.bib. No primary PDFs are bundled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from response_distance.code.response_certificates import (
    certify_bv_storage_example,
    planar_corner_certificate,
)
from response_distance.code.response_storage import optimize_euler_storage

BRANCH = "response_distance"
GRID = BRANCH + "/results/runs/response_storage/canonical_v2_20260930/bv_optima.csv"
SOURCE = BRANCH + "/note.tex"


def _closing_brace(text: str, start: int) -> int:
    depth, index = 0, start
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError("unbalanced BibTeX braces")


def bibliography_entries(text: str) -> dict[str, dict[str, str]]:
    entries = {}
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", text):
        key = match.group(1)
        if key in entries:
            raise ValueError(f"duplicate bibliography key: {key}")
        start = text.index("{", match.start())
        block = text[match.end() : _closing_brace(text, start)]
        fields = {}
        for field in re.finditer(r"(?m)^\s*(\w+)\s*=\s*\{", block):
            opening = field.end() - 1
            value = block[opening + 1 : _closing_brace(block, opening)]
            fields[field.group(1).lower()] = " ".join(value.split())
        entries[key] = fields
    return entries


def reference_block(source: str, bibliography: str) -> str:
    keys = list(
        dict.fromkeys(
            key.strip()
            for cite in re.findall(r"\\cite\{([^}]+)\}", source)
            for key in cite.split(",")
        )
    )
    entries = bibliography_entries(bibliography)
    output = [r"\begin{thebibliography}{99}"]
    for key in keys:
        entry = entries[key]
        author, title, year = (entry[field] for field in ("author", "title", "year"))
        author = author.replace(" and ", "; ")
        lines = [
            rf"\bibitem{{{key}}}",
            author if author.endswith(".") else author + ".",
        ]
        lines.append(r"\emph{" + title + "}.")
        if "journal" in entry:
            journal = entry["journal"]
            if "volume" in entry:
                journal += " " + entry["volume"]
            if "pages" in entry:
                journal += ", " + entry["pages"]
            lines.append(journal + f" ({year}).")
        else:
            lines.append(year + ".")
        url = entry.get("url")
        if not url and "eprint" in entry:
            url = "https://arxiv.org/abs/" + entry["eprint"]
        if not url and "doi" in entry:
            url = "https://doi.org/" + entry["doi"]
        if url:
            lines.append(r"\url{" + url + "}.")
        output.extend(lines)
    output.append(r"\end{thebibliography}")
    return "\n".join(output)


def replace_block(source: str, name: str, replacement: str) -> str:
    pattern = rf"(?s)(% BEGIN GENERATED {name}\n).*?(\n% END GENERATED {name})"
    updated, count = re.subn(
        pattern, lambda match: match[1] + replacement + match[2], source
    )
    if count != 1:
        raise ValueError(f"expected one generated {name} block")
    return updated


def verify_storage_grid(path: Path) -> dict:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("storage grid is empty")
    moved = cheaper = 0
    max_saving = Fraction(0)
    for row in rows:
        args = int(row["channel"]), float(row["scale"]), float(row["epsilon"])
        sets, costs = [], []
        for prefix, criterion in (("exact", "exact"), ("fpl", "fpl")):
            best, profile = optimize_euler_storage(*args, criterion=criterion)
            actual = best.depth, best.intervals, best.storage
            retained = tuple(
                int(row[f"{prefix}_{field}"])
                for field in ("depth", "intervals", "storage")
            )
            if actual != retained:
                raise ValueError(f"retained optimum differs at {args}: {prefix}")
            ties = {
                (choice.depth, choice.intervals)
                for choice in profile
                if choice.storage == best.storage
            }
            if ties != {tuple(pair) for pair in json.loads(row[f"{prefix}_ties"])}:
                raise ValueError(f"retained ties differ at {args}: {prefix}")
            sets.append(ties)
            costs.append(best.storage)
        changed = sets[0] != sets[1]
        if changed != (row["same_optimum_set"] == "False"):
            raise ValueError("retained optimum-set flag differs")
        moved += changed
        cheaper += costs[0] < costs[1]
        max_saving = max(max_saving, 1 - Fraction(costs[0], costs[1]))
    return {
        "cases": len(rows),
        "optimum_sets_changed": moved,
        "strictly_cheaper": cheaper,
        "max_saving_fraction": str(max_saving),
        "max_saving_percent": f"{100 * float(max_saving):.1f}",
        "verification": "current integer search; high-precision boundaries, not interval arithmetic",
    }


def archive_members(root: Path) -> list[str]:
    """Curate branch material plus its two shared, non-training dependencies."""
    names = [
        SOURCE,
        BRANCH + "/README.md",
        BRANCH + "/technical.md",
        BRANCH + "/literature.md",
        BRANCH + "/references.bib",
        BRANCH + "/notebooks/examples.ipynb",
        BRANCH + "/archive/response_distance_2026-10-14_before_reorganization.zip",
        "scripts/check_math.js",
        "tests/test_math_checker.py",
        "src/pathloss/__init__.py",
        "src/pathloss/brownian_messages.py",
        "src/pathloss/provenance.py",
        "pyproject.toml",
        "package.json",
        BRANCH + "/__init__.py",
    ]
    for directory in ("code", "scripts", "tests"):
        names.extend(
            str(path.relative_to(root))
            for path in sorted((root / BRANCH / directory).glob("*.py"))
        )
    retained = (
        "response_separation/canonical_20261002_seed211",
        "response_gap/canonical_20261002_seed0",
        "response_composition/canonical_v2_depth2_seed211",
        "response_composition/canonical_v2_depth3_seed211",
        "response_compression/canonical_seed211",
        "response_logode_merge/canonical_v6_seed211",
        "response_area_gauges/canonical_v3_seed211",
        "response_storage/canonical_v2_20260930",
        "response_mixed/canonical_v2_20261002",
        "response_mixed/review_v2_20261014",
        "response_mixed/consolidated_20261014",
        "response_mixed/sending_20261014",
        "response_mixed/organized_20261014",
    )
    for run in retained:
        directory = root / BRANCH / "results/runs" / run
        if not directory.is_dir():
            raise ValueError(f"missing retained run: {run}")
        names.extend(
            str(path.relative_to(root))
            for path in sorted(directory.glob("*"))
            if path.suffix in {".csv", ".json"}
        )
    names = list(dict.fromkeys(names))
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"missing or out-of-scope archive input: {name}")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    stats = verify_storage_grid(ROOT / GRID)
    certificate = planar_corner_certificate()
    if (
        not certificate.response_upper
        < Fraction(1, 50000)
        < certificate.certificate_lower
    ):
        raise AssertionError("corner certificate fails")
    if certify_bv_storage_example() != {"exact": (4, 1, 8), "fpl": (5, 1, 14)}:
        raise AssertionError("rational storage check fails")
    source_path = ROOT / SOURCE
    source = source_path.read_text()
    source = replace_block(
        source,
        "REFERENCES",
        reference_block(source, (ROOT / BRANCH / "references.bib").read_text()),
    )
    source_path.write_text(source)
    print(json.dumps(stats, indent=2))
    if args.prepare_only:
        return
    compiler = shutil.which("pdflatex")
    if compiler is None:
        raise SystemExit("pdflatex unavailable; standalone source is preserved")
    output = ROOT / BRANCH / "output"
    output.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "tmp/pdfs"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="response-note-", dir=scratch) as temporary:
        command = [
            compiler,
            "-no-shell-escape",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-jobname=response_distance_note",
            "-output-directory=" + temporary,
            str(source_path),
        ]
        for _ in range(3):
            completed = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, check=False
            )
            if completed.returncode:
                raise RuntimeError(completed.stdout[-8000:])
        log = (Path(temporary) / "response_distance_note.log").read_text()
        for error in (
            "Overfull",
            "undefined references",
            "undefined citations",
            "Missing character",
            "Label(s) may have changed",
        ):
            if error in log:
                index = log.index(error)
                raise RuntimeError(
                    f"PDF layout/reference check failed: {log[index : index + 1200]}"
                )
        pdf = output / "response_distance_note.pdf"
        shutil.copyfile(Path(temporary) / pdf.name, pdf)
    members = archive_members(ROOT)
    readme = (
        "# Response-distance source package\n\n"
        "Start with [response_distance/README.md](response_distance/README.md) "
        "for the document map, status and reproduction commands.\n\n"
        "The extracted layout matches the repository root. Shared Chen and "
        "bookkeeping helpers are included in src/pathloss; the response branch "
        "is under response_distance/. No GPU or torch is needed.\n"
    )
    manifest = {
        "format_version": 2,
        "branch_root": BRANCH,
        "grid": stats,
        "corner_certificate": {
            "scale": str(certificate.scale),
            "response_upper_rational": str(certificate.response_upper),
            "HS_lower_rational": str(certificate.certificate_lower),
            "tolerance": "1/50000",
        },
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in members
        },
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "package_readme_sha256": hashlib.sha256(readme.encode()).hexdigest(),
    }
    metadata = output / "response_distance_manifest.json"
    metadata.write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output / "response_distance_source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(pdf, BRANCH + "/output/" + pdf.name)
        bundle.write(metadata, BRANCH + "/output/" + metadata.name)
        bundle.writestr("README.md", readme)
        for name in members:
            bundle.write(ROOT / name, name)
    print(f"PDF: {pdf}")
    print(f"Source archive: {archive}")


if __name__ == "__main__":
    main()
