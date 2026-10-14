from pathlib import Path

import pytest
from scripts.build_response_note import (
    bibliography_entries,
    reference_block,
    replace_block,
)


def test_bibliography_parser_preserves_nested_math_and_accents() -> None:
    text = r"""
@misc{sample,
  author = {Perr{\'e}e, Oliver and B{\"o}ttcher, Albrecht},
  title = {A {$C^*$} example},
  year = {2026},
  eprint = {1234.5678}
}
"""
    entry = bibliography_entries(text)["sample"]
    assert entry["title"] == r"A {$C^*$} example"
    result = reference_block(r"\cite{sample}", text)
    assert r"Perr{\'e}e" in result
    assert r"https://arxiv.org/abs/1234.5678" in result
    with pytest.raises(ValueError, match="duplicate"):
        bibliography_entries(text + text)


def test_generated_block_preserves_other_source_and_requires_unique_marker() -> None:
    source = "Before\n% BEGIN GENERATED GRID\nold\n% END GENERATED GRID\nAfter"
    assert replace_block(source, "GRID", r"\newcommand{\X}{1}") == (
        "Before\n% BEGIN GENERATED GRID\n\\newcommand{\\X}{1}\n% END GENERATED GRID\nAfter"
    )
    with pytest.raises(ValueError):
        replace_block("missing", "GRID", "new")


def test_note_citations_all_resolve_from_canonical_bibliography() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "papers/response_distance_note.tex").read_text()
    bibliography = (root / "papers/references.bib").read_text()
    references = reference_block(source, bibliography)
    assert r"\bibitem{ferrucci2026concise}" in references
    assert r"\bibitem{bottcher2008frobenius}" in references
