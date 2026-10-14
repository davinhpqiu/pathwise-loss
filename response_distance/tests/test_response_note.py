import re
from pathlib import Path

import pytest

from response_distance.scripts.build_response_note import (
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
    root = Path(__file__).resolve().parents[2]
    source = (root / "response_distance/note.tex").read_text()
    bibliography = (root / "response_distance/references.bib").read_text()
    references = reference_block(source, bibliography)
    assert r"\bibitem{ferrucci2026concise}" in references
    assert r"\bibitem{bottcher2008frobenius}" in references


def test_note_explains_decoder_before_theorem_and_defers_tail_bookkeeping() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "response_distance/note.tex").read_text()
    main, appendix = source.split(r"\appendix", 1)
    opening = main.split(r"\label{thm:bv}", 1)[0]
    assert r"\ell_i^{\le2}" in opening
    assert r"\widehat\Phi^A_\pi(X)" in opening
    assert r"\xrightarrow{\text{exponentiate and compose}}" in opening
    for symbol in ("T_4", "Q_R", "Q_H", "C_T", "t_*"):
        assert symbol not in main
        assert symbol in appendix


def test_moved_note_equations_keep_unique_resolving_labels() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "response_distance/note.tex").read_text()
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    references = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", source)
    assert len(labels) == len(set(labels))
    assert set(references) <= set(labels)


def test_note_defines_norms_and_tensor_operations_before_criteria() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "response_distance/note.tex").read_text()
    opening = source.split(r"\label{eq:criterion}", 1)[0]
    for definition in (
        r"\norm{X}_{1\text{-var}}",
        r"S(X)_0=1",
        r"e_\otimes^U=\exp_\otimes U=",
        r"[U,V]=U\otimes V-V\otimes U",
        r"\norm{A_i}_F^2=\operatorname{tr}(A_i^\top A_i)",
    ):
        assert definition in opening
    assert r"\norm{M}_{\op}=\sup_{\norm{v}_2=1}\norm{Mv}_2" in source
