# Response-distance discussion package

Open `output/pdf/response_distance_note.pdf` first. It contains definitions, theorems,
examples, full supporting proofs and references, without local-file dependencies.
The source archive additionally supplies detailed research notes, code,
notebooks and retained numerical tables. Optional formula-render checks use
Node.js and KaTeX.

## Reproduce

From the extracted archive root, use Python 3.12 with NumPy, SciPy, Matplotlib
and pytest. SymPy is optional for exact algebra tests; Jupyter is optional for
notebooks. GPU and torch are unnecessary for this branch.

```bash
PYTHONPATH=src python -m pytest -q tests/test_response*.py
PYTHONPATH=src python scripts/study_response_mixed.py --out results/runs/response_mixed/new_run
PYTHONPATH=src python scripts/study_response_storage.py --out results/runs/response_storage/new_run
python scripts/build_response_note.py
```

The build command additionally needs installed `pdflatex`. It regenerates
reference entries from `papers/references.bib`, rechecks every retained BV
storage optimum and tie set, certifies the displayed examples with rational
arithmetic, and exports PDF plus source archive. `--prepare-only` refreshes
the standalone LaTeX source without requiring a compiler.

`response_distance_manifest.json` records source/PDF hashes, grid statistics
and exact rational corner bounds. Original study metadata records historical
source snapshots. New output directories preserve historical runs.

The note is a discussion draft. Expert correctness review and exact-ratio
novelty confirmation remain outstanding. General numerical searches are lower
estimates or evaluated bounds, without an interval-certification claim.
