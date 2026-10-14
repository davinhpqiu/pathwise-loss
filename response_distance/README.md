# Response distance

A separate research branch on response-aware path compression. Updated
14 October 2026. The question is whether preserving a path's effect on bounded
linear systems permits cheaper representations than tensor-error certificates.

## Read only these

- [Discussion note](output/response_distance_note.pdf): the self-contained
  sending copy, including selected proofs and references.
- [Technical appendix](technical.md): definitions and the complete working
  proofs, including constants, homogeneous merges, storage and planar BV results.
- [Literature review](literature.md): primary-source locators and novelty limits.

[Examples notebook](notebooks/examples.ipynb) is for running the examples,
not a second exposition. [Editable note](note.tex) accompanies the PDF.
The source package is `output/response_distance_source.zip`; the builder
regenerates it, so it is sent with the PDF but not versioned. Rebuilding after
extraction recreates the ZIP rather than nesting it inside itself.

## What has been established

| result | scope and limitation |
|---|---|
| Sharp planar ratio $2\sqrt2/9$ | Degree-two log-ODE, fixed partition, planar BV, nonzero cubic discrepancy; uniform in response dimension $n\ge2$, with explicit scale and perturbation radii. |
| All-scale homogeneous merge separation | Adjacent area chains of every length $\ell\ge3$; the comparison is between the stated decoded factors, not automatically the original path. |
| Exact BV Euler minimax $e^r-P_N(r/m)^m$ | Path-dependent, coefficient-independent partitions; finite optima can move; leading asymptotics persist. Scalar expression appears in $(\varepsilon,r)$-representation paper. |
| Pure Engel storage bracket | Euler decoding and dense generic layout only; a single pure log-ODE factor is exact under a different decoder. |
| Exact directional gauges | Planar levels two/three, Engel directions, single words, mixed level four at $n=2$. Higher-dimensional mixed level-four attainment is open. |

The corner example at $t=1/20$ has response upper bound below
$1.934\times10^{-5}$ and HS lower bound above $3.608\times10^{-5}$.
Tolerance $2\times10^{-5}$ therefore distinguishes the two decisions.
The displayed bounds and the small storage example have rational certificates.
On the retained 135-case BV grid, 85 optimum sets move and 84 costs fall;
the largest saving is $3/7$ (42.9%). This is a grid result, not a universal rate.

## Status and evidence discipline

This is a discussion draft prepared with AI assistance. Expert correctness
review and exact-ratio novelty confirmation remain outstanding. Classical
matrix identities, development separation, commutator inequalities and leading
log-ODE error terms are attributed in the literature review. The candidate
addition is the quantitative response/certificate comparison and its finite
compression decisions, not those foundations.

Analytic bounds, explicit matrix witnesses, nonconvex lower estimates and modular
ranks are different evidence types. Floating-point agreement proves neither an
exact supremum nor dimension sufficiency. General grid searches use
high-precision boundary refinement, not interval arithmetic. Pre-29 September
terminal-Frobenius runs are superseded by operator-output runs.

Open: independent proof reading, rough input with $2<p<3$, mesh-uniform
nondegeneracy, general cubic-zero behavior, exact finite-scale response values,
and mixed level-four attainment above $n=2$. A linear coefficient budget does
not by itself certify nonlinear neural CDE errors.

## Folder and reproduction

Everything specific to this branch lives here: `code/`, `scripts/`, `tests/`,
`notebooks/`, `results/`, `output/` and `archive/`. The only shared Python helpers
are Chen products and run bookkeeping in the parent repository's `pathloss`
package; the source package includes them. `references.bib` is this branch's
bibliography. Main loss-project work is otherwise unchanged.

Run from the repository root in an activated Python 3.12 environment, with
NumPy, SciPy, Matplotlib and pytest. Install with `python -m pip install -e .`, or set
`PYTHONPATH=src` for the commands below. SymPy is optional for algebra tests,
Jupyter for the notebook. GPU and torch are unnecessary.

```bash
PYTHONPATH=src python -m pytest -q response_distance/tests
python -m response_distance.scripts.build_response_note
```

The builder needs `pdflatex`, refreshes references from `references.bib`, checks
all retained BV optimum/tie sets and rational examples, then exports PDF,
manifest and source ZIP. `--prepare-only` needs no compiler. The manifest records
file hashes; historical study metadata remains unchanged.

Every study must use a fresh output directory:

```bash
PYTHONPATH=src python -m response_distance.scripts.check_response_separation --kmax 8 --seed 211 --out response_distance/results/runs/response_separation/new_run
PYTHONPATH=src python -m response_distance.scripts.study_response_gap --seed 211 --out response_distance/results/runs/response_gap/new_run
PYTHONPATH=src python -m response_distance.scripts.study_response_composition --seed 211 --out response_distance/results/runs/response_composition/new_run
PYTHONPATH=src python -m response_distance.scripts.study_response_compression --seed 211 --out response_distance/results/runs/response_compression/new_run
PYTHONPATH=src python -m response_distance.scripts.study_response_logode_merge --seed 211 --out response_distance/results/runs/response_logode_merge/new_run
PYTHONPATH=src python -m response_distance.scripts.study_response_area_gauges --response-dims 2 3 --seed 211 --out response_distance/results/runs/response_area_gauges/new_run
PYTHONPATH=src python -m response_distance.scripts.study_response_storage --out response_distance/results/runs/response_storage/new_run
PYTHONPATH=src python -m response_distance.scripts.study_response_mixed --out response_distance/results/runs/response_mixed/new_run
node scripts/check_math.js --root response_distance
```

## Retained evidence

Paths below are relative to `results/runs/`. All old runs were moved intact;
their metadata still records the original commands and pre-move source paths.
Those hashes describe historical snapshots, not current source files.

| study | retained run | interpretation |
|---|---|---|
| Identity kernels | `response_separation/canonical_20261002_seed211` | Overflow-safe modular ranks; deficient rank needs symbolic confirmation. |
| Directional gaps | `response_gap/canonical_20261002_seed0` | Exact benchmarks plus lower estimates. |
| Common tuple/composition | `response_composition/canonical_v2_depth2_seed211`, `canonical_v2_depth3_seed211` | Lower estimates and an exact loop-order anchor. |
| Tensor Euler merge | `response_compression/canonical_seed211` | Exact formulas and lower estimates. |
| Log-ODE merge | `response_logode_merge/canonical_v6_seed211` | Analytic response sandwich, HS tails and lower estimates. |
| Area gauges | `response_area_gauges/canonical_v3_seed211` | Spectral upper bounds versus lower estimates. |
| Storage | `response_storage/canonical_v2_20260930` | Global BV integer search; pure response costs bracketed. |
| Planar/mixed bounds | `response_mixed/canonical_v2_20261002`, `consolidated_20261014`, `sending_20261014`, `organized_20261014` | Analytic envelopes and explicit witnesses; 48 deterministic cases. |

Orthogonal families, Gram extensions and arbitrary chain lengths have analytic
proofs and formula regressions, not additional ascent runs. Original designs,
detailed tables and dated audits are preserved in
`archive/response_distance_2026-10-14_before_reorganization.zip`.
This archive is historical, not a second set of current documents. Raw run
directories remain Git-ignored; curated CSV/JSON are bundled for reproduction.
Superseded and exploratory runs are kept locally in `results/runs/_superseded/`.

## Verification checkpoint

Checks: 351 project tests pass (one skip); all 135 retained BV optima and
tie sets reproduce; 1,134 branch Markdown/notebook expressions render.
Fresh mixed-block run has 48 rows, byte-identical to canonical table,
with current source hashes. Rebuilt PDF is visually checked.
Independent proof review remains pending.
