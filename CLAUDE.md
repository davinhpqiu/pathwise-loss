# Conventions for this repo

## Orientation

**Aim.** Design and evaluate loss functions for path-to-path learning, against
pointwise MSE. Oxford summer project, six weeks, one student (Davin). Original
specification is `papers/` project proposal: sequence-to-sequence and neural CDE
baselines, synthetic trajectories with controllable irregular sampling, one small
real dataset at the end.

**Reading order for anyone arriving cold.**

1. This file.
2. `README.md`: layout, how to run.
3. Newest file in `docs/logbook/`: current state, decisions in force.
4. `docs/open_questions.md`: what is undecided, with definitions attached. Every
   entry is resumable without reading its source.
5. Notebooks in order, as needed. Each states its own mathematics.

**State of play is the newest logbook entry, not this file.** Directions change:
$p$-variation was central for a week and is now a diagnostic (17/08); a
pre-registered protocol was written and deleted inside two days. Check the date
on a claim before acting on it. Length of a notebook indicates effort spent, not
current priority.

## Working practice

- **Davin runs tests and training.** Write them, leave execution to him, and say
  plainly which parts are unverified.
- **Davin decides what to raise with supervisors.** Record open items in
  `docs/open_questions.md`; do not draft agendas or supervisor questions unasked.
- **Ask before adding a dependency.** `requirements.txt` stays installable
  without a compiler, GPU, or git clone. Modelling stack is separate, in
  `requirements-ml.txt`.
- **Correct rather than narrate.** A wrong claim is removed. Saying so once in
  chat suffices.

## Verification

**Anchor against something independent.** A test comparing one implementation
against another written by the same author checks consistency and nothing else.
Use a closed form, a published number, a brute-force enumerator, or a second
route to the same quantity. `p_variation_brute` anchors `pvar.py`;
`tests/test_pipeline.py` anchors the torch losses against the NumPy ones in
`norms.py`; Corollary 1 of `2026-08-16.md` would anchor a signature
implementation against `integral_norm`.

**Suspect the harness first.** Recorded failures in this repo were, in order: an
adequacy test that held for every path and so tested nothing; a test suite that
compared code against its own second implementation; an acceptance test granting
300 optimiser steps where thousands were needed. In each the code was sound and
the check was not.

**A threshold met by lowering the threshold measures nothing.** When an
acceptance test fails, fix the cause or raise the budget. Loosening the criterion
hides whatever else was wrong.

**Justify additions with a test that could remove them.** Fourier features in
`models.py` are kept because `test_fourier_features_help` shows the raw-scalar
decoder losing under an equal budget and seed. Complexity without such a test
gets deleted.

## Environment

Run from repo root; `pyproject.toml` sets `pythonpath = ["src"]`, so no install
step.

- `pytest -q` for everything. `tests/test_pipeline.py` needs torch and skips
  without it.
- `python scripts/run_experiment.py --config configs/<name>.yaml --out results/runs/<name>`
- `npm install katex && node scripts/check_math.js` renders every formula in the
  repo through KaTeX. Run after editing mathematics: KaTeX implements a subset of
  LaTeX and `\unicode`, `\shuffle` and `\mathscr` fail silently in a notebook.
- `aeon` pins `numpy<2.5`, `scipy<1.18`, `pandas<2.4`, and installing it
  downgrades all three. Bounds are in `requirements.txt` so a fresh install
  resolves once.

## Writing

Applies to markdown files and notebooks.

Keep all text written (in markdown files / notebooks) concise and well-defined. That means generally as few words as suffices. Never use the positive, negative sentence structure for emphasis (e.g. This is sth, it is not sth.), i.e. do not add an extra clause just to emphasise the previous point that does not add additional information. Do not use sentence structures such as "this is the important one" or "The one that matters" these are pointless emphasis that clutters reading and affects interpretation. Any non-objective claim can only be presented if it is backed by rigorous reasoning. Define all relevant mathematical formulae and notation. Use fewer definite articles e.g. "the" or "an". Grammatical rules can be suspended to make way for conciseness. Do not pose a question, as a title or as a regular sentence. No question structures unless it is a real open question, in which case say that it is open. Do not open a sentence or a heading with a negation: put the negated part in brackets or after the subject, e.g. "the one-dimensional algorithm (not implemented)" rather than "Not implemented: the one-dimensional algorithm".

Also:

- Few hyphens and dashes. Colons or brackets instead. Dash beside mathematics reads as minus.
- Define at point of use, not in preamble.
- Quote standard results, do not demonstrate them: citation plus test, not table of numbers.
- No number a test already asserts. Closed forms preferred: they do not go stale.
- No assumption about data. Not necessarily SDE trajectories, smooth, or uniformly sampled. State assumption where used.
- Cite at point of use. `papers/references.bib` is single list.

## Structure

| | goes in |
|---|---|
| mathematics, experiments, results | relevant **notebook** |
| function used more than once | **`src/pathloss/`**, with test |
| decisions, changes, findings | **`docs/logbook/`**, dated |
| how to run something | **`README.md`** |

Anything that must be correct lives in `src/pathloss/` with a test. Notebook cell is never sole copy of a function.

## Experiment design

An experiment's design is written down **before** the run: question, what varies,
what is held fixed, and what each outcome would mean. For a comparison that
produces a claim, that goes in the notebook that will report it, committed ahead
of the numbers. Departures are recorded in the logbook with a date and a reason,
so a prediction stays distinguishable from a post-hoc rationalisation.

Pre-registration in a separate file was tried (`docs/protocol_01_*.md`, 16/08)
and dropped on 17/08: it suited a confirmatory comparison the project turned out
not to need yet, and a standalone file drifted from the code it described.

## Logbook

Records decisions and changes, dated. Not word counts, code cell counts, test counts, or editorial edits to notebooks: rewording a notebook is not a project event.

Entries are not rewritten to reflect later opinion. Corrected in place: cross-references into notebooks since renumbered.

A claim later **superseded** gets a dated line saying so. A claim that was simply **wrong** is removed, with no trace and no narration of the error: a note is for what is true, not a record of drafting.
