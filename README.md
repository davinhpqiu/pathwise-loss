# Principled loss functions for continuous-time learning

Summer project. Design and evaluate loss functions for **path-to-path**
learning, and compare them against pointwise MSE.

Supervision: Monday (informal) / Thursday 13:00 (formal).

This file is about **how to run things and where they live.** The mathematics
and the results live in the notebooks; what was decided and when lives in the
logbook.

---

## Start here

New to the project: read `CLAUDE.md` first (aim, conventions, reading order),
then the newest entry in `docs/logbook/` for what is currently in force, then
`docs/open_questions.md` for what is undecided.

> `docs/logbook/` and `docs/open_questions.md` are local working notes, kept
> untracked since 20/08. Versioned exception
> `docs/neural_ode_operator_experiments.md` records detailed procedure for
> notebooks 05 and 06. Orientation available from a fresh clone: this file,
> `CLAUDE.md`, the versioned procedure and notebooks, each of which states its
> own mathematics.

Current state, 08/09: all experiments used in the final evidence chain are
complete. Fixed-path Neural ODE study includes its hundred-block local-signature
refinement; Brownian-to-OU operator study is closed; Brownian-message sensitivity
includes core acceptance, independent RoughPy verification, 10,000-stream main
study and independent 10,000-stream confirmation. Classification is deferred;
the early GRU/Linear-Neural-CDE reconstruction branch is preliminary. Final
claims exclude both branches.

Recommended report reading order is notebook 01 for the sampling-measure
argument, notebook 05 for the controlled fixed-path comparison, notebook 06 for
transfer to stream-to-stream operator learning, and notebook 07 for direct
sensitivity to first- and second-level Brownian rough-path messages. Notebook 02
is a roughness diagnostic; notebooks 03 and 04 document earlier or deferred
branches and state their evidence limits at the top.

### Experimental story

Project asks what closeness between paths should mean when training or comparing
stream-valued objects. [Notebook 01](notebooks/01_integral_norms.ipynb) shows
that equal sample weights average error under observation distribution, while
elapsed-time quadrature approximates an integral over time.
[Notebook 05](notebooks/05_neural_ode_path.ipynb) turns that distinction into a
controlled Neural ODE study: uneven sampling separates MSE from $J_2$;
derivative supervision is powerful for one smooth target; signature behavior
depends on optimization, truncation and local partition.
[Notebook 06](notebooks/06_brownian_ou_operator.ipynb) transfers sampling result
to a causal Brownian-driver to OU-response operator.
[Notebook 07](notebooks/07_brownian_message_sensitivity.ipynb) then isolates
what coordinate norms cannot see: area-only rough-path messages leave
coordinate paths unchanged, while lift-aware signature comparisons respond,
with local sensitivity controlled by window placement. Combined result is a
map of what each discrepancy controls and misses, not one universally best
loss.

## Setup: first time

```bash
cd "path/to/Pathwise Loss/pathwise-loss"

python3 -m venv .venv                # note: python3, not python; see below
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # installs pathloss + numpy/scipy/matplotlib
pip install -r requirements.txt      # jupyter, pytest, pyyaml, the rest

python -m ipykernel install --user --name pathwise-loss
```

`requirements.txt` is **core only** and needs no compiler, GPU or git clone.
The modelling stack for Neural ODE, Neural CDE and signature experiments is in
`requirements-ml.txt`; install it only when running those notebooks or fits.

Check it worked:

```bash
pytest -q                            # expect: all pass
```

## Running things

### Tests

```bash
pytest -q                  # all
pytest -q -k p_variation   # one group
pytest -q -v               # see the names: the names are documentation
```

Run before trusting notebook output. If they pass, every formula the notebooks
rely on does what it claims.

### Maths rendering

```bash
npm install katex && node scripts/check_math.js
```

Renders every `$...$` and `$$...$$` in the logbook, README and notebooks through
KaTeX. KaTeX implements a subset of LaTeX, so expressions that are valid TeX can
still fail to display.

### Notebooks

```bash
jupyter lab notebooks/01_integral_norms.ipynb
```

Then *Run All*: a few seconds, no GPU, nothing external. Notebook 01 has no
code; everything it claims is proved in the text or asserted in `tests/`.

Headless re-run, for checking nothing broke after editing the library:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_integral_norms.ipynb
```

### Training experiments

Each runner has one task. Each notebook repeats the commands producing its
own results, so the runs below are the reference list rather than the only
place they appear.

| script | task |
|---|---|
| `run_reconstruction.py` | one fit: reconstruct a path at query times from irregular context |
| `run_integral_study.py` | grid of paired reconstruction fits: MSE against weighted $J_2$ |
| `run_fixed_path.py` | fit a Neural ODE to one fixed target path under one loss |
| `run_fixed_path_study.py` | run one fixed-path stage locally and stream progress |
| `run_ou_operator.py` | Brownian-to-OU acceptance, audit or one Neural CDE fit |
| `run_ou_operator_study.py` | run one Brownian-to-OU stage locally |
| `run_classification.py` | 1-NN path-distance classification, no trained model |
| `run_brownian_message_study.py` | controlled Brownian rough-path message sensitivity |

`run_reconstruction.py` generates data, trains, evaluates, and writes
configuration, history, metrics, and provenance:

```bash
python scripts/run_reconstruction.py --config configs/baseline_mse.yaml --out results/runs/test
```

Use a new output directory for each run.

Core paired study, locally for one seed:

```bash
python scripts/run_integral_study.py \
  --config configs/integral_core_study.yaml \
  --out results/runs/integral_core \
  --seed 0
```

Fixed-path Neural ODE adequacy check:

```bash
python scripts/run_fixed_path.py \
  --config configs/neural_ode_fixed_path.yaml \
  --out results/runs/neural_ode_fixed_path/adequacy \
  --adequacy
```

Run one paired comparison member after adequacy passes:

```bash
python scripts/run_fixed_path.py \
  --config configs/neural_ode_fixed_path.yaml \
  --out results/runs/neural_ode_fixed_path/restricted/seed0/clustered/j2 \
  --seed 0 \
  --capacity restricted \
  --condition clustered \
  --loss j2
```

Every run saves initial-state fingerprint, history, dense metrics, fitted path,
derivatives, model state and diagnostic plot. Configurations with
`evaluation_checkpoints` also save common checkpoint metrics and fitted paths.
Equal seed and capacity must give the same fingerprint across losses.

Run a complete stage locally and watch each fit update in sequence:

```bash
caffeinate -i python scripts/run_fixed_path_study.py \
  --config configs/neural_ode_fixed_path.yaml \
  --out results/runs/neural_ode_fixed_path \
  --stage primary
```

Stages are `primary`, `h1`, `signature`, `signature_pilot` and `configured`.
Add `--seed 0` or `--capacity restricted` for a smaller subset. Completed runs
are skipped.

Seed-zero 5,000-update pilot triggered budget check in notebook 05. Paired
expressive clustered MSE and $J_2$ diagnostic ran at 10,000 updates:

```bash
sbatch scripts/arc/submit_fixed_path_budget_diagnostic.slurm
```

It writes under `results/runs/neural_ode_fixed_path_budget_10k/`. Ten thousand
updates are fixed as a finite compute budget; late-window ratios prevent
describing terminal fits as converged.

Signature implementation must pass tests and levelwise audit before training:

```bash
python scripts/run_fixed_path.py \
  --config configs/neural_ode_fixed_path_signature_10k.yaml \
  --out results/runs/neural_ode_fixed_path_signature_10k/signature_audit \
  --signature-audit
```

Audit passed on 28 August with unit output scaling. It records finite, nonzero
gradients together with strong depth-dependent imbalance; checkpointed training
tests whether structural terms become active. Run seed-zero uniform MSE, $J_2$,
global depth-four signature and ten-block local depth-two fits in both
capacities. ARC launches all eight members in parallel:

```bash
sbatch scripts/arc/submit_fixed_path_signature_pilot.slurm
```

Local sequential equivalent is:

```bash
caffeinate -i python scripts/run_fixed_path_study.py \
  --config configs/neural_ode_fixed_path_signature_10k.yaml \
  --out results/runs/neural_ode_fixed_path_signature_10k \
  --stage signature_pilot
```

Experiment A closeout uses three seeds, two capacities and its exact configured
case list. Existing 10,000-update pilot and budget runs are reused by metadata;
older 5,000-update results cannot satisfy this matrix.

```bash
python scripts/check_fixed_path_study.py \
  --config configs/neural_ode_fixed_path_closeout_10k.yaml \
  --out results/runs/neural_ode_fixed_path_closeout_10k

sbatch scripts/arc/submit_fixed_path_closeout_array.slurm
```

Completion checker exits with status 2 and lists missing exact identities until
matrix is complete.

Learning-rate sensitivity has a separate configuration and result root:

```bash
sbatch scripts/arc/submit_fixed_path_lr_sensitivity_array.slurm
```

After model files are local, evaluate Simpson, Romberg and three RK4 step sizes
for one fit:

```bash
python scripts/evaluate_fixed_path_resolution.py \
  --run results/runs/neural_ode_fixed_path_signature_10k/restricted/seed0/uniform/mse
```

The completed final Experiment A refinement replaces ten local signature blocks by 100
while applying the levelwise homogeneity correction derived in notebook 05.
It keeps the same target, 64 observations, models, seeds, optimizer and
10,000-update budget. The value-gradient audit passed; this command reproduces it:

```bash
sbatch scripts/arc/submit_fixed_path_local_fine_audit.slurm
```

Audit passed on 7 September: all components and gradients are finite, and
homogeneity scaling restores fine level terms to the same numerical order as
ten-block terms. Six independent fits then completed; this command reproduces
the array:

```bash
sbatch scripts/arc/submit_fixed_path_local_fine_array.slurm
```

Check exact completion with:

```bash
python scripts/check_fixed_path_study.py \
  --config configs/neural_ode_fixed_path_local_fine_10k.yaml \
  --out results/runs/neural_ode_fixed_path_local_fine_10k
```

After downloading the six run directories, build the paired comparison used by
notebook 05:

```bash
python scripts/analyze_fixed_path_local_refinement.py \
  --refinement-root results/runs/neural_ode_fixed_path_local_fine_10k \
  --baseline-root results/runs/neural_ode_fixed_path_closeout_10k \
  --baseline-root results/runs/neural_ode_fixed_path_signature_10k \
  --baseline-root results/runs/neural_ode_fixed_path_budget_10k \
  --out results/runs/neural_ode_fixed_path_local_fine_10k/analysis
```

Brownian-to-OU stream operator begins with implementation gates:

```bash
python scripts/run_ou_operator.py \
  --config configs/neural_cde_brownian_ou.yaml \
  --out results/runs/neural_cde_brownian_ou/acceptance \
  --acceptance
```

After `acceptance.json` reports `passed: true`, run paired MSE and $J_2$ fits
locally or on ARC:

```bash
python scripts/run_ou_operator_study.py \
  --config configs/neural_cde_brownian_ou.yaml \
  --out results/runs/neural_cde_brownian_ou \
  --stage primary \
  --evaluate-test
sbatch scripts/arc/submit_ou_primary_array.slurm
```

Optional OU signature calibration starts with its training-data scaling audit:

```bash
python scripts/run_ou_operator.py \
  --config configs/neural_cde_brownian_ou.yaml \
  --out results/runs/neural_cde_brownian_ou/signature_audit \
  --signature-audit
```

Review audit, set `signature.audit_accepted: true`, then run signature stage or
`scripts/arc/submit_ou_signature_array.slurm`.

Brownian message study uses supplied four-dimensional step-two rough streams.
Core output includes variance-normalized signature discrepancies,
response-derived $r\in\{0.5,1,2\}$ family and normalized finite depth-four
signature-kernel distance. These use cached explicit signatures and add no
dependency; optional Goursat-PDE rough kernel remains separate.
Run core algebra checks, independent RoughPy reference, pilot preparation and
pilot conditions in that order:

```bash
python scripts/run_brownian_message_study.py \
  --config configs/brownian_message.yaml \
  --out results/runs/brownian_message \
  --stage acceptance
python scripts/run_brownian_message_study.py \
  --config configs/brownian_message.yaml \
  --out results/runs/brownian_message \
  --stage reference
python scripts/run_brownian_message_study.py \
  --config configs/brownian_message.yaml \
  --out results/runs/brownian_message \
  --stage prepare-pilot
sbatch scripts/arc/submit_brownian_message_pilot_array.slurm
```

`reference` requires separately installed RoughPy and locks main jobs until its
comparison passes. Main cache is shared across 48 condition tasks:

```bash
sbatch scripts/arc/submit_brownian_message_prepare.slurm
sbatch scripts/arc/submit_brownian_message_main_array.slurm
sbatch scripts/arc/submit_brownian_message_four_class.slurm
python scripts/check_brownian_message_study.py \
  --config configs/brownian_message.yaml \
  --out results/runs/brownian_message \
  --stage main
python scripts/run_brownian_message_study.py \
  --config configs/brownian_message.yaml \
  --out results/runs/brownian_message \
  --stage aggregate
```

The transition and predecessor amplitudes are now frozen in configuration.
Prepare the untouched second-ensemble cache and then submit the 42-condition
confirmation array:

```bash
sbatch scripts/arc/submit_brownian_message_prepare_confirmation.slurm
sbatch scripts/arc/submit_brownian_message_confirmation.slurm
```

Submit the second command only after preparation completes successfully.

Supervisor-provided BasicMotions classification. Raw data provide the archive
anchor; alternative preprocessing has a separate configuration:

```bash
python scripts/run_classification.py \
  --config configs/classification_basicmotions.yaml
python scripts/run_classification.py \
  --config configs/classification_basicmotions_training_channel.yaml
python scripts/run_classification.py \
  --config configs/classification_basicmotions_per_series.yaml
```

### Presentation

```bash
python scripts/build_presentation_figures.py      # regenerate every deck figure
cd presentation && latexmk -pdf pathwise_loss_presentation.tex
```

Deck references `presentation/assets/` only. That folder is written entirely by
the figure script, which reads stored runs, verifies each run's configuration
against what the slide caption claims, and records provenance in
`assets/figure_manifest.json`. Editing a caption that names a capacity, seed or
condition means updating `FIXED_PATH_RUN` or `OU_RUN` in the script, which then
fails loudly if the named run disagrees.

Figures need `results/runs/`, which is gitignored, so a fresh clone rebuilds the
deck from committed assets and regenerates them only after rerunning the studies.

### On ARC

ARC becomes relevant once model training starts. See [`docs/arc_guide.md`](docs/arc_guide.md).
Short version:

```bash
ssh username@htc-login.arc.ox.ac.uk
cd $DATA/pathwise-loss
srun -p interactive --pty /bin/bash        # builds go on interactive nodes
bash scripts/arc/setup_env.sh              # once
sbatch scripts/arc/submit_gpu.slurm configs/<name>.yaml
squeue -u $USER
```

---

## Notebooks

| notebook | what it covers | status |
|---|---|---|
| `01_integral_norms.ipynb` | The estimator and why: quadrature rules, convergence rates, **why MSE is inconsistent under non-uniform sampling**, choice of $p$. Exposition; verification is in `tests/` | complete |
| `02_p_variation.ipynb` | roughness of a path: definition and three implemented estimators; index estimation is documented future work | supporting diagnostic |
| `03_loss_comparison.ipynb` | early matched MSE against weighted-$J_2$ reconstruction study: pilot and held-out seed-0 GRU/Linear-NCDE results | preliminary; further seeds deferred |
| `04_classification.ipynb` | fixed 1-NN path-distance benchmark design and one retained historical output | deferred; not final evidence |
| `05_neural_ode_path.ipynb` | fixed-target Neural ODE loss comparison, including three-seed closeout and hundred-block local-signature refinement | complete |
| `06_brownian_ou_operator.ipynb` | causal Brownian-driver to OU-response Neural CDE: algorithm, gates and paired loss analysis | complete |
| `07_brownian_message_sensitivity.ipynb` | controlled increment and Lévy-area messages: direct discrepancy curves and detection | complete |

Each notebook records experiment stages, mathematics and results. Launch
commands live in this README. Preliminary missingness check is part of notebook
03 rather than a separate experiment.

Neural ODE and stream-to-stream experiments are specified in
[`docs/neural_ode_operator_experiments.md`](docs/neural_ode_operator_experiments.md).
Fixed-path implementation and final results use notebook 05.
Brownian-to-OU implementation and final results use notebook 06.

---

## Layout

```
pathwise-loss/
├── README.md                # how to run things (this file)
├── requirements.txt
├── pyproject.toml           # makes `pip install -e .` work
├── .gitignore
│
├── src/pathloss/            # THE LIBRARY. Everything that must be correct.
│   │                        # NumPy, no training dependency:
│   ├── norms.py             # quadrature, L^p integral norms and distances
│   ├── pvar.py              # p-variation: brute force, O(N^2) DP, pruned
│   ├── classification.py    # fixed 1-NN evaluation on labelled archives
│   ├── brownian_messages.py # BCH messages and rough-stream discrepancies
│   ├── brownian_message_study.py # caches, gates and message analysis
│   │                        # data:
│   ├── paths.py             # generators, irregular sampling, missingness
│   ├── datasets.py          # context / target / fine-grid training examples
│   │                        # torch:
│   ├── losses.py            # differentiable MSE, weighted L^p, Sobolev H^1
│   ├── models.py            # GRU query and Linear Neural CDE baselines
│   ├── train.py             # training loop and evaluation
│   ├── fixed_path.py        # fixed-target Neural ODE: target, model, fitting
│   ├── fixed_path_study.py  # exact fixed-path run identities and completion
│   ├── operator.py          # path-output Neural CDE and Brownian-to-OU fitting
│   ├── signatures.py        # differentiable piecewise-linear signatures
│   │                        # bookkeeping:
│   └── provenance.py        # git state, config loading, run metadata
│
├── tests/                   # pytest. Run before trusting notebook output.
├── notebooks/               # THE EXPERIMENTS: maths, code, results, together
├── scripts/                 # runners: argument parsing and file output only
│   ├── run_reconstruction.py    # one config -> data -> model -> loss -> results
│   ├── run_integral_study.py# job grid for the paired integral-loss study
│   ├── run_fixed_path.py    # fixed-path fit, adequacy check or signature audit
│   ├── run_fixed_path_study.py # staged local fixed-path runs
│   ├── check_fixed_path_study.py # exact closeout completion report
│   ├── evaluate_fixed_path_resolution.py # quadrature and solver refinement
│   ├── run_ou_operator.py   # OU acceptance, audit or one fit
│   ├── run_ou_operator_study.py # staged local OU runs
│   ├── run_classification.py# 1-NN path-distance benchmark
│   ├── check_math.js        # render every formula through KaTeX
│   └── arc/                 # SLURM submission scripts
├── configs/                 # one YAML per experiment; never hardcode in scripts
├── data/{raw,synthetic}/    # gitignored. Regenerate, don't commit.
├── results/{runs,logs,figures}/
├── papers/                  # PDFs + references.bib
└── docs/                    # local notes plus one versioned procedure
    ├── arc_guide.md         # Oxford ARC: accounts, SLURM, storage
    ├── neural_ode_operator_experiments.md # versioned design and execution record
    ├── open_questions.md    # register of what is undecided
    └── logbook/             # dated notes and findings. Append-only.
```

### Placement

| | goes in |
|---|---|
| A derivation, an experiment, a plot, the reading of a result | the relevant **notebook** |
| A function used more than once | **`src/pathloss/`**, with a test |
| What was decided, results, interpretations | **`docs/logbook/`**, dated |
| Something undecided, with its definitions | **`docs/open_questions.md`** |
| How to run something | **this README** |

**The one structural rule:** anything that must be correct lives in
`src/pathloss/` and has a test in `tests/`. Notebooks import it. A notebook cell
must never be the only copy of a function.

The same rule applies to `scripts/`. A runner parses arguments, loads a
configuration and writes files; anything it computes belongs in the library. A
helper needed by a second runner moves to `src/pathloss/` rather than being
copied: `provenance.py` exists because `git_sha` had drifted into three
versions, one of which caught a narrower set of exceptions than the others.

**Working habit:** when a notebook produces something you didn't expect, write a
dated paragraph in `docs/logbook/` the same day. The notebook records *what the
result is*; the logbook records *interpretations and relevant decisions*. The two
drift apart quickly if the second is left until the write-up.

---

## Status

| | status |
|---|---|
| Quadrature, $L^p$ norms, convergence studies | done: notebook 01 |
| $p$-variation algorithms | implemented and retained as notebook 02 diagnostic |
| Early GRU and Linear Neural CDE reconstruction | seed-0 pilot complete; preliminary evidence only: notebook 03 |
| BasicMotions 1-NN classification | corrected design retained; fresh runs deferred and excluded: notebook 04 |
| Fixed-path Neural ODE loss comparison | complete, including three seeds, numerical checks and hundred-block refinement: notebook 05 |
| Brownian-to-OU path operator | complete for three paired seeds: notebook 06 |
| Brownian message sensitivity | complete main and independent confirmation analyses: notebook 07 |

No further run is required for a coherent final report. Optional extensions are
listed in `docs/open_questions.md`; they should not be mixed into the completed
evidence chain without a new design decision.

---

## Conventions

A path is a pair `(t, x)` with `t` of shape `(T,)` and `x` of shape
`(..., T, d)`: time is the second-to-last axis, matching `torchcde` and
`signatory`'s `(batch, time, channel)`.

Synthetic data is always generated on a **fine grid** treated as ground truth,
then subsampled to produce what the model sees. Keeping those two objects
separate is what makes the robustness experiments well-defined.

---

## References

`papers/` holds the PDFs and `papers/references.bib`: one list, used for the
report.

**Citations go at the point of use**, in the notebook cell or logbook entry
where the paper actually changed a decision, not in a separate index. Notebook
01 ends with the works it cites and the section each one bears on; a reading
order is in `docs/logbook/2026-08-09.md`.

When adding a paper: drop the PDF in `papers/`, add the BibTeX entry with a
`note` giving the filename, and cite it where it mattered.
