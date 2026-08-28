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

> `docs/` is working notes, kept locally and deliberately untracked since
> 20/08. A fresh clone has no `docs/` directory, so every reference to it below
> resolves only in the author's working copy. Orientation available from the
> repository alone: this file, `CLAUDE.md`, and the notebooks, each of which
> states its own mathematics.

Current state, 22/08: short-term focus is continuous path parameterisation by a
Neural ODE, followed by Brownian-driver to OU-response stream learning with a
Neural CDE. Detailed procedure is
[`docs/neural_ode_operator_experiments.md`](docs/neural_ode_operator_experiments.md).
Existing reconstruction, classification and ARC work supplies supporting
calibration. $p$-variation is a diagnostic.

## Setup: first time

```bash
cd "path/to/Pathwise Loss/pathwise-loss"

python3 -m venv .venv                # note: python3, not python; see below
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # installs pathloss + numpy/scipy/matplotlib
pip install -r requirements.txt      # jupyter, pytest, pyyaml, the rest

python -m ipykernel install --user --name pathwise-loss
```

`requirements.txt` is **core only**, does not need compiler, GPU, or
a git clone. The modelling stack (torch, neural CDEs, signatures) is in `requirements-ml.txt`. See [Troubleshooting](#troubleshooting) before installing that one.

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

Stages are `primary`, `h1`, `signature` and `signature_pilot`. Add `--seed 0` or
`--capacity restricted` for a smaller subset. Completed runs are skipped.

Seed-zero 5,000-update pilot triggered budget check in notebook 05. Paired
expressive clustered MSE and $J_2$ diagnostic ran at 10,000 updates:

```bash
sbatch scripts/arc/submit_fixed_path_budget_diagnostic.slurm
```

It writes under `results/runs/neural_ode_fixed_path_budget_10k/`. Ten thousand
updates are now fixed as a finite compute budget; late-window ratios prevent
describing terminal fits as converged. Extended primary and $H^1$ arrays remain
deferred while signature pilot is run.

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
| `02_p_variation.ipynb` | roughness of a path: definition, and the three implementations, one section each | complete |
| `03_loss_comparison.ipynb` | matched MSE against weighted-$J_2$ experiment: GRU and Linear NCDE, uniform and clustered targets, pilot and held-out seed-0 results | in progress |
| `04_classification.ipynb` | fixed 1-NN path-distance benchmark: explicit preprocessing and dependent/independent DTW controls, with signature extension defined | in progress |
| `05_neural_ode_path.ipynb` | fixed-target Neural ODE loss comparison: design, target inspection, acceptance criteria and result analysis | adequacy complete; comparisons unrun |
| `06_brownian_ou_operator.ipynb` | causal Brownian-driver to OU-response Neural CDE: algorithm, gates and paired loss analysis | implemented; runs absent |

Each notebook records experiment stages, mathematics and results. Launch
commands live in this README. Preliminary missingness check is part of notebook
03 rather than a separate experiment.

Neural ODE and stream-to-stream experiments are specified in
[`docs/neural_ode_operator_experiments.md`](docs/neural_ode_operator_experiments.md).
Fixed-path implementation and future results use notebook 05.
Brownian-to-OU implementation and future results use notebook 06.

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
│   │                        # data:
│   ├── paths.py             # generators, irregular sampling, missingness
│   ├── datasets.py          # context / target / fine-grid training examples
│   │                        # torch:
│   ├── losses.py            # differentiable MSE, weighted L^p, Sobolev H^1
│   ├── models.py            # GRU query and Linear Neural CDE baselines
│   ├── train.py             # training loop and evaluation
│   ├── fixed_path.py        # fixed-target Neural ODE: target, model, fitting
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
│   ├── run_ou_operator.py   # OU acceptance, audit or one fit
│   ├── run_ou_operator_study.py # staged local OU runs
│   ├── run_classification.py# 1-NN path-distance benchmark
│   ├── check_math.js        # render every formula through KaTeX
│   └── arc/                 # SLURM submission scripts
├── configs/                 # one YAML per experiment; never hardcode in scripts
├── data/{raw,synthetic}/    # gitignored. Regenerate, don't commit.
├── results/{runs,logs,figures}/
├── papers/                  # PDFs + references.bib
└── docs/                    # untracked since 20/08: absent from a fresh clone
    ├── arc_guide.md         # Oxford ARC: accounts, SLURM, storage
    ├── neural_ode_operator_experiments.md # next experiment procedure
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
| $p$-variation: brute force, $O(N^2)$ DP, pruned $O(N\log N)$ | done, tests pass 16/08 |
| $p$-variation index estimator | deferred: diagnostic only |
| MSE vs integral norm estimator under irregular sampling | done: notebook 01 §3 |
| Learned effect of target sampling mechanism | seed 0 complete for GRU and matched Linear NCDE; two seeds remain |
| Effect of exponent $p$ | implementation available; application-specific experiments deferred |
| Top-down segmentation (adequacy + similarity) | designed, not written: `docs/logbook/2026-08-12.md` |
| `src/pathloss/losses.py` (torch, differentiable) | done: MSE + weighted $L^p$ |
| Baseline model (GRU encoder + query-time decoder) | done: `src/pathloss/models.py` |
| Linear NCDE baseline | parameter matched core study implemented; seed 0 complete |
| `scripts/run_reconstruction.py` training loop | done, needs torch installed |
| 1-NN path-distance classification benchmark | corrected BasicMotions configurations written; fresh raw and preprocessing runs remain |
| Signatures | differentiable global/local fixed-path losses and independent checks written; value and gradient audit is next |
| Fixed-path Neural ODE | Fourier adequacy passes; 10,000-update finite-budget signature pilot prepared |
| Brownian-to-OU path operator | data, causal Neural CDE and acceptance gates implemented; work deferred |
| Controlled missingness | evaluator implemented; one-seed pipeline check only |
| Real dataset | not obtained |
| ARC | fixed-path eight-job signature pilot prepared; earlier integral and extended arrays retained |

Next: run fixed-path signature value and gradient audit, then matched eight-fit
signature pilot at 10,000 updates.
Remaining reconstruction and
classification runs are supporting tasks. Exponent-specific integral-norm
studies remain deferred.
`docs/open_questions.md` contains the remaining decisions.

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
