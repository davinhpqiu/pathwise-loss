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

Current state, 19/08: project week 2 of a 3 August to 11 September project.
Baseline pipeline trains end to end on synthetic paths under MSE or time-weighted
$L^p$ losses. Fine-grid evaluation, a three-seed matched loss comparison,
controlled missingness, and an initial Linear Neural CDE baseline are working.
$p$-variation is a diagnostic.

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

The experiment script generates data, trains, evaluates, and writes configuration,
history, metrics, and provenance:

```bash
python scripts/run_experiment.py --config configs/baseline_mse.yaml --out results/runs/test
```

Use a new output directory for each run.

Core paired study, locally for one seed:

```bash
python scripts/run_integral_study.py \
  --config configs/integral_core_study.yaml \
  --out results/runs/integral_core \
  --seed 0
```

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

Each notebook states its own mathematics, runs its own experiments, and reads
its own results. The preliminary missingness check is part of notebook 03 rather
than a separate experiment. That is where to look for a derivation.

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
│   ├── datasets.py          # synthetic generators and irregular sampling
│   ├── models.py            # sequence and continuous-time baselines
│   ├── train.py             # training and evaluation
│   ├── norms.py             # quadrature, L^p integral norms (NumPy)
│   ├── pvar.py              # p-variation: brute force, O(N^2) DP, pruned
│   └── losses.py            # differentiable MSE and weighted L^p losses
│
├── tests/                   # pytest. Run before trusting notebook output.
├── notebooks/               # THE EXPERIMENTS: maths, code, results, together
├── scripts/
│   ├── run_experiment.py    # config -> data -> model -> loss -> results
│   └── arc/                 # SLURM submission scripts
├── configs/                 # one YAML per experiment; never hardcode in scripts
├── data/{raw,synthetic}/    # gitignored. Regenerate, don't commit.
├── results/{runs,logs,figures}/
├── papers/                  # PDFs + references.bib
└── docs/
    ├── arc_guide.md         # Oxford ARC: accounts, SLURM, storage
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
| `scripts/run_experiment.py` training loop | done, needs torch installed |
| 1-NN path-distance classification benchmark | corrected BasicMotions configurations written; fresh raw and preprocessing runs remain |
| Signatures | intended main later direction: fixed features first, training loss second |
| Controlled missingness | evaluator implemented; one-seed pipeline check only |
| Real dataset | not obtained |
| ARC | 24-job core-study array prepared; submission remains |

Next: run seeds 1 and 2 of the core MSE against weighted-$J_2$ study through the
ARC array, run corrected BasicMotions controls, then introduce signatures in the
classification benchmark. Exponent-specific integral-norm studies remain
deferred.
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
