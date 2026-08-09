# Principled loss functions for continuous-time learning

Summer project, 6 weeks. Design and evaluate loss functions for **path-to-path**
learning, and compare them against pointwise MSE.

Supervision: Monday (informal) / Thursday 13:00 (formal).

This file is about **how to run things and where they live.** The mathematics
and the results live in the notebooks; what was decided and when lives in the
logbook.

---

## Setup — first time

```bash
cd "path/to/Pathwise Loss/pathwise-loss"

python3 -m venv .venv                # note: python3, not python — see below
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # installs pathloss + numpy/scipy/matplotlib
pip install -r requirements.txt      # jupyter, pytest, pyyaml, the rest

python -m ipykernel install --user --name pathwise-loss
```

`requirements.txt` is **core only**, does not need compiler, GPU, or
a git clone. The modelling stack (torch, neural CDEs, signatures) is in `requirements-ml.txt`. See [Troubleshooting](#troubleshooting) before installing that one.

Check it worked:

```bash
pytest -q                            # expect: 19 passed
```

## Running things

### Tests

```bash
pytest -q                  # all
pytest -q -k p_variation   # one group
pytest -q -v               # see the names — the names are documentation
```

Run these before trusting any notebook output. If the 19 pass, every formula the
notebooks rely on is doing what it claims.

### Notebooks

```bash
jupyter lab notebooks/01_integral_norms.ipynb
```

Then *Run All* — a few seconds, no GPU, nothing external. Notebooks are
committed **with outputs**, so they can be read without being run.

Headless re-run, for checking nothing broke after editing the library:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_integral_norms.ipynb
```

### Training experiments

Not yet possible — `scripts/run_experiment.py` is a stub. It parses the config,
writes provenance to `meta.json`, and prints what it *would* do:

```bash
python scripts/run_experiment.py --config configs/baseline_mse.yaml --out results/runs/test
```

The harness exists so that when the model arrives in week 2 there is somewhere
for it to go.

### On ARC

Nothing here needs ARC yet. When it does, see [`docs/arc_guide.md`](docs/arc_guide.md).
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
| `01_integral_norms.ipynb` | Quadrature rules and $L^p$ integral norms; verification against closed forms; convergence rates on smooth and rough paths; **MSE vs integral norm under irregular sampling**; sensitivity to $p$; $p$-variation, dyadic vs exact | complete |
| `02_losses_torch.ipynb` | differentiable losses, NumPy/torch agreement | not written |
| `03_signatures.ipynb` | signature features, signature kernel | not written |

Each notebook states its own mathematics, runs its own experiments, and reads
its own results. That is where to look for a derivation.

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
│   ├── paths.py             # synthetic generators, irregular sampling, missingness
│   ├── norms.py             # quadrature, L^p integral norms, p-variation (NumPy)
│   └── losses.py            # differentiable torch losses (week 3+, not written)
│
├── tests/                   # pytest. Run before trusting notebook output.
├── notebooks/               # THE EXPERIMENTS: maths, code, results, together
├── scripts/
│   ├── run_experiment.py    # config -> data -> model -> loss -> results (stub)
│   └── arc/                 # SLURM submission scripts
├── configs/                 # one YAML per experiment; never hardcode in scripts
├── data/{raw,synthetic}/    # gitignored. Regenerate, don't commit.
├── results/{runs,logs,figures}/
├── papers/                  # PDFs + references.bib + reading index
└── docs/
    ├── arc_guide.md         # Oxford ARC: accounts, SLURM, storage
    └── logbook/             # dated notes and findings. Append-only.
```

### Where things go

| | goes in |
|---|---|
| A derivation, an experiment, a plot, the reading of a result | the relevant **notebook** |
| A function used more than once | **`src/pathloss/`**, with a test |
| What was decided, results, interpretations | **`docs/logbook/`**, dated |
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
| Quadrature, $L^p$ norms, convergence studies | done — notebook 01 |
| $p$-variation, dyadic + exact DP | done — notebook 01 §6 |
| MSE vs integral norm under irregular sampling | done — notebook 01 §4 |
| `src/pathloss/losses.py` (torch, differentiable) | not written |
| Models (LSTM / Transformer / Linear NCDE) | not written |
| `scripts/run_experiment.py` training loop | stub |
| Signature features / signature kernel | not started |
| Real dataset | not obtained |
| Anything on ARC | not run — account not yet requested |

Next: torch versions of `integral_distance` and `pointwise_mse` in `losses.py`,
with a test that they match the NumPy versions to floating-point tolerance. That
is the bridge from "we can measure this" to "we can train against it".

---

## Conventions

A path is a pair `(t, x)` with `t` of shape `(T,)` and `x` of shape
`(..., T, d)` — time is the second-to-last axis, matching `torchcde` and
`signatory`'s `(batch, time, channel)`.

Synthetic data is always generated on a **fine grid** treated as ground truth,
then subsampled to produce what the model sees. Keeping those two objects
separate is what makes the robustness experiments well-defined.

---

## References

`papers/` holds the PDFs and `papers/references.bib` — one list, used for the
report.

**Citations go at the point of use**, in the notebook cell or logbook entry
where the paper actually changed a decision, not in a separate index. Notebook
01 ends with the works it cites and the section each one bears on; the reading
order for the weeks ahead is in `docs/logbook/2026-08-09.md`.

When adding a paper: drop the PDF in `papers/`, add the BibTeX entry with a
`note` giving the filename, and cite it where it mattered.
