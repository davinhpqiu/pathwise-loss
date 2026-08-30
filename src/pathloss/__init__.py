"""pathloss : loss functions for path-to-path learning.

Anything that must be correct lives here and has a test in `tests/`. Notebooks
and `scripts/` import from here; a notebook cell is never the sole copy of a
function.

Measurement, NumPy, no training dependency
------------------------------------------
norms.py           quadrature weights, L^p integral norms and distances
pvar.py            p-variation: brute force, O(N^2) DP, pruned search
classification.py  fixed 1-nearest-neighbour evaluation on labelled archives

Data
----
paths.py           path generators (smooth, Brownian, OU), irregular
                   subsampling, missingness
datasets.py        assembles generated paths into context/target/fine-grid
                   training examples

Training, requires torch
------------------------
losses.py          differentiable MSE, weighted L^p and Sobolev H^1
models.py          GRU query and Linear Neural CDE baselines
train.py           training loop and evaluation
fixed_path.py      fixed-target Neural ODE: target, model and fitting
operator.py        path-output Neural CDE and Brownian-to-OU operator fitting
signatures.py      differentiable piecewise-linear signatures and losses

Bookkeeping
-----------
provenance.py      git state, config loading, run metadata
fixed_path_study.py exact fixed-path run identities and completion
"""

__version__ = "0.1.0"
