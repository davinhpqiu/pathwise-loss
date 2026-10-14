# Principled loss functions for continuous-time learning

Summer project, Mathematical Institute, University of Oxford, and DataSig,
3 August to 10 September 2026.

Haoping Qiu. Supervisors: Dr Sam Morley, Dr Ben Walker, Prof. Terry Lyons.

Models mapping one time series to another are usually trained with pointwise
mean squared error. This project compares pointwise, elapsed-time, Sobolev and
path-signature losses for *path-to-path* learning, where inputs and outputs are
whole continuous-time trajectories. Main finding: two paths count as close only
relative to structure relevant to the task. Observation density, regularity,
signature truncation and localisation each change which loss works.

## Contents

```
pathwise_losses/
├── pathwise_losses.ipynb   summary notebook, stored with outputs
├── figures.py              one plotting function per figure
├── pathwise.mplstyle       DataSig colours, presentation sizes
├── data/                   frozen extract of the original runs, 0.4 MB
│   └── PROVENANCE.json     source commit and SHA-256 of every file read
└── requirements.txt
```

Notebook sections:

1. **Motivation.** Why pointwise agreement does not determine path error;
   losses compared; path signatures.
2. **Experiment A.** A Neural ODE fitted to one fixed path under each loss:
   sampling density, derivative supervision, learning dynamics and local
   signature partitions.
3. **Experiment B.** A Neural CDE learning the Brownian-to-Ornstein-Uhlenbeck
   operator, and the stability of signature training.
4. **Experiment C.** Messages planted in Brownian rough paths, and which
   distances detect them.
5. **Conclusions**, scope and references.

## Viewing and running

GitHub renders the notebook with all figures and tables. To run it:

```bash
pip install -r requirements.txt
jupyter notebook pathwise_losses.ipynb
```

It reads only `data/`, needs no GPU or deep-learning library, and runs in
seconds. Each number in the text is recomputed from the data, and assertion
cells fail if the data and the stated findings disagree.

## Data

`data/` holds the arrays and tables behind each figure: fitted paths at
selected checkpoints, final and checkpoint metrics for every training run in
Experiment A, held-out metrics and training histories for Experiment B, and
sensitivity, detection and threshold results for Experiment C. It is written by
`scripts/export_public_bundle.py` in the project repository, which reads the
full run outputs and records their hashes in `PROVENANCE.json`. Training code,
configurations and complete analysis notebooks are in that repository.

## Citation

> H. Qiu. *Principled loss functions for continuous-time learning.* Summer
> project, Mathematical Institute, University of Oxford, 2026.
