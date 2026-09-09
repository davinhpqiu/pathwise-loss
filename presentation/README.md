# Pathwise-loss presentation

The slide deck is self-contained: all figures required by the LaTeX source are
stored in `assets/`, so a clean clone can build the PDF without downloading the
experiment outputs.

## Build

From this directory, run:

```sh
make
```

Equivalently:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error pathwise_loss_presentation.tex
```

The resulting file is `pathwise_loss_presentation.pdf`.

In VS Code, open `pathwise_loss_presentation.tex`, select the LaTeX Workshop
extension's build command, and use `pathwise_loss_presentation.tex` as the root
document.

## Figure provenance

The committed PNG files are the build inputs for the deck. Their source runs
and settings are recorded in `assets/figure_manifest.json`.

To regenerate them after restoring the corresponding ignored experiment
outputs under `results/runs/`, run from the repository root:

```sh
source .venv/bin/activate
python scripts/build_presentation_figures.py
```

The script checks the fixed-path and OU captions against the stored run
metadata before writing the figures. Regenerating figures is not required for
ordinary editing or compilation of the slides.
