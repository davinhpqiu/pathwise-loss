DECK := pathwise_loss_presentation
FIGURES := $(wildcard assets/*.png)
THEME_ASSETS := $(wildcard theme-assets/*.png)

.PHONY: all figures clean

all: $(DECK).pdf

$(DECK).pdf: $(DECK).tex beamerthemeoxfordmaths.sty $(FIGURES) $(THEME_ASSETS)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(DECK).tex

# Figure regeneration is separate from the normal deck build because the
# underlying experiment outputs are intentionally not stored in Git.
figures:
	cd .. && .venv/bin/python scripts/build_presentation_figures.py

clean:
	latexmk -c $(DECK).tex
