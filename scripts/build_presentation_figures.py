#!/usr/bin/env python
"""Regenerate every figure used by the LaTeX deck, from stored runs.

Deck references `presentation/assets/` only, so this script is the single
producer of its figures. Each records its source run in
`presentation/assets/figure_manifest.json`, so a slide caption can be checked
against the run that produced it.

    python scripts/build_presentation_figures.py
    python scripts/build_presentation_figures.py --check   # provenance only

Every figure is drawn at a size and font matched to its slide, so the deck
scales images by `width=\\linewidth` alone and never shrinks text. Reads stored
`paths.npz`, `test_paths.npz` and per-condition `result.json`. No fitting here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / "runs"
ASSETS = REPO / "presentation" / "assets"

EVENT_INTERVAL = (0.15, 0.35)
AMPLITUDES = [0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]

# Beamer 16:9 at 9pt gives roughly 15.0cm usable width. Figures are drawn at
# their final printed size, so the deck never rescales text.
#
# Typeface matches the deck: beamer's default is Computer Modern Sans (CMSS)
# with Computer Modern maths, and matplotlib bundles both, so slide text and
# figure text are one family.
STYLE = {
    "font.family": "cmss10",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.unicode_minus": False,  # cmss10 has no U+2212
    "figure.dpi": 200,
    "savefig.dpi": 200,
}

FIXED_PATH_RUN = {
    "root": "neural_ode_fixed_path_closeout_10k",
    "capacity": "restricted",
    "seed": 1,
    "condition": "clustered",
    "updates": 10000,
}
OU_RUN = {
    "root": "neural_cde_brownian_ou",
    "seed": 1,
    "condition": "clustered",
    "loss": "j2",
    "test_path_index": 0,
}
REFINEMENT_CAPACITY = "restricted"
# Series worth naming on a slide. Others are drawn faint and unlabelled.
HIGHLIGHT = {
    "mse": ("tab:blue", "MSE"),
    "j2": ("tab:cyan", r"$J_2$"),
    "linf": ("tab:brown", "maximum error"),
    "area": ("tab:green", "direct area"),
    "global": ("tab:orange", "global depth 4"),
    "local_aligned": ("tab:purple", "local aligned"),
    "local_offset": ("tab:red", "local offset"),
}



def _log_ticks(axis) -> None:
    """Write log tick labels as mathtext powers.

    Matplotlib's default log formatter emits U+2212, which cmss10 lacks, so the
    exponent renders as a box. Writing the label directly keeps the minus inside
    mathtext, where Computer Modern supplies it.
    """
    import math  # noqa: PLC0415
    from matplotlib.ticker import FuncFormatter  # noqa: PLC0415

    def label(value, _position):
        if value <= 0:
            return ""
        exponent = int(round(math.log10(value)))
        return f"$10^{{{exponent}}}$"

    axis.yaxis.set_major_formatter(FuncFormatter(label))


def fixed_path_dir(loss: str) -> Path:
    run = FIXED_PATH_RUN
    return RUNS / run["root"] / run["capacity"] / f"seed{run['seed']}" / run["condition"] / loss


def ou_dir() -> Path:
    return RUNS / OU_RUN["root"] / f"seed{OU_RUN['seed']}" / OU_RUN["condition"] / OU_RUN["loss"]


def verify_fixed_path(loss: str) -> dict:
    """Confirm a run's stored configuration matches what captions will claim."""
    meta = json.loads((fixed_path_dir(loss) / "meta.json").read_text())
    config = meta["fit_config"]
    expected = {
        "seed": FIXED_PATH_RUN["seed"],
        "condition": FIXED_PATH_RUN["condition"],
        "loss": loss,
        "updates": FIXED_PATH_RUN["updates"],
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(
                f"{fixed_path_dir(loss)}: {key} is {config[key]!r}, caption claims {value!r}"
            )
    return {
        "run": str(fixed_path_dir(loss).relative_to(REPO)),
        "capacity": FIXED_PATH_RUN["capacity"],
        **expected,
        "initial_fingerprint": meta["initial_fingerprint"],
    }


def build_fixed_path_residuals() -> list[dict]:
    """Both residual panels in one wide figure, sharing a vertical scale."""
    provenance = [verify_fixed_path(loss) for loss in ("mse", "j2")]
    stored = {loss: np.load(fixed_path_dir(loss) / "paths.npz") for loss in ("mse", "j2")}
    residual = {
        loss: np.linalg.norm(a["prediction"] - a["target"], axis=-1) for loss, a in stored.items()
    }
    ceiling = max(r.max() for r in residual.values()) * 1.06

    figure, axes = plt.subplots(1, 2, figsize=(14.0, 4.3), sharey=True)
    for axis, loss, title in zip(axes, ("mse", "j2"), ("MSE-trained fit", "$J_2$-trained fit")):
        axis.plot(stored[loss]["time"], residual[loss], color="tab:red", linewidth=1.3)
        axis.axvspan(*EVENT_INTERVAL, color="tab:orange", alpha=0.20, label="event interval")
        axis.set_ylim(0.0, ceiling)
        axis.set_xlabel("time")
        axis.set_title(title)
    axes[0].set_ylabel("residual magnitude")
    axes[0].legend(loc="upper right", frameon=False)
    figure.tight_layout()
    figure.savefig(ASSETS / "fixed_clustered_residuals.png")
    plt.close(figure)

    for entry in provenance:
        entry["shared_y_limit"] = float(ceiling)
    return provenance


def build_ou_response() -> dict:
    directory = ou_dir()
    config = json.loads((directory / "meta.json").read_text())["fit_config"]
    for key in ("seed", "condition", "loss"):
        if config[key] != OU_RUN[key]:
            raise ValueError(
                f"{directory}: {key} is {config[key]!r}, caption claims {OU_RUN[key]!r}"
            )
    stored = np.load(directory / "test_paths.npz")
    index = OU_RUN["test_path_index"]
    time = stored["time"]
    target = stored["target"][index, :, 0]
    prediction = stored["prediction"][index, :, 0]

    figure, axes = plt.subplots(2, 1, figsize=(8.2, 5.6), sharex=True, height_ratios=[2, 1])
    axes[0].plot(time, target, color="black", linewidth=1.4, label="target")
    axes[0].plot(time, prediction, color="tab:blue", linewidth=1.2, linestyle="--", label="prediction")
    axes[0].set_ylabel("OU response")
    axes[0].legend(loc="best", frameon=False)
    axes[1].plot(time, np.abs(prediction - target), color="tab:red", linewidth=1.2)
    axes[1].set_yscale("log")
    _log_ticks(axes[1])
    axes[1].set_ylabel("absolute residual")
    axes[1].set_xlabel("time")
    figure.tight_layout()
    figure.savefig(ASSETS / "ou_clustered_j2_response_path0.png")
    plt.close(figure)

    return {
        "run": str(directory.relative_to(REPO)),
        "seed": config["seed"],
        "condition": config["condition"],
        "loss": config["loss"],
        "test_path_index": index,
        "max_absolute_residual": float(np.abs(prediction - target).max()),
    }


def _amplitude_directory(template: str, message: str, amplitude: float) -> Path:
    text = ("%g" % amplitude).replace(".", "p")
    return RUNS / "brownian_message" / "main" / "conditions" / template / message / f"rho-{text}"


def _series(template: str, message: str, section: str, field: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for amplitude in AMPLITUDES:
        path = _amplitude_directory(template, message, amplitude) / "result.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} absent; rerun scripts/run_brownian_message_study.py")
        block = json.loads(path.read_text())[section]
        for discrepancy, entry in block.items():
            values.setdefault(discrepancy, []).append(float(entry[field]))
    return values


def _message_panels(section: str, field: str, ylabel: str, chance, filename: str) -> dict:
    template = "balanced"
    messages = ("area", "combined", "increment")
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)
    for axis, message in zip(axes, messages):
        series = _series(template, message, section, field)
        for discrepancy, curve in sorted(series.items()):
            if discrepancy in HIGHLIGHT:
                colour, label = HIGHLIGHT[discrepancy]
                axis.plot(AMPLITUDES, curve, marker="o", markersize=4, linewidth=1.6,
                          color=colour, label=label)
            else:
                axis.plot(AMPLITUDES, curve, linewidth=0.9, color="0.75", zorder=1)
        if chance is not None:
            axis.axhline(chance, color="0.35", linestyle=":", linewidth=1.2)
        axis.set_title(f"{message} message")
        axis.set_xlabel(r"normalised amplitude $\rho$")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                  bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(ASSETS / filename)
    plt.close(figure)
    return {"asset": filename, "template": template, "messages": list(messages),
            "source": "results/runs/brownian_message/main/conditions/*/result.json"}


def build_refinement_paths() -> dict:
    """Ten-block against hundred-block local-signature fits, restricted capacity."""
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 4.9))
    for column, seed in enumerate(range(3)):
        coarse_dir = None
        for root in ("neural_ode_fixed_path_signature_10k", "neural_ode_fixed_path_closeout_10k"):
            candidate = RUNS / root / REFINEMENT_CAPACITY / f"seed{seed}" / "uniform" / "sig_local"
            if (candidate / "paths.npz").exists():
                coarse_dir = candidate
                break
        fine_dir = (RUNS / "neural_ode_fixed_path_local_fine_10k" / REFINEMENT_CAPACITY
                    / f"seed{seed}" / "uniform" / "sig_local_fine")
        if coarse_dir is None or not (fine_dir / "paths.npz").exists():
            raise FileNotFoundError(f"refinement paths absent for seed {seed}")
        coarse, fine = np.load(coarse_dir / "paths.npz"), np.load(fine_dir / "paths.npz")

        top = axes[0, column]
        top.plot(coarse["target"][:, 0], coarse["target"][:, 1], color="black", linewidth=1.4,
                 label="target")
        top.plot(coarse["prediction"][:, 0], coarse["prediction"][:, 1], color="tab:orange",
                 linewidth=1.2, label="10 blocks")
        top.plot(fine["prediction"][:, 0], fine["prediction"][:, 1], color="tab:blue",
                 linewidth=1.2, linestyle="--", label="100 blocks")
        top.set_aspect("equal")
        top.set_title(f"seed {seed}")

        bottom = axes[1, column]
        for arrays, colour, style in ((coarse, "tab:orange", "-"), (fine, "tab:blue", "--")):
            residual = np.linalg.norm(arrays["prediction"] - arrays["target"], axis=-1)
            bottom.plot(arrays["time"], residual, color=colour, linestyle=style, linewidth=1.2)
        bottom.axvspan(*EVENT_INTERVAL, color="tab:orange", alpha=0.18)
        bottom.set_yscale("log")
        _log_ticks(bottom)
        bottom.set_xlabel("time")
    axes[0, 0].set_ylabel("$Y_2$")
    axes[0, 0].legend(loc="lower right", frameon=False)
    axes[1, 0].set_ylabel("residual magnitude")
    figure.tight_layout()
    figure.savefig(ASSETS / "path_comparison_restricted.png")
    plt.close(figure)
    return {"asset": "path_comparison_restricted.png", "capacity": REFINEMENT_CAPACITY,
            "seeds": [0, 1, 2], "condition": "uniform", "updates": 10000}



def build_target_design() -> dict:
    """Fixed target and the two observation conditions used in experiment A."""
    import sys
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    try:
        from pathloss.fixed_path import fixed_target, observation_times  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as error:  # torch lives in requirements-ml.txt
        raise ModuleNotFoundError(
            "target_design needs pathloss.fixed_path, which imports torch. "
            "Run from the project environment with requirements-ml.txt installed."
        ) from error

    grid = torch.linspace(0.0, 1.0, 513)
    target = fixed_target(grid).numpy()
    uniform = observation_times(64, "uniform").numpy()
    clustered = observation_times(64, "clustered").numpy()

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), width_ratios=[1, 1.5])
    axes[0].plot(target[:, 0], target[:, 1], color="black", linewidth=1.4)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("$Y_1^\\star$")
    axes[0].set_ylabel("$Y_2^\\star$")
    axes[0].set_title("fixed target path")
    axes[1].scatter(uniform, np.zeros_like(uniform), s=16, color="tab:blue", label="uniform")
    axes[1].scatter(clustered, np.ones_like(clustered), s=16, color="tab:red", label="clustered")
    axes[1].axvspan(*EVENT_INTERVAL, color="tab:orange", alpha=0.20, label="oscillatory burst")
    axes[1].set_yticks([0, 1], ["uniform", "clustered"])
    axes[1].set_ylim(-0.6, 1.6)
    axes[1].set_xlabel("time")
    axes[1].set_title("64 training observation times")
    axes[1].legend(loc="center right", frameon=False)
    figure.tight_layout()
    figure.savefig(ASSETS / "target_design.png")
    plt.close(figure)
    return {"asset": "target_design.png", "source": "src/pathloss/fixed_path.py"}


def build_learning_dynamics() -> dict:
    """Common metrics against update count, restricted capacity, uniform, seed 0."""
    losses = ("mse", "j2", "h1", "sig_global", "sig_local")
    colours = {"mse": "tab:blue", "j2": "tab:cyan", "h1": "tab:green",
               "sig_global": "tab:orange", "sig_local": "tab:red"}
    labels = {"mse": "MSE", "j2": "$J_2$", "h1": "$H^1$",
              "sig_global": "global signature", "sig_local": "local signature"}
    panels = [("mse", "dense MSE"), ("local_j2", "event-region $J_2$"),
              ("sig_local", "local-signature discrepancy")]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    used = {}
    for loss in losses:
        directory = None
        for root in ("neural_ode_fixed_path_signature_10k", "neural_ode_fixed_path_closeout_10k"):
            candidate = RUNS / root / "restricted" / "seed0" / "uniform" / loss
            if (candidate / "checkpoints.json").exists():
                directory = candidate
                break
        if directory is None:
            raise FileNotFoundError(f"checkpoints absent for {loss}")
        used[loss] = str(directory.relative_to(REPO))
        rows = json.loads((directory / "checkpoints.json").read_text())
        updates = [r["updates_completed"] for r in rows]
        for axis, (metric, _) in zip(axes, panels):
            axis.plot(updates, [r["metrics"][metric] for r in rows], marker="o", markersize=3,
                      linewidth=1.5, color=colours[loss], label=labels[loss])
    for axis, (_, title) in zip(axes, panels):
        axis.set_xscale("symlog", linthresh=100)
        axis.set_yscale("log")
        _log_ticks(axis)
        axis.set_xlabel("Adam updates")
        axis.set_title(title)
    axes[0].legend(loc="best", frameon=False)
    figure.tight_layout()
    figure.savefig(ASSETS / "learning_dynamics.png")
    plt.close(figure)
    return {"asset": "learning_dynamics.png", "capacity": "restricted", "seed": 0,
            "condition": "uniform", "runs": used}


def build_ou_signature_histories() -> dict:
    """Training-loss histories showing signature optimizer instability."""
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 3.7), sharex=True)
    for axis, loss, title in zip(axes, ("sig_global", "sig_local"),
                                 ("global depth-4 signature", "local depth-2 signature")):
        for seed in range(3):
            history = json.loads(
                (RUNS / "neural_cde_brownian_ou" / f"seed{seed}" / "uniform" / loss
                 / "history.json").read_text())
            axis.plot([r["epoch"] for r in history], [r["train_loss"] for r in history],
                      linewidth=1.3, label=f"seed {seed}")
        axis.set_yscale("log")
        _log_ticks(axis)
        axis.set_xlabel("epoch")
        axis.set_title(title)
    axes[0].set_ylabel("epoch-average training loss")
    axes[0].legend(loc="best", frameon=False)
    figure.tight_layout()
    figure.savefig(ASSETS / "ou_signature_histories.png")
    plt.close(figure)
    return {"asset": "ou_signature_histories.png", "condition": "uniform", "seeds": [0, 1, 2]}


def build_template_comparison() -> dict:
    """Net against balanced template, combined message, paired sensitivity."""
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 3.9), sharey=True)
    for axis, template in zip(axes, ("net", "balanced")):
        series = _series(template, "combined", "paired", "normalized_median")
        for discrepancy, curve in sorted(series.items()):
            if discrepancy in HIGHLIGHT:
                colour, label = HIGHLIGHT[discrepancy]
                axis.plot(AMPLITUDES, curve, marker="o", markersize=4, linewidth=1.6,
                          color=colour, label=label)
            else:
                axis.plot(AMPLITUDES, curve, linewidth=0.9, color="0.75", zorder=1)
        axis.set_xlabel(r"normalised amplitude $\rho$")
        axis.set_title(f"{template} template, combined message")
    axes[0].set_ylabel("normalised paired sensitivity")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                  bbox_to_anchor=(0.5, -0.03))
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    figure.savefig(ASSETS / "template_comparison.png")
    plt.close(figure)
    return {"asset": "template_comparison.png", "templates": ["net", "balanced"],
            "message": "combined"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify run provenance without writing figures")
    arguments = parser.parse_args()
    if arguments.check:
        for loss in ("mse", "j2"):
            print(verify_fixed_path(loss))
        print(json.loads((ou_dir() / "meta.json").read_text())["fit_config"])
        return 0

    ASSETS.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        manifest = {
            "fixed_path_residuals": build_fixed_path_residuals(),
            "ou_response": build_ou_response(),
            "refinement_paths": build_refinement_paths(),
            "paired_sensitivity": _message_panels(
                "paired", "normalized_median", "normalised paired sensitivity",
                None, "paired_balanced.png"),
            "detection": _message_panels(
                "detection", "balanced_accuracy", "balanced accuracy",
                0.5, "detection_balanced.png"),
            "learning_dynamics": build_learning_dynamics(),
            "ou_signature_histories": build_ou_signature_histories(),
            "template_comparison": build_template_comparison(),
        }
        try:
            manifest["target_design"] = build_target_design()
        except ModuleNotFoundError as error:
            manifest["target_design"] = {"pending": str(error)}
            print("skipped target_design:", error)
    (ASSETS / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    written = {
        "fixed_clustered_residuals.png",
        "ou_clustered_j2_response_path0.png",
        "path_comparison_restricted.png",
        "paired_balanced.png",
        "detection_balanced.png",
        "target_design.png",
        "learning_dynamics.png",
        "ou_signature_histories.png",
        "template_comparison.png",
    }
    stale = sorted(p.name for p in ASSETS.glob("*.png") if p.name not in written)
    print(f"wrote {len(written)} figures to {ASSETS}")
    if stale:
        print("stale files no longer used by the deck, safe to delete:", *stale, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
