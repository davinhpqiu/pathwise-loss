"""Figures for the public summary notebook.

Each function reads the frozen bundle in ``data/`` and draws one figure from the
presentation. Plot choices follow ``scripts/build_presentation_figures.py`` in
the main repository, so figures here match the slides.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
STYLE = HERE / "pathwise.mplstyle"

EVENT_INTERVAL = (0.15, 0.35)
CHECKPOINT_UPDATES = (100, 500, 1000, 2500, 10000)
HIGHLIGHT = {
    "mse": ("tab:blue", "MSE"),
    "j2": ("tab:cyan", r"$J_2$"),
    "linf": ("tab:brown", "maximum error"),
    "area": ("tab:green", "direct area distance"),
    "global": ("tab:orange", "global signature"),
    "local_aligned": ("tab:purple", "local signature, aligned"),
    "local_offset": ("tab:red", "local signature, offset"),
}


def use_style() -> None:
    plt.style.use(STYLE)


def load():
    """All bundle files, keyed by short name."""
    return {
        "fixed": np.load(DATA / "fixed_path.npz"),
        "fixed_final": pd.read_csv(DATA / "fixed_path_final_metrics.csv"),
        "fixed_checkpoints": pd.read_csv(DATA / "fixed_path_checkpoint_metrics.csv"),
        "refinement": pd.read_csv(DATA / "refinement_metrics.csv"),
        "ou": np.load(DATA / "ou_operator.npz"),
        "ou_test": pd.read_csv(DATA / "ou_test_metrics.csv"),
        "messages": pd.read_csv(DATA / "brownian_messages.csv"),
        "thresholds": pd.read_csv(DATA / "brownian_thresholds.csv"),
    }


def _log_ticks(axis) -> None:
    """Powers of ten in mathtext; cmss10 lacks the Unicode minus."""
    def label(value, _position):
        return "" if value <= 0 else f"$10^{{{int(round(math.log10(value)))}}}$"
    axis.yaxis.set_major_formatter(FuncFormatter(label))


def partition_schematic():
    """Aligned and offset block partitions around one balanced message."""
    figure, axis = plt.subplots(figsize=(10.0, 2.2))
    message = (40, 56)
    rows = (("aligned: message stays inside one block", 1.0, range(0, 129, 32)),
            ("offset: a boundary divides the message", 0.0, range(16, 113, 32)))
    for label, height, boundaries in rows:
        start, stop = min(boundaries), max(boundaries)
        axis.plot([start, stop], [height, height], color="#0A1446", linewidth=2.2)
        for x in boundaries:
            axis.plot([x, x], [height - 0.12, height + 0.12], color="#0A1446", linewidth=1.4)
        axis.fill_between(message, height - 0.06, height + 0.06, color="tab:blue", alpha=0.55)
        axis.text(stop + 4, height, label, va="center", fontsize=10)
    axis.plot([48, 48], [-0.16, 0.16], color="tab:red", linewidth=2.0)
    axis.text(48, 1.22, "balanced message", ha="center", fontsize=10)
    axis.set_xlim(-2, 210)
    axis.set_ylim(-0.4, 1.45)
    axis.axis("off")
    figure.tight_layout()
    return figure


def target_design(data):
    fixed = data["fixed"]
    has_times = "observation_times_uniform" in fixed.files
    widths = [1, 1.5] if has_times else [1]
    figure, axes = plt.subplots(1, len(widths), figsize=(13.5 if has_times else 5.5, 4.4),
                                width_ratios=widths, squeeze=False)
    axis = axes[0, 0]
    axis.plot(fixed["target"][:, 0], fixed["target"][:, 1], color="black", linewidth=1.4)
    axis.set_aspect("equal")
    axis.set_xlabel(r"$Y_1^\star$")
    axis.set_ylabel(r"$Y_2^\star$")
    axis.set_title("fixed target path")
    if has_times:
        uniform = fixed["observation_times_uniform"]
        clustered = fixed["observation_times_clustered"]
        axis = axes[0, 1]
        axis.scatter(uniform, np.zeros_like(uniform), s=16, color="tab:blue", label="uniform")
        axis.scatter(clustered, np.ones_like(clustered), s=16, color="tab:red", label="clustered")
        axis.axvspan(*EVENT_INTERVAL, color="tab:orange", alpha=0.20, label="oscillatory burst")
        axis.set_yticks([0, 1], ["uniform", "clustered"])
        axis.set_ylim(-0.6, 1.6)
        axis.set_xlabel("time")
        axis.set_title("64 training observation times")
        axis.legend(loc="center right", frameon=False)
    figure.tight_layout()
    return figure


def learning_paths(data, capacity: str):
    fixed = data["fixed"]
    losses = ("mse", "j2", "sig_global", "sig_local")
    labels = {"mse": "MSE", "j2": "$J_2$", "sig_global": "global signature",
              "sig_local": "local signature"}
    colours = {"mse": "tab:blue", "j2": "tab:orange", "sig_global": "tab:green",
               "sig_local": "tab:red"}
    figure, axes = plt.subplots(len(losses), len(CHECKPOINT_UPDATES), figsize=(15.0, 10.7),
                                sharex=True, sharey=True)
    for row, loss in enumerate(losses):
        for column, update in enumerate(CHECKPOINT_UPDATES):
            axis = axes[row, column]
            prediction = fixed[f"checkpoint_{capacity}_{loss}_{update:05d}"]
            axis.plot(fixed["target"][:, 0], fixed["target"][:, 1], color="black", linewidth=1.15)
            axis.plot(prediction[:, 0], prediction[:, 1], color=colours[loss], linewidth=1.15)
            axis.set_aspect("equal")
            if row == 0:
                axis.set_title(f"{update:,} updates")
        axes[row, 0].set_ylabel(labels[loss])
    figure.suptitle(f"{capacity} model: target (black) and fitted path", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    return figure


def clustered_residuals(data):
    fixed = data["fixed"]
    residual = {loss: np.linalg.norm(fixed[f"clustered_prediction_{loss}"] - fixed["target"], axis=-1)
                for loss in ("mse", "j2")}
    positive = np.concatenate([r[r > 0] for r in residual.values()])
    limits = (10 ** np.floor(np.log10(positive.min())), 10 ** np.ceil(np.log10(positive.max())))
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 4.3), sharey=True)
    for axis, loss, title in zip(axes, ("mse", "j2"), ("MSE-trained fit", "$J_2$-trained fit")):
        axis.plot(fixed["time"], residual[loss], color="tab:red", linewidth=1.3)
        axis.axvspan(*EVENT_INTERVAL, color="tab:orange", alpha=0.20, label="event interval")
        axis.set_yscale("log")
        axis.set_ylim(*limits)
        _log_ticks(axis)
        axis.set_xlabel("time")
        axis.set_title(title)
    axes[0].set_ylabel("residual magnitude")
    axes[0].legend(loc="upper right", frameon=False)
    figure.tight_layout()
    return figure


def learning_dynamics(data):
    table = data["fixed_checkpoints"]
    subset = table[(table["capacity"] == "restricted") & (table["seed"] == 0)]
    colours = {"mse": "tab:blue", "j2": "tab:cyan", "h1": "tab:green",
               "sig_global": "tab:orange", "sig_local": "tab:red"}
    labels = {"mse": "MSE", "j2": "$J_2$", "h1": "$H^1$",
              "sig_global": "global signature", "sig_local": "local signature"}
    panels = [("mse", "dense MSE"), ("local_j2", "event-region $J_2$"),
              ("sig_local", "local-signature discrepancy")]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for loss in colours:
        rows = subset[subset["loss"] == loss].sort_values("update")
        for axis, (metric, _) in zip(axes, panels):
            axis.plot(rows["update"], rows[metric], marker="o", markersize=3, linewidth=1.5,
                      color=colours[loss], label=labels[loss])
    for axis, (_, title) in zip(axes, panels):
        axis.set_xscale("symlog", linthresh=100)
        axis.set_yscale("log")
        _log_ticks(axis)
        axis.set_xlabel("Adam updates")
        axis.set_title(title)
    axes[0].legend(loc="best", frameon=False)
    figure.tight_layout()
    return figure


def refinement_paths(data):
    fixed = data["fixed"]
    figure, axes = plt.subplots(2, 3, figsize=(15.0, 4.9), sharey="row")
    target = fixed["target"]
    for column, seed in enumerate(range(3)):
        coarse = fixed[f"refinement_seed{seed}_blocks10"]
        fine = fixed[f"refinement_seed{seed}_blocks100"]
        top = axes[0, column]
        top.plot(target[:, 0], target[:, 1], color="black", linewidth=1.4, label="target")
        top.plot(coarse[:, 0], coarse[:, 1], color="tab:orange", linewidth=1.2, label="10 blocks")
        top.plot(fine[:, 0], fine[:, 1], color="tab:blue", linewidth=1.2, linestyle="--",
                 label="100 blocks")
        top.set_aspect("equal")
        top.set_title(f"seed {seed}")
        bottom = axes[1, column]
        for prediction, colour, style in ((coarse, "tab:orange", "-"), (fine, "tab:blue", "--")):
            residual = np.linalg.norm(prediction - target, axis=-1)
            bottom.plot(fixed["time"], residual, color=colour, linestyle=style, linewidth=1.2)
        bottom.axvspan(*EVENT_INTERVAL, color="tab:orange", alpha=0.18)
        bottom.set_yscale("log")
        _log_ticks(bottom)
        bottom.set_xlabel("time")
    axes[0, 0].set_ylabel("$Y_2$")
    axes[0, 0].legend(loc="lower right", frameon=False)
    axes[1, 0].set_ylabel("residual magnitude")
    figure.tight_layout()
    return figure


def ou_response(data):
    ou = data["ou"]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.3), width_ratios=[1.7, 1])
    axes[0].plot(ou["time"], ou["target"], color="black", linewidth=1.4, label="target")
    axes[0].plot(ou["time"], ou["prediction"], color="tab:blue", linewidth=1.2, linestyle="--",
                 label="prediction")
    axes[0].set_ylabel("OU response")
    axes[0].set_xlabel("time")
    axes[0].legend(loc="best", frameon=False)
    axes[1].plot(ou["time"], np.abs(ou["prediction"] - ou["target"]), color="tab:red", linewidth=1.2)
    axes[1].set_yscale("log")
    _log_ticks(axes[1])
    axes[1].set_ylabel("absolute residual")
    axes[1].set_xlabel("time")
    figure.tight_layout()
    return figure


def ou_signature_histories(data):
    ou = data["ou"]
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 3.7), sharex=True)
    for axis, loss, title in zip(axes, ("sig_global", "sig_local"),
                                 ("global depth-4 signature", "local depth-2 signature")):
        for seed in range(3):
            axis.plot(ou[f"history_{loss}_seed{seed}_epoch"],
                      ou[f"history_{loss}_seed{seed}_train_loss"],
                      linewidth=1.3, label=f"seed {seed}")
        axis.set_yscale("log")
        _log_ticks(axis)
        axis.set_xlabel("epoch")
        axis.set_title(title)
    axes[0].set_ylabel("epoch-average training loss")
    axes[0].legend(loc="best", frameon=False)
    figure.tight_layout()
    return figure


def _curves(messages, template, message, section):
    rows = messages[(messages["template"] == template) & (messages["message"] == message)
                    & (messages["section"] == section)]
    return {name: group.sort_values("amplitude") for name, group in rows.groupby("discrepancy")}


def _draw_curves(axis, curves):
    for name, group in sorted(curves.items()):
        if name in HIGHLIGHT:
            colour, label = HIGHLIGHT[name]
            axis.plot(group["amplitude"], group["value"], marker="o", markersize=4,
                      linewidth=1.6, color=colour, label=label)
        else:
            axis.plot(group["amplitude"], group["value"], linewidth=0.9, color="0.75", zorder=1)


def template_comparison(data):
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 3.9), sharey=True)
    for axis, template in zip(axes, ("net", "balanced")):
        _draw_curves(axis, _curves(data["messages"], template, "combined", "paired"))
        axis.set_xlabel(r"normalised amplitude $\rho$")
        axis.set_title(f"{template} template, combined message")
    axes[0].set_ylabel("normalised paired sensitivity")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                  bbox_to_anchor=(0.5, -0.03))
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    return figure


def message_panels(data, section: str):
    ylabel = "normalised paired sensitivity" if section == "paired" else "balanced accuracy"
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)
    for axis, message in zip(axes, ("area", "combined", "increment")):
        _draw_curves(axis, _curves(data["messages"], "balanced", message, section))
        if section == "detection":
            axis.axhline(0.5, color="0.35", linestyle=":", linewidth=1.2)
        axis.set_title(f"{message} message")
        axis.set_xlabel(r"normalised amplitude $\rho$")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
                  bbox_to_anchor=(0.5, -0.02))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure
