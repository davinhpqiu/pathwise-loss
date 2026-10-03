"""Exact BV Euler minimax and certified pure-Engel storage comparison.

Design and proof scope are recorded before execution in
docs/response_distance/storage.md. All searches use exact Witt dimensions;
pure-path response envelopes are bounds, not computed response suprema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from pathloss.response_storage import (
    engel_scales,
    euler_error_feasible,
    log_fpl_euler_bound,
    log_scalar_euler_error,
    minimum_intervals,
    optimize_euler_storage,
)


def _source_provenance(root: Path) -> dict:
    sources = [
        "scripts/study_response_storage.py",
        "src/pathloss/response_storage.py",
        "tests/test_response_storage.py",
    ]
    optional = "docs/response_distance/storage.md"
    present = (root / optional).is_file()
    if present:
        sources.append(optional)
    return {
        "source_sha256": {
            source: hashlib.sha256((root / source).read_bytes()).hexdigest()
            for source in sources
        },
        "unavailable_optional_sources": [] if present else [optional],
    }


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _search(channel, scale, epsilon, criterion, degree, context, profiles):
    best, rows = optimize_euler_storage(
        channel, scale, epsilon, criterion=criterion, degree=degree
    )
    error = log_scalar_euler_error if criterion == "exact" else log_fpl_euler_bound
    for row in rows:
        previous = (
            error(row.order, row.intervals - 1, scale) if row.intervals > 1 else None
        )
        if not euler_error_feasible(
            row.order, row.intervals, scale, epsilon, criterion=criterion
        ) or (
            previous is not None
            and euler_error_feasible(
                row.order, row.intervals - 1, scale, epsilon, criterion=criterion
            )
        ):
            raise AssertionError("interval search failed minimality check")
        profiles.append(
            {**context, **asdict(row), "log_error_at_m_minus_one": previous}
        )
    ties = [(row.depth, row.intervals) for row in rows if row.storage == best.storage]
    return best, ties


def _bv(profiles):
    rows = []
    for channel, scale, epsilon in product(
        (2, 3, 5),
        (0.1, 0.5, 1, 2, 5, 10, 20, 50, 100),
        (1e-1, 1e-2, 1e-4, 1e-8, 1e-12),
    ):
        context = {
            "family": "bv",
            "channel": channel,
            "scale": scale,
            "epsilon": epsilon,
            "degree": 1,
        }
        exact, exact_ties = _search(
            channel,
            scale,
            epsilon,
            "exact",
            1,
            {**context, "criterion": "exact"},
            profiles,
        )
        bound, bound_ties = _search(
            channel,
            scale,
            epsilon,
            "fpl",
            1,
            {**context, "criterion": "fpl"},
            profiles,
        )
        if exact.storage > bound.storage:
            raise AssertionError("exact constraint lost a certified feasible point")
        rows.append(
            {
                **context,
                "exact_depth": exact.depth,
                "exact_intervals": exact.intervals,
                "exact_storage": exact.storage,
                "fpl_depth": bound.depth,
                "fpl_intervals": bound.intervals,
                "fpl_storage": bound.storage,
                "exact_to_fpl_storage": exact.storage / bound.storage,
                "exact_ties": json.dumps(exact_ties),
                "fpl_ties": json.dumps(bound_ties),
                "same_optimum_set": set(exact_ties) == set(bound_ties),
                "canonical_depth_changed": exact.depth != bound.depth,
            }
        )
    return rows


def _engel(profiles):
    rows = []
    for degree, hs_budget, epsilon in product(
        (2, 3, 4, 6, 8),
        (0.25, 1, 4, 10, 25),
        (1e-3, 1e-6, 1e-12),
    ):
        hs, gauge, witness = engel_scales(degree)
        amplitude = hs_budget / hs
        if epsilon >= math.expm1(amplitude * witness):
            raise AssertionError("depths below the pure degree may be feasible")
        context = {
            "family": "engel",
            "channel": 2,
            "scale": hs_budget,
            "epsilon": epsilon,
            "degree": degree,
        }
        choices = {}
        for label, effective in (
            ("hs", hs_budget),
            ("upper", amplitude * gauge),
            ("lower", amplitude * witness),
        ):
            choices[label], _ = _search(
                2,
                effective,
                epsilon,
                "exact",
                degree,
                {**context, "criterion": label},
                profiles,
            )
        hs_choice, upper, lower = (choices[key] for key in ("hs", "upper", "lower"))
        if not lower.storage <= upper.storage <= hs_choice.storage:
            raise AssertionError("structured storage sandwich failed")
        row = {
            **context,
            "amplitude": amplitude,
            "hs_size": hs,
            "response_gauge": gauge,
            "witness_eigenvalue": witness,
        }
        for label, choice in choices.items():
            row.update(
                {f"{label}_{key}": value for key, value in asdict(choice).items()}
            )
        row.update(
            {
                "upper_to_hs_storage": upper.storage / hs_choice.storage,
                "lower_to_upper_storage": lower.storage / upper.storage,
                "certified_depth_changed": upper.depth != hs_choice.depth,
            }
        )
        rows.append(row)
    return rows


def _asymptotic(profiles):
    rows = []
    hs, gauge, _ = engel_scales(2)
    for amplitude in (1, 4, 16, 64, 256, 1024):
        costs = {}
        for label, scale in (("hs", amplitude * hs), ("upper", amplitude * gauge)):
            best, _ = _search(
                2,
                scale,
                1e-6,
                "exact",
                2,
                {
                    "family": "engel_amplitude",
                    "channel": 2,
                    "scale": scale,
                    "epsilon": 1e-6,
                    "degree": 2,
                    "criterion": label,
                },
                profiles,
            )
            costs[label] = best.storage
        rows.append(
            {
                "amplitude": amplitude,
                "hs_storage": costs["hs"],
                "upper_storage": costs["upper"],
                "log_cost_gap_over_root_amplitude": (
                    math.log(costs["hs"]) - math.log(costs["upper"])
                )
                / math.sqrt(amplitude),
                "limiting_certified_gap": 2
                * math.sqrt(2 * math.log(2))
                * (math.sqrt(hs) - math.sqrt(gauge)),
            }
        )
    return rows


def _figures(out, bv, engel, profiles, asymptotic):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    # Plot a shared depth grid, including depths beyond the search stop rule.
    chosen = [
        {
            "criterion": criterion,
            "depth": depth,
            "intervals": minimum_intervals(depth, 1, 0.1, criterion=criterion),
        }
        for criterion in ("fpl", "exact")
        for depth in range(1, 7)
    ]
    for criterion, label in (("fpl", "FPL bound"), ("exact", "Exact minimax")):
        subset = [row for row in chosen if row["criterion"] == criterion]
        axes[0].plot(
            [row["depth"] for row in subset],
            [row["intervals"] for row in subset],
            "o-",
            label=label,
        )
    axes[0].set(
        xlabel="Truncation depth N",
        ylabel="Least feasible intervals m",
        yscale="log",
        title="BV feasible boundary: d=2, r=1, ε=0.1",
    )
    axes[0].legend()
    for epsilon in (0.1, 1e-4, 1e-12):
        subset = [
            row for row in bv if row["channel"] == 2 and row["epsilon"] == epsilon
        ]
        axes[1].plot(
            [row["scale"] for row in subset],
            [row["exact_to_fpl_storage"] for row in subset],
            "o-",
            label=f"ε={epsilon:g}",
        )
    axes[1].set(
        xlabel="Response budget r",
        ylabel="Exact / FPL optimal storage",
        xscale="log",
        title="Finite savings and asymptotic agreement",
    )
    axes[1].legend()
    fig.savefig(out / "bv_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    subset = [row for row in engel if row["scale"] == 10 and row["epsilon"] == 1e-6]
    degree = [row["degree"] for row in subset]
    for key, label in (
        ("hs", "HS certificate"),
        ("upper", "Response: feasible upper"),
        ("lower", "Response: witness lower"),
    ):
        axes[0].plot(
            degree, [row[f"{key}_storage"] for row in subset], "o-", label=label
        )
    axes[0].set(
        xlabel="Pure Lie degree k",
        ylabel="Optimal dense storage",
        yscale="log",
        title="Engel paths: a‖Eₖ‖HS=10, ε=10⁻⁶",
    )
    axes[0].legend()
    axes[1].plot(
        [row["amplitude"] for row in asymptotic],
        [row["log_cost_gap_over_root_amplitude"] for row in asymptotic],
        "o-",
    )
    axes[1].axhline(
        asymptotic[0]["limiting_certified_gap"],
        color="black",
        linestyle="--",
        label="Proved asymptotic limit",
    )
    axes[1].set(
        xlabel="Pure-area amplitude a",
        ylabel="log(HS / feasible cost) / √a",
        xscale="log",
        title="Growing storage separation, k=2",
    )
    axes[1].legend()
    fig.savefig(out / "engel_comparison.png", dpi=180)
    plt.close(fig)


def main():
    from pathloss.provenance import create_run_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    create_run_directory(args.out)
    profiles = []
    bv = _bv(profiles)
    engel = _engel(profiles)
    asymptotic = _asymptotic(profiles)
    _write_csv(args.out / "bv_optima.csv", bv)
    _write_csv(args.out / "engel_optima.csv", engel)
    _write_csv(args.out / "engel_amplitude.csv", asymptotic)
    _write_csv(args.out / "depth_profiles.csv", profiles)
    _figures(args.out, bv, engel, profiles, asymptotic)
    summary = {
        "bv_cases": len(bv),
        "bv_optimum_sets_changed": sum(not row["same_optimum_set"] for row in bv),
        "bv_strict_cost_improvements": sum(
            row["exact_to_fpl_storage"] < 1 for row in bv
        ),
        "bv_canonical_depth_changes": sum(row["canonical_depth_changed"] for row in bv),
        "bv_min_cost_ratio": min(row["exact_to_fpl_storage"] for row in bv),
        "engel_cases": len(engel),
        "engel_certified_depth_changes": sum(
            row["certified_depth_changed"] for row in engel
        ),
        "engel_min_upper_to_hs_ratio": min(row["upper_to_hs_storage"] for row in engel),
    }
    root = Path(__file__).resolve().parents[1]
    metadata = {
        "command": [sys.executable, *sys.argv],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "storage": "m times exact cumulative Witt dimension; dense generic slots",
        "decoder": "Euler polynomial; no log-ODE substitution",
        "pure_response_status": "rigorous lower and upper envelopes; supremum unknown",
        **_source_provenance(root),
        "summary": summary,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
