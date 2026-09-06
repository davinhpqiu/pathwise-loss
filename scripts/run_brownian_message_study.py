#!/usr/bin/env python
"""Run staged Brownian rough-path message sensitivity experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np

from pathloss.brownian_message_study import (
    acceptance_checks,
    aggregate_condition_results,
    prepare_feature_cache,
    roughpy_reference_checks,
    run_four_class_condition,
    run_message_condition,
)
from pathloss.brownian_messages import (
    bch_insert_message,
    file_sha256,
    message_arrays,
    message_conditions,
    reconstruct_path,
    resolve_data_path,
)
from pathloss.provenance import load_config, run_metadata, utc_now


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _peak_memory_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / 1024**2
    return value / 1024.0


def _software_versions() -> dict[str, str]:
    return {"numpy": np.__version__}


def _data_path(config: dict, role: str) -> Path:
    key = "main_candidates" if role == "main" else "confirmation_candidates"
    return resolve_data_path(config["data"][key])


def _load_streams(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _load_small_streams(config: dict) -> np.ndarray:
    roots = [Path(value).expanduser() for value in config["data"]["small_roots"]]
    files = []
    for root in roots:
        if root.is_dir():
            files = sorted(root.glob("stream-*.npy"))
            if files:
                break
    if not files:
        raise FileNotFoundError("no configured single-stream directory exists")
    limit = int(config["acceptance"]["small_stream_count"])
    arrays = [np.load(path, allow_pickle=False) for path in files[:limit]]
    return np.stack(arrays)


def _save_message_examples(path: Path, streams: np.ndarray, config: dict) -> None:
    cache = path.parent / ".matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    data = config["data"]
    message = config["message"]
    width = int(data["brownian_width"])
    steps = int(data["steps"])
    base = streams[:1]
    base_path = reconstruct_path(base, width=width)[0]
    time_grid = np.linspace(0.0, float(data["duration"]), steps + 1)
    cases = [
        (template, message_type)
        for template in message["templates"]
        for message_type in message["types"]
    ]
    figure, axes = plt.subplots(len(cases), 2, figsize=(12, 2.5 * len(cases)))
    support = slice(
        int(message["support_start"]),
        int(message["support_start"]) + int(message["support_length"]),
    )
    for row, (template, message_type) in enumerate(cases):
        first, area = message_arrays(
            message_type=message_type,
            template=template,
            amplitude=4.0,
            support_length=int(message["support_length"]),
            sigma_first=1.0 / np.sqrt(steps),
            sigma_area=1.0 / (2.0 * steps),
            first_direction=message["first_direction"],
            area_direction=message["area_direction"],
        )
        changed = bch_insert_message(
            base,
            start=int(message["support_start"]),
            first_message=first,
            area_message=area,
            width=width,
            area_pairs=data["area_pairs"],
        )
        changed_path = reconstruct_path(changed, width=width)[0]
        axes[row, 0].plot(time_grid, base_path[:, 0], color="black", label="original")
        axes[row, 0].plot(time_grid, changed_path[:, 0], label="modified")
        axes[row, 0].set_ylabel(f"{template}\n{message_type}")
        axes[row, 1].plot(
            np.arange(support.start, support.stop),
            changed[0, support, width] - base[0, support, width],
        )
        axes[row, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].legend()
    axes[0, 0].set_title("first coordinate")
    axes[0, 1].set_title("first area-coordinate change on support")
    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("increment index")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _condition_records(root: Path) -> list[dict]:
    return [
        json.loads(path.read_text())
        for path in sorted(root.glob("conditions/*/*/rho-*/result.json"))
    ]


def _save_aggregate_outputs(root: Path) -> None:
    records = _condition_records(root)
    if not records:
        return
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    cache = figure_dir / ".matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    with (root / "paired_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "template",
                "message_type",
                "amplitude",
                "discrepancy",
                "median",
                "normalized_median",
            ]
        )
        for record in records:
            condition = record["condition"]
            for name, result in record["paired"].items():
                writer.writerow(
                    [
                        condition["template"],
                        condition["message_type"],
                        condition["amplitude"],
                        name,
                        result["median"],
                        result["normalized_median"],
                    ]
                )

    with (root / "compute_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "template",
                "message_type",
                "amplitude",
                "runtime_seconds",
                "peak_memory_mb",
                "truncated_kernel_status",
                "rough_kernel_status",
            ]
        )
        for record in records:
            condition = record["condition"]
            writer.writerow(
                [
                    condition["template"],
                    condition["message_type"],
                    condition["amplitude"],
                    record.get("runtime_seconds"),
                    record.get("peak_memory_mb"),
                    record.get("truncated_signature_kernel", {}).get("status"),
                    record.get("rough_kernel", {}).get("status"),
                ]
            )

    templates = sorted({record["condition"]["template"] for record in records})
    message_types = sorted({record["condition"]["message_type"] for record in records})
    for template in templates:
        figure, axes = plt.subplots(
            1, len(message_types), figsize=(6 * len(message_types), 4), squeeze=False
        )
        for axis, message_type in zip(axes[0], message_types):
            subset = sorted(
                (
                    record
                    for record in records
                    if record["condition"]["template"] == template
                    and record["condition"]["message_type"] == message_type
                ),
                key=lambda record: record["condition"]["amplitude"],
            )
            amplitudes = [record["condition"]["amplitude"] for record in subset]
            core_names = [
                name
                for name in subset[0]["paired"]
                if not name.startswith("response_")
                and name != "signature_kernel_truncated"
            ]
            for name in core_names:
                values = [
                    record["paired"][name]["normalized_median"] for record in subset
                ]
                axis.plot(amplitudes, values, marker="o", label=name)
            axis.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
            axis.set_title(message_type)
            axis.set_xlabel("normalized amplitude")
            axis.set_ylabel("normalized paired sensitivity")
        axes[0, -1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        figure.suptitle(f"Paired sensitivity: {template} template")
        figure.tight_layout()
        figure.savefig(
            figure_dir / f"paired_{template}.png", dpi=180, bbox_inches="tight"
        )
        plt.close(figure)

        response_groups = (
            ("global", "global response family"),
            ("local_aligned", "aligned local response family"),
            ("local_offset", "offset local response family"),
        )
        for representation, title in response_groups:
            names = [
                name
                for name in records[0]["paired"]
                if name == representation
                or name.startswith(f"response_{representation}_")
                or (representation == "global" and name == "signature_kernel_truncated")
            ]
            figure, axes = plt.subplots(
                1,
                len(message_types),
                figsize=(6 * len(message_types), 4),
                squeeze=False,
            )
            for axis, message_type in zip(axes[0], message_types):
                message_subset = sorted(
                    (
                        record
                        for record in records
                        if record["condition"]["template"] == template
                        and record["condition"]["message_type"] == message_type
                    ),
                    key=lambda record: record["condition"]["amplitude"],
                )
                for name in names:
                    values = [
                        record["paired"][name]["normalized_median"]
                        for record in message_subset
                    ]
                    axis.plot(
                        [record["condition"]["amplitude"] for record in message_subset],
                        values,
                        marker="o",
                        label=name,
                    )
                axis.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
                axis.set_title(message_type)
                axis.set_xlabel("normalized amplitude")
                axis.set_ylabel("normalized paired sensitivity")
            axes[0, -1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
            figure.suptitle(f"{title}: {template} template")
            figure.tight_layout()
            figure.savefig(
                figure_dir / f"paired_response_{representation}_{template}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(figure)

        detected = [record for record in records if record.get("detection")]
        if detected:
            figure, axes = plt.subplots(
                1,
                len(message_types),
                figsize=(6 * len(message_types), 4),
                squeeze=False,
            )
            for axis, message_type in zip(axes[0], message_types):
                subset = sorted(
                    (
                        record
                        for record in detected
                        if record["condition"]["template"] == template
                        and record["condition"]["message_type"] == message_type
                    ),
                    key=lambda record: record["condition"]["amplitude"],
                )
                if not subset:
                    continue
                amplitudes = [record["condition"]["amplitude"] for record in subset]
                for name in subset[0]["detection"]:
                    values = [
                        record["detection"][name]["balanced_accuracy"]
                        for record in subset
                    ]
                    axis.plot(amplitudes, values, marker="o", label=name)
                axis.axhline(0.5, color="black", linewidth=0.8, linestyle=":")
                axis.set_ylim(0.4, 1.02)
                axis.set_title(message_type)
                axis.set_xlabel("normalized amplitude")
                axis.set_ylabel("balanced accuracy")
            axes[0, -1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
            figure.suptitle(f"Single-stream detection: {template} template")
            figure.tight_layout()
            figure.savefig(
                figure_dir / f"detection_{template}.png", dpi=180, bbox_inches="tight"
            )
            plt.close(figure)

    component_records = sorted(
        (
            record
            for record in records
            if record["condition"]["template"] == "balanced"
            and record["condition"]["message_type"] == "area"
        ),
        key=lambda record: record["condition"]["amplitude"],
    )
    if component_records:
        figure, axes = plt.subplots(1, 3, figsize=(16, 4))
        for axis, representation in zip(
            axes, ("global", "local_aligned", "local_offset")
        ):
            names = component_records[0]["paired"][representation]["components"]
            amplitudes = [
                record["condition"]["amplitude"] for record in component_records
            ]
            for name in names:
                values = [
                    record["paired"][representation]["components"][name]
                    for record in component_records
                ]
                axis.plot(amplitudes, values, marker="o", label=name)
            axis.set_title(representation)
            axis.set_xlabel("normalized amplitude")
            axis.set_ylabel("normalized squared contribution")
            axis.legend(fontsize=8)
        figure.suptitle("Balanced area message: signature components")
        figure.tight_layout()
        figure.savefig(figure_dir / "balanced_area_components.png", dpi=180)
        plt.close(figure)


def _save_confirmation_comparison(out: Path) -> None:
    main = _condition_records(out / "main")
    confirmation = _condition_records(out / "confirmation")
    if not main or not confirmation:
        return
    key = lambda record: (
        record["condition"]["template"],
        record["condition"]["message_type"],
        float(record["condition"]["amplitude"]),
    )
    main_map = {key(record): record for record in main}
    rows = []
    for record in confirmation:
        condition_key = key(record)
        if condition_key not in main_map:
            continue
        for name, result in record["paired"].items():
            rows.append(
                [
                    *condition_key,
                    name,
                    main_map[condition_key]["paired"][name]["normalized_median"],
                    result["normalized_median"],
                ]
            )
    with (out / "confirmation_comparison.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "template",
                "message_type",
                "amplitude",
                "discrepancy",
                "ensemble_0",
                "ensemble_1",
            ]
        )
        writer.writerows(rows)


def _save_four_class_figures(out: Path) -> None:
    paths = sorted((out / "main" / "four_class").glob("*/result.json"))
    if not paths:
        return
    figure_dir = out / "main" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    cache = figure_dir / ".matplotlib"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    for path in paths:
        record = json.loads(path.read_text())
        names = list(record["detection"])
        columns = 3
        rows = int(np.ceil(len(names) / columns))
        figure, axes = plt.subplots(
            rows, columns, figsize=(4 * columns, 4 * rows), squeeze=False
        )
        for axis, name in zip(axes.flat, names):
            matrix = np.asarray(record["detection"][name]["confusion_matrix"])
            image = axis.imshow(matrix, cmap="Blues")
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    axis.text(
                        column, row, str(matrix[row, column]), ha="center", va="center"
                    )
            axis.set_title(name)
            axis.set_xlabel("predicted class")
            axis.set_ylabel("true class")
            figure.colorbar(image, ax=axis, fraction=0.046)
        for axis in axes.flat[len(names) :]:
            axis.axis("off")
        figure.suptitle(f"Four-class detection: {record['template']} template")
        figure.tight_layout()
        figure.savefig(figure_dir / f"four_class_{record['template']}.png", dpi=180)
        plt.close(figure)


def _require_core_acceptance(out: Path) -> dict:
    path = out / "acceptance" / "acceptance.json"
    if not path.exists():
        raise SystemExit("run --stage acceptance before preparing experiment data")
    report = json.loads(path.read_text())
    if not report.get("passed"):
        raise SystemExit("core acceptance checks failed; inspect acceptance.json")
    return report


def _require_main_gate(out: Path, config: dict) -> None:
    _require_core_acceptance(out)
    reference_path = out / "acceptance" / "reference.json"
    recorded = (
        json.loads(reference_path.read_text()).get("passed", False)
        if reference_path.exists()
        else False
    )
    if not recorded and not config["acceptance"].get("reference_accepted", False):
        raise SystemExit(
            "main stage is locked until independent rough-path reference check "
            "passes (run --stage reference)"
        )


def _run_acceptance(config_path: Path, config: dict, out: Path) -> int:
    target = out / "acceptance"
    target.mkdir(parents=True, exist_ok=True)
    meta = run_metadata(config_path, config)
    started = time.perf_counter()
    small_streams = _load_small_streams(config)
    result = acceptance_checks(small_streams, config=config)
    meta.update(result)
    meta["software_versions"] = _software_versions()
    meta["runtime_seconds"] = time.perf_counter() - started
    meta["peak_memory_mb"] = _peak_memory_mb()
    meta["finished"] = utc_now()
    _write_json(target / "acceptance.json", meta)
    _save_message_examples(target / "message_examples.png", small_streams, config)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


def _run_reference(config_path: Path, config: dict, out: Path) -> int:
    _require_core_acceptance(out)
    target = out / "acceptance"
    report = run_metadata(config_path, config)
    started = time.perf_counter()
    result = roughpy_reference_checks(_load_small_streams(config), config=config)
    report.update(result)
    report["software_versions"] = _software_versions()
    report["runtime_seconds"] = time.perf_counter() - started
    report["peak_memory_mb"] = _peak_memory_mb()
    report["finished"] = utc_now()
    _write_json(target / "reference.json", report)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 2


def _run_prepare(
    config_path: Path,
    config: dict,
    out: Path,
    *,
    role: str,
    pilot: bool,
) -> int:
    _require_core_acceptance(out)
    path = _data_path(config, role)
    streams = _load_streams(path)
    count = (
        int(config["analysis"]["pilot_paths"])
        if pilot
        else int(config["data"]["expected_paths"])
    )
    cache_name = f"{'pilot' if pilot else role}_cache"
    cache = out / cache_name
    started = time.perf_counter()
    digest = file_sha256(path)
    expected_digest = config["data"]["expected_sha256"][role]
    if digest != expected_digest:
        raise SystemExit(
            f"{role} archive hash mismatch: expected {expected_digest}, found {digest}"
        )
    manifest = prepare_feature_cache(
        streams,
        data_path=path,
        cache=cache,
        config=config,
        count=count,
        data_sha256=digest,
        split_seed=int(
            config["splits"][
                "confirmation_seed" if role == "confirmation" else "main_seed"
            ]
        ),
    )
    manifest["config"] = str(config_path)
    manifest["git_metadata"] = run_metadata(config_path, config)
    manifest["software_versions"] = _software_versions()
    manifest["runtime_seconds"] = time.perf_counter() - started
    manifest["peak_memory_mb"] = _peak_memory_mb()
    manifest["finished"] = utc_now()
    _write_json(cache / "manifest.json", manifest)
    print(json.dumps({"cache": str(cache), "count": count, "complete": True}, indent=2))
    return 0


def _select_conditions(config: dict, stage: str):
    if stage == "confirmation":
        amplitudes = config["confirmation"].get("amplitudes", [])
        if not amplitudes:
            raise SystemExit(
                "confirmation.amplitudes is empty; populate it from frozen main summary"
            )
        message = dict(config["message"])
        message["amplitudes"] = amplitudes
        return message_conditions(message)
    return message_conditions(config["message"])


def _run_conditions(
    config_path: Path,
    config: dict,
    out: Path,
    *,
    stage: str,
    task_id: int | None,
    paired_only: bool,
) -> int:
    if stage == "main":
        _require_main_gate(out, config)
        role = "main"
        count = int(config["data"]["expected_paths"])
        cache = out / "main_cache"
    elif stage == "confirmation":
        _require_main_gate(out, config)
        role = "confirmation"
        count = int(config["data"]["expected_paths"])
        cache = out / "confirmation_cache"
    else:
        _require_core_acceptance(out)
        role = "main"
        count = int(config["analysis"]["pilot_paths"])
        cache = out / "pilot_cache"
    path = _data_path(config, role)
    streams = _load_streams(path)
    conditions = _select_conditions(config, stage)
    if task_id is not None:
        if task_id < 0 or task_id >= len(conditions):
            raise SystemExit(f"--task-id must lie in [0, {len(conditions) - 1}]")
        indexed = [(task_id, conditions[task_id])]
    else:
        indexed = list(enumerate(conditions))
    result_root = out / stage / "conditions"
    for index, condition in indexed:
        target = result_root / condition.slug
        result_path = target / "result.json"
        if result_path.exists():
            print(f"[{index + 1}/{len(conditions)}] skip {condition.slug}", flush=True)
            continue
        print(f"[{index + 1}/{len(conditions)}] start {condition.slug}", flush=True)
        started = time.perf_counter()
        payload = run_message_condition(
            streams,
            data_path=path,
            cache=cache,
            out=target,
            config=config,
            condition=condition,
            count=count,
            include_detection=not paired_only,
        )
        payload["config"] = str(config_path)
        payload["git_metadata"] = run_metadata(config_path, config)
        payload["software_versions"] = _software_versions()
        payload["runtime_seconds"] = time.perf_counter() - started
        payload["peak_memory_mb"] = _peak_memory_mb()
        payload["finished"] = utc_now()
        _write_json(result_path, payload)
        print(
            json.dumps(
                {
                    "condition": payload["condition"],
                    "runtime_seconds": payload["runtime_seconds"],
                    "complete": True,
                },
                indent=2,
            ),
            flush=True,
        )
    return 0


def _run_four_class(
    config_path: Path,
    config: dict,
    out: Path,
    *,
    task_id: int | None,
) -> int:
    _require_main_gate(out, config)
    templates = tuple(config["message"]["templates"])
    if task_id is not None and (task_id < 0 or task_id >= len(templates)):
        raise SystemExit(f"--task-id must lie in [0, {len(templates) - 1}]")
    selected = (
        [(task_id, templates[task_id])]
        if task_id is not None
        else list(enumerate(templates))
    )
    path = _data_path(config, "main")
    streams = _load_streams(path)
    count = int(config["data"]["expected_paths"])
    cache = out / "main_cache"
    amplitude = float(config["analysis"]["four_class_amplitude"])
    for index, template in selected:
        target = out / "main" / "four_class" / template
        if (target / "result.json").exists():
            print(f"[{index + 1}/{len(templates)}] skip {template}", flush=True)
            continue
        started = time.perf_counter()
        payload = run_four_class_condition(
            streams,
            data_path=path,
            cache=cache,
            out=target,
            config=config,
            template=template,
            amplitude=amplitude,
            count=count,
        )
        payload["config"] = str(config_path)
        payload["git_metadata"] = run_metadata(config_path, config)
        payload["software_versions"] = _software_versions()
        payload["runtime_seconds"] = time.perf_counter() - started
        payload["peak_memory_mb"] = _peak_memory_mb()
        payload["finished"] = utc_now()
        _write_json(target / "result.json", payload)
        print(json.dumps({"template": template, "complete": True}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "acceptance",
            "reference",
            "prepare",
            "prepare-pilot",
            "pilot",
            "main",
            "four-class",
            "confirmation",
            "aggregate",
        ),
        required=True,
    )
    parser.add_argument("--ensemble", choices=("main", "confirmation"), default="main")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--paired-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage == "acceptance":
        return _run_acceptance(args.config, config, args.out)
    if args.stage == "reference":
        return _run_reference(args.config, config, args.out)
    if args.stage == "prepare":
        return _run_prepare(
            args.config,
            config,
            args.out,
            role=args.ensemble,
            pilot=False,
        )
    if args.stage == "prepare-pilot":
        return _run_prepare(
            args.config,
            config,
            args.out,
            role="main",
            pilot=True,
        )
    if args.stage == "aggregate":
        for stage in ("pilot", "main", "confirmation"):
            root = args.out / stage
            if (root / "conditions").exists():
                summary = aggregate_condition_results(root)
                _save_aggregate_outputs(root)
                print(json.dumps({stage: summary}, indent=2))
        _save_confirmation_comparison(args.out)
        _save_four_class_figures(args.out)
        return 0
    if args.stage == "four-class":
        return _run_four_class(args.config, config, args.out, task_id=args.task_id)
    if (
        args.stage == "pilot"
        and not (args.out / "pilot_cache" / "manifest.json").exists()
    ):
        raise SystemExit("run --stage prepare-pilot before pilot array tasks")
    return _run_conditions(
        args.config,
        config,
        args.out,
        stage=args.stage,
        task_id=args.task_id,
        paired_only=args.paired_only,
    )


if __name__ == "__main__":
    raise SystemExit(main())
