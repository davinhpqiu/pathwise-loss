"""Preparation and analysis for notebook 07's Brownian-message study.

The input archive contains step-two Brownian rough-path increments. Preparation
validates that archive and caches global/local signature factors for unmodified
stream identities. A condition then inserts a controlled increment, area, or
combined message; compares original and modified streams under every configured
discrepancy; and runs the fixed 1-nearest-neighbour detection check. Aggregation
writes the tables consumed directly by notebook 07. No neural model is trained
in this experiment.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .brownian_messages import (
    MessageCondition,
    bch_insert_message,
    binary_roc_auc,
    bootstrap_interval,
    brownian_scales,
    chen_product_numpy,
    chen_reduce,
    classification_split,
    coordinate_discrepancies,
    derangement,
    file_sha256,
    local_rough_signatures,
    message_arrays,
    nearest_neighbour,
    reconstruct_path,
    response_signature_distance,
    rough_signature,
    segment_signature,
    truncated_signature_features,
    truncated_signature_kernel_distance,
    validate_stream_array,
)

__all__ = [
    "acceptance_checks",
    "aggregate_condition_results",
    "prepare_feature_cache",
    "roughpy_reference_checks",
    "run_four_class_condition",
    "run_message_condition",
]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _positive_levels(levels: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    return levels[1:]


def _with_level_zero(levels: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    batch_shape = levels[0].shape[:-1]
    zero = np.ones(batch_shape + (1,), dtype=levels[0].dtype)
    return (zero, *levels)


def _combine_three(
    left: tuple[np.ndarray, ...],
    middle: tuple[np.ndarray, ...],
    right: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    return _positive_levels(
        chen_product_numpy(
            chen_product_numpy(_with_level_zero(left), _with_level_zero(middle)),
            _with_level_zero(right),
        )
    )


def _open_level_maps(
    cache: Path,
    prefix: str,
    depth: int,
    mode: str = "r",
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.load(cache / f"{prefix}_level{level}.npy", mmap_mode=mode)
        for level in range(1, depth + 1)
    )


def _create_level_maps(
    cache: Path,
    prefix: str,
    shape: tuple[int, ...],
    width: int,
    depth: int,
) -> tuple[np.memmap, ...]:
    return tuple(
        np.lib.format.open_memmap(
            cache / f"{prefix}_level{level}.npy",
            mode="w+",
            dtype="float32",
            shape=shape + (width**level,),
        )
        for level in range(1, depth + 1)
    )


def _feature_slices(width: int, depth: int) -> tuple[slice, ...]:
    starts = np.cumsum([0, *[width**level for level in range(1, depth + 1)]])
    return tuple(
        slice(int(starts[level - 1]), int(starts[level]))
        for level in range(1, depth + 1)
    )


def _number_slug(value: float) -> str:
    return format(float(value), ".8g").replace(".", "p")


def _response_name(representation: str, response_bound: float) -> str:
    return f"response_{representation}_r{_number_slug(response_bound)}"


def _response_bounds(config: dict) -> tuple[float, ...]:
    values = tuple(
        float(value) for value in config["signature"].get("response_bounds", ())
    )
    if any(value <= 0 or not np.isfinite(value) for value in values):
        raise ValueError("response bounds must be finite and positive")
    if len(values) != len(set(values)):
        raise ValueError("response bounds contain duplicates")
    return values


def _truncated_kernel_config(config: dict) -> dict:
    return config["signature"].get(
        "truncated_kernel", {"enabled": False, "level_weight": 1.0}
    )


def _scale_feature_levels(
    levels: tuple[np.ndarray, ...],
    q2: list[float],
    *,
    width: int,
    delta: float,
    blocks: int = 1,
) -> np.ndarray:
    scaled = [
        level.reshape(level.shape[0], -1)
        / math.sqrt(blocks * (width**index) * (q2[index - 1] + delta))
        for index, level in enumerate(levels, start=1)
    ]
    return np.concatenate(scaled, axis=1).astype(np.float32, copy=False)


def _compute_q2(
    levels: tuple[np.ndarray, ...],
    training_indices: np.ndarray,
    *,
    local: bool,
) -> list[float]:
    result = []
    for level in levels:
        training = np.asarray(level[training_indices], dtype=np.float64)
        mean_axis = 0
        centered = training - np.mean(training, axis=mean_axis, keepdims=True)
        result.append(float(np.mean(centered**2)))
    return result


def _distance_scale(kind: str, values: np.ndarray) -> np.ndarray:
    if kind in {"mse", "j2", "area", "increment", "feature"}:
        return np.sqrt(np.maximum(values, 0.0))
    return values


def _natural_scales(
    streams: np.ndarray,
    cache: Path,
    *,
    config: dict,
    count: int,
    width: int,
    duration: float,
    seed: int,
    batch_size: int,
) -> dict[str, float]:
    paired = derangement(count, seed=seed)
    samples: dict[str, list[np.ndarray]] = {
        "mse": [],
        "j2": [],
        "l1": [],
        "linf": [],
        "increment": [],
        "area": [],
    }
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        ids = np.arange(start, stop)
        first = np.asarray(streams[ids])
        second = np.asarray(streams[paired[ids]])
        paths = coordinate_discrepancies(
            reconstruct_path(first, width=width),
            reconstruct_path(second, width=width),
            duration=duration,
        )
        for name, value in paths.items():
            samples[name].append(_distance_scale(name, value))
        increment = np.mean(
            np.sum((first[..., :width] - second[..., :width]) ** 2, axis=-1),
            axis=-1,
        )
        area = np.mean(
            np.sum((first[..., width:] - second[..., width:]) ** 2, axis=-1),
            axis=-1,
        )
        samples["increment"].append(np.sqrt(np.maximum(increment, 0.0)))
        samples["area"].append(np.sqrt(np.maximum(area, 0.0)))
    result = {
        name: float(np.median(np.concatenate(values)))
        for name, values in samples.items()
    }
    for name in ("global", "local_aligned", "local_offset"):
        features = np.load(cache / f"{name}_features.npy", mmap_mode="r")
        distances = np.empty(count, dtype=np.float64)
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            difference = np.asarray(
                features[start:stop], dtype=np.float64
            ) - np.asarray(features[paired[start:stop]], dtype=np.float64)
            distances[start:stop] = np.linalg.norm(difference, axis=1)
        result[name] = float(np.median(distances))
    for name in ("global", "local_aligned", "local_offset"):
        depth = int(
            config["signature"]["global_depth" if name == "global" else "local_depth"]
        )
        levels = _open_level_maps(cache, name, depth)
        for response_bound in _response_bounds(config):
            pieces = []
            for start in range(0, count, batch_size):
                stop = min(start + batch_size, count)
                ids = np.arange(start, stop)
                pieces.append(
                    response_signature_distance(
                        tuple(np.asarray(level[ids]) for level in levels),
                        tuple(np.asarray(level[paired[ids]]) for level in levels),
                        response_bound=response_bound,
                        average_blocks=name != "global",
                    )
                )
            result[_response_name(name, response_bound)] = float(
                np.median(np.concatenate(pieces))
            )
    kernel = _truncated_kernel_config(config)
    if kernel.get("enabled", False):
        levels = _open_level_maps(
            cache, "global", int(config["signature"]["global_depth"])
        )
        pieces = []
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            ids = np.arange(start, stop)
            pieces.append(
                truncated_signature_kernel_distance(
                    tuple(np.asarray(level[ids]) for level in levels),
                    tuple(np.asarray(level[paired[ids]]) for level in levels),
                    level_weight=float(kernel["level_weight"]),
                )
            )
        result["signature_kernel_truncated"] = float(np.median(np.concatenate(pieces)))
    return result


def prepare_feature_cache(
    streams: np.ndarray,
    *,
    data_path: Path,
    cache: Path,
    config: dict,
    count: int,
    data_sha256: str | None = None,
    split_seed: int | None = None,
) -> dict:
    """Create reusable rough-signature factors, features, and scales."""
    data = config["data"]
    message = config["message"]
    signature = config["signature"]
    split_cfg = config["splits"]
    width = int(data["brownian_width"])
    steps = int(data["steps"])
    area_pairs = tuple(tuple(pair) for pair in data["area_pairs"])
    validate_stream_array(
        streams,
        expected_paths=data["expected_paths"],
        expected_steps=steps,
        expected_width=width + len(area_pairs),
        expected_dtype=data["dtype"],
    )
    if count < 2 or count > streams.shape[0]:
        raise ValueError("cache count lies outside available paths")
    support_start = int(message["support_start"])
    support_length = int(message["support_length"])
    block_size = int(signature["local_block_size"])
    if support_start % block_size or support_length != block_size:
        raise ValueError("primary support must equal one aligned local block")
    if steps % block_size:
        raise ValueError("aligned block size must divide stream length")
    block_index = support_start // block_size
    block_count = steps // block_size
    offset = int(signature["local_offset"])
    offset_blocks = (steps - offset) // block_size
    global_depth = int(signature["global_depth"])
    local_depth = int(signature["local_depth"])
    batch_size = int(signature["prepare_batch_size"])
    if global_depth > 4 or local_depth > global_depth:
        raise ValueError("unsupported signature depth configuration")

    cache.mkdir(parents=True, exist_ok=True)
    digest = data_sha256 or file_sha256(data_path)
    split = classification_split(
        count,
        seed=int(split_cfg["main_seed"] if split_seed is None else split_seed),
        proportions=split_cfg["proportions"],
        classes=2,
    )
    training_indices = split["train_indices"]
    scales = brownian_scales(
        streams,
        training_indices,
        width=width,
        batch_size=int(data["statistics_batch_size"]),
    )

    global_levels = _create_level_maps(cache, "global", (count,), width, global_depth)
    left_levels = _create_level_maps(cache, "left", (count,), width, global_depth)
    right_levels = _create_level_maps(cache, "right", (count,), width, global_depth)
    aligned_levels = _create_level_maps(
        cache, "local_aligned", (count, block_count), width, local_depth
    )
    offset_levels = _create_level_maps(
        cache, "local_offset", (count, offset_blocks), width, local_depth
    )

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = np.asarray(streams[start:stop])
        blocked = batch.reshape(stop - start, block_count, block_size, batch.shape[-1])
        block_levels = rough_signature(
            blocked,
            global_depth,
            width=width,
            area_pairs=area_pairs,
        )
        left = _positive_levels(
            chen_reduce(
                _with_level_zero(tuple(x[:, :block_index] for x in block_levels))
            )
        )
        right = _positive_levels(
            chen_reduce(
                _with_level_zero(tuple(x[:, block_index + 1 :] for x in block_levels))
            )
        )
        middle = tuple(level[:, block_index] for level in block_levels)
        whole = _combine_three(left, middle, right)
        local_offset = local_rough_signatures(
            batch,
            block_size=block_size,
            offset=offset,
            depth=local_depth,
            width=width,
            area_pairs=area_pairs,
        )
        for target, source in zip(global_levels, whole):
            target[start:stop] = source
        for target, source in zip(left_levels, left):
            target[start:stop] = source
        for target, source in zip(right_levels, right):
            target[start:stop] = source
        for target, source in zip(aligned_levels, block_levels[:local_depth]):
            target[start:stop] = source
        for target, source in zip(offset_levels, local_offset):
            target[start:stop] = source

    for arrays in (
        global_levels,
        left_levels,
        right_levels,
        aligned_levels,
        offset_levels,
    ):
        for array in arrays:
            array.flush()

    delta = float(signature["scale_delta"])
    scale_data = {
        "brownian": scales,
        "global_q2": _compute_q2(global_levels, training_indices, local=False),
        "local_aligned_q2": _compute_q2(aligned_levels, training_indices, local=True),
        "local_offset_q2": _compute_q2(offset_levels, training_indices, local=True),
    }
    for name, levels, q_key, blocks in (
        ("global", global_levels, "global_q2", 1),
        ("local_aligned", aligned_levels, "local_aligned_q2", block_count),
        ("local_offset", offset_levels, "local_offset_q2", offset_blocks),
    ):
        feature_count = sum(width**level for level in range(1, len(levels) + 1))
        if blocks > 1:
            feature_count *= blocks
        features = np.lib.format.open_memmap(
            cache / f"{name}_features.npy",
            mode="w+",
            dtype="float32",
            shape=(count, feature_count),
        )
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            features[start:stop] = _scale_feature_levels(
                tuple(np.asarray(level[start:stop]) for level in levels),
                scale_data[q_key],
                width=width,
                delta=delta,
                blocks=blocks,
            )
        features.flush()

    natural = _natural_scales(
        streams,
        cache,
        config=config,
        count=count,
        width=width,
        duration=float(data["duration"]),
        seed=int(config["analysis"]["derangement_seed"]),
        batch_size=int(data["statistics_batch_size"]),
    )
    manifest = {
        "schema_version": 2,
        "complete": True,
        "data_path": str(data_path),
        "data_sha256": digest,
        "data_size_bytes": data_path.stat().st_size,
        "count": count,
        "width": width,
        "steps": steps,
        "area_pairs": [list(pair) for pair in area_pairs],
        "support_start": support_start,
        "support_length": support_length,
        "block_index": block_index,
        "block_count": block_count,
        "offset": offset,
        "offset_blocks": offset_blocks,
        "global_depth": global_depth,
        "local_depth": local_depth,
        "split": {name: value.tolist() for name, value in split.items()},
        "scales": scale_data,
        "natural_scales": natural,
    }
    _write_json(cache / "manifest.json", manifest)
    return manifest


def _load_manifest(cache: Path, data_path: Path, count: int) -> dict:
    path = cache / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"run prepare stage first: {path} is absent")
    manifest = json.loads(path.read_text())
    if not manifest.get("complete") or int(manifest["count"]) != count:
        raise ValueError("feature cache is incomplete or has wrong path count")
    if int(manifest.get("schema_version", 0)) != 2:
        raise ValueError(
            "feature cache predates response and kernel losses; rerun prepare"
        )
    if data_path.stat().st_size != int(manifest["data_size_bytes"]):
        raise ValueError("feature cache data size does not match archive")
    return manifest


def _message_for_condition(
    condition: MessageCondition, manifest: dict, config: dict
) -> tuple[np.ndarray, np.ndarray]:
    message = config["message"]
    scales = manifest["scales"]["brownian"]
    return message_arrays(
        message_type=condition.message_type,
        template=condition.template,
        amplitude=condition.amplitude,
        support_length=int(message["support_length"]),
        sigma_first=float(scales["sigma_first"]),
        sigma_area=float(scales["sigma_area"]),
        first_direction=message["first_direction"],
        area_direction=message["area_direction"],
    )


def _modified_global_levels(
    streams: np.ndarray,
    identities: np.ndarray,
    labels: np.ndarray,
    *,
    cache: Path,
    manifest: dict,
    config: dict,
    condition: MessageCondition,
) -> tuple[np.ndarray, ...]:
    signature = config["signature"]
    data = config["data"]
    width = int(data["brownian_width"])
    depth = int(signature["global_depth"])
    area_pairs = tuple(tuple(pair) for pair in data["area_pairs"])
    base = _open_level_maps(cache, "global", depth)
    result = tuple(np.asarray(level[identities]).copy() for level in base)
    selected = np.flatnonzero(labels != 0)
    if selected.size == 0 or condition.amplitude == 0:
        return result
    selected_ids = identities[selected]
    first_message, area_message = _message_for_condition(condition, manifest, config)
    support_start = int(manifest["support_start"])
    support_stop = support_start + int(manifest["support_length"])
    left_cache = _open_level_maps(cache, "left", depth)
    right_cache = _open_level_maps(cache, "right", depth)
    batch_size = int(signature["condition_batch_size"])
    for start in range(0, len(selected), batch_size):
        stop = min(start + batch_size, len(selected))
        positions = selected[start:stop]
        ids = selected_ids[start:stop]
        support = np.asarray(streams[ids, support_start:support_stop])
        modified = bch_insert_message(
            support,
            start=0,
            first_message=first_message,
            area_message=area_message,
            width=width,
            area_pairs=area_pairs,
        )
        middle = rough_signature(modified, depth, width=width, area_pairs=area_pairs)
        left = tuple(np.asarray(level[ids]) for level in left_cache)
        right = tuple(np.asarray(level[ids]) for level in right_cache)
        combined = _combine_three(left, middle, right)
        for target, source in zip(result, combined):
            target[positions] = source
    return result


def _modified_local_levels(
    streams: np.ndarray,
    identities: np.ndarray,
    labels: np.ndarray,
    *,
    cache: Path,
    manifest: dict,
    config: dict,
    condition: MessageCondition,
    offset: bool,
) -> tuple[np.ndarray, ...]:
    data = config["data"]
    signature = config["signature"]
    width = int(data["brownian_width"])
    depth = int(signature["local_depth"])
    area_pairs = tuple(tuple(pair) for pair in data["area_pairs"])
    name = "local_offset" if offset else "local_aligned"
    base = _open_level_maps(cache, name, depth)
    result = tuple(np.asarray(level[identities]).copy() for level in base)
    selected = np.flatnonzero(labels != 0)
    if selected.size == 0 or condition.amplitude == 0:
        return result
    ids = identities[selected]
    first_message, area_message = _message_for_condition(condition, manifest, config)
    support_start = int(manifest["support_start"])
    support_stop = support_start + int(manifest["support_length"])
    block_size = int(signature["local_block_size"])
    partition_offset = int(signature["local_offset"]) if offset else 0
    first_block = (support_start - partition_offset) // block_size
    last_block = (support_stop - 1 - partition_offset) // block_size
    affected = tuple(range(first_block, last_block + 1))
    batch_size = int(signature["condition_batch_size"])
    for start in range(0, len(selected), batch_size):
        stop = min(start + batch_size, len(selected))
        positions = selected[start:stop]
        batch_ids = ids[start:stop]
        window_start = partition_offset + first_block * block_size
        window_stop = partition_offset + (last_block + 1) * block_size
        window = np.asarray(streams[batch_ids, window_start:window_stop])
        modified = bch_insert_message(
            window,
            start=support_start - window_start,
            first_message=first_message,
            area_message=area_message,
            width=width,
            area_pairs=area_pairs,
        )
        local = local_rough_signatures(
            modified,
            block_size=block_size,
            offset=0,
            depth=depth,
            width=width,
            area_pairs=area_pairs,
        )
        for target, source in zip(result, local):
            target[
                np.ix_(positions, np.asarray(affected), np.arange(source.shape[-1]))
            ] = source
    return result


def _scaled_condition_features(
    streams: np.ndarray,
    identities: np.ndarray,
    labels: np.ndarray,
    *,
    cache: Path,
    manifest: dict,
    config: dict,
    condition: MessageCondition,
    representation: str,
) -> np.ndarray:
    signature = config["signature"]
    width = int(config["data"]["brownian_width"])
    delta = float(signature["scale_delta"])
    if representation == "global":
        levels = _modified_global_levels(
            streams,
            identities,
            labels,
            cache=cache,
            manifest=manifest,
            config=config,
            condition=condition,
        )
        q2 = manifest["scales"]["global_q2"]
        blocks = 1
    else:
        offset = representation == "local_offset"
        levels = _modified_local_levels(
            streams,
            identities,
            labels,
            cache=cache,
            manifest=manifest,
            config=config,
            condition=condition,
            offset=offset,
        )
        q2 = manifest["scales"][f"{representation}_q2"]
        blocks = int(manifest["offset_blocks"] if offset else manifest["block_count"])
    return _scale_feature_levels(levels, q2, width=width, delta=delta, blocks=blocks)


def _raw_condition_signature_levels(
    streams: np.ndarray,
    identities: np.ndarray,
    labels: np.ndarray,
    *,
    cache: Path,
    manifest: dict,
    config: dict,
    condition: MessageCondition,
    representation: str,
) -> tuple[np.ndarray, ...]:
    if representation == "global":
        return _modified_global_levels(
            streams,
            identities,
            labels,
            cache=cache,
            manifest=manifest,
            config=config,
            condition=condition,
        )
    if representation not in {"local_aligned", "local_offset"}:
        raise ValueError(f"unknown signature representation {representation!r}")
    return _modified_local_levels(
        streams,
        identities,
        labels,
        cache=cache,
        manifest=manifest,
        config=config,
        condition=condition,
        offset=representation == "local_offset",
    )


def _response_condition_features(
    streams: np.ndarray,
    identities: np.ndarray,
    labels: np.ndarray,
    *,
    cache: Path,
    manifest: dict,
    config: dict,
    condition: MessageCondition,
    representation: str,
    response_bound: float,
) -> np.ndarray:
    levels = _raw_condition_signature_levels(
        streams,
        identities,
        labels,
        cache=cache,
        manifest=manifest,
        config=config,
        condition=condition,
        representation=representation,
    )
    blocks = (
        int(
            manifest[
                "offset_blocks" if representation == "local_offset" else "block_count"
            ]
        )
        if representation != "global"
        else 1
    )
    pieces = [
        response_bound**level * value.reshape(value.shape[0], -1) / blocks
        for level, value in enumerate(levels, start=1)
    ]
    return np.concatenate(pieces, axis=1).astype(np.float32, copy=False)


def _kernel_condition_features(
    streams: np.ndarray,
    identities: np.ndarray,
    labels: np.ndarray,
    *,
    cache: Path,
    manifest: dict,
    config: dict,
    condition: MessageCondition,
) -> np.ndarray:
    levels = _raw_condition_signature_levels(
        streams,
        identities,
        labels,
        cache=cache,
        manifest=manifest,
        config=config,
        condition=condition,
        representation="global",
    )
    kernel = _truncated_kernel_config(config)
    return truncated_signature_features(
        levels,
        level_weight=float(kernel["level_weight"]),
        include_scalar=True,
        normalize=True,
    ).astype(np.float32, copy=False)


def _parse_response_representation(name: str, config: dict) -> tuple[str, float] | None:
    for representation in ("global", "local_aligned", "local_offset"):
        for response_bound in _response_bounds(config):
            if name == _response_name(representation, response_bound):
                return representation, response_bound
    return None


def _condition_stream_values(
    streams: np.ndarray,
    identities: np.ndarray,
    labels: np.ndarray,
    *,
    manifest: dict,
    config: dict,
    condition: MessageCondition,
    representation: str,
) -> np.ndarray:
    width = int(config["data"]["brownian_width"])
    values = np.asarray(streams[identities]).copy()
    selected = np.flatnonzero(labels != 0)
    if selected.size and condition.amplitude != 0:
        first_message, area_message = _message_for_condition(
            condition, manifest, config
        )
        values[selected] = bch_insert_message(
            values[selected],
            start=int(manifest["support_start"]),
            first_message=first_message,
            area_message=area_message,
            width=width,
            area_pairs=config["data"]["area_pairs"],
        )
    if representation == "coordinate":
        return reconstruct_path(values, width=width)
    if representation == "increment":
        return values[..., :width]
    if representation == "area":
        return values[..., width:]
    raise ValueError(f"unknown stream representation {representation!r}")


def _balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    labels = np.unique(truth)
    return float(
        np.mean([np.mean(prediction[truth == label] == label) for label in labels])
    )


def _balanced_accuracy_interval(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = rng.integers(0, len(truth), size=len(truth))
        estimates[index] = _balanced_accuracy(truth[selected], prediction[selected])
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def _classification_result(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    kind: str,
    config: dict,
) -> dict:
    analysis = config["analysis"]
    result = nearest_neighbour(
        train_x,
        train_y,
        test_x,
        kind=kind,
        duration=float(config["data"]["duration"]),
        query_batch=int(analysis["nn_query_batch"]),
        reference_batch=int(analysis["nn_reference_batch"]),
    )
    prediction = result["prediction"]
    balanced = _balanced_accuracy(test_y, prediction)
    correct = (prediction == test_y).astype(float)
    low, high = _balanced_accuracy_interval(
        test_y,
        prediction,
        replicates=int(analysis["bootstrap_replicates"]),
        seed=int(analysis["bootstrap_seed"]),
    )
    payload = {
        "balanced_accuracy": balanced,
        "accuracy": float(np.mean(correct)),
        "balanced_accuracy_interval": [low, high],
        "prediction": prediction.tolist(),
        "nearest_train_index": result["nearest_index"].tolist(),
        "confusion_matrix": _confusion_matrix(test_y, prediction),
    }
    if set(np.unique(test_y)) == {0, 1}:
        class_distance = result["class_distance"]
        if kind in {"mse", "j2", "area", "increment", "feature_squared"}:
            class_distance = np.sqrt(np.maximum(class_distance, 0.0))
        score = class_distance[:, 0] - class_distance[:, 1]
        payload["roc_auc"] = binary_roc_auc(test_y, score)
        payload["roc_score"] = score.tolist()
    return payload


def _paired_results(
    streams: np.ndarray,
    *,
    cache: Path,
    manifest: dict,
    config: dict,
    condition: MessageCondition,
) -> dict:
    count = int(manifest["count"])
    width = int(config["data"]["brownian_width"])
    duration = float(config["data"]["duration"])
    first_message, area_message = _message_for_condition(condition, manifest, config)
    batch_size = int(config["signature"]["condition_batch_size"])
    raw: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "mse",
            "j2",
            "l1",
            "linf",
            "increment",
            "area",
            "global",
            "local_aligned",
            "local_offset",
        )
    }
    for representation in ("global", "local_aligned", "local_offset"):
        for response_bound in _response_bounds(config):
            raw[_response_name(representation, response_bound)] = []
    kernel_config = _truncated_kernel_config(config)
    if kernel_config.get("enabled", False):
        raw["signature_kernel_truncated"] = []
    component_samples = {
        "global": {
            f"level_{level}": []
            for level in range(1, int(config["signature"]["global_depth"]) + 1)
        },
        "local_aligned": {
            **{
                f"level_{level}": []
                for level in range(1, int(config["signature"]["local_depth"]) + 1)
            },
            "message_blocks": [],
        },
        "local_offset": {
            **{
                f"level_{level}": []
                for level in range(1, int(config["signature"]["local_depth"]) + 1)
            },
            "message_blocks": [],
        },
    }
    base_level_maps = {
        "global": _open_level_maps(
            cache, "global", int(config["signature"]["global_depth"])
        ),
        "local_aligned": _open_level_maps(
            cache, "local_aligned", int(config["signature"]["local_depth"])
        ),
        "local_offset": _open_level_maps(
            cache, "local_offset", int(config["signature"]["local_depth"])
        ),
    }
    signature_delta = float(config["signature"]["scale_delta"])
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        ids = np.arange(start, stop)
        labels = np.ones(stop - start, dtype=np.int64)
        original = np.asarray(streams[start:stop])
        modified = bch_insert_message(
            original,
            start=int(manifest["support_start"]),
            first_message=first_message,
            area_message=area_message,
            width=width,
            area_pairs=config["data"]["area_pairs"],
        )
        coordinate = coordinate_discrepancies(
            reconstruct_path(original, width=width),
            reconstruct_path(modified, width=width),
            duration=duration,
        )
        for name, values in coordinate.items():
            raw[name].append(_distance_scale(name, values))
        for name, first, second in (
            ("increment", original[..., :width], modified[..., :width]),
            ("area", original[..., width:], modified[..., width:]),
        ):
            value = np.mean(np.sum((first - second) ** 2, axis=-1), axis=-1)
            raw[name].append(np.sqrt(np.maximum(value, 0.0)))
        for name in ("global", "local_aligned", "local_offset"):
            if name == "global":
                changed_levels = _modified_global_levels(
                    streams,
                    ids,
                    labels,
                    cache=cache,
                    manifest=manifest,
                    config=config,
                    condition=condition,
                )
                blocks = 1
                q2 = manifest["scales"]["global_q2"]
            else:
                changed_levels = _modified_local_levels(
                    streams,
                    ids,
                    labels,
                    cache=cache,
                    manifest=manifest,
                    config=config,
                    condition=condition,
                    offset=name == "local_offset",
                )
                blocks = int(
                    manifest[
                        "offset_blocks" if name == "local_offset" else "block_count"
                    ]
                )
                q2 = manifest["scales"][f"{name}_q2"]
            total = np.zeros(stop - start, dtype=np.float64)
            block_total = np.zeros(stop - start, dtype=np.float64)
            response_total = {
                response_bound: np.zeros(stop - start, dtype=np.float64)
                for response_bound in _response_bounds(config)
            }
            base_levels = tuple(
                np.asarray(base_map[ids]) for base_map in base_level_maps[name]
            )
            for level_index, (base_map, changed_level) in enumerate(
                zip(base_level_maps[name], changed_levels), start=1
            ):
                difference = np.asarray(base_map[ids]) - changed_level
                axes = tuple(range(1, difference.ndim))
                component = np.sum(difference**2, axis=axes) / (
                    blocks
                    * width**level_index
                    * (q2[level_index - 1] + signature_delta)
                )
                component_samples[name][f"level_{level_index}"].append(component)
                total += component
                response_component = np.sum(np.abs(difference), axis=axes) / blocks
                for response_bound in response_total:
                    response_total[response_bound] += (
                        response_bound**level_index * response_component
                    )
                if name != "global":
                    if name == "local_aligned":
                        affected = [int(manifest["block_index"])]
                    else:
                        support_start = int(manifest["support_start"])
                        support_stop = support_start + int(manifest["support_length"])
                        offset = int(manifest["offset"])
                        block_size = int(config["signature"]["local_block_size"])
                        affected = list(
                            range(
                                (support_start - offset) // block_size,
                                (support_stop - 1 - offset) // block_size + 1,
                            )
                        )
                    selected_difference = difference[:, affected]
                    block_axes = tuple(range(1, selected_difference.ndim))
                    block_total += np.sum(selected_difference**2, axis=block_axes) / (
                        blocks
                        * width**level_index
                        * (q2[level_index - 1] + signature_delta)
                    )
            if name != "global":
                component_samples[name]["message_blocks"].append(block_total)
            raw[name].append(np.sqrt(np.maximum(total, 0.0)))
            for response_bound, values in response_total.items():
                raw[_response_name(name, response_bound)].append(values)
            if name == "global" and kernel_config.get("enabled", False):
                raw["signature_kernel_truncated"].append(
                    truncated_signature_kernel_distance(
                        base_levels,
                        changed_levels,
                        level_weight=float(kernel_config["level_weight"]),
                    )
                )
    output = {}
    for index, (name, pieces) in enumerate(raw.items()):
        values = np.concatenate(pieces).astype(float)
        low, high = bootstrap_interval(
            values,
            replicates=int(config["analysis"]["bootstrap_replicates"]),
            seed=int(config["analysis"]["bootstrap_seed"]) + index,
        )
        median = float(np.median(values))
        normalizer = float(manifest["natural_scales"][name])
        output[name] = {
            "median": median,
            "iqr": [
                float(np.quantile(values, 0.25)),
                float(np.quantile(values, 0.75)),
            ],
            "median_interval": [low, high],
            "natural_scale": normalizer,
            "normalized_median": median / normalizer if normalizer > 0 else None,
        }
        if name in component_samples:
            output[name]["components"] = {
                component: float(np.median(np.concatenate(values)))
                for component, values in component_samples[name].items()
            }
    return output


def _detection_results(
    streams: np.ndarray,
    *,
    cache: Path,
    manifest: dict,
    config: dict,
    condition: MessageCondition,
) -> dict:
    split = {name: np.asarray(value) for name, value in manifest["split"].items()}
    result = {}
    for representation in config["analysis"]["detection_representations"]:
        train_ids = split["train_indices"]
        train_y = split["train_labels"]
        test_ids = split["test_indices"]
        test_y = split["test_labels"]
        response = _parse_response_representation(representation, config)
        if response is not None:
            signature_representation, response_bound = response
            train_x = _response_condition_features(
                streams,
                train_ids,
                train_y,
                cache=cache,
                manifest=manifest,
                config=config,
                condition=condition,
                representation=signature_representation,
                response_bound=response_bound,
            )
            test_x = _response_condition_features(
                streams,
                test_ids,
                test_y,
                cache=cache,
                manifest=manifest,
                config=config,
                condition=condition,
                representation=signature_representation,
                response_bound=response_bound,
            )
            result[representation] = _classification_result(
                train_x,
                train_y,
                test_x,
                test_y,
                kind="feature_l1",
                config=config,
            )
        elif representation == "signature_kernel_truncated":
            train_x = _kernel_condition_features(
                streams,
                train_ids,
                train_y,
                cache=cache,
                manifest=manifest,
                config=config,
                condition=condition,
            )
            test_x = _kernel_condition_features(
                streams,
                test_ids,
                test_y,
                cache=cache,
                manifest=manifest,
                config=config,
                condition=condition,
            )
            result[representation] = _classification_result(
                train_x,
                train_y,
                test_x,
                test_y,
                kind="feature_squared",
                config=config,
            )
        elif representation in {"global", "local_aligned", "local_offset"}:
            train_x = _scaled_condition_features(
                streams,
                train_ids,
                train_y,
                cache=cache,
                manifest=manifest,
                config=config,
                condition=condition,
                representation=representation,
            )
            test_x = _scaled_condition_features(
                streams,
                test_ids,
                test_y,
                cache=cache,
                manifest=manifest,
                config=config,
                condition=condition,
                representation=representation,
            )
            result[representation] = _classification_result(
                train_x,
                train_y,
                test_x,
                test_y,
                kind="feature_squared",
                config=config,
            )
        elif representation == "coordinate":
            train_x = _condition_stream_values(
                streams,
                train_ids,
                train_y,
                manifest=manifest,
                config=config,
                condition=condition,
                representation="coordinate",
            )
            test_x = _condition_stream_values(
                streams,
                test_ids,
                test_y,
                manifest=manifest,
                config=config,
                condition=condition,
                representation="coordinate",
            )
            for metric in config["analysis"]["coordinate_detection_metrics"]:
                result[metric] = _classification_result(
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    kind=metric,
                    config=config,
                )
        else:
            train_x = _condition_stream_values(
                streams,
                train_ids,
                train_y,
                manifest=manifest,
                config=config,
                condition=condition,
                representation=representation,
            )
            test_x = _condition_stream_values(
                streams,
                test_ids,
                test_y,
                manifest=manifest,
                config=config,
                condition=condition,
                representation=representation,
            )
            result[representation] = _classification_result(
                train_x,
                train_y,
                test_x,
                test_y,
                kind=representation,
                config=config,
            )
    return result


def run_message_condition(
    streams: np.ndarray,
    *,
    data_path: Path,
    cache: Path,
    out: Path,
    config: dict,
    condition: MessageCondition,
    count: int,
    include_detection: bool,
) -> dict:
    """Run paired sensitivity and optional 1-NN detection for one condition."""
    manifest = _load_manifest(cache, data_path, count)
    payload = {
        "condition": {
            "message_type": condition.message_type,
            "template": condition.template,
            "amplitude": condition.amplitude,
        },
        "count": count,
        "data_sha256": manifest["data_sha256"],
        "area_pairs": manifest["area_pairs"],
        "split": manifest["split"],
        "scales": manifest["scales"],
        "paired": _paired_results(
            streams,
            cache=cache,
            manifest=manifest,
            config=config,
            condition=condition,
        ),
        "detection": None,
        "truncated_signature_kernel": {
            "status": (
                "run"
                if _truncated_kernel_config(config).get("enabled", False)
                else "disabled"
            ),
            "depth": int(config["signature"]["global_depth"]),
            "level_weight": float(
                _truncated_kernel_config(config).get("level_weight", 1.0)
            ),
        },
        "rough_kernel": {
            "status": "not_run",
            "reason": "optional rough-path kernel backend is not configured",
        },
    }
    if include_detection:
        payload["detection"] = _detection_results(
            streams,
            cache=cache,
            manifest=manifest,
            config=config,
            condition=condition,
        )
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "result.json", payload)
    return payload


def _multiclass_representation(
    streams: np.ndarray,
    identities: np.ndarray,
    labels: np.ndarray,
    *,
    cache: Path,
    manifest: dict,
    config: dict,
    template: str,
    amplitude: float,
    representation: str,
) -> np.ndarray:
    zero = np.zeros_like(labels)
    base_condition = MessageCondition("increment", template, 0.0)
    response = _parse_response_representation(representation, config)
    if response is not None:
        signature_representation, response_bound = response
        result = _response_condition_features(
            streams,
            identities,
            zero,
            cache=cache,
            manifest=manifest,
            config=config,
            condition=base_condition,
            representation=signature_representation,
            response_bound=response_bound,
        )
        for class_label, message_type in enumerate(
            ("increment", "area", "combined"), start=1
        ):
            selected = (labels == class_label).astype(np.int64)
            changed = _response_condition_features(
                streams,
                identities,
                selected,
                cache=cache,
                manifest=manifest,
                config=config,
                condition=MessageCondition(message_type, template, amplitude),
                representation=signature_representation,
                response_bound=response_bound,
            )
            result[labels == class_label] = changed[labels == class_label]
        return result
    if representation == "signature_kernel_truncated":
        result = _kernel_condition_features(
            streams,
            identities,
            zero,
            cache=cache,
            manifest=manifest,
            config=config,
            condition=base_condition,
        )
        for class_label, message_type in enumerate(
            ("increment", "area", "combined"), start=1
        ):
            selected = (labels == class_label).astype(np.int64)
            changed = _kernel_condition_features(
                streams,
                identities,
                selected,
                cache=cache,
                manifest=manifest,
                config=config,
                condition=MessageCondition(message_type, template, amplitude),
            )
            result[labels == class_label] = changed[labels == class_label]
        return result
    if representation in {"global", "local_aligned", "local_offset"}:
        result = _scaled_condition_features(
            streams,
            identities,
            zero,
            cache=cache,
            manifest=manifest,
            config=config,
            condition=base_condition,
            representation=representation,
        )
        for class_label, message_type in enumerate(
            ("increment", "area", "combined"), start=1
        ):
            selected = (labels == class_label).astype(np.int64)
            changed = _scaled_condition_features(
                streams,
                identities,
                selected,
                cache=cache,
                manifest=manifest,
                config=config,
                condition=MessageCondition(message_type, template, amplitude),
                representation=representation,
            )
            result[labels == class_label] = changed[labels == class_label]
        return result
    result = _condition_stream_values(
        streams,
        identities,
        zero,
        manifest=manifest,
        config=config,
        condition=base_condition,
        representation=representation,
    )
    for class_label, message_type in enumerate(
        ("increment", "area", "combined"), start=1
    ):
        selected = (labels == class_label).astype(np.int64)
        changed = _condition_stream_values(
            streams,
            identities,
            selected,
            manifest=manifest,
            config=config,
            condition=MessageCondition(message_type, template, amplitude),
            representation=representation,
        )
        result[labels == class_label] = changed[labels == class_label]
    return result


def run_four_class_condition(
    streams: np.ndarray,
    *,
    data_path: Path,
    cache: Path,
    out: Path,
    config: dict,
    template: str,
    amplitude: float,
    count: int,
) -> dict:
    """Classify original, increment, area, and combined message streams."""
    manifest = _load_manifest(cache, data_path, count)
    split = classification_split(
        count,
        seed=int(config["splits"]["main_seed"]),
        proportions=config["splits"]["proportions"],
        classes=4,
    )
    output = {}
    for representation in config["analysis"]["detection_representations"]:
        train_x = _multiclass_representation(
            streams,
            split["train_indices"],
            split["train_labels"],
            cache=cache,
            manifest=manifest,
            config=config,
            template=template,
            amplitude=amplitude,
            representation=representation,
        )
        test_x = _multiclass_representation(
            streams,
            split["test_indices"],
            split["test_labels"],
            cache=cache,
            manifest=manifest,
            config=config,
            template=template,
            amplitude=amplitude,
            representation=representation,
        )
        if representation == "coordinate":
            for metric in config["analysis"]["coordinate_detection_metrics"]:
                output[metric] = _classification_result(
                    train_x,
                    split["train_labels"],
                    test_x,
                    split["test_labels"],
                    kind=metric,
                    config=config,
                )
        else:
            response = _parse_response_representation(representation, config)
            if response is not None:
                kind = "feature_l1"
            elif representation in {
                "global",
                "local_aligned",
                "local_offset",
                "signature_kernel_truncated",
            }:
                kind = "feature_squared"
            else:
                kind = representation
            output[representation] = _classification_result(
                train_x,
                split["train_labels"],
                test_x,
                split["test_labels"],
                kind=kind,
                config=config,
            )
    payload = {
        "template": template,
        "amplitude": amplitude,
        "count": count,
        "data_sha256": manifest["data_sha256"],
        "area_pairs": manifest["area_pairs"],
        "split": {name: value.tolist() for name, value in split.items()},
        "scales": manifest["scales"],
        "class_order": ["unmodified", "increment", "area", "combined"],
        "detection": output,
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "result.json", payload)
    return payload


def _confusion_matrix(truth: np.ndarray, prediction: np.ndarray) -> list[list[int]]:
    labels = np.unique(np.concatenate((truth, prediction)))
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    mapping = {label: index for index, label in enumerate(labels)}
    for wanted, actual in zip(truth, prediction):
        matrix[mapping[wanted], mapping[actual]] += 1
    return matrix.tolist()


def acceptance_checks(
    small_streams: np.ndarray,
    *,
    config: dict,
) -> dict:
    """Algebraic and shape checks that do not require optional libraries."""
    data = config["data"]
    message = config["message"]
    width = int(data["brownian_width"])
    steps = int(data["steps"])
    pairs = tuple(tuple(pair) for pair in data["area_pairs"])
    expected_lie = width + len(pairs)
    checks: dict[str, bool] = {}
    measurements: dict[str, float | list] = {}
    checks["shape"] = small_streams.ndim == 3 and small_streams.shape[1:] == (
        steps,
        expected_lie,
    )
    checks["finite"] = bool(np.isfinite(small_streams).all())
    checks["basis_dimensions"] = len(pairs) == width * (width - 1) // 2
    first_std = np.std(small_streams[..., :width], axis=(0, 1), dtype=np.float64)
    area_std = np.std(small_streams[..., width:], axis=(0, 1), dtype=np.float64)
    expected_first = 1.0 / math.sqrt(steps)
    expected_area = 1.0 / (2.0 * steps)
    measurements["first_coordinate_std"] = first_std.tolist()
    measurements["area_coordinate_std"] = area_std.tolist()
    first_relative = np.abs(first_std / expected_first - 1.0)
    area_relative = np.abs(area_std / expected_area - 1.0)
    checks["first_coordinate_scales"] = bool(
        np.all(first_relative <= config["acceptance"]["first_scale_relative_tolerance"])
    )
    checks["area_coordinate_scales"] = bool(
        np.all(area_relative <= config["acceptance"]["area_scale_relative_tolerance"])
    )

    sigma_first = 1.0 / math.sqrt(steps)
    sigma_area = 1.0 / (2.0 * steps)
    first, area = message_arrays(
        message_type="combined",
        template="net",
        amplitude=0.0,
        support_length=int(message["support_length"]),
        sigma_first=sigma_first,
        sigma_area=sigma_area,
        first_direction=message["first_direction"],
        area_direction=message["area_direction"],
    )
    unchanged = bch_insert_message(
        small_streams,
        start=int(message["support_start"]),
        first_message=first,
        area_message=area,
        width=width,
        area_pairs=pairs,
    )
    checks["rho_zero_bitwise"] = bool(np.array_equal(unchanged, small_streams))

    first, area = message_arrays(
        message_type="area",
        template="balanced",
        amplitude=4.0,
        support_length=int(message["support_length"]),
        sigma_first=sigma_first,
        sigma_area=sigma_area,
        first_direction=message["first_direction"],
        area_direction=message["area_direction"],
    )
    changed = bch_insert_message(
        small_streams,
        start=int(message["support_start"]),
        first_message=first,
        area_message=area,
        width=width,
        area_pairs=pairs,
    )
    checks["area_message_preserves_first_level"] = bool(
        np.array_equal(changed[..., :width], small_streams[..., :width])
    )
    difference_rows = np.any(changed != small_streams, axis=(0, 2))
    wanted_rows = np.zeros(steps, dtype=bool)
    wanted_rows[
        int(message["support_start"]) : int(message["support_start"])
        + int(message["support_length"])
    ] = True
    checks["support_only"] = bool(np.array_equal(difference_rows, wanted_rows))

    start = int(message["support_start"])
    stop = start + int(message["support_length"])
    original_support = small_streams[:, start:stop]
    changed_support = changed[:, start:stop]
    original_depth2 = rough_signature(
        original_support, 2, width=width, area_pairs=pairs
    )
    changed_depth2 = rough_signature(changed_support, 2, width=width, area_pairs=pairs)
    aligned_error = max(
        float(np.max(np.abs(first_level - second_level)))
        for first_level, second_level in zip(original_depth2, changed_depth2)
    )
    measurements["balanced_area_aligned_depth2_max_abs"] = aligned_error
    tolerance = float(config["acceptance"]["absolute_tolerance"])
    checks["balanced_area_aligned_depth2_zero"] = aligned_error <= tolerance

    offset = int(config["signature"]["local_offset"])
    block = int(config["signature"]["local_block_size"])
    offset_original = local_rough_signatures(
        small_streams,
        block_size=block,
        offset=offset,
        depth=2,
        width=width,
        area_pairs=pairs,
    )
    offset_changed = local_rough_signatures(
        changed,
        block_size=block,
        offset=offset,
        depth=2,
        width=width,
        area_pairs=pairs,
    )
    offset_response = float(
        np.median(np.sum((offset_original[1] - offset_changed[1]) ** 2, axis=(-2, -1)))
    )
    measurements["balanced_area_offset_depth2_median"] = offset_response
    checks["balanced_area_offset_depth2_positive"] = offset_response > tolerance**2

    rng = np.random.default_rng(20260906)
    tiny = rng.normal(scale=0.01, size=(8, expected_lie))
    tiny[:, width:] *= 0.01
    direct = rough_signature(tiny[None], 2, width=width, area_pairs=pairs)
    segments = segment_signature(tiny[None], 2, width=width, area_pairs=pairs)
    reduced = _positive_levels(chen_reduce(segments))
    reduction_error = max(
        float(np.max(np.abs(first_level - second_level)))
        for first_level, second_level in zip(direct, reduced)
    )
    measurements["balanced_reduction_max_abs"] = reduction_error
    checks["balanced_reduction_consistent"] = reduction_error <= tolerance
    kernel_self = truncated_signature_kernel_distance(
        direct,
        direct,
        level_weight=float(_truncated_kernel_config(config).get("level_weight", 1.0)),
    )
    measurements["truncated_kernel_self_distance_max"] = float(np.max(kernel_self))
    checks["truncated_kernel_self_distance_zero"] = bool(
        np.max(kernel_self) <= tolerance
    )
    response_self = response_signature_distance(
        direct,
        direct,
        response_bound=1.0,
    )
    measurements["response_signature_self_distance_max"] = float(np.max(response_self))
    checks["response_signature_self_distance_zero"] = bool(
        np.max(response_self) <= tolerance
    )
    return {
        "checks": checks,
        "measurements": measurements,
        "passed": all(checks.values()),
        "optional_reference_check": {
            "status": "pending",
            "reason": "RoughPy or RoughPy-JAX is not installed",
        },
        "optional_rough_kernel_check": {
            "status": "pending",
            "reason": "rough-path signature-kernel backend is not configured",
        },
    }


def roughpy_reference_checks(
    small_streams: np.ndarray,
    *,
    config: dict,
) -> dict:
    """Compare tensor signatures against independently maintained RoughPy."""
    try:
        import roughpy as rp
    except ImportError as error:
        raise RuntimeError(
            "RoughPy reference check requires optional package roughpy"
        ) from error

    data = config["data"]
    message = config["message"]
    width = int(data["brownian_width"])
    depth = int(config["signature"]["global_depth"])
    pairs = tuple(tuple(pair) for pair in data["area_pairs"])
    sample = np.asarray(
        small_streams[0, : int(message["support_length"])], dtype=np.float64
    )
    sigma_first = 1.0 / math.sqrt(int(data["steps"]))
    sigma_area = 1.0 / (2.0 * int(data["steps"]))
    first, area = message_arrays(
        message_type="combined",
        template="balanced",
        amplitude=1.0,
        support_length=len(sample),
        sigma_first=sigma_first,
        sigma_area=sigma_area,
        first_direction=message["first_direction"],
        area_direction=message["area_direction"],
    )
    modified = bch_insert_message(
        sample,
        start=0,
        first_message=first,
        area_message=area,
        width=width,
        area_pairs=pairs,
    )
    context = rp.get_context(width, depth, rp.DPReal)

    def reference_signature(increments: np.ndarray) -> np.ndarray:
        count = len(increments)
        stream = rp.LieIncrementStream.from_increments(
            increments,
            indices=np.arange(count, dtype=np.float64) / count,
            ctx=context,
            resolution=math.ceil(math.log2(count)),
        )
        return np.asarray(stream.signature(rp.RealInterval(0.0, 1.0))).reshape(-1)

    comparisons = {}
    for name, increments in (("original", sample), ("modified", modified)):
        ours = rough_signature(increments[None], depth, width=width, area_pairs=pairs)
        ours_flat = np.concatenate(
            (np.ones(1, dtype=np.float64), *(level[0] for level in ours))
        )
        reference = reference_signature(increments)
        if reference.shape != ours_flat.shape:
            raise ValueError(
                f"RoughPy tensor basis size {reference.shape} differs from {ours_flat.shape}"
            )
        comparisons[name] = float(np.max(np.abs(reference - ours_flat)))
    tolerance = float(config["acceptance"].get("reference_tolerance", 1.0e-9))
    basis = [str(key) for key in context.lie_basis]
    checks = {
        "basis_dimension": len(basis) >= width + len(pairs),
        "original_signature": comparisons["original"] <= tolerance,
        "modified_signature": comparisons["modified"] <= tolerance,
    }
    return {
        "checks": checks,
        "measurements": comparisons,
        "basis_prefix": basis[:10],
        "tolerance": tolerance,
        "passed": all(checks.values()),
    }


def aggregate_condition_results(root: Path) -> dict:
    """Collect condition JSON files and determine binary transition amplitudes."""
    records = []
    for path in sorted(root.glob("conditions/*/*/rho-*/result.json")):
        records.append(json.loads(path.read_text()))
    if not records:
        raise FileNotFoundError("no condition results found")
    transitions = []
    representations = sorted(
        {
            name
            for record in records
            if record.get("detection")
            for name in record["detection"]
        }
    )
    null_controls = []
    failed_null_representations = set()
    for representation in representations:
        zero_records = [
            record
            for record in records
            if float(record["condition"]["amplitude"]) == 0.0
            and record.get("detection", {}).get(representation)
        ]
        if not zero_records:
            continue
        values = zero_records[0]["detection"][representation]
        control = {
            "representation": representation,
            "balanced_accuracy": values["balanced_accuracy"],
            "balanced_accuracy_interval": values["balanced_accuracy_interval"],
            "roc_auc": values.get("roc_auc"),
            "passed": values["balanced_accuracy_interval"][0] <= 0.5,
        }
        null_controls.append(control)
        if not control["passed"]:
            failed_null_representations.add(representation)
    for template in sorted({r["condition"]["template"] for r in records}):
        for message_type in sorted({r["condition"]["message_type"] for r in records}):
            subset = sorted(
                (
                    r
                    for r in records
                    if r["condition"]["template"] == template
                    and r["condition"]["message_type"] == message_type
                ),
                key=lambda value: value["condition"]["amplitude"],
            )
            for representation in representations:
                if representation in failed_null_representations:
                    continue
                eligible = [
                    r
                    for r in subset
                    if float(r["condition"]["amplitude"]) > 0.0
                    if r.get("detection", {}).get(representation)
                    and r["detection"][representation]["balanced_accuracy_interval"][0]
                    > 0.5
                ]
                if eligible:
                    chosen = eligible[0]
                    index = subset.index(chosen)
                    preceding = subset[max(0, index - 1)]["condition"]["amplitude"]
                    transitions.append(
                        {
                            "template": template,
                            "message_type": message_type,
                            "representation": representation,
                            "amplitude": chosen["condition"]["amplitude"],
                            "preceding_amplitude": preceding,
                        }
                    )
    summary = {
        "condition_count": len(records),
        "null_controls": null_controls,
        "null_control_failures": sorted(failed_null_representations),
        "transitions": transitions,
    }
    _write_json(root / "summary.json", summary)
    return summary
