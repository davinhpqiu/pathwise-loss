"""Controlled messages and discrepancies for supplied Brownian rough streams.

Input rows are step-two Lie increments: four first-level coordinates followed
by six antisymmetric area coordinates. Tensor signatures are obtained by
embedding each Lie increment in tensor algebra, exponentiating, and applying
Chen multiplication. No rough-path library is required for this core route.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "AREA_PAIRS",
    "MessageCondition",
    "bch_insert_message",
    "binary_roc_auc",
    "bootstrap_interval",
    "brownian_scales",
    "chen_product_numpy",
    "chen_reduce",
    "classification_split",
    "coordinate_discrepancies",
    "derangement",
    "distance_values",
    "embed_area",
    "file_sha256",
    "local_rough_signatures",
    "message_arrays",
    "message_conditions",
    "nearest_neighbour",
    "reconstruct_path",
    "resolve_data_path",
    "response_signature_distance",
    "rough_signature",
    "segment_signature",
    "truncated_signature_features",
    "truncated_signature_kernel_distance",
    "validate_stream_array",
]


AREA_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass(frozen=True)
class MessageCondition:
    """One fixed perturbation condition."""

    message_type: str
    template: str
    amplitude: float

    @property
    def slug(self) -> str:
        amplitude = format(self.amplitude, ".8g").replace(".", "p")
        return f"{self.template}/{self.message_type}/rho-{amplitude}"


def resolve_data_path(candidates: Sequence[str | Path]) -> Path:
    """Return first existing data path from an ordered candidate list."""
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file():
            return path
    rendered = ", ".join(str(Path(value).expanduser()) for value in candidates)
    raise FileNotFoundError(f"none of the configured data paths exists: {rendered}")


def file_sha256(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Compute SHA-256 without loading a large archive into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stream_array(
    values: np.ndarray,
    *,
    expected_paths: int | None = None,
    expected_steps: int = 2048,
    expected_width: int = 10,
    expected_dtype: str | np.dtype = "float32",
) -> None:
    """Validate shape, dtype, and finite values of rough increment data."""
    if values.ndim != 3:
        raise ValueError(f"stream array must have three axes, got {values.shape}")
    if expected_paths is not None and values.shape[0] != expected_paths:
        raise ValueError(f"expected {expected_paths} paths, found {values.shape[0]}")
    if values.shape[1:] != (expected_steps, expected_width):
        raise ValueError(
            f"expected trailing shape {(expected_steps, expected_width)}, "
            f"found {values.shape[1:]}"
        )
    if values.dtype != np.dtype(expected_dtype):
        raise TypeError(f"expected dtype {expected_dtype}, found {values.dtype}")
    for start in range(0, values.shape[0], 256):
        if not np.isfinite(values[start : start + 256]).all():
            raise ValueError("stream array contains a non-finite value")


def _validate_area_pairs(
    area_pairs: Sequence[Sequence[int]], width: int
) -> tuple[tuple[int, int], ...]:
    pairs = tuple((int(pair[0]), int(pair[1])) for pair in area_pairs)
    expected = width * (width - 1) // 2
    if len(pairs) != expected:
        raise ValueError(f"width {width} requires {expected} area pairs")
    if len(set(pairs)) != len(pairs):
        raise ValueError("area pairs contain duplicates")
    if any(left < 0 or right >= width or left >= right for left, right in pairs):
        raise ValueError("area pairs must satisfy 0 <= left < right < width")
    if set(pairs) != {
        (left, right) for left in range(width) for right in range(left + 1, width)
    }:
        raise ValueError("area pairs do not cover every antisymmetric coordinate")
    return pairs


def embed_area(
    area: np.ndarray,
    *,
    width: int = 4,
    area_pairs: Sequence[Sequence[int]] = AREA_PAIRS,
) -> np.ndarray:
    """Embed Lie-bracket coefficients as flattened antisymmetric tensors."""
    area = np.asarray(area)
    pairs = _validate_area_pairs(area_pairs, width)
    if area.shape[-1] != len(pairs):
        raise ValueError("last area axis does not match configured basis")
    result = np.zeros(area.shape[:-1] + (width, width), dtype=area.dtype)
    for index, (left, right) in enumerate(pairs):
        result[..., left, right] = area[..., index]
        result[..., right, left] = -area[..., index]
    return result.reshape(area.shape[:-1] + (width * width,))


def _outer_flat(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[:-1] != right.shape[:-1]:
        raise ValueError("tensor factors must have matching batch axes")
    return (left[..., :, None] * right[..., None, :]).reshape(
        left.shape[:-1] + (left.shape[-1] * right.shape[-1],)
    )


def segment_signature(
    lie_increments: np.ndarray,
    depth: int,
    *,
    width: int = 4,
    area_pairs: Sequence[Sequence[int]] = AREA_PAIRS,
) -> tuple[np.ndarray, ...]:
    """Tensor exponential of step-two Lie increments through depth four."""
    values = np.asarray(lie_increments)
    if depth < 1 or depth > 4:
        raise ValueError("implemented rough signature depth must lie in [1, 4]")
    expected = width + width * (width - 1) // 2
    if values.shape[-1] != expected:
        raise ValueError(f"expected {expected} Lie coordinates")
    first = values[..., :width]
    second_lie = embed_area(values[..., width:], width=width, area_pairs=area_pairs)
    levels = [np.ones(values.shape[:-1] + (1,), dtype=values.dtype), first]
    if depth == 1:
        return tuple(levels)

    first2 = _outer_flat(first, first)
    levels.append(second_lie + 0.5 * first2)
    if depth == 2:
        return tuple(levels)

    first3 = _outer_flat(first2, first)
    level3 = (
        0.5 * (_outer_flat(first, second_lie) + _outer_flat(second_lie, first))
        + first3 / 6.0
    )
    levels.append(level3)
    if depth == 3:
        return tuple(levels)

    first4 = _outer_flat(first3, first)
    level4 = 0.5 * _outer_flat(second_lie, second_lie)
    level4 += (
        _outer_flat(first2, second_lie)
        + _outer_flat(_outer_flat(first, second_lie), first)
        + _outer_flat(second_lie, first2)
    ) / 6.0
    level4 += first4 / 24.0
    levels.append(level4)
    return tuple(levels)


def chen_product_numpy(
    left: Sequence[np.ndarray], right: Sequence[np.ndarray]
) -> tuple[np.ndarray, ...]:
    """Truncated Chen product for flattened tensor levels."""
    if len(left) != len(right) or not left:
        raise ValueError("signatures must contain same nonzero level count")
    if left[0].shape[:-1] != right[0].shape[:-1]:
        raise ValueError("signature batch axes differ")
    result = []
    for level in range(len(left)):
        terms = [
            _outer_flat(left[index], right[level - index]) for index in range(level + 1)
        ]
        result.append(np.sum(np.stack(terms, axis=0), axis=0))
    return tuple(result)


def chen_reduce(levels: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Reduce a time axis of tensor signatures by balanced Chen products."""
    if not levels or levels[0].shape[-2] < 1:
        raise ValueError("signature sequence requires a nonempty time axis")
    current = tuple(np.asarray(level) for level in levels)
    if any(level.shape[:-1] != current[0].shape[:-1] for level in current):
        raise ValueError("signature levels have different batch or time axes")
    while current[0].shape[-2] > 1:
        count = current[0].shape[-2]
        paired = count // 2
        left = tuple(value[..., : 2 * paired : 2, :] for value in current)
        right = tuple(value[..., 1 : 2 * paired : 2, :] for value in current)
        combined = chen_product_numpy(left, right)
        if count % 2:
            current = tuple(
                np.concatenate((value, old[..., -1:, :]), axis=-2)
                for value, old in zip(combined, current)
            )
        else:
            current = combined
    return tuple(level[..., 0, :] for level in current)


def rough_signature(
    lie_increments: np.ndarray,
    depth: int,
    *,
    width: int = 4,
    area_pairs: Sequence[Sequence[int]] = AREA_PAIRS,
) -> tuple[np.ndarray, ...]:
    """Signature of one or a batch of rough paths by balanced Chen reduction.

    Return contains positive levels only. Input shape is ``(..., time, lie)``.
    Balanced reduction has logarithmic Python depth and is much faster than a
    Python loop over 2,048 increments.
    """
    values = np.asarray(lie_increments)
    if values.ndim < 2 or values.shape[-2] < 1:
        raise ValueError("rough increments require a nonempty time axis")
    levels = segment_signature(values, depth, width=width, area_pairs=area_pairs)
    return chen_reduce(levels)[1:]


def local_rough_signatures(
    lie_increments: np.ndarray,
    *,
    block_size: int,
    offset: int,
    depth: int = 2,
    width: int = 4,
    area_pairs: Sequence[Sequence[int]] = AREA_PAIRS,
) -> tuple[np.ndarray, ...]:
    """Signatures on complete fixed-width blocks after a fixed offset."""
    values = np.asarray(lie_increments)
    if block_size < 1 or offset < 0 or offset >= values.shape[-2]:
        raise ValueError("invalid block size or offset")
    block_count = (values.shape[-2] - offset) // block_size
    if block_count < 1:
        raise ValueError("partition contains no complete block")
    stopped = offset + block_count * block_size
    blocked = values[..., offset:stopped, :].reshape(
        values.shape[:-2] + (block_count, block_size, values.shape[-1])
    )
    return rough_signature(blocked, depth, width=width, area_pairs=area_pairs)


def response_signature_distance(
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
    *,
    response_bound: float,
    average_blocks: bool = False,
) -> np.ndarray:
    """Finite response-derived signature discrepancy on paired paths.

    Each input contains positive tensor levels with batch axis first. For local
    signatures, set ``average_blocks=True`` when axis one indexes blocks.
    """
    if len(first) != len(second) or not first:
        raise ValueError("signature tuples must have equal positive depth")
    if response_bound <= 0 or not np.isfinite(response_bound):
        raise ValueError("response bound must be finite and positive")
    total = None
    for level, (left, right) in enumerate(zip(first, second), start=1):
        left = np.asarray(left)
        right = np.asarray(right)
        if left.shape != right.shape or left.ndim < 2:
            raise ValueError("paired signature levels must have matching batch shape")
        axes = tuple(range(1, left.ndim))
        contribution = response_bound**level * np.sum(np.abs(left - right), axis=axes)
        if average_blocks:
            if left.ndim < 3:
                raise ValueError("local signature levels require a block axis")
            contribution = contribution / left.shape[1]
        total = contribution if total is None else total + contribution
    return np.asarray(total)


def truncated_signature_features(
    levels: Sequence[np.ndarray],
    *,
    level_weight: float = 1.0,
    include_scalar: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """Flatten a finite signature into its explicit tensor feature map."""
    if not levels:
        raise ValueError("at least one positive signature level is required")
    if level_weight <= 0 or not np.isfinite(level_weight):
        raise ValueError("level weight must be finite and positive")
    batch = np.asarray(levels[0]).shape[0]
    pieces = []
    if include_scalar:
        pieces.append(np.ones((batch, 1), dtype=np.asarray(levels[0]).dtype))
    for level, value in enumerate(levels, start=1):
        value = np.asarray(value)
        if value.shape[0] != batch or value.ndim < 2:
            raise ValueError("signature levels must share a nonempty batch axis")
        pieces.append(level_weight**level * value.reshape(batch, -1))
    features = np.concatenate(pieces, axis=1)
    if normalize:
        norms = np.linalg.norm(features.astype(np.float64), axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero feature cannot be kernel normalized")
        features = features / norms
    return features


def truncated_signature_kernel_distance(
    first: Sequence[np.ndarray],
    second: Sequence[np.ndarray],
    *,
    level_weight: float = 1.0,
) -> np.ndarray:
    """Distance induced by normalized finite linear signature kernel."""
    left = truncated_signature_features(
        first, level_weight=level_weight, include_scalar=True, normalize=True
    )
    right = truncated_signature_features(
        second, level_weight=level_weight, include_scalar=True, normalize=True
    )
    if left.shape != right.shape:
        raise ValueError("paired truncated signature features have different shape")
    squared = np.sum((left - right) ** 2, axis=1)
    return np.sqrt(np.maximum(squared, 0.0))


def _area_bracket(
    first: np.ndarray,
    message: np.ndarray,
    area_pairs: Sequence[Sequence[int]],
) -> np.ndarray:
    values = [
        first[..., left] * message[..., right] - first[..., right] * message[..., left]
        for left, right in area_pairs
    ]
    return np.stack(values, axis=-1)


def message_arrays(
    *,
    message_type: str,
    template: str,
    amplitude: float,
    support_length: int,
    sigma_first: float,
    sigma_area: float,
    first_direction: Sequence[float],
    area_direction: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """First-level and area message rows at natural Brownian scales."""
    if message_type not in {"increment", "area", "combined"}:
        raise ValueError("unknown message type")
    if template not in {"net", "balanced"}:
        raise ValueError("unknown temporal template")
    if amplitude < 0 or not np.isfinite(amplitude):
        raise ValueError("amplitude must be finite and nonnegative")
    if support_length < 1 or (template == "balanced" and support_length % 2):
        raise ValueError("balanced support length must be positive and even")
    signs = np.ones(support_length, dtype=np.float64)
    if template == "balanced":
        signs[support_length // 2 :] = -1.0
    first_direction = np.asarray(first_direction, dtype=np.float64)
    area_direction = np.asarray(area_direction, dtype=np.float64)
    if not np.isclose(np.linalg.norm(first_direction), 1.0):
        raise ValueError("first-level direction must have unit norm")
    if not np.isclose(np.linalg.norm(area_direction), 1.0):
        raise ValueError("area direction must have unit norm")
    first = amplitude * sigma_first * signs[:, None] * first_direction
    area = amplitude * sigma_area * signs[:, None] * area_direction
    if message_type == "increment":
        area.fill(0.0)
    elif message_type == "area":
        first.fill(0.0)
    return first, area


def bch_insert_message(
    lie_increments: np.ndarray,
    *,
    start: int,
    first_message: np.ndarray,
    area_message: np.ndarray,
    width: int = 4,
    area_pairs: Sequence[Sequence[int]] = AREA_PAIRS,
) -> np.ndarray:
    """Right-compose selected step-two Lie increments with message rows."""
    values = np.asarray(lie_increments)
    pairs = _validate_area_pairs(area_pairs, width)
    length = first_message.shape[0]
    if first_message.shape != (length, width):
        raise ValueError("first message has wrong shape")
    if area_message.shape != (length, len(pairs)):
        raise ValueError("area message has wrong shape")
    if start < 0 or start + length > values.shape[-2]:
        raise ValueError("message support lies outside stream")
    result = values.copy()
    original_first = values[..., start : start + length, :width]
    broadcast_shape = (1,) * (values.ndim - 2) + first_message.shape
    first = first_message.reshape(broadcast_shape)
    area = area_message.reshape((1,) * (values.ndim - 2) + area_message.shape)
    result[..., start : start + length, :width] = original_first + first
    result[..., start : start + length, width:] = (
        values[..., start : start + length, width:]
        + area
        + 0.5 * _area_bracket(original_first, first, pairs)
    )
    return result


def reconstruct_path(lie_increments: np.ndarray, *, width: int = 4) -> np.ndarray:
    """Reconstruct first-level coordinate path beginning at zero."""
    increments = np.asarray(lie_increments)[..., :width]
    zero = np.zeros(increments.shape[:-2] + (1, width), dtype=increments.dtype)
    return np.concatenate((zero, np.cumsum(increments, axis=-2)), axis=-2)


def coordinate_discrepancies(
    first: np.ndarray, second: np.ndarray, *, duration: float = 1.0
) -> dict[str, np.ndarray]:
    """Per-path coordinate MSE, trapezoidal L2/L1, and supremum errors."""
    first = np.asarray(first)
    second = np.asarray(second)
    if first.shape != second.shape or first.ndim < 2:
        raise ValueError("coordinate paths must have identical path shape")
    error_norm = np.linalg.norm(first - second, axis=-1)
    intervals = first.shape[-2] - 1
    if intervals < 1:
        raise ValueError("coordinate path needs at least two points")
    h = duration / intervals
    trapezoid_squared = h * (
        0.5 * error_norm[..., 0] ** 2
        + np.sum(error_norm[..., 1:-1] ** 2, axis=-1)
        + 0.5 * error_norm[..., -1] ** 2
    )
    trapezoid_absolute = h * (
        0.5 * error_norm[..., 0]
        + np.sum(error_norm[..., 1:-1], axis=-1)
        + 0.5 * error_norm[..., -1]
    )
    return {
        "mse": np.mean(error_norm**2, axis=-1),
        "j2": trapezoid_squared,
        "l1": trapezoid_absolute,
        "linf": np.max(error_norm, axis=-1),
    }


def _batched_moments(
    values: np.ndarray, indices: np.ndarray, columns: slice, batch_size: int
) -> tuple[np.ndarray, np.ndarray, int]:
    width = (
        values.shape[-1] if columns == slice(None) else values[..., columns].shape[-1]
    )
    total = np.zeros(width, dtype=np.float64)
    square = np.zeros(width, dtype=np.float64)
    count = 0
    for start in range(0, len(indices), batch_size):
        batch = np.asarray(values[indices[start : start + batch_size], :, columns])
        flat = batch.reshape(-1, width).astype(np.float64, copy=False)
        total += flat.sum(axis=0)
        square += np.square(flat).sum(axis=0)
        count += flat.shape[0]
    mean = total / count
    variance = np.maximum(square / count - mean**2, 0.0)
    return mean, np.sqrt(variance), count


def brownian_scales(
    values: np.ndarray,
    training_indices: Iterable[int],
    *,
    width: int = 4,
    batch_size: int = 128,
) -> dict[str, object]:
    """Training-only pooled RMS scales and coordinate standard deviations."""
    indices = np.asarray(tuple(training_indices), dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("training indices must be nonempty and one-dimensional")
    first_mean, first_std, _ = _batched_moments(
        values, indices, slice(0, width), batch_size
    )
    area_mean, area_std, _ = _batched_moments(
        values, indices, slice(width, None), batch_size
    )
    first_rms = math.sqrt(float(np.mean(first_std**2 + first_mean**2)))
    area_rms = math.sqrt(float(np.mean(area_std**2 + area_mean**2)))
    return {
        "sigma_first": first_rms,
        "sigma_area": area_rms,
        "first_mean": first_mean.tolist(),
        "first_std": first_std.tolist(),
        "area_mean": area_mean.tolist(),
        "area_std": area_std.tolist(),
    }


def message_conditions(config: dict) -> tuple[MessageCondition, ...]:
    """Ordered Cartesian product of configured templates, types, amplitudes."""
    conditions = tuple(
        MessageCondition(message_type, template, float(amplitude))
        for template in config["templates"]
        for message_type in config["types"]
        for amplitude in config["amplitudes"]
    )
    if len(conditions) != len(set(conditions)):
        raise ValueError("message configuration contains duplicate conditions")
    return conditions


def classification_split(
    count: int,
    *,
    seed: int,
    proportions: Sequence[float] = (0.6, 0.2, 0.2),
    classes: int = 2,
) -> dict[str, np.ndarray]:
    """Disjoint deterministic identities with balanced classes in each split."""
    if count < classes or classes < 2:
        raise ValueError("count must accommodate at least two classes")
    proportions = np.asarray(proportions, dtype=float)
    if proportions.shape != (3,) or np.any(proportions <= 0):
        raise ValueError("split proportions must contain three positive values")
    proportions = proportions / proportions.sum()
    first = round(count * proportions[0])
    second = round(count * proportions[1])
    sizes = [first, second, count - first - second]
    sizes = [size - size % classes for size in sizes]
    sizes[-1] = count - sizes[0] - sizes[1]
    sizes[-1] -= sizes[-1] % classes
    used = sum(sizes)
    if min(sizes) < classes:
        raise ValueError("each split must contain every class")
    permutation = np.random.default_rng(seed).permutation(count)[:used]
    result: dict[str, np.ndarray] = {}
    cursor = 0
    for name, size in zip(("train", "validation", "test"), sizes):
        identities = permutation[cursor : cursor + size]
        labels = np.tile(np.arange(classes, dtype=np.int64), size // classes)
        result[f"{name}_indices"] = identities
        result[f"{name}_labels"] = labels
        cursor += size
    return result


def derangement(count: int, *, seed: int) -> np.ndarray:
    """Deterministic random permutation with no fixed points."""
    if count < 2:
        raise ValueError("derangement requires at least two items")
    rng = np.random.default_rng(seed)
    values = np.arange(count)
    for _ in range(100):
        candidate = rng.permutation(count)
        if np.all(candidate != values):
            return candidate
    shift = int(rng.integers(1, count))
    return np.roll(values, shift)


def _feature_squared_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.sum((first - second) ** 2, axis=-1)


def distance_values(
    first: np.ndarray,
    second: np.ndarray,
    *,
    kind: str,
    duration: float = 1.0,
) -> np.ndarray:
    """Paired discrepancy values for feature, coordinate, or Lie arrays."""
    if first.shape != second.shape:
        raise ValueError("paired inputs must have identical shape")
    if kind in {"mse", "j2", "l1", "linf"}:
        return coordinate_discrepancies(first, second, duration=duration)[kind]
    if kind in {"area", "increment"}:
        return np.mean(np.sum((first - second) ** 2, axis=-1), axis=-1)
    if kind == "feature_l1":
        return np.sum(np.abs(first - second), axis=-1)
    if kind.startswith("feature"):
        return _feature_squared_distance(first, second)
    raise ValueError(f"unknown discrepancy kind {kind!r}")


def _distance_matrix(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    kind: str,
    duration: float,
) -> np.ndarray:
    if kind == "feature_l1":
        difference = query[:, None, ...] - reference[None, :, ...]
        return np.sum(np.abs(difference), axis=tuple(range(2, difference.ndim)))
    if kind.startswith("feature") or kind in {"mse", "j2", "area", "increment"}:
        if kind == "j2":
            intervals = query.shape[-2] - 1
            weights = np.full(query.shape[-2], duration / intervals)
            weights[[0, -1]] *= 0.5
            query = query * np.sqrt(weights)[None, :, None]
            reference = reference * np.sqrt(weights)[None, :, None]
            normalizer = 1.0
        elif kind in {"mse", "area", "increment"}:
            normalizer = float(query.shape[-2])
        else:
            normalizer = 1.0
        query_flat = query.reshape(query.shape[0], -1).astype(np.float64, copy=False)
        ref_flat = reference.reshape(reference.shape[0], -1).astype(
            np.float64, copy=False
        )
        result = (
            np.sum(query_flat**2, axis=1)[:, None]
            + np.sum(ref_flat**2, axis=1)[None, :]
            - 2.0 * query_flat @ ref_flat.T
        ) / normalizer
        return np.maximum(result, 0.0)
    difference = query[:, None, ...] - reference[None, :, ...]
    if kind in {"mse", "j2", "l1", "linf"}:
        norm = np.linalg.norm(difference, axis=-1)
        intervals = query.shape[-2] - 1
        h = duration / intervals
        if kind == "l1":
            return h * (
                0.5 * norm[..., 0]
                + np.sum(norm[..., 1:-1], axis=-1)
                + 0.5 * norm[..., -1]
            )
        return np.max(norm, axis=-1)
    raise ValueError(f"unknown distance kind {kind!r}")


def nearest_neighbour(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    kind: str,
    duration: float = 1.0,
    query_batch: int = 16,
    reference_batch: int = 128,
) -> dict[str, np.ndarray]:
    """Exact chunked 1-NN prediction and nearest distance in each class."""
    train_x = np.asarray(train_x)
    test_x = np.asarray(test_x)
    train_y = np.asarray(train_y)
    if train_x.shape[1:] != test_x.shape[1:] or len(train_y) != len(train_x):
        raise ValueError("1-NN feature or label shapes disagree")
    classes = np.unique(train_y)
    nearest_index = np.full(len(test_x), -1, dtype=np.int64)
    nearest_distance = np.full(len(test_x), np.inf, dtype=np.float64)
    class_distance = np.full((len(test_x), len(classes)), np.inf, dtype=np.float64)
    for query_start in range(0, len(test_x), query_batch):
        query_stop = min(query_start + query_batch, len(test_x))
        query = test_x[query_start:query_stop]
        best = nearest_distance[query_start:query_stop]
        best_index = nearest_index[query_start:query_stop]
        best_class = class_distance[query_start:query_stop]
        for ref_start in range(0, len(train_x), reference_batch):
            ref_stop = min(ref_start + reference_batch, len(train_x))
            distances = _distance_matrix(
                query,
                train_x[ref_start:ref_stop],
                kind=kind,
                duration=duration,
            )
            local = np.argmin(distances, axis=1)
            local_distance = distances[np.arange(len(query)), local]
            improve = local_distance < best
            best[improve] = local_distance[improve]
            best_index[improve] = ref_start + local[improve]
            for class_index, label in enumerate(classes):
                mask = train_y[ref_start:ref_stop] == label
                if np.any(mask):
                    best_class[:, class_index] = np.minimum(
                        best_class[:, class_index], np.min(distances[:, mask], axis=1)
                    )
        nearest_distance[query_start:query_stop] = best
        nearest_index[query_start:query_stop] = best_index
        class_distance[query_start:query_stop] = best_class
    return {
        "prediction": train_y[nearest_index],
        "nearest_index": nearest_index,
        "nearest_distance": nearest_distance,
        "class_distance": class_distance,
        "classes": classes,
    }


def binary_roc_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Binary ROC AUC with average ranks for tied scores."""
    truth = np.asarray(truth)
    score = np.asarray(score, dtype=float)
    if truth.shape != score.shape or set(np.unique(truth)) != {0, 1}:
        raise ValueError("ROC AUC requires matching binary labels and scores")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=float)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    positives = truth == 1
    positive_count = int(np.sum(positives))
    negative_count = len(truth) - positive_count
    return float(
        (np.sum(ranks[positives]) - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def bootstrap_interval(
    values: np.ndarray,
    *,
    statistic=np.median,
    replicates: int = 1000,
    seed: int = 20260905,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval over path identities."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap values must be a nonempty vector")
    if replicates < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sample = values[rng.integers(0, len(values), size=len(values))]
        estimates[index] = statistic(sample)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, (tail, 1.0 - tail))
    return float(low), float(high)
