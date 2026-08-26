"""Differentiable signatures and signature losses for piecewise-linear paths.

Tensor levels use flattened lexicographic word order. Paths have shape
``(..., time, channel)``. Implementation uses Chen multiplication, so gradients
flow from every signature coordinate to path values through ordinary PyTorch
operations.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

__all__ = [
    "chen_product",
    "piecewise_linear_signature",
    "signature_feature_count",
    "time_augmented_path",
    "anchored_coordinate_mean_components",
    "anchored_coordinate_mean_signature_loss",
]


def _validate_path(path: torch.Tensor, *, name: str) -> None:
    if path.ndim < 2:
        raise ValueError(f"{name} must have shape (..., time, channel)")
    if path.shape[-2] < 2:
        raise ValueError(f"{name} must contain at least two times")
    if path.shape[-1] < 1:
        raise ValueError(f"{name} must contain at least one channel")
    if not torch.is_floating_point(path):
        raise TypeError(f"{name} must have floating dtype")


def _tensor_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Flattened tensor product over final coordinate axes."""
    if left.shape[:-1] != right.shape[:-1]:
        raise ValueError("tensor-product batch shapes must match")
    return (left.unsqueeze(-1) * right.unsqueeze(-2)).flatten(-2)


def _segment_signature(increment: torch.Tensor, depth: int) -> tuple[torch.Tensor, ...]:
    """Levels zero to ``depth`` for one straight segment."""
    levels = [increment.new_ones(increment.shape[:-1] + (1,))]
    for level in range(1, depth + 1):
        levels.append(_tensor_product(levels[-1], increment) / float(level))
    return tuple(levels)


def chen_product(
    left: Sequence[torch.Tensor], right: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """Truncated signature of concatenation from two signatures.

    Inputs contain levels zero through ``depth``. Level zero is a final axis of
    length one. Formula is Chen identity

    ``combined[k] = sum(left[j] tensor right[k-j], j=0,...,k)``.
    """
    if len(left) != len(right) or len(left) < 1:
        raise ValueError("left and right must contain same nonzero number of levels")
    combined = []
    for level in range(len(left)):
        terms = [
            _tensor_product(left[j], right[level - j])
            for j in range(level + 1)
        ]
        combined.append(torch.stack(terms, dim=0).sum(dim=0))
    return tuple(combined)


def piecewise_linear_signature(
    path: torch.Tensor,
    depth: int,
    *,
    include_level_zero: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Signature levels of piecewise-linear interpolation through ``path``."""
    _validate_path(path, name="path")
    if depth < 1:
        raise ValueError(f"depth must be positive, got {depth}")
    channel = path.shape[-1]
    batch_shape = path.shape[:-2]
    levels = [path.new_ones(batch_shape + (1,))]
    levels.extend(
        path.new_zeros(batch_shape + (channel**k,))
        for k in range(1, depth + 1)
    )
    signature = tuple(levels)
    increments = path[..., 1:, :] - path[..., :-1, :]
    for index in range(increments.shape[-2]):
        signature = chen_product(
            signature,
            _segment_signature(increments[..., index, :], depth),
        )
    return signature if include_level_zero else signature[1:]


def signature_feature_count(channel: int, depth: int, *, intervals: int = 1) -> int:
    """Number of positive-level coordinates in a truncated representation."""
    if channel < 1 or depth < 1 or intervals < 1:
        raise ValueError("channel, depth and intervals must be positive")
    return intervals * sum(channel**level for level in range(1, depth + 1))


def _output_scale(
    path: torch.Tensor,
    scale: float | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(scale, device=path.device, dtype=path.dtype)
    if value.ndim > 1 or (value.ndim == 1 and value.numel() not in (1, path.shape[-1])):
        raise ValueError(
            "output_scale must be scalar or have one value per output channel"
        )
    if bool(torch.any(~torch.isfinite(value))) or bool(torch.any(value <= 0)):
        raise ValueError("output_scale must contain finite positive values")
    return value


def time_augmented_path(
    time: torch.Tensor,
    path: torch.Tensor,
    *,
    output_scale: float | Sequence[float] | torch.Tensor = 1.0,
    time_origin: torch.Tensor | float | None = None,
    time_span: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Return ``(normalised time, scaled output)`` with path batch dimensions."""
    _validate_path(path, name="path")
    if time.ndim != 1 or time.numel() != path.shape[-2]:
        raise ValueError("time must be one-dimensional and match path length")
    if bool(torch.any(time[1:] <= time[:-1])):
        raise ValueError("time must be strictly increasing")
    origin = time[0] if time_origin is None else torch.as_tensor(
        time_origin, device=time.device, dtype=time.dtype
    )
    span = time[-1] - time[0] if time_span is None else torch.as_tensor(
        time_span, device=time.device, dtype=time.dtype
    )
    if not bool(torch.isfinite(span)) or bool(span <= 0):
        raise ValueError("time_span must be finite and positive")
    normalised_time = ((time - origin) / span).to(device=path.device, dtype=path.dtype)
    view_shape = (1,) * len(path.shape[:-2]) + (path.shape[-2], 1)
    normalised_time = normalised_time.reshape(view_shape).expand(path.shape[:-1] + (1,))
    scaled_path = path / _output_scale(path, output_scale)
    return torch.cat((normalised_time, scaled_path), dim=-1)


def _interpolate_at(
    time: torch.Tensor,
    path: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Value of piecewise-linear ``path`` at one fixed query time."""
    index = int(torch.searchsorted(time, query, right=False))
    if index == 0:
        return path[..., 0, :]
    if index == time.numel():
        return path[..., -1, :]
    if bool(time[index] == query):
        return path[..., index, :]
    left = index - 1
    weight = ((query - time[left]) / (time[index] - time[left])).to(path.dtype)
    return path[..., left, :] + weight * (path[..., index, :] - path[..., left, :])


def _restrict_path(
    time: torch.Tensor,
    path: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict piecewise-linear path to closed interval, inserting endpoints."""
    interior = (time > start) & (time < end)
    restricted_time = torch.cat((start.reshape(1), time[interior], end.reshape(1)))
    restricted_path = torch.cat(
        (
            _interpolate_at(time, path, start).unsqueeze(-2),
            path[..., interior, :],
            _interpolate_at(time, path, end).unsqueeze(-2),
        ),
        dim=-2,
    )
    return restricted_time, restricted_path


def anchored_coordinate_mean_components(
    time: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    depth: int,
    intervals: int = 1,
    output_scale: float | Sequence[float] | torch.Tensor = 1.0,
) -> dict[str, torch.Tensor]:
    """Anchor and level terms of global or equally partitioned signature loss."""
    _validate_path(prediction, name="prediction")
    _validate_path(target, name="target")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if time.ndim != 1 or time.numel() != prediction.shape[-2]:
        raise ValueError("time must be one-dimensional and match path length")
    if depth < 1 or intervals < 1:
        raise ValueError("depth and intervals must be positive")
    scale = _output_scale(prediction, output_scale)
    anchor = ((prediction[..., 0, :] - target[..., 0, :]) / scale).square().mean(dim=-1)
    level_terms = [prediction.new_zeros(prediction.shape[:-2]) for _ in range(depth)]
    time_origin = time[0]
    time_span = time[-1] - time[0]

    for interval in range(intervals):
        start = time_origin + time_span * (interval / intervals)
        end = time_origin + time_span * ((interval + 1) / intervals)
        local_time, local_prediction = _restrict_path(time, prediction, start, end)
        _, local_target = _restrict_path(time, target, start, end)
        augmented_prediction = time_augmented_path(
            local_time,
            local_prediction,
            output_scale=scale,
            time_origin=time_origin,
            time_span=time_span,
        )
        augmented_target = time_augmented_path(
            local_time,
            local_target,
            output_scale=scale,
            time_origin=time_origin,
            time_span=time_span,
        )
        prediction_signature = piecewise_linear_signature(augmented_prediction, depth)
        target_signature = piecewise_linear_signature(augmented_target, depth)
        channel = augmented_prediction.shape[-1]
        for level, (predicted, wanted) in enumerate(
            zip(prediction_signature, target_signature), start=1
        ):
            level_terms[level - 1] = level_terms[level - 1] + (
                (predicted - wanted).square().sum(dim=-1) / (channel**level)
            )

    components = {"anchor": anchor}
    components.update(
        {
            f"level_{level}": value / intervals
            for level, value in enumerate(level_terms, start=1)
        }
    )
    return components


def anchored_coordinate_mean_signature_loss(
    time: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    depth: int,
    intervals: int = 1,
    output_scale: float | Sequence[float] | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Mean anchored coordinate loss, reduced across path batch dimensions."""
    components = anchored_coordinate_mean_components(
        time,
        prediction,
        target,
        depth=depth,
        intervals=intervals,
        output_scale=output_scale,
    )
    per_path = torch.stack(tuple(components.values()), dim=0).sum(dim=0)
    return per_path.mean()
