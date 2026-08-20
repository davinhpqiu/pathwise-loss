"""Fixed 1-nearest-neighbour evaluation for labelled path archives."""

from __future__ import annotations

import numpy as np
from aeon.datasets import load_from_ts_file
from aeon.distances import pairwise_distance

__all__ = [
    "classification_scores",
    "classify_1nn",
    "load_ts_split",
    "preprocess_classification",
    "standardise_per_series",
    "standardise_from_training",
]


def load_ts_split(path) -> tuple[np.ndarray, np.ndarray]:
    """Load one UCR/UEA ``.ts`` split as (cases, channels, time)."""
    x, y = load_from_ts_file(path)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if x.ndim != 3:
        raise ValueError(f"expected equal-length 3-D archive data, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("classification baseline requires complete finite series")
    return x, y


def standardise_from_training(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Standardise each channel using training cases and times only."""
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    if train.ndim != 3 or test.ndim != 3:
        raise ValueError("train and test must have shape (cases, channels, time)")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train and test channel counts differ")
    mean = train.mean(axis=(0, 2), keepdims=True)
    scale = train.std(axis=(0, 2), keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    metadata = {
        "mean": mean.reshape(-1).tolist(),
        "scale": scale.reshape(-1).tolist(),
    }
    return (train - mean) / scale, (test - mean) / scale, metadata


def standardise_per_series(values: np.ndarray) -> np.ndarray:
    """Standardise each channel of each series over its own time axis."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise ValueError("values must have shape (cases, channels, time)")
    mean = values.mean(axis=2, keepdims=True)
    scale = values.std(axis=2, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    return (values - mean) / scale


def preprocess_classification(
    train: np.ndarray,
    test: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply one explicit preprocessing rule to archive splits."""
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    if train.ndim != 3 or test.ndim != 3:
        raise ValueError("train and test must have shape (cases, channels, time)")
    if train.shape[1] != test.shape[1]:
        raise ValueError("train and test channel counts differ")

    if method == "none":
        return train.copy(), test.copy(), {"kind": "none"}
    if method == "training_channel":
        train_scaled, test_scaled, statistics = standardise_from_training(train, test)
        return train_scaled, test_scaled, {
            "kind": "training_channel",
            **statistics,
        }
    if method == "per_series":
        return (
            standardise_per_series(train),
            standardise_per_series(test),
            {"kind": "per_series", "axis": "time"},
        )
    raise ValueError(
        "normalisation must be one of: none, training_channel, per_series"
    )


def classification_scores(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Accuracy and macro-averaged class recall."""
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if truth.shape != prediction.shape:
        raise ValueError("truth and prediction shapes differ")
    classes = np.unique(truth)
    recalls = [float(np.mean(prediction[truth == label] == label)) for label in classes]
    return {
        "accuracy": float(np.mean(prediction == truth)),
        "balanced_accuracy": float(np.mean(recalls)),
    }


def classify_1nn(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    distance: str,
    n_jobs: int = 1,
    **distance_kwargs,
) -> dict:
    """Classify official test cases using a fixed 1-NN rule."""
    if distance == "dtw_dependent":
        distances = pairwise_distance(
            test_x,
            train_x,
            method="dtw",
            symmetric=False,
            n_jobs=n_jobs,
            **distance_kwargs,
        )
    elif distance == "dtw_independent":
        distances = sum(
            pairwise_distance(
                test_x[:, channel, :],
                train_x[:, channel, :],
                method="dtw",
                symmetric=False,
                n_jobs=n_jobs,
                **distance_kwargs,
            )
            for channel in range(train_x.shape[1])
        )
    else:
        distances = pairwise_distance(
            test_x,
            train_x,
            method=distance,
            symmetric=False,
            n_jobs=n_jobs,
            **distance_kwargs,
        )
    nearest = np.argmin(distances, axis=1)
    prediction = np.asarray(train_y)[nearest]
    return {
        "distance": distance,
        "distance_kwargs": distance_kwargs,
        "scores": classification_scores(test_y, prediction),
        "prediction": prediction.tolist(),
        "nearest_train_index": nearest.tolist(),
    }
