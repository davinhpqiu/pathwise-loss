"""Fixed 1-nearest-neighbour evaluation for labelled path archives."""

from __future__ import annotations

import numpy as np
from aeon.datasets import load_from_ts_file
from aeon.distances import pairwise_distance

__all__ = [
    "classification_scores",
    "classify_1nn",
    "load_ts_split",
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
