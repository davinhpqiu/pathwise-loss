"""Tests for fixed archive classification utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pathloss.classification import (
    classification_scores,
    classify_1nn,
    standardise_from_training,
)


def test_standardisation_uses_training_statistics():
    train = np.array([[[0.0, 2.0]], [[2.0, 4.0]]])
    test = np.array([[[100.0, 100.0]]])
    train_scaled, test_scaled, metadata = standardise_from_training(train, test)
    assert train_scaled.mean() == pytest.approx(0.0)
    assert train_scaled.std() == pytest.approx(1.0)
    assert metadata["mean"] == [2.0]
    assert test_scaled.mean() > 50.0


def test_scores_include_balanced_accuracy():
    truth = np.array(["a", "a", "a", "b"])
    prediction = np.array(["a", "a", "b", "b"])
    scores = classification_scores(truth, prediction)
    assert scores["accuracy"] == pytest.approx(0.75)
    assert scores["balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)


def test_euclidean_1nn_on_separated_paths():
    train_x = np.array([[[0.0, 0.0]], [[10.0, 10.0]]])
    train_y = np.array(["low", "high"])
    test_x = np.array([[[0.1, -0.1]], [[9.9, 10.1]]])
    test_y = np.array(["low", "high"])
    result = classify_1nn(
        train_x, train_y, test_x, test_y, distance="euclidean"
    )
    assert result["scores"]["accuracy"] == 1.0
