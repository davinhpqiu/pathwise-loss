"""Tests for fixed archive classification utilities."""

from __future__ import annotations

import numpy as np
import pytest

from pathloss.classification import (
    classification_scores,
    classify_1nn,
    preprocess_classification,
    standardise_per_series,
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


def test_no_preprocessing_preserves_archive_values():
    train = np.array([[[0.0, 2.0]]])
    test = np.array([[[3.0, 5.0]]])
    train_out, test_out, metadata = preprocess_classification(train, test, "none")
    np.testing.assert_array_equal(train_out, train)
    np.testing.assert_array_equal(test_out, test)
    assert metadata == {"kind": "none"}


def test_per_series_standardisation_uses_each_time_axis():
    values = np.array(
        [
            [[0.0, 1.0, 2.0], [10.0, 12.0, 14.0]],
            [[100.0, 102.0, 104.0], [-3.0, 0.0, 3.0]],
        ]
    )
    scaled = standardise_per_series(values)
    np.testing.assert_allclose(scaled.mean(axis=2), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled.std(axis=2), 1.0, atol=1e-12)


def test_unknown_preprocessing_is_rejected():
    values = np.zeros((1, 1, 2))
    with pytest.raises(ValueError, match="normalisation"):
        preprocess_classification(values, values, "unknown")


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


def test_multivariate_dtw_variants_are_named_explicitly():
    train_x = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[10.0, 10.0], [10.0, 10.0]],
        ]
    )
    train_y = np.array(["low", "high"])
    test_x = np.array([[[0.1, 0.0], [0.0, 0.1]]])
    test_y = np.array(["low"])
    for distance in ("dtw_dependent", "dtw_independent"):
        result = classify_1nn(
            train_x, train_y, test_x, test_y, distance=distance
        )
        assert result["distance"] == distance
        assert result["scores"]["accuracy"] == 1.0
