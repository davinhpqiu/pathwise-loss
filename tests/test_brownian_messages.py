import json
import math

import numpy as np
import pytest

from pathloss.brownian_message_study import (
    prepare_feature_cache,
    roughpy_reference_checks,
    run_message_condition,
)
from pathloss.brownian_messages import (
    AREA_PAIRS,
    MessageCondition,
    bch_insert_message,
    binary_roc_auc,
    chen_product_numpy,
    classification_split,
    coordinate_discrepancies,
    embed_area,
    local_rough_signatures,
    message_arrays,
    nearest_neighbour,
    reconstruct_path,
    response_signature_distance,
    segment_signature,
    truncated_signature_features,
    truncated_signature_kernel_distance,
)


def test_embed_area_is_antisymmetric():
    area = np.arange(1.0, 7.0)[None]
    tensor = embed_area(area).reshape(1, 4, 4)
    np.testing.assert_allclose(tensor + tensor.swapaxes(-1, -2), 0.0)
    np.testing.assert_allclose(tensor[0][np.triu_indices(4, 1)], area[0])


def test_segment_signature_matches_closed_form_for_straight_segment():
    increment = np.zeros((1, 10), dtype=np.float64)
    increment[0, :4] = [0.2, -0.3, 0.1, 0.4]
    levels = segment_signature(increment, 4)
    first = increment[:, :4]
    tensor = first
    for level in range(1, 5):
        if level > 1:
            tensor = (tensor[..., :, None] * first[..., None, :]).reshape(1, 4**level)
        np.testing.assert_allclose(levels[level], tensor / math.factorial(level))


def test_pure_area_segment_has_only_even_tensor_terms_through_depth_four():
    increment = np.zeros((1, 10), dtype=np.float64)
    increment[0, 4] = 0.25
    levels = segment_signature(increment, 4)
    area = embed_area(increment[:, 4:])
    np.testing.assert_allclose(levels[1], 0.0)
    np.testing.assert_allclose(levels[2], area)
    np.testing.assert_allclose(levels[3], 0.0)
    expected_four = 0.5 * (area[..., :, None] * area[..., None, :]).reshape(1, 256)
    np.testing.assert_allclose(levels[4], expected_four)


def test_bch_message_step_two_exponential_equals_concatenation():
    rng = np.random.default_rng(14)
    original = rng.normal(scale=0.1, size=(3, 1, 10))
    original[..., 4:] *= 0.1
    first = rng.normal(scale=0.03, size=(1, 4))
    area = rng.normal(scale=0.003, size=(1, 6))
    combined = bch_insert_message(
        original,
        start=0,
        first_message=first,
        area_message=area,
    )
    left = segment_signature(original[..., 0, :], 2)
    message = np.concatenate((first, area), axis=1)
    right = segment_signature(np.broadcast_to(message, (3, 10)), 2)
    concatenated = chen_product_numpy(left, right)
    wanted = segment_signature(combined[..., 0, :], 2)
    for actual, expected in zip(concatenated, wanted):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_balanced_area_cancels_aligned_depth_two_but_not_offset_blocks():
    rng = np.random.default_rng(8)
    values = rng.normal(scale=0.01, size=(5, 32, 10))
    values[..., 4:] *= 0.01
    first, area = message_arrays(
        message_type="area",
        template="balanced",
        amplitude=2.0,
        support_length=8,
        sigma_first=0.1,
        sigma_area=0.01,
        first_direction=[1, 0, 0, 0],
        area_direction=[1, 0, 0, 0, 0, 0],
    )
    changed = bch_insert_message(
        values,
        start=16,
        first_message=first,
        area_message=area,
    )
    original_aligned = local_rough_signatures(values, block_size=8, offset=0)
    changed_aligned = local_rough_signatures(changed, block_size=8, offset=0)
    for actual, expected in zip(original_aligned, changed_aligned):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    original_offset = local_rough_signatures(values, block_size=8, offset=4)
    changed_offset = local_rough_signatures(changed, block_size=8, offset=4)
    assert np.max(np.abs(original_offset[1] - changed_offset[1])) > 0.0


def test_response_signature_distance_uses_level_weights_and_block_average():
    first = (
        np.array([[[1.0, -2.0], [3.0, 0.0]]]),
        np.array([[[1.0], [-1.0]]]),
    )
    second = tuple(np.zeros_like(level) for level in first)
    value = response_signature_distance(
        first, second, response_bound=2.0, average_blocks=True
    )
    np.testing.assert_allclose(value, [10.0])


def test_normalized_truncated_signature_kernel_distance_has_closed_value():
    first = (np.array([[1.0, 0.0]]),)
    second = (np.array([[-1.0, 0.0]]),)
    features = truncated_signature_features(first, normalize=True)
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0)
    np.testing.assert_allclose(
        truncated_signature_kernel_distance(first, first), 0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        truncated_signature_kernel_distance(first, second), math.sqrt(2.0)
    )


def test_reconstruction_and_coordinate_discrepancies_have_closed_values():
    first = np.zeros((1, 4, 10))
    second = first.copy()
    second[:, 1, 0] = 1.0
    path_first = reconstruct_path(first)
    path_second = reconstruct_path(second)
    metrics = coordinate_discrepancies(path_first, path_second)
    np.testing.assert_allclose(path_second[0, :, 0], [0, 0, 1, 1, 1])
    np.testing.assert_allclose(metrics["mse"], 3 / 5)
    np.testing.assert_allclose(metrics["j2"], 5 / 8)
    np.testing.assert_allclose(metrics["l1"], 5 / 8)
    np.testing.assert_allclose(metrics["linf"], 1.0)


def test_classification_split_is_disjoint_and_balanced():
    split = classification_split(100, seed=3, classes=2)
    identity_sets = [
        set(split[f"{name}_indices"].tolist())
        for name in ("train", "validation", "test")
    ]
    assert identity_sets[0].isdisjoint(identity_sets[1])
    assert identity_sets[0].isdisjoint(identity_sets[2])
    assert identity_sets[1].isdisjoint(identity_sets[2])
    for name in ("train", "validation", "test"):
        labels, counts = np.unique(split[f"{name}_labels"], return_counts=True)
        np.testing.assert_array_equal(labels, [0, 1])
        assert counts[0] == counts[1]


def test_nearest_neighbour_and_binary_auc():
    train = np.array([[0.0], [10.0], [1.0], [11.0]])
    labels = np.array([0, 1, 0, 1])
    test = np.array([[0.2], [10.2]])
    result = nearest_neighbour(
        train,
        labels,
        test,
        kind="feature_squared",
        query_batch=1,
        reference_batch=2,
    )
    np.testing.assert_array_equal(result["prediction"], [0, 1])
    score = result["class_distance"][:, 0] - result["class_distance"][:, 1]
    assert binary_roc_auc(np.array([0, 1]), score) == 1.0
    l1_result = nearest_neighbour(
        train,
        labels,
        test,
        kind="feature_l1",
        query_batch=1,
        reference_batch=2,
    )
    np.testing.assert_array_equal(l1_result["prediction"], [0, 1])


def _small_config(path, count=20):
    return {
        "data": {
            "expected_paths": count,
            "steps": 32,
            "duration": 1.0,
            "dtype": "float32",
            "brownian_width": 4,
            "area_pairs": [list(pair) for pair in AREA_PAIRS],
            "statistics_batch_size": 5,
        },
        "message": {
            "support_start": 16,
            "support_length": 8,
            "first_direction": [1.0, 0.0, 0.0, 0.0],
            "area_direction": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "splits": {"proportions": [0.6, 0.2, 0.2], "main_seed": 4},
        "signature": {
            "global_depth": 2,
            "local_depth": 2,
            "local_block_size": 8,
            "local_offset": 4,
            "response_bounds": [0.5, 1.0, 2.0],
            "truncated_kernel": {"enabled": True, "level_weight": 1.0},
            "scale_delta": 1e-12,
            "prepare_batch_size": 5,
            "condition_batch_size": 5,
        },
        "analysis": {
            "derangement_seed": 5,
            "bootstrap_seed": 6,
            "bootstrap_replicates": 10,
            "nn_query_batch": 2,
            "nn_reference_batch": 4,
            "detection_representations": [
                "global",
                "local_offset",
                "response_global_r1",
                "signature_kernel_truncated",
            ],
            "coordinate_detection_metrics": ["mse"],
        },
    }


def test_small_cache_and_condition_pipeline(tmp_path):
    count = 20
    rng = np.random.default_rng(40)
    values = rng.normal(scale=0.05, size=(count, 32, 10)).astype("float32")
    values[..., 4:] *= 0.05
    data_path = tmp_path / "streams.npy"
    np.save(data_path, values)
    config = _small_config(data_path, count)
    cache = tmp_path / "cache"
    manifest = prepare_feature_cache(
        values,
        data_path=data_path,
        cache=cache,
        config=config,
        count=count,
    )
    assert manifest["complete"]
    assert np.load(cache / "global_features.npy").shape == (20, 20)
    result = run_message_condition(
        values,
        data_path=data_path,
        cache=cache,
        out=tmp_path / "condition",
        config=config,
        condition=MessageCondition("area", "net", 1.0),
        count=count,
        include_detection=True,
    )
    assert result["paired"]["mse"]["median"] == 0.0
    assert result["paired"]["area"]["median"] > 0.0
    assert result["paired"]["response_global_r1"]["median"] > 0.0
    assert result["paired"]["signature_kernel_truncated"]["median"] > 0.0
    assert set(result["detection"]) == {
        "global",
        "local_offset",
        "response_global_r1",
        "signature_kernel_truncated",
    }
    assert json.loads((tmp_path / "condition" / "result.json").read_text())


def test_roughpy_reference_when_available():
    pytest.importorskip("roughpy")
    rng = np.random.default_rng(12)
    values = rng.normal(scale=0.01, size=(2, 32, 10)).astype("float32")
    values[..., 4:] *= 0.01
    config = _small_config(None, 2)
    config["acceptance"] = {"reference_tolerance": 1e-8}
    result = roughpy_reference_checks(values, config=config)
    assert result["passed"]
