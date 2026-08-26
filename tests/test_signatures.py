"""Independent checks for differentiable piecewise-linear signatures."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch is in requirements-ml.txt")

from pathloss.signatures import (  # noqa: E402
    anchored_coordinate_mean_components,
    anchored_coordinate_mean_signature_loss,
    chen_product,
    piecewise_linear_signature,
    signature_feature_count,
    time_augmented_path,
)
from pathloss.fixed_path import fixed_path_loss  # noqa: E402


def tensor_power(vector: torch.Tensor, level: int) -> torch.Tensor:
    result = vector.new_ones(1)
    for _ in range(level):
        result = torch.outer(result, vector).reshape(-1)
    return result


def test_straight_line_signature_matches_closed_form():
    increment = torch.tensor([0.5, -0.25, 1.0], dtype=torch.float64)
    path = torch.stack((torch.zeros_like(increment), increment / 3.0, increment))
    got = piecewise_linear_signature(path, depth=4)
    for level, value in enumerate(got, start=1):
        want = tensor_power(increment, level) / math.factorial(level)
        assert torch.allclose(value, want, atol=1e-12, rtol=1e-12)


def test_chen_concatenation_matches_direct_piecewise_linear_signature():
    path = torch.tensor(
        [[0.0, 0.0], [0.4, -0.2], [0.1, 0.8]], dtype=torch.float64
    )
    full = piecewise_linear_signature(path, depth=4, include_level_zero=True)
    first = piecewise_linear_signature(path[:2], depth=4, include_level_zero=True)
    second = piecewise_linear_signature(path[1:], depth=4, include_level_zero=True)
    combined = chen_product(first, second)
    for got, want in zip(combined, full):
        assert torch.allclose(got, want, atol=1e-12, rtol=1e-12)


def test_signature_matches_iisignature_reference_when_installed():
    iisignature = pytest.importorskip("iisignature")
    path = torch.tensor(
        [[0.0, 0.0], [0.2, 0.5], [0.7, -0.1], [1.0, 0.3]],
        dtype=torch.float64,
    )
    depth = 4
    got = torch.cat(piecewise_linear_signature(path, depth)).numpy()
    want = iisignature.sig(path.numpy(), depth)
    assert got == pytest.approx(want, abs=1e-11, rel=1e-11)


def test_time_augmentation_normalises_time_and_scales_outputs():
    time = torch.tensor([2.0, 3.0, 6.0], dtype=torch.float64)
    path = torch.tensor([[2.0, -3.0], [4.0, 1.0], [6.0, 5.0]], dtype=torch.float64)
    got = time_augmented_path(time, path, output_scale=torch.tensor([2.0, 4.0]))
    expected_time = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64)
    assert torch.allclose(got[:, 0], expected_time)
    assert torch.allclose(got[:, 1:], path / torch.tensor([2.0, 4.0]))


def test_translation_changes_anchor_but_not_signature_levels():
    time = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    target = torch.stack((torch.sin(time), time.square()), dim=-1)
    prediction = target + torch.tensor([2.0, -3.0], dtype=torch.float64)
    components = anchored_coordinate_mean_components(
        time, prediction, target, depth=3
    )
    assert components["anchor"].item() == pytest.approx(6.5, abs=1e-12)
    for level in range(1, 4):
        assert components[f"level_{level}"].item() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("intervals,depth", [(1, 4), (10, 2)])
def test_signature_loss_is_zero_on_identical_paths(intervals: int, depth: int):
    time = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    path = torch.stack((torch.cos(time), torch.sin(2.0 * time)), dim=-1)
    loss = anchored_coordinate_mean_signature_loss(
        time, path, path, depth=depth, intervals=intervals
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("intervals,depth", [(1, 4), (10, 2)])
def test_signature_loss_has_finite_nonzero_path_gradient(intervals: int, depth: int):
    time = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    target = torch.stack((torch.cos(time), torch.sin(2.0 * time)), dim=-1)
    perturbation = 0.05 * torch.stack((time, time.square()), dim=-1)
    prediction = (target + perturbation).requires_grad_()
    loss = anchored_coordinate_mean_signature_loss(
        time, prediction, target, depth=depth, intervals=intervals
    )
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert float(torch.linalg.vector_norm(prediction.grad)) > 0.0


def test_global_and_local_representations_store_120_coordinates():
    assert signature_feature_count(3, 4) == 120
    assert signature_feature_count(3, 2, intervals=10) == 120


def test_local_loss_inserts_partition_boundary_on_piecewise_linear_path():
    time = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64)
    target = torch.tensor(
        [[0.0, 0.0], [0.5, -0.2], [0.8, 0.7]], dtype=torch.float64
    )
    prediction = torch.tensor(
        [[0.1, -0.1], [0.4, 0.1], [1.0, 0.5]], dtype=torch.float64
    )
    weight = (0.5 - 0.37) / (1.0 - 0.37)
    target_mid = target[1] + weight * (target[2] - target[1])
    prediction_mid = prediction[1] + weight * (prediction[2] - prediction[1])
    refined_time = torch.tensor([0.0, 0.37, 0.5, 1.0], dtype=torch.float64)
    refined_target = torch.stack((target[0], target[1], target_mid, target[2]))
    refined_prediction = torch.stack(
        (prediction[0], prediction[1], prediction_mid, prediction[2])
    )
    coarse = anchored_coordinate_mean_signature_loss(
        time, prediction, target, depth=2, intervals=2
    )
    refined = anchored_coordinate_mean_signature_loss(
        refined_time, refined_prediction, refined_target, depth=2, intervals=2
    )
    assert torch.allclose(coarse, refined, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "name,depth,intervals", [("sig_global", 4, 1), ("sig_local", 2, 10)]
)
def test_fixed_path_registry_uses_specified_signature_loss(
    name: str, depth: int, intervals: int
):
    time = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    target = torch.stack((torch.cos(time), torch.sin(time)), dim=-1)
    prediction = target + 0.02 * torch.stack((time, time.square()), dim=-1)
    got = fixed_path_loss(name, time, prediction, target)
    want = anchored_coordinate_mean_signature_loss(
        time, prediction, target, depth=depth, intervals=intervals
    )
    assert torch.allclose(got, want, atol=1e-12, rtol=1e-12)
