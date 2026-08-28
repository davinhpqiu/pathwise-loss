"""Independent checks for fixed-path Neural ODE experiment A."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch", reason="torch is in requirements-ml.txt")

from pathloss.fixed_path import (  # noqa: E402
    FixedPathTrainConfig,
    LatentNeuralODE,
    fixed_path_loss,
    fixed_target,
    fixed_target_derivative,
    h1_balance,
    make_paired_model,
    observation_times,
    state_fingerprint,
    train_fixed_path,
)
from pathloss.losses import sobolev_h1  # noqa: E402


def test_fixed_target_has_prespecified_value_at_event_centre():
    t = torch.tensor([0.25], dtype=torch.float64)
    got = fixed_target(t)[0]
    assert got[0].item() == pytest.approx(-0.3, abs=1e-12)
    assert got[1].item() == pytest.approx(1.0, abs=1e-12)


def test_fixed_target_derivative_has_closed_form_at_event_centre():
    t = torch.tensor([0.25], dtype=torch.float64)
    got = fixed_target_derivative(t)[0]
    assert got[0].item() == pytest.approx(-2.0 * math.pi, abs=1e-12)
    assert got[1].item() == pytest.approx(-3.6 * math.pi, abs=1e-12)


def test_observation_grids_match_definition_and_starve_local_event():
    n = 64
    uniform = observation_times(n, "uniform", dtype=torch.float64)
    clustered = observation_times(n, "clustered", dtype=torch.float64)
    u = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    assert torch.equal(uniform, u)
    assert torch.allclose(clustered, 1.0 - (1.0 - u) ** 3)
    assert uniform[0] == clustered[0] == 0.0
    assert uniform[-1] == clustered[-1] == 1.0
    assert torch.all(uniform[1:] > uniform[:-1])
    assert torch.all(clustered[1:] > clustered[:-1])
    uniform_event = ((uniform >= 0.15) & (uniform <= 0.35)).sum()
    clustered_event = ((clustered >= 0.15) & (clustered <= 0.35)).sum()
    assert clustered_event < uniform_event


def test_h1_balance_is_fixed_by_target_and_positive():
    t = observation_times(64, "uniform", dtype=torch.float64)
    target = fixed_target(t)
    first = h1_balance(t, target)
    second = h1_balance(t, target)
    assert torch.equal(first, second)
    assert 0.0 < float(first) < 1.0


def constant_velocity_model() -> LatentNeuralODE:
    model = LatentNeuralODE(
        hidden=2, width=4, output_dim=2, n_fourier=0, max_step=0.01
    ).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.initial.copy_(torch.tensor([1.0, -2.0], dtype=torch.float64))
        model.vector_net[-1].bias.copy_(torch.tensor([0.5, 1.25], dtype=torch.float64))
        model.decoder.weight.copy_(torch.eye(2, dtype=torch.float64))
    return model


def test_rk4_matches_constant_velocity_closed_form():
    model = constant_velocity_model()
    t = torch.tensor([0.0, 0.2, 0.73, 1.0], dtype=torch.float64)
    prediction, derivative = model.forward_with_derivative(t)
    velocity = torch.tensor([0.5, 1.25], dtype=torch.float64)
    want = torch.tensor([1.0, -2.0], dtype=torch.float64) + t[:, None] * velocity
    assert torch.allclose(prediction, want, atol=1e-12, rtol=1e-12)
    assert torch.allclose(derivative, velocity.expand_as(derivative), atol=1e-12)


def test_fixed_path_model_reaches_every_parameter_by_gradient():
    torch.manual_seed(5)
    model = LatentNeuralODE(hidden=3, width=7, n_fourier=2, max_step=1.0 / 16.0)
    t = torch.linspace(0.0, 1.0, 17)
    loss = (model(t) - fixed_target(t)).square().mean()
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_paired_initialization_has_exact_fingerprint():
    kwargs = dict(hidden=2, width=8, n_fourier=2, max_step=1.0 / 32.0)
    first = make_paired_model(7, **kwargs)
    second = make_paired_model(7, **kwargs)
    third = make_paired_model(8, **kwargs)
    assert state_fingerprint(first) == state_fingerprint(second)
    assert state_fingerprint(first) != state_fingerprint(third)


def test_sobolev_loss_matches_constant_derivative_closed_form():
    t = torch.linspace(0.0, 1.0, 33, dtype=torch.float64)[None]
    target = torch.zeros(1, 33, 2, dtype=torch.float64)
    derivative = torch.ones_like(target)
    got = sobolev_h1(t, target, target, derivative, target, rho=0.5)
    assert got.item() == pytest.approx(1.0, abs=1e-12)


def test_h1_requires_derivatives_and_is_uniform_only():
    t = observation_times(8)
    target = fixed_target(t)
    with pytest.raises(ValueError, match="requires both derivatives"):
        fixed_path_loss("h1", t, target, target, rho=1.0)
    with pytest.raises(ValueError, match="uniform-observation"):
        train_fixed_path(
            FixedPathTrainConfig(
                condition="clustered",
                loss="h1",
                updates=1,
                hidden=1,
                width=1,
                n_fourier=0,
                max_step=1.0,
            )
        )


@pytest.mark.parametrize("loss", ["sig_global", "sig_local"])
def test_initial_signature_comparison_is_uniform_only(loss: str):
    with pytest.raises(ValueError, match="uniform observations"):
        train_fixed_path(
            FixedPathTrainConfig(
                condition="clustered",
                loss=loss,
                updates=1,
                hidden=1,
                width=1,
                n_fourier=0,
                max_step=1.0,
            )
        )


def test_training_records_common_metrics_at_requested_checkpoints():
    result = train_fixed_path(
        FixedPathTrainConfig(
            updates=2,
            evaluation_checkpoints=(0, 1, 2),
            n_target=4,
            n_fine=5,
            hidden=1,
            width=2,
            n_fourier=0,
            max_step=1.0,
        )
    )
    assert [row["updates_completed"] for row in result["checkpoints"]] == [0, 1, 2]
    assert set(result["checkpoint_predictions"]) == {0, 1, 2}
    for row in result["checkpoints"]:
        assert set(row["metrics"]) == {
            "mse",
            "j2",
            "h1",
            "linf",
            "sig_global",
            "sig_local",
            "local_j2",
        }
        assert all(math.isfinite(value) for value in row["metrics"].values())
