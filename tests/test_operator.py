"""Implementation checks for Brownian-to-OU path-output operator learning."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is in requirements-ml.txt")

from pathloss.fixed_path import state_fingerprint
from pathloss.operator import (
    OUOperatorTrainConfig,
    evaluate_ou_operator,
    interpolate_shared_grid,
    make_ou_operator_model,
    make_ou_operator_splits,
    operator_loss,
    ou_dynamics_residual,
    ou_signature_gradient_audit,
    train_ou_operator,
)


def tiny_config(**kwargs) -> OUOperatorTrainConfig:
    base = OUOperatorTrainConfig(
        n_train=8,
        n_val=4,
        n_test=4,
        n_steps=8,
        n_target=5,
        hidden=4,
        width=8,
        epochs=1,
        batch_size=4,
    )
    return replace(base, **kwargs)


def test_shared_grid_interpolation_matches_linear_closed_form():
    time = torch.tensor([0.0, 0.5, 1.0])
    values = torch.tensor([[[0.0], [1.0], [0.0]], [[1.0], [2.0], [3.0]]])
    query = torch.tensor([0.0, 0.25, 0.75, 1.0])
    got = interpolate_shared_grid(time, values, query)
    want = torch.tensor([[[0.0], [0.5], [0.5], [0.0]], [[1.0], [1.5], [2.5], [3.0]]])
    assert torch.allclose(got, want)


def test_path_output_neural_cde_shape_and_all_parameter_gradients():
    cfg = tiny_config()
    data = make_ou_operator_splits(cfg)["train"]
    model = make_ou_operator_model(cfg)
    query = torch.linspace(0.0, 1.0, 5)
    output = model(data["time"], data["driver"], query)
    assert output.shape == (8, 5, 1)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_path_output_neural_cde_is_causal_and_driver_sensitive():
    cfg = tiny_config(seed=4)
    data = make_ou_operator_splits(cfg)["train"]
    model = make_ou_operator_model(cfg)
    changed = data["driver"].clone()
    split = 4
    changed[:, split + 1 :] += 0.75
    original = model.forward_fine(data["time"], data["driver"])
    perturbed = model.forward_fine(data["time"], changed)
    assert torch.equal(original[:, : split + 1], perturbed[:, : split + 1])
    assert not torch.allclose(original[:, split + 1 :], perturbed[:, split + 1 :])


def test_operator_losses_are_differentiable_on_shared_target_grid():
    time = torch.tensor([0.0, 0.1, 0.4, 1.0])
    target = torch.randn(3, 4, 1)
    for name in ("mse", "j2", "sig_global", "sig_local"):
        prediction = torch.zeros_like(target, requires_grad=True)
        loss = operator_loss(
            name,
            time,
            prediction,
            target,
            signature_output_scale=1.0,
            signature_global_depth=4,
            signature_local_depth=2,
            signature_local_intervals=2,
        )
        loss.backward()
        assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_true_ou_path_has_zero_dynamics_residual_to_float_tolerance():
    cfg = tiny_config()
    data = make_ou_operator_splits(cfg)["train"]
    residual = ou_dynamics_residual(
        data["time"],
        data["driver"],
        data["target"],
        lambd=cfg.lambd,
        sigma=cfg.sigma,
    )
    assert float(residual) < 1.0e-13


def test_operator_split_initialization_and_training_order_reproduce():
    cfg = tiny_config(seed=7)
    first_data = make_ou_operator_splits(cfg)
    second_data = make_ou_operator_splits(cfg)
    assert torch.equal(first_data["train"]["driver"], second_data["train"]["driver"])
    assert torch.equal(first_data["train"]["target"], second_data["train"]["target"])
    assert state_fingerprint(make_ou_operator_model(cfg)) == state_fingerprint(
        make_ou_operator_model(cfg)
    )
    first = train_ou_operator(cfg, verbose=False)
    second = train_ou_operator(cfg, verbose=False)
    assert first["target_time_fingerprint"] == second["target_time_fingerprint"]
    assert first["order_fingerprint"] == second["order_fingerprint"]
    assert first["history"] == second["history"]


def test_evaluation_reports_independent_and_signature_metrics():
    cfg = tiny_config()
    split = make_ou_operator_splits(cfg)["val"]
    model = make_ou_operator_model(cfg)
    target_time = torch.linspace(0.0, 1.0, cfg.n_target)
    metrics, paths = evaluate_ou_operator(
        model,
        split,
        target_time,
        lambd=cfg.lambd,
        sigma=cfg.sigma,
        signature_output_scale=float(split["target"].std(unbiased=False)),
        keep_paths=2,
    )
    assert set(metrics) == {
        "target_mse",
        "target_j2",
        "target_sig_global",
        "target_sig_local",
        "fine_mse",
        "fine_j1",
        "fine_j2",
        "fine_j4",
        "fine_linf",
        "sig_global",
        "sig_local",
        "dynamics_residual",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert paths["driver"].shape[0] == 2


def test_ou_signature_audit_matches_thirty_coordinate_design():
    report = ou_signature_gradient_audit(tiny_config(seed=991))
    assert report["scale_rule"] == "training_target_std"
    assert report["representations"]["global"]["feature_count"] == 30
    assert report["representations"]["local"]["feature_count"] == 30
    for representation in report["representations"].values():
        for component in representation["components"].values():
            assert np.isfinite(component["value"])
            assert np.isfinite(component["parameter_gradient_l2"])
