"""Paired stream-to-stream operator learning for Brownian drivers and OU targets.

Experiment B supplies a complete Brownian control path on a fixed grid. A
path-output Neural CDE evolves causally along that control and decodes its hidden
state throughout the interval. Training changes output-path discrepancy only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn

from .datasets import brownian_ou_pairs
from .fixed_path import observation_times, state_fingerprint
from .losses import integral_lp, pointwise_mse
from .signatures import (
    anchored_coordinate_mean_components,
    anchored_coordinate_mean_signature_loss,
    signature_feature_count,
)

__all__ = [
    "OUOperatorTrainConfig",
    "PathOutputNeuralCDE",
    "evaluate_ou_operator",
    "interpolate_shared_grid",
    "make_ou_operator_model",
    "make_ou_operator_splits",
    "operator_loss",
    "ou_dynamics_residual",
    "ou_signature_gradient_audit",
    "run_ou_acceptance",
    "train_ou_operator",
]


@dataclass(frozen=True)
class OUOperatorTrainConfig:
    """One Brownian-to-OU fit, including data and optimization settings."""

    seed: int = 0
    device: str = "cpu"
    condition: str = "uniform"
    loss: str = "mse"
    n_train: int = 512
    n_val: int = 128
    n_test: int = 256
    n_steps: int = 256
    n_target: int = 64
    T: float = 1.0
    lambd: float = 2.0
    sigma: float = 0.5
    y0: float = 0.0
    train_data_seed: int = 12001
    val_data_seed: int = 12002
    test_data_seed: int = 12003
    hidden: int = 16
    width: int = 64
    epochs: int = 500
    batch_size: int = 64
    lr: float = 1.0e-3
    signature_global_depth: int = 4
    signature_local_depth: int = 2
    signature_local_intervals: int = 5


def _validate_config(cfg: OUOperatorTrainConfig) -> None:
    if cfg.condition not in {"uniform", "clustered"}:
        raise ValueError("condition must be uniform or clustered")
    if cfg.loss not in {"mse", "j2", "sig_global", "sig_local"}:
        raise ValueError(f"unknown operator loss {cfg.loss!r}")
    if min(cfg.n_train, cfg.n_val, cfg.n_steps, cfg.n_target) < 1:
        raise ValueError("data sizes and grid sizes must be positive")
    if cfg.n_target < 2 or cfg.n_steps < 1:
        raise ValueError("n_target must be at least 2 and n_steps positive")
    if cfg.n_test < 0 or cfg.hidden < 1 or cfg.width < 1:
        raise ValueError("n_test must be non-negative; model sizes must be positive")
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.lr <= 0:
        raise ValueError("epochs, batch_size and learning rate must be positive")


def _tensor_fingerprint(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def interpolate_shared_grid(
    time: torch.Tensor,
    values: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Piecewise-linear interpolation on one grid shared by a path batch."""
    if time.ndim != 1 or query.ndim != 1:
        raise ValueError("time and query must be one-dimensional")
    if values.ndim != 3 or values.shape[1] != time.numel():
        raise ValueError("values must have shape (batch, len(time), channel)")
    if time.numel() < 2 or bool(torch.any(time[1:] <= time[:-1])):
        raise ValueError("time must contain at least two strictly increasing values")
    if bool(torch.any(query < time[0])) or bool(torch.any(query > time[-1])):
        raise ValueError("query times must lie inside time grid")

    right = torch.searchsorted(time, query, right=False).clamp(1, time.numel() - 1)
    left = right - 1
    weight = (query - time[left]) / (time[right] - time[left])
    weight = weight.to(values.dtype).reshape(1, -1, 1)
    return values[:, left, :] + weight * (values[:, right, :] - values[:, left, :])


def _to_tensor_split(
    data: dict[str, np.ndarray], device: str
) -> dict[str, torch.Tensor]:
    return {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in data.items()
    }


def make_ou_operator_splits(
    cfg: OUOperatorTrainConfig,
) -> dict[str, dict[str, torch.Tensor]]:
    """Generate deterministic train, validation and optional test pairs."""
    _validate_config(cfg)
    common = {
        "n_steps": cfg.n_steps,
        "T": cfg.T,
        "lambd": cfg.lambd,
        "sigma": cfg.sigma,
        "y0": cfg.y0,
    }
    splits = {
        "train": _to_tensor_split(
            brownian_ou_pairs(cfg.n_train, rng=cfg.train_data_seed, **common),
            cfg.device,
        ),
        "val": _to_tensor_split(
            brownian_ou_pairs(cfg.n_val, rng=cfg.val_data_seed, **common),
            cfg.device,
        ),
    }
    if cfg.n_test:
        splits["test"] = _to_tensor_split(
            brownian_ou_pairs(cfg.n_test, rng=cfg.test_data_seed, **common),
            cfg.device,
        )
    return splits


class PathOutputNeuralCDE(nn.Module):
    """Causal hidden path driven by piecewise-linear control ``(time, W)``.

    On interval ``[t_m,t_{m+1}]``, one RK4 step integrates the control-linear
    equation after reparameterising interval to unit length. Its vector field is
    ``V_0(h) * dt + V_1(h) * dW``. Decoder is applied at every grid point.
    """

    def __init__(self, hidden: int = 16, width: int = 64) -> None:
        super().__init__()
        if hidden < 1 or width < 1:
            raise ValueError("hidden and width must be positive")
        self.hidden = int(hidden)
        self.initial = nn.Parameter(torch.zeros(self.hidden))
        self.vector_net = nn.Sequential(
            nn.Linear(self.hidden, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, self.hidden * 2),
        )
        self.decoder = nn.Linear(self.hidden, 1)

    def vector_fields(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 2 or h.shape[-1] != self.hidden:
            raise ValueError(f"h must have shape (batch, {self.hidden})")
        return self.vector_net(h).reshape(h.shape[0], self.hidden, 2)

    def _interval_increment(
        self, h: torch.Tensor, control_increment: torch.Tensor
    ) -> torch.Tensor:
        fields = self.vector_fields(h)
        return torch.einsum("bhc,bc->bh", fields, control_increment)

    def hidden_trajectory(
        self, time: torch.Tensor, driver: torch.Tensor
    ) -> torch.Tensor:
        if time.ndim != 1 or time.numel() < 2:
            raise ValueError(
                "time must be a one-dimensional grid with at least two points"
            )
        if bool(torch.any(time[1:] <= time[:-1])):
            raise ValueError("time must be strictly increasing")
        if driver.ndim != 3 or driver.shape[1:] != (time.numel(), 1):
            raise ValueError("driver must have shape (batch, len(time), 1)")
        if not torch.is_floating_point(driver):
            raise TypeError("driver must have floating dtype")

        batch = driver.shape[0]
        h = self.initial.unsqueeze(0).expand(batch, -1)
        states = [h]
        for step in range(time.numel() - 1):
            dt = (time[step + 1] - time[step]).to(driver.dtype)
            dW = driver[:, step + 1, 0] - driver[:, step, 0]
            increment = torch.stack((dt.expand_as(dW), dW), dim=-1)
            k1 = self._interval_increment(h, increment)
            k2 = self._interval_increment(h + k1 / 2.0, increment)
            k3 = self._interval_increment(h + k2 / 2.0, increment)
            k4 = self._interval_increment(h + k3, increment)
            h = h + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            states.append(h)
        return torch.stack(states, dim=1)

    def forward_fine(self, time: torch.Tensor, driver: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.hidden_trajectory(time, driver))

    def forward(
        self,
        time: torch.Tensor,
        driver: torch.Tensor,
        query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        fine = self.forward_fine(time, driver)
        return fine if query is None else interpolate_shared_grid(time, fine, query)


def make_ou_operator_model(cfg: OUOperatorTrainConfig) -> PathOutputNeuralCDE:
    """Deterministic model initialization shared across paired loss runs."""
    torch.manual_seed(cfg.seed)
    return PathOutputNeuralCDE(hidden=cfg.hidden, width=cfg.width).to(cfg.device)


def _target_times(cfg: OUOperatorTrainConfig, reference: torch.Tensor) -> torch.Tensor:
    return (
        observation_times(
            cfg.n_target,
            cfg.condition,
            device=reference.device,
            dtype=reference.dtype,
        )
        * cfg.T
    )


def _signature_scale(target: torch.Tensor) -> float:
    """Training-target standard deviation used for OU output-channel scaling."""
    scale = float(target.detach().std(unbiased=False))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "training targets must have positive finite standard deviation"
        )
    return scale


def operator_loss(
    name: str,
    time: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    signature_output_scale: float,
    signature_global_depth: int = 4,
    signature_local_depth: int = 2,
    signature_local_intervals: int = 5,
) -> torch.Tensor:
    """One Experiment B output-path discrepancy."""
    batch_time = time.unsqueeze(0).expand(prediction.shape[0], -1)
    if name == "mse":
        return pointwise_mse(batch_time, prediction, target)
    if name == "j2":
        return integral_lp(batch_time, prediction, target, p=2.0)
    if name == "sig_global":
        return anchored_coordinate_mean_signature_loss(
            time,
            prediction,
            target,
            depth=signature_global_depth,
            intervals=1,
            output_scale=signature_output_scale,
        )
    if name == "sig_local":
        return anchored_coordinate_mean_signature_loss(
            time,
            prediction,
            target,
            depth=signature_local_depth,
            intervals=signature_local_intervals,
            output_scale=signature_output_scale,
        )
    raise ValueError(f"unknown operator loss {name!r}")


def ou_dynamics_residual(
    time: torch.Tensor,
    driver: torch.Tensor,
    prediction: torch.Tensor,
    *,
    lambd: float,
    sigma: float,
) -> torch.Tensor:
    """Mean squared residual of known OU Euler recurrence."""
    dt = (time[1:] - time[:-1]).reshape(1, -1, 1)
    dW = driver[:, 1:, :] - driver[:, :-1, :]
    residual = (
        prediction[:, 1:, :]
        - prediction[:, :-1, :]
        + float(lambd) * prediction[:, :-1, :] * dt
        - float(sigma) * dW
    )
    return residual.square().mean()


@torch.no_grad()
def _evaluate_target_grid(
    model: PathOutputNeuralCDE,
    split: dict[str, torch.Tensor],
    target_time: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    prediction = model(split["time"], split["driver"], target_time)
    target = interpolate_shared_grid(split["time"], split["target"], target_time)
    batch_time = target_time.unsqueeze(0).expand(prediction.shape[0], -1)
    return {
        "mse": float(pointwise_mse(batch_time, prediction, target)),
        "j2": float(integral_lp(batch_time, prediction, target, p=2.0)),
    }


@torch.no_grad()
def evaluate_ou_operator(
    model: PathOutputNeuralCDE,
    split: dict[str, torch.Tensor],
    target_time: torch.Tensor,
    *,
    lambd: float,
    sigma: float,
    signature_output_scale: float,
    signature_global_depth: int = 4,
    signature_local_depth: int = 2,
    signature_local_intervals: int = 5,
    keep_paths: int = 0,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Target-grid and fine-grid metrics for one fitted OU operator."""
    model.eval()
    time = split["time"]
    prediction = model.forward_fine(time, split["driver"])
    target = split["target"]
    prediction_target = interpolate_shared_grid(time, prediction, target_time)
    target_observed = interpolate_shared_grid(time, target, target_time)
    observed_time = target_time.unsqueeze(0).expand(prediction.shape[0], -1)
    fine_time = time.unsqueeze(0).expand(prediction.shape[0], -1)
    metrics = {
        "target_mse": float(
            pointwise_mse(observed_time, prediction_target, target_observed)
        ),
        "target_j2": float(
            integral_lp(observed_time, prediction_target, target_observed, p=2.0)
        ),
        "fine_mse": float(pointwise_mse(fine_time, prediction, target)),
        "fine_j1": float(integral_lp(fine_time, prediction, target, p=1.0)),
        "fine_j2": float(integral_lp(fine_time, prediction, target, p=2.0)),
        "fine_j4": float(integral_lp(fine_time, prediction, target, p=4.0)),
        "fine_linf": float(integral_lp(fine_time, prediction, target, p=float("inf"))),
        "sig_global": float(
            anchored_coordinate_mean_signature_loss(
                time,
                prediction,
                target,
                depth=signature_global_depth,
                intervals=1,
                output_scale=signature_output_scale,
            )
        ),
        "sig_local": float(
            anchored_coordinate_mean_signature_loss(
                time,
                prediction,
                target,
                depth=signature_local_depth,
                intervals=signature_local_intervals,
                output_scale=signature_output_scale,
            )
        ),
        "dynamics_residual": float(
            ou_dynamics_residual(
                time,
                split["driver"],
                prediction,
                lambd=lambd,
                sigma=sigma,
            )
        ),
    }
    count = min(int(keep_paths), prediction.shape[0])
    paths = {
        "time": time.detach().cpu(),
        "target_time": target_time.detach().cpu(),
        "driver": split["driver"][:count].detach().cpu(),
        "target": target[:count].detach().cpu(),
        "prediction": prediction[:count].detach().cpu(),
    }
    return metrics, paths


def _component_gradient_norm(
    value: torch.Tensor,
    parameters: tuple[nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        value,
        parameters,
        allow_unused=True,
        retain_graph=retain_graph,
    )
    squared = value.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.square().sum()
    return float(torch.sqrt(squared).detach())


def ou_signature_gradient_audit(cfg: OUOperatorTrainConfig) -> dict:
    """Initial OU signature components and parameter-gradient magnitudes."""
    audit_cfg = replace(cfg, condition="uniform", loss="sig_global")
    splits = make_ou_operator_splits(audit_cfg)
    train = splits["train"]
    model = make_ou_operator_model(audit_cfg)
    target_time = _target_times(audit_cfg, train["time"])
    driver = train["driver"][: min(8, audit_cfg.n_train)]
    target = interpolate_shared_grid(
        train["time"], train["target"][: driver.shape[0]], target_time
    )
    output_scale = _signature_scale(train["target"])
    parameters = tuple(model.parameters())
    specifications = {
        "global": (audit_cfg.signature_global_depth, 1),
        "local": (
            audit_cfg.signature_local_depth,
            audit_cfg.signature_local_intervals,
        ),
    }
    report = {
        "output_scale": output_scale,
        "scale_rule": "training_target_std",
        "representations": {},
    }
    for representation, (depth, intervals) in specifications.items():
        prediction = model(train["time"], driver, target_time)
        components = anchored_coordinate_mean_components(
            target_time,
            prediction,
            target,
            depth=depth,
            intervals=intervals,
            output_scale=output_scale,
        )
        scalar = {name: value.mean() for name, value in components.items()}
        scalar["total"] = torch.stack(tuple(scalar.values())).sum()
        records = {}
        items = tuple(scalar.items())
        for index, (name, value) in enumerate(items):
            records[name] = {
                "value": float(value.detach()),
                "parameter_gradient_l2": _component_gradient_norm(
                    value,
                    parameters,
                    retain_graph=index < len(items) - 1,
                ),
            }
        report["representations"][representation] = {
            "depth": depth,
            "intervals": intervals,
            "feature_count": signature_feature_count(2, depth, intervals=intervals),
            "components": records,
        }
    return report


def _orders(n: int, epochs: int, seed: int) -> tuple[list[np.ndarray], str]:
    generator = np.random.default_rng(seed)
    digest = hashlib.sha256()
    orders = []
    for _ in range(epochs):
        order = generator.permutation(n)
        orders.append(order)
        digest.update(order.tobytes())
    return orders, digest.hexdigest()


def train_ou_operator(
    cfg: OUOperatorTrainConfig,
    *,
    include_test: bool = False,
    verbose: bool = False,
) -> dict:
    """Train one paired Experiment B model and return paths and cross-metrics."""
    _validate_config(cfg)
    splits = make_ou_operator_splits(cfg)
    train = splits["train"]
    val = splits["val"]
    model = make_ou_operator_model(cfg)
    initial_fingerprint = state_fingerprint(model)
    data_fingerprint = _tensor_fingerprint(train["driver"], train["target"])
    target_time = _target_times(cfg, train["time"])
    target_time_fingerprint = _tensor_fingerprint(target_time)
    train_target = interpolate_shared_grid(train["time"], train["target"], target_time)
    signature_output_scale = _signature_scale(train["target"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    epoch_orders, order_fingerprint = _orders(cfg.n_train, cfg.epochs, cfg.seed)
    untrained_validation = _evaluate_target_grid(model, val, target_time)

    history = []
    for epoch, order in enumerate(epoch_orders):
        model.train()
        total = 0.0
        count = 0
        for start in range(0, cfg.n_train, cfg.batch_size):
            selection = torch.as_tensor(
                order[start : start + cfg.batch_size],
                dtype=torch.long,
                device=train["driver"].device,
            )
            prediction = model(train["time"], train["driver"][selection], target_time)
            loss = operator_loss(
                cfg.loss,
                target_time,
                prediction,
                train_target[selection],
                signature_output_scale=signature_output_scale,
                signature_global_depth=cfg.signature_global_depth,
                signature_local_depth=cfg.signature_local_depth,
                signature_local_intervals=cfg.signature_local_intervals,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_count = selection.numel()
            total += float(loss.detach()) * batch_count
            count += batch_count
        validation = _evaluate_target_grid(model, val, target_time)
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            "val_mse": validation["mse"],
            "val_j2": validation["j2"],
        }
        history.append(row)
        if verbose and (
            epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1
        ):
            print(
                f"epoch {epoch:4d}/{cfg.epochs - 1}: train {row['train_loss']:.6f}, "
                f"val mse {row['val_mse']:.6f}, val J2 {row['val_j2']:.6f}",
                flush=True,
            )

    validation_metrics, validation_paths = evaluate_ou_operator(
        model,
        val,
        target_time,
        lambd=cfg.lambd,
        sigma=cfg.sigma,
        signature_output_scale=signature_output_scale,
        signature_global_depth=cfg.signature_global_depth,
        signature_local_depth=cfg.signature_local_depth,
        signature_local_intervals=cfg.signature_local_intervals,
        keep_paths=4,
    )
    result = {
        "config": cfg,
        "initial_fingerprint": initial_fingerprint,
        "data_fingerprint": data_fingerprint,
        "target_time_fingerprint": target_time_fingerprint,
        "order_fingerprint": order_fingerprint,
        "signature_output_scale": signature_output_scale,
        "untrained_validation": untrained_validation,
        "history": history,
        "validation_metrics": validation_metrics,
        "validation_paths": validation_paths,
        "model": model,
    }
    if include_test:
        if "test" not in splits:
            raise ValueError("include_test requires n_test > 0")
        test_metrics, test_paths = evaluate_ou_operator(
            model,
            splits["test"],
            target_time,
            lambd=cfg.lambd,
            sigma=cfg.sigma,
            signature_output_scale=signature_output_scale,
            signature_global_depth=cfg.signature_global_depth,
            signature_local_depth=cfg.signature_local_depth,
            signature_local_intervals=cfg.signature_local_intervals,
            keep_paths=8,
        )
        result["test_metrics"] = test_metrics
        result["test_paths"] = test_paths
    return result


def run_ou_acceptance(
    base: OUOperatorTrainConfig,
    *,
    overfit_epochs: int = 300,
    overfit_max_ratio: float = 0.1,
    overfit_max_loss: float = 1.0e-3,
    heldout_epochs: int = 120,
    heldout_max_ratio: float = 0.8,
    verbose: bool = False,
) -> dict:
    """Run Experiment B implementation gates before comparative training."""
    check_cfg = replace(
        base,
        seed=991,
        condition="uniform",
        loss="mse",
        n_train=8,
        n_val=8,
        n_test=0,
        n_steps=32,
        n_target=16,
        epochs=1,
        batch_size=8,
    )
    raw = brownian_ou_pairs(
        check_cfg.n_train,
        n_steps=check_cfg.n_steps,
        T=check_cfg.T,
        lambd=check_cfg.lambd,
        sigma=check_cfg.sigma,
        y0=check_cfg.y0,
        rng=check_cfg.train_data_seed,
    )
    raw_dt = check_cfg.T / check_cfg.n_steps
    raw_recurrence = (
        raw["target"][:, :-1, :]
        - check_cfg.lambd * raw["target"][:, :-1, :] * raw_dt
        + check_cfg.sigma * raw["increments"]
    )
    recurrence_exact = np.array_equal(raw["target"][:, 1:, :], raw_recurrence)
    recurrence_error = float(np.max(np.abs(raw["target"][:, 1:, :] - raw_recurrence)))

    splits = make_ou_operator_splits(check_cfg)
    data = splits["train"]

    model = make_ou_operator_model(check_cfg)
    target_time = _target_times(check_cfg, data["time"])
    output = model(data["time"], data["driver"], target_time)
    output.square().mean().backward()
    gradients_reach_all = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )

    changed = data["driver"].clone()
    changed[:, changed.shape[1] // 2 + 1 :, :] += 0.5
    with torch.no_grad():
        original_fine = model.forward_fine(data["time"], data["driver"])
        changed_fine = model.forward_fine(data["time"], changed)
    split_index = changed.shape[1] // 2
    causal_error = float(
        (original_fine[:, : split_index + 1] - changed_fine[:, : split_index + 1])
        .abs()
        .max()
    )
    driver_effect = float(
        (original_fine[:, split_index + 1 :] - changed_fine[:, split_index + 1 :])
        .abs()
        .max()
    )

    second_splits = make_ou_operator_splits(check_cfg)
    second_model = make_ou_operator_model(check_cfg)
    second_target_time = _target_times(check_cfg, second_splits["train"]["time"])
    first_orders, first_order_fingerprint = _orders(
        check_cfg.n_train, 3, check_cfg.seed
    )
    second_orders, second_order_fingerprint = _orders(
        check_cfg.n_train, 3, check_cfg.seed
    )
    reproducible = (
        _tensor_fingerprint(data["driver"], data["target"])
        == _tensor_fingerprint(
            second_splits["train"]["driver"], second_splits["train"]["target"]
        )
        and state_fingerprint(model) == state_fingerprint(second_model)
        and torch.equal(target_time, second_target_time)
        and first_order_fingerprint == second_order_fingerprint
        and all(np.array_equal(a, b) for a, b in zip(first_orders, second_orders))
    )

    overfit_cfg = replace(
        check_cfg,
        epochs=overfit_epochs,
        lr=3.0e-3,
    )
    if verbose:
        print("acceptance: small-batch overfit", flush=True)
    overfit = train_ou_operator(overfit_cfg, verbose=verbose)
    overfit_ratio = (
        overfit["history"][-1]["train_loss"] / overfit["history"][0]["train_loss"]
    )
    overfit_final_loss = overfit["history"][-1]["train_loss"]

    heldout_cfg = replace(
        base,
        seed=992,
        condition="uniform",
        loss="mse",
        n_train=128,
        n_val=32,
        n_test=0,
        n_steps=64,
        n_target=32,
        epochs=heldout_epochs,
        batch_size=32,
    )
    if verbose:
        print("acceptance: held-out improvement", flush=True)
    heldout = train_ou_operator(heldout_cfg, verbose=verbose)
    heldout_ratio = (
        heldout["validation_metrics"]["target_mse"]
        / heldout["untrained_validation"]["mse"]
    )

    checks = {
        "recurrence_exact": recurrence_exact,
        "output_shape": tuple(output.shape) == (8, 16, 1),
        "finite_gradients_reach_all_parameters": gradients_reach_all,
        "driver_changes_future_prediction": driver_effect > 0.0,
        "causal_prefix_unchanged": causal_error <= 1.0e-7,
        "small_batch_fit": (
            overfit_ratio <= overfit_max_ratio
            and overfit_final_loss <= overfit_max_loss
        ),
        "heldout_improves": heldout_ratio <= heldout_max_ratio,
        "paired_inputs_initialization_and_order_reproduce": reproducible,
    }
    return {
        "checks": checks,
        "measurements": {
            "recurrence_max_abs_error": recurrence_error,
            "causal_prefix_max_abs_error": causal_error,
            "future_driver_effect_max_abs": driver_effect,
            "overfit_loss_ratio": overfit_ratio,
            "overfit_max_ratio": overfit_max_ratio,
            "overfit_final_loss": overfit_final_loss,
            "overfit_max_loss": overfit_max_loss,
            "heldout_mse_ratio": heldout_ratio,
            "heldout_max_ratio": heldout_max_ratio,
        },
        "passed": all(checks.values()),
    }
