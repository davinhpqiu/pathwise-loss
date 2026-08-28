"""Fixed-path Neural ODE experiment from the 22 August procedure.

One learned initial state and one neural vector field generate a continuous
hidden trajectory. An affine decoder maps it to the two-dimensional target
path. Training changes only the path discrepancy.

This module contains experiment mathematics and training code. The command-line
wrapper only handles configuration and output files.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .losses import integral_lp, pointwise_mse, sobolev_h1, trapezoid_weights
from .signatures import (
    anchored_coordinate_mean_components,
    anchored_coordinate_mean_signature_loss,
    signature_feature_count,
)

__all__ = [
    "FixedPathTrainConfig",
    "LatentNeuralODE",
    "fixed_target",
    "fixed_target_derivative",
    "observation_times",
    "h1_balance",
    "make_paired_model",
    "state_fingerprint",
    "fixed_path_loss",
    "evaluate_fixed_path",
    "signature_gradient_audit",
    "train_fixed_path",
]


@dataclass(frozen=True)
class FixedPathTrainConfig:
    """Optimization settings for one loss, capacity, condition and seed."""

    seed: int = 0
    device: str = "cpu"
    condition: str = "uniform"
    loss: str = "mse"
    n_target: int = 64
    n_fine: int = 513
    hidden: int = 2
    width: int = 16
    n_fourier: int = 5
    max_step: float = 1.0 / 512.0
    updates: int = 5000
    lr: float = 1.0e-3
    signature_output_scale: float = 1.0
    signature_global_depth: int = 4
    signature_local_depth: int = 2
    signature_local_intervals: int = 10
    evaluation_checkpoints: tuple[int, ...] = ()


def fixed_target(t: torch.Tensor) -> torch.Tensor:
    """Target Y star evaluated at times t, with output shape (..., 2)."""
    a = 0.3 * torch.exp(-100.0 * (t - 0.25) ** 2)
    return torch.stack(
        (
            torch.cos(2.0 * torch.pi * t) + a * torch.cos(12.0 * torch.pi * t),
            torch.sin(2.0 * torch.pi * t) + a * torch.sin(12.0 * torch.pi * t),
        ),
        dim=-1,
    )


def fixed_target_derivative(t: torch.Tensor) -> torch.Tensor:
    """Analytic time derivative of fixed_target."""
    a = 0.3 * torch.exp(-100.0 * (t - 0.25) ** 2)
    da = -200.0 * (t - 0.25) * a
    return torch.stack(
        (
            -2.0 * torch.pi * torch.sin(2.0 * torch.pi * t)
            + da * torch.cos(12.0 * torch.pi * t)
            - 12.0 * torch.pi * a * torch.sin(12.0 * torch.pi * t),
            2.0 * torch.pi * torch.cos(2.0 * torch.pi * t)
            + da * torch.sin(12.0 * torch.pi * t)
            + 12.0 * torch.pi * a * torch.cos(12.0 * torch.pi * t),
        ),
        dim=-1,
    )


def observation_times(
    n_target: int = 64,
    condition: str = "uniform",
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic uniform or late-clustered target times on [0, 1]."""
    if n_target < 2:
        raise ValueError(f"n_target must be at least 2, got {n_target}")
    u = torch.linspace(0.0, 1.0, n_target, device=device, dtype=dtype)
    if condition == "uniform":
        return u
    if condition == "clustered":
        return 1.0 - (1.0 - u) ** 3
    raise ValueError(f"condition must be 'uniform' or 'clustered', got {condition!r}")


def h1_balance(t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Rho balancing target value variation and derivative energy."""
    if t.ndim != 1 or target.shape != (t.numel(), 2):
        raise ValueError("t must be (T,) and target must be (T, 2)")
    w = trapezoid_weights(t.unsqueeze(0))[0]
    mean = (w.unsqueeze(-1) * target).sum(0)
    centred_energy = (w * (target - mean).square().sum(-1)).sum()
    derivative = fixed_target_derivative(t)
    derivative_energy = (w * derivative.square().sum(-1)).sum()
    if float(derivative_energy) <= 0.0:
        raise ValueError("target derivative energy must be positive")
    return centred_energy / derivative_energy


class LatentNeuralODE(nn.Module):
    """Learned initial state, non-autonomous neural vector field and decoder."""

    def __init__(
        self,
        hidden: int = 2,
        width: int = 16,
        output_dim: int = 2,
        n_fourier: int = 5,
        max_step: float = 1.0 / 512.0,
    ) -> None:
        super().__init__()
        if hidden < 1 or width < 1 or output_dim < 1:
            raise ValueError("hidden, width and output_dim must be positive")
        if n_fourier < 0:
            raise ValueError("n_fourier must be non-negative")
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        self.hidden = int(hidden)
        self.output_dim = int(output_dim)
        self.n_fourier = int(n_fourier)
        self.max_step = float(max_step)
        feature_dim = 1 + 2 * self.n_fourier
        self.initial = nn.Parameter(torch.zeros(self.hidden))
        self.vector_net = nn.Sequential(
            nn.Linear(self.hidden + feature_dim, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, self.hidden),
        )
        self.decoder = nn.Linear(self.hidden, self.output_dim)

    def time_features(self, t: torch.Tensor) -> torch.Tensor:
        """Fourier map gamma(t) for one scalar time."""
        scalar = t.reshape(())
        features = [scalar.reshape(1)]
        for k in range(self.n_fourier):
            frequency = (2.0**k) * torch.pi
            features.extend(
                (
                    torch.sin(frequency * scalar).reshape(1),
                    torch.cos(frequency * scalar).reshape(1),
                )
            )
        return torch.cat(features)

    def vector_field(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if h.shape != (self.hidden,):
            raise ValueError(
                f"h must have shape ({self.hidden},), got {tuple(h.shape)}"
            )
        return self.vector_net(torch.cat((self.time_features(t), h)))

    def hidden_trajectory(self, t_eval: torch.Tensor) -> torch.Tensor:
        """Fixed-step RK4 solution evaluated exactly at sorted t_eval."""
        if t_eval.ndim != 1:
            raise ValueError(
                f"t_eval must be one-dimensional, got {tuple(t_eval.shape)}"
            )
        if t_eval.numel() < 1:
            raise ValueError("t_eval must contain at least one time")
        if bool(torch.any(t_eval < 0.0)) or bool(torch.any(t_eval > 1.0)):
            raise ValueError("t_eval must lie in [0, 1]")
        if t_eval.numel() > 1 and bool(torch.any(t_eval[1:] <= t_eval[:-1])):
            raise ValueError("t_eval must be strictly increasing")

        current_t = t_eval.new_zeros(())
        h = self.initial
        states = []
        for next_t in t_eval:
            span = float((next_t - current_t).detach().cpu())
            if span < 0.0:
                raise ValueError("first evaluation time must be non-negative")
            n_steps = max(1, math.ceil(span / self.max_step)) if span > 0.0 else 0
            if n_steps:
                step = (next_t - current_t) / n_steps
                for _ in range(n_steps):
                    k1 = self.vector_field(current_t, h)
                    k2 = self.vector_field(current_t + step / 2.0, h + step * k1 / 2.0)
                    k3 = self.vector_field(current_t + step / 2.0, h + step * k2 / 2.0)
                    k4 = self.vector_field(current_t + step, h + step * k3)
                    h = h + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                    current_t = current_t + step
                current_t = next_t
            states.append(h)
        return torch.stack(states)

    def forward(self, t_eval: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.hidden_trajectory(t_eval))

    def forward_with_derivative(
        self, t_eval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.hidden_trajectory(t_eval)
        prediction = self.decoder(hidden)
        hidden_derivative = torch.stack(
            [self.vector_field(t, h) for t, h in zip(t_eval, hidden)]
        )
        output_derivative = hidden_derivative @ self.decoder.weight.transpose(0, 1)
        return prediction, output_derivative


def make_paired_model(
    seed: int,
    *,
    hidden: int,
    width: int,
    n_fourier: int,
    max_step: float,
    device: str = "cpu",
) -> LatentNeuralODE:
    """Deterministic initialization shared by every loss in one comparison."""
    torch.manual_seed(seed)
    return LatentNeuralODE(
        hidden=hidden,
        width=width,
        n_fourier=n_fourier,
        max_step=max_step,
    ).to(device)


def state_fingerprint(model: nn.Module) -> str:
    """Hash an initial state so paired runs can verify exact equality."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fixed_path_loss(
    name: str,
    t: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    prediction_derivative: torch.Tensor | None = None,
    target_derivative: torch.Tensor | None = None,
    rho: torch.Tensor | float | None = None,
    signature_output_scale: float = 1.0,
    signature_global_depth: int = 4,
    signature_local_depth: int = 2,
    signature_local_intervals: int = 10,
) -> torch.Tensor:
    """One Experiment A training loss."""
    batch_t = t.unsqueeze(0)
    batch_prediction = prediction.unsqueeze(0)
    batch_target = target.unsqueeze(0)
    if name == "mse":
        return pointwise_mse(batch_t, batch_prediction, batch_target)
    if name in {"j2", "integral_l2"}:
        return integral_lp(batch_t, batch_prediction, batch_target, p=2.0)
    if name == "h1":
        if prediction_derivative is None or target_derivative is None or rho is None:
            raise ValueError("h1 loss requires both derivatives and rho")
        return sobolev_h1(
            batch_t,
            batch_prediction,
            batch_target,
            prediction_derivative.unsqueeze(0),
            target_derivative.unsqueeze(0),
            rho,
        )
    if name == "sig_global":
        return anchored_coordinate_mean_signature_loss(
            t,
            prediction,
            target,
            depth=signature_global_depth,
            intervals=1,
            output_scale=signature_output_scale,
        )
    if name == "sig_local":
        return anchored_coordinate_mean_signature_loss(
            t,
            prediction,
            target,
            depth=signature_local_depth,
            intervals=signature_local_intervals,
            output_scale=signature_output_scale,
        )
    raise ValueError(f"unknown fixed-path loss {name!r}")


@torch.no_grad()
def evaluate_fixed_path(
    model: LatentNeuralODE,
    *,
    n_fine: int = 513,
    rho: torch.Tensor | float | None = None,
    signature_output_scale: float = 1.0,
    signature_global_depth: int = 4,
    signature_local_depth: int = 2,
    signature_local_intervals: int = 10,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Dense metrics and paths for one fitted model."""
    if n_fine < 2:
        raise ValueError("n_fine must be at least 2")
    parameter = next(model.parameters())
    t = torch.linspace(0.0, 1.0, n_fine, device=parameter.device, dtype=parameter.dtype)
    target = fixed_target(t)
    target_derivative = fixed_target_derivative(t)
    prediction, prediction_derivative = model.forward_with_derivative(t)
    batch_t = t.unsqueeze(0)
    batch_prediction = prediction.unsqueeze(0)
    batch_target = target.unsqueeze(0)

    if rho is None:
        rho = h1_balance(t, target)
    metrics = {
        "mse": float(pointwise_mse(batch_t, batch_prediction, batch_target)),
        "j2": float(integral_lp(batch_t, batch_prediction, batch_target, p=2.0)),
        "h1": float(
            sobolev_h1(
                batch_t,
                batch_prediction,
                batch_target,
                prediction_derivative.unsqueeze(0),
                target_derivative.unsqueeze(0),
                rho,
            )
        ),
        "linf": float(torch.linalg.vector_norm(prediction - target, dim=-1).max()),
        "sig_global": float(
            anchored_coordinate_mean_signature_loss(
                t,
                prediction,
                target,
                depth=signature_global_depth,
                intervals=1,
                output_scale=signature_output_scale,
            )
        ),
        "sig_local": float(
            anchored_coordinate_mean_signature_loss(
                t,
                prediction,
                target,
                depth=signature_local_depth,
                intervals=signature_local_intervals,
                output_scale=signature_output_scale,
            )
        ),
    }

    local_t = torch.linspace(
        0.15, 0.35, 103, device=parameter.device, dtype=parameter.dtype
    )
    local_prediction = model(local_t)
    local_target = fixed_target(local_t)
    metrics["local_j2"] = float(
        integral_lp(
            local_t.unsqueeze(0),
            local_prediction.unsqueeze(0),
            local_target.unsqueeze(0),
            p=2.0,
        )
    )
    paths = {
        "time": t.detach().cpu(),
        "target": target.detach().cpu(),
        "prediction": prediction.detach().cpu(),
        "target_derivative": target_derivative.detach().cpu(),
        "prediction_derivative": prediction_derivative.detach().cpu(),
    }
    return metrics, paths


def _component_gradient_norm(
    value: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        value,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = value.new_zeros(())
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.square().sum()
    return float(torch.sqrt(squared).detach())


def signature_gradient_audit(cfg: FixedPathTrainConfig) -> dict:
    """Levelwise initial losses and parameter gradients before signature runs."""
    if cfg.condition != "uniform":
        raise ValueError("initial signature audit uses uniform observations")
    model = make_paired_model(
        cfg.seed,
        hidden=cfg.hidden,
        width=cfg.width,
        n_fourier=cfg.n_fourier,
        max_step=cfg.max_step,
        device=cfg.device,
    )
    parameter = next(model.parameters())
    parameters = tuple(model.parameters())
    t = observation_times(
        cfg.n_target,
        "uniform",
        device=cfg.device,
        dtype=parameter.dtype,
    )
    target = fixed_target(t)
    specifications = {
        "global": (cfg.signature_global_depth, 1),
        "local": (cfg.signature_local_depth, cfg.signature_local_intervals),
    }
    audit = {
        "output_scale": cfg.signature_output_scale,
        "representations": {},
    }
    for representation, (depth, intervals) in specifications.items():
        prediction = model(t)
        components = anchored_coordinate_mean_components(
            t,
            prediction,
            target,
            depth=depth,
            intervals=intervals,
            output_scale=cfg.signature_output_scale,
        )
        total = torch.stack(tuple(components.values()), dim=0).sum(dim=0).mean()
        scalar_components = {name: value.mean() for name, value in components.items()}
        scalar_components["total"] = total
        records = {}
        items = tuple(scalar_components.items())
        for index, (name, value) in enumerate(items):
            records[name] = {
                "value": float(value.detach()),
                "parameter_gradient_l2": _component_gradient_norm(
                    value,
                    parameters,
                    retain_graph=index < len(items) - 1,
                ),
            }
        audit["representations"][representation] = {
            "depth": depth,
            "intervals": intervals,
            "feature_count": signature_feature_count(
                target.shape[-1] + 1,
                depth,
                intervals=intervals,
            ),
            "components": records,
        }
    return audit


def train_fixed_path(
    cfg: FixedPathTrainConfig,
    *,
    initial_model: LatentNeuralODE | None = None,
    verbose: bool = False,
) -> dict:
    """Train one paired Experiment A run and return model, history and metrics."""
    if cfg.loss == "h1" and cfg.condition != "uniform":
        raise ValueError("h1 is a secondary uniform-observation comparator")
    if cfg.loss in {"sig_global", "sig_local"} and cfg.condition != "uniform":
        raise ValueError("initial signature comparison uses uniform observations")
    if cfg.updates < 1:
        raise ValueError("updates must be positive")
    if cfg.lr <= 0:
        raise ValueError("lr must be positive")
    checkpoints = tuple(sorted(set(cfg.evaluation_checkpoints)))
    if any(value < 0 or value > cfg.updates for value in checkpoints):
        raise ValueError("evaluation checkpoints must lie between 0 and updates")

    model = (
        make_paired_model(
            cfg.seed,
            hidden=cfg.hidden,
            width=cfg.width,
            n_fourier=cfg.n_fourier,
            max_step=cfg.max_step,
            device=cfg.device,
        )
        if initial_model is None
        else copy.deepcopy(initial_model).to(cfg.device)
    )
    fingerprint = state_fingerprint(model)
    parameter = next(model.parameters())
    t = observation_times(
        cfg.n_target,
        cfg.condition,
        device=cfg.device,
        dtype=parameter.dtype,
    )
    target = fixed_target(t)
    target_derivative = fixed_target_derivative(t)
    rho = h1_balance(t, target)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    checkpoint_records = []
    checkpoint_predictions = {}
    checkpoint_evaluations = {}

    def record_checkpoint(updates_completed: int) -> None:
        model.eval()
        checkpoint_metrics, checkpoint_paths = evaluate_fixed_path(
            model,
            n_fine=cfg.n_fine,
            rho=rho,
            signature_output_scale=cfg.signature_output_scale,
            signature_global_depth=cfg.signature_global_depth,
            signature_local_depth=cfg.signature_local_depth,
            signature_local_intervals=cfg.signature_local_intervals,
        )
        checkpoint_records.append(
            {
                "updates_completed": updates_completed,
                "metrics": checkpoint_metrics,
            }
        )
        checkpoint_predictions[updates_completed] = checkpoint_paths[
            "prediction"
        ]
        checkpoint_evaluations[updates_completed] = (
            checkpoint_metrics,
            checkpoint_paths,
        )

    if 0 in checkpoints:
        record_checkpoint(0)

    history = []
    for update in range(cfg.updates):
        model.train()
        if cfg.loss == "h1":
            prediction, prediction_derivative = model.forward_with_derivative(t)
        else:
            prediction = model(t)
            prediction_derivative = None
        loss = fixed_path_loss(
            cfg.loss,
            t,
            prediction,
            target,
            prediction_derivative=prediction_derivative,
            target_derivative=target_derivative,
            rho=rho,
            signature_output_scale=cfg.signature_output_scale,
            signature_global_depth=cfg.signature_global_depth,
            signature_local_depth=cfg.signature_local_depth,
            signature_local_intervals=cfg.signature_local_intervals,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({"update": update, "train_loss": float(loss.detach())})
        updates_completed = update + 1
        if updates_completed in checkpoints:
            record_checkpoint(updates_completed)
        if verbose and (
            update % max(1, cfg.updates // 10) == 0 or update == cfg.updates - 1
        ):
            print(
                f"update {update:5d}/{cfg.updates - 1}: "
                f"train loss {history[-1]['train_loss']:.6f}",
                flush=True,
            )

    model.eval()
    if cfg.updates in checkpoint_evaluations:
        metrics, paths = checkpoint_evaluations[cfg.updates]
    else:
        metrics, paths = evaluate_fixed_path(
            model,
            n_fine=cfg.n_fine,
            rho=rho,
            signature_output_scale=cfg.signature_output_scale,
            signature_global_depth=cfg.signature_global_depth,
            signature_local_depth=cfg.signature_local_depth,
            signature_local_intervals=cfg.signature_local_intervals,
        )
    return {
        "config": cfg,
        "initial_fingerprint": fingerprint,
        "rho": float(rho.detach()),
        "history": history,
        "metrics": metrics,
        "paths": paths,
        "checkpoints": checkpoint_records,
        "checkpoint_predictions": checkpoint_predictions,
        "model": model,
    }
