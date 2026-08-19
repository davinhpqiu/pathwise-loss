"""Training loop for the baseline. Config in, metrics and predictions out.

Kept separate from `scripts/run_experiment.py` so it can be imported by tests.
The overfit test in `tests/test_pipeline.py` is the acceptance criterion for the
pipeline: a model that cannot drive the loss near zero on a single batch it sees
repeatedly is miswired, and no result computed on top of it means anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .datasets import make_dataset
from .losses import get_loss, integral_lp, pointwise_mse
from .models import build_model

__all__ = ["TrainConfig", "to_tensors", "evaluate", "evaluate_missingness", "train"]


@dataclass
class TrainConfig:
    """Everything the loop needs. Mirrors `configs/*.yaml`."""

    seed: int = 0
    device: str = "cpu"
    # data
    generator: str = "ornstein_uhlenbeck"
    n_train: int = 512
    n_val: int = 128
    n_test: int = 0
    n_fine: int = 513
    n_ctx: int = 64
    n_tgt: int = 64
    mode: str = "clustered"
    density_bias: float = 3.0
    context_mode: str | None = None
    target_mode: str | None = None
    context_density_bias: float | None = None
    target_density_bias: float | None = None
    noise: float = 0.0
    missing_rate: float = 0.0
    d: int = 1
    # model
    model: str = "gru_query"
    model_kwargs: dict = field(default_factory=lambda: {"hidden": 64, "layers": 2})
    # loss and optimisation
    loss: str = "mse"
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    eval_missingness_rates: tuple[float, ...] = ()


def to_tensors(data: dict, device: str = "cpu") -> dict:
    """Convert a `make_dataset` dict to float32 tensors on `device`."""
    out = {}
    for k, v in data.items():
        out[k] = torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
    return out


@torch.no_grad()
def evaluate(model, batch: dict, include_fine: bool = False) -> dict[str, float]:
    """Report every metric on one batch, whatever loss was trained against.

    Reporting all of them together is a standing rule of the project: a loss
    value is uninterpretable without the others beside it, since the losses
    disagree only through the weights and the exponent.
    """
    model.eval()
    pred = model(
        batch["t_ctx"], batch["x_ctx"], batch["t_tgt"], batch.get("m_ctx")
    )
    t, y = batch["t_tgt"], batch["x_tgt"]
    metrics = {
        "mse": float(pointwise_mse(t, pred, y)),
        "integral_l2": float(integral_lp(t, pred, y, p=2.0)),
        "integral_l1": float(integral_lp(t, pred, y, p=1.0)),
        "integral_l4": float(integral_lp(t, pred, y, p=4.0)),
        "integral_linf": float(integral_lp(t, pred, y, p=float("inf"))),
    }
    if include_fine:
        t_fine = batch["t_fine"]
        if t_fine.ndim == 1:
            t_fine = t_fine.unsqueeze(0).expand(batch["x_fine"].shape[0], -1)
        pred_fine = model(
            batch["t_ctx"], batch["x_ctx"], t_fine, batch.get("m_ctx")
        )
        y_fine = batch["x_fine"]
        metrics.update(
            {
                "fine_mse": float(pointwise_mse(t_fine, pred_fine, y_fine)),
                "fine_integral_l2": float(integral_lp(t_fine, pred_fine, y_fine, p=2.0)),
                "fine_integral_l1": float(integral_lp(t_fine, pred_fine, y_fine, p=1.0)),
                "fine_integral_l4": float(integral_lp(t_fine, pred_fine, y_fine, p=4.0)),
                "fine_integral_linf": float(
                    integral_lp(t_fine, pred_fine, y_fine, p=float("inf"))
                ),
            }
        )
    return metrics


@torch.no_grad()
def evaluate_missingness(
    model, batch: dict, rates: tuple[float, ...], seed: int = 0
) -> dict[str, dict[str, float]]:
    """Evaluate one clean batch under nested context-missingness masks."""
    if torch.any(batch["m_ctx"] != 1):
        raise ValueError("missingness sweep requires a clean evaluation batch")
    generator = torch.Generator(device=batch["x_ctx"].device).manual_seed(seed)
    uniforms = torch.rand(
        batch["x_ctx"].shape,
        generator=generator,
        device=batch["x_ctx"].device,
        dtype=batch["x_ctx"].dtype,
    )
    out = {}
    for rate in rates:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"missingness rate must lie in [0, 1], got {rate}")
        mask = (uniforms >= rate).to(batch["x_ctx"].dtype)
        corrupted = dict(batch)
        corrupted["m_ctx"] = mask
        corrupted["x_ctx"] = batch["x_ctx"] * mask
        out[str(rate)] = evaluate(model, corrupted, include_fine=True)
    return out


def train(cfg: TrainConfig, verbose: bool = True) -> dict:
    """Train one model under one loss. Returns history and final metrics."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    common = dict(
        generator=cfg.generator,
        n_fine=cfg.n_fine,
        n_ctx=cfg.n_ctx,
        n_tgt=cfg.n_tgt,
        mode=cfg.mode,
        density_bias=cfg.density_bias,
        context_mode=cfg.context_mode,
        target_mode=cfg.target_mode,
        context_density_bias=cfg.context_density_bias,
        target_density_bias=cfg.target_density_bias,
        noise=cfg.noise,
        missing_rate=cfg.missing_rate,
        d=cfg.d,
    )
    train_set = to_tensors(make_dataset(cfg.n_train, rng=rng, **common), cfg.device)
    val_set = to_tensors(make_dataset(cfg.n_val, rng=rng, **common), cfg.device)
    test_set = None
    if cfg.n_test:
        test_set = to_tensors(make_dataset(cfg.n_test, rng=rng, **common), cfg.device)

    model = build_model(cfg.model, d=cfg.d, **cfg.model_kwargs).to(cfg.device)
    loss_fn = get_loss(cfg.loss)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    n = cfg.n_train
    history = []
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(n, device=cfg.device)
        running, nb = 0.0, 0
        for start in range(0, n, cfg.batch_size):
            sel = order[start : start + cfg.batch_size]
            pred = model(
                train_set["t_ctx"][sel],
                train_set["x_ctx"][sel],
                train_set["t_tgt"][sel],
                train_set["m_ctx"][sel],
            )
            loss = loss_fn(train_set["t_tgt"][sel], pred, train_set["x_tgt"][sel])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            nb += 1

        row = {"epoch": epoch, "train_loss": running / nb}
        row.update({f"val_{k}": v for k, v in evaluate(model, val_set).items()})
        history.append(row)
        if verbose and (epoch % max(1, cfg.epochs // 10) == 0 or epoch == cfg.epochs - 1):
            print(
                f"epoch {epoch:4d}  train {row['train_loss']:.5f}  "
                f"val mse {row['val_mse']:.5f}  val L2 {row['val_integral_l2']:.5f}"
            )

    robustness = {}
    if cfg.eval_missingness_rates:
        robustness["missingness"] = evaluate_missingness(
            model, val_set, cfg.eval_missingness_rates, seed=cfg.seed
        )

    out = {
        "config": cfg.__dict__,
        "history": history,
        "final": evaluate(model, val_set, include_fine=True),
        "model": model,
        "robustness": robustness,
    }
    if test_set is not None:
        out["test"] = evaluate(model, test_set, include_fine=True)
    return out
