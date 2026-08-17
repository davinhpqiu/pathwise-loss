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

__all__ = ["TrainConfig", "to_tensors", "evaluate", "train"]


@dataclass
class TrainConfig:
    """Everything the loop needs. Mirrors `configs/*.yaml`."""

    seed: int = 0
    device: str = "cpu"
    # data
    generator: str = "ornstein_uhlenbeck"
    n_train: int = 512
    n_val: int = 128
    n_fine: int = 513
    n_ctx: int = 64
    n_tgt: int = 64
    mode: str = "clustered"
    density_bias: float = 3.0
    noise: float = 0.0
    d: int = 1
    # model
    model: str = "gru_query"
    model_kwargs: dict = field(default_factory=lambda: {"hidden": 64, "layers": 2})
    # loss and optimisation
    loss: str = "mse"
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3


def to_tensors(data: dict, device: str = "cpu") -> dict:
    """Convert a `make_dataset` dict to float32 tensors on `device`."""
    out = {}
    for k, v in data.items():
        out[k] = torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
    return out


@torch.no_grad()
def evaluate(model, batch: dict) -> dict[str, float]:
    """Report every metric on one batch, whatever loss was trained against.

    Reporting all of them together is a standing rule of the project: a loss
    value is uninterpretable without the others beside it, since the losses
    disagree only through the weights and the exponent.
    """
    model.eval()
    pred = model(batch["t_ctx"], batch["x_ctx"], batch["t_tgt"])
    t, y = batch["t_tgt"], batch["x_tgt"]
    return {
        "mse": float(pointwise_mse(t, pred, y)),
        "integral_l2": float(integral_lp(t, pred, y, p=2.0)),
        "integral_l1": float(integral_lp(t, pred, y, p=1.0)),
        "integral_linf": float(integral_lp(t, pred, y, p=float("inf"))),
    }


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
        noise=cfg.noise,
        d=cfg.d,
    )
    train_set = to_tensors(make_dataset(cfg.n_train, rng=rng, **common), cfg.device)
    val_set = to_tensors(make_dataset(cfg.n_val, rng=rng, **common), cfg.device)

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
                train_set["t_ctx"][sel], train_set["x_ctx"][sel], train_set["t_tgt"][sel]
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

    return {
        "config": cfg.__dict__,
        "history": history,
        "final": evaluate(model, val_set),
        "model": model,
    }
