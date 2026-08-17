"""Baseline path-to-path models: read context points, answer at query times.

Interface every model shares:

    forward(t_ctx, x_ctx, t_query) -> (B, Q, d)

with t_ctx (B, C), x_ctx (B, C, d), t_query (B, Q). Query times are arbitrary
and need not lie in the context, which is what allows the loss to be evaluated
on an irregular grid of its own (`datasets.py`).

`GRUQuery` is deliberately the dumbest model that satisfies that interface: a
recurrent encoder summarising the context into one vector, and a feed-forward
decoder mapping (summary, time) to a value. It exists to make the pipeline run
end to end, and its accuracy is beside the point. The Linear Neural CDE named in
the proposal replaces the encoder later without touching the interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GRUQuery", "MODELS", "build_model"]


class GRUQuery(nn.Module):
    """GRU encoder over context points, MLP decoder at query times.

    Encoder input per step is [t, dt, x], where dt is the gap to the previous
    context time (0 at the first). Feeding dt explicitly is what lets a
    discrete-time recurrent net see irregular spacing at all; without it the
    sequence carries no information about when observations happened.

    Parameters
    ----------
    d : channels of the path.
    hidden : GRU hidden width.
    layers : GRU depth.
    width : decoder hidden width.
    """

    def __init__(
        self,
        d: int = 1,
        hidden: int = 64,
        layers: int = 2,
        width: int = 128,
    ) -> None:
        super().__init__()
        self.d = d
        self.encoder = nn.GRU(
            input_size=d + 2, hidden_size=hidden, num_layers=layers, batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, d),
        )

    def forward(
        self, t_ctx: torch.Tensor, x_ctx: torch.Tensor, t_query: torch.Tensor
    ) -> torch.Tensor:
        b, c, d = x_ctx.shape
        if d != self.d:
            raise ValueError(f"model built for d = {self.d}, got {d}")

        dt = torch.zeros_like(t_ctx)
        dt[:, 1:] = t_ctx[:, 1:] - t_ctx[:, :-1]
        enc_in = torch.cat([t_ctx.unsqueeze(-1), dt.unsqueeze(-1), x_ctx], dim=-1)
        _, h = self.encoder(enc_in)
        summary = h[-1]                                   # (B, hidden)

        q = t_query.shape[1]
        summary = summary.unsqueeze(1).expand(b, q, summary.shape[-1])
        dec_in = torch.cat([summary, t_query.unsqueeze(-1)], dim=-1)
        return self.decoder(dec_in)


MODELS = {"gru_query": GRUQuery}


def build_model(kind: str, **kwargs) -> nn.Module:
    """Construct a model by the name used in a config file."""
    if kind not in MODELS:
        raise ValueError(f"unknown model {kind!r}; have {sorted(MODELS)}")
    return MODELS[kind](**kwargs)
