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

    Query times enter the decoder through Fourier features,

        gamma(t) = [t, sin(2^0 pi t), cos(2^0 pi t), ..., sin(2^{K-1} pi t), cos(2^{K-1} pi t)],

    rather than as a raw scalar. An MLP on a raw scalar input fits low
    frequencies far faster than high ones (spectral bias), so it resists exactly
    the wiggle a sample path consists of. Fourier features supply the high
    frequencies directly and are the standard remedy. `n_fourier=0` recovers the
    raw-scalar decoder, which is what `test_fourier_features_help` compares
    against.

    Parameters
    ----------
    d : channels of the path.
    hidden : GRU hidden width.
    layers : GRU depth.
    width : decoder hidden width.
    n_fourier : number of octaves K above. 0 disables the feature map.
    """

    def __init__(
        self,
        d: int = 1,
        hidden: int = 64,
        layers: int = 2,
        width: int = 128,
        n_fourier: int = 8,
    ) -> None:
        super().__init__()
        self.d = d
        self.n_fourier = int(n_fourier)
        self.encoder = nn.GRU(
            input_size=d + 2, hidden_size=hidden, num_layers=layers, batch_first=True
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden + 1 + 2 * self.n_fourier, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, d),
        )

    def time_features(self, t: torch.Tensor) -> torch.Tensor:
        """gamma(t) of the class docstring. Shape (..., 1 + 2 * n_fourier)."""
        feats = [t.unsqueeze(-1)]
        for k in range(self.n_fourier):
            w = (2.0**k) * torch.pi
            feats += [torch.sin(w * t).unsqueeze(-1), torch.cos(w * t).unsqueeze(-1)]
        return torch.cat(feats, dim=-1)

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
        dec_in = torch.cat([summary, self.time_features(t_query)], dim=-1)
        return self.decoder(dec_in)


MODELS = {"gru_query": GRUQuery}


def build_model(kind: str, **kwargs) -> nn.Module:
    """Construct a model by the name used in a config file."""
    if kind not in MODELS:
        raise ValueError(f"unknown model {kind!r}; have {sorted(MODELS)}")
    return MODELS[kind](**kwargs)
