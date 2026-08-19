"""Tests for the baseline pipeline: losses, model, training loop.

Needs torch, an optional dependency (`requirements-ml.txt`), so the whole module
skips without it. Dataset tests live in `test_datasets.py` and need NumPy only.

The acceptance criterion is `test_overfits_one_batch`: one batch shown
repeatedly must drive the loss near zero. Failure means a wiring fault, and
every number computed on top of the pipeline is then meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from pathloss.norms import integral_distance, quadrature_weights

torch = pytest.importorskip("torch", reason="torch is in requirements-ml.txt")

from pathloss.losses import (  # noqa: E402
    integral_lp,
    pointwise_mse,
    trapezoid_weights,
)
from pathloss.models import build_model  # noqa: E402
from pathloss.train import TrainConfig, evaluate_missingness, to_tensors, train  # noqa: E402
from pathloss.datasets import make_dataset  # noqa: E402


# --- losses: agreement with the NumPy definitions --------------------------


def test_trapezoid_weights_match_numpy():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 1, 40))
    t[0], t[-1] = 0.0, 1.0
    want = quadrature_weights(t)                       # unnormalised, horizon 1
    got = trapezoid_weights(torch.tensor(t)[None]).numpy()[0]
    assert np.allclose(got, want, atol=1e-12)


def test_integral_l2_matches_numpy_integral_distance():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 1, 64))
    t[0], t[-1] = 0.0, 1.0
    a, b = rng.normal(size=(64, 1)), rng.normal(size=(64, 1))
    want = float(integral_distance(t, a, b, p=2.0, normalise=True)) ** 2
    got = float(
        integral_lp(
            torch.tensor(t)[None], torch.tensor(a)[None], torch.tensor(b)[None], p=2.0
        )
    )
    assert got == pytest.approx(want, rel=1e-6)


def test_mse_and_integral_l2_agree_on_an_even_grid():
    """For diffuse bounded residuals, endpoint reweighting vanishes with N."""
    rng = np.random.default_rng(2)
    for n in (64, 256, 1024):
        t = torch.linspace(0, 1, n)[None]
        a = torch.tensor(rng.normal(size=(1, n, 1)), dtype=torch.float32)
        b = torch.zeros_like(a)
        mse = float(pointwise_mse(t, a, b))
        l2 = float(integral_lp(t, a, b, p=2.0))
        assert abs(mse - l2) / mse < 5.0 / n


def test_mse_and_integral_l2_disagree_on_an_uneven_grid():
    """And the converse, which is why target times are irregular by design."""
    n = 256
    t = torch.cat([torch.linspace(0, 0.1, n // 2), torch.linspace(0.11, 1.0, n // 2)])[None]
    a = torch.zeros(1, n, 1)
    a[:, : n // 2] = 1.0                    # error confined to the dense stretch
    b = torch.zeros_like(a)
    mse = float(pointwise_mse(t, a, b))
    l2 = float(integral_lp(t, a, b, p=2.0))
    assert mse == pytest.approx(0.5, abs=1e-6)   # half the samples carry error
    assert l2 < 0.2                              # but only a tenth of the horizon


def test_losses_are_zero_on_exact_predictions_and_positive_otherwise():
    t = torch.linspace(0, 1, 32)[None]
    y = torch.randn(1, 32, 2)
    for p in (1.0, 2.0, 4.0, float("inf")):
        assert float(integral_lp(t, y, y, p=p)) == pytest.approx(0.0, abs=1e-8)
        assert float(integral_lp(t, y + 0.3, y, p=p)) > 0


def test_unsorted_times_raise():
    t = torch.tensor([[0.0, 0.5, 0.4, 1.0]])
    y = torch.zeros(1, 4, 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        integral_lp(t, y, y, p=2.0)


def test_losses_are_differentiable():
    t = torch.linspace(0, 1, 16)[None]
    y = torch.randn(1, 16, 1)
    for name, fn in (("mse", pointwise_mse), ("l2", integral_lp)):
        pred = torch.zeros_like(y, requires_grad=True)
        fn(t, pred, y).backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all(), name
        assert float(pred.grad.abs().sum()) > 0, name


# --- model -----------------------------------------------------------------


def test_model_output_shape_and_arbitrary_query_times():
    model = build_model("gru_query", d=2, hidden=16, layers=1, width=32)
    t_ctx, x_ctx = torch.rand(4, 10).sort(dim=-1).values, torch.randn(4, 10, 2)
    for q in (1, 7, 33):
        t_q = torch.rand(4, q).sort(dim=-1).values
        assert model(t_ctx, x_ctx, t_q).shape == (4, q, 2)


def test_model_response_depends_on_query_time():
    """A model ignoring t_query would make every path-shaped loss meaningless."""
    torch.manual_seed(0)
    model = build_model("gru_query", d=1, hidden=16, layers=1, width=32)
    t_ctx, x_ctx = torch.rand(2, 10).sort(dim=-1).values, torch.randn(2, 10, 1)
    out = model(t_ctx, x_ctx, torch.tensor([[0.1, 0.9], [0.1, 0.9]]))
    assert not torch.allclose(out[:, 0], out[:, 1], atol=1e-4)


def test_linear_cde_model_shape_and_gradients():
    pytest.importorskip("torchcde")
    model = build_model(
        "linear_cde_query", d=1, hidden=8, width=16, n_fourier=2
    )
    t_ctx = torch.rand(3, 8).sort(dim=-1).values
    x_ctx = torch.randn(3, 8, 1)
    t_query = torch.rand(3, 5).sort(dim=-1).values
    out = model(t_ctx, x_ctx, t_query)
    assert out.shape == (3, 5, 1)
    out.square().mean().backward()
    assert model.func.weight.grad is not None
    assert torch.isfinite(model.func.weight.grad).all()


def test_model_rejects_wrong_channel_count():
    model = build_model("gru_query", d=1, hidden=8, layers=1, width=16)
    with pytest.raises(ValueError, match="built for d = 1"):
        model(torch.rand(2, 5).sort(dim=-1).values, torch.randn(2, 5, 3), torch.rand(2, 4))


def test_model_uses_missingness_mask():
    torch.manual_seed(0)
    model = build_model("gru_query", d=1, hidden=8, layers=1, width=16)
    t = torch.linspace(0, 1, 5)[None]
    x = torch.ones(1, 5, 1)
    q = torch.tensor([[0.25, 0.75]])
    observed = model(t, x, q, torch.ones_like(x))
    missing = model(t, x, q, torch.zeros_like(x))
    assert not torch.allclose(observed, missing)


# --- training --------------------------------------------------------------


@pytest.mark.parametrize("loss", ["mse", "integral_l2"])
def test_overfits_one_batch(loss):
    """Acceptance criterion for the pipeline.

    One batch, shown until the optimiser has had 2000 steps, under a model with
    capacity to memorise it. The loss must fall by two orders of magnitude.
    Failure means a wiring fault: detached gradients, a shuffled target, or a
    model ignoring its query times.

    The step budget is the point, and an earlier version of this test got it
    wrong: 16 samples at batch size 16 gives one optimiser step per epoch, so
    300 epochs was 300 steps, far too few to memorise anything. Loosening the
    threshold would have hidden that; raising the budget is the honest fix.
    """
    cfg = TrainConfig(
        n_train=8,
        n_val=8,
        n_fine=129,
        n_ctx=32,
        n_tgt=32,
        batch_size=8,
        epochs=2000,
        lr=3e-3,
        loss=loss,
        model_kwargs={"hidden": 64, "layers": 1, "width": 128, "n_fourier": 8},
    )
    out = train(cfg, verbose=False)
    first = out["history"][0]["train_loss"]
    last = out["history"][-1]["train_loss"]
    assert last < first / 100.0, f"{loss}: {first:.4g} -> {last:.4g}"


def test_fourier_features_help():
    """Justification for the Fourier feature map, rather than an assertion of it.

    Same budget, same data, same seed; the only difference is whether the
    decoder sees gamma(t) or the raw scalar t. If the raw-scalar model were as
    good, the feature map would be unearned complexity and should go.
    """
    common = dict(
        n_train=8, n_val=8, n_fine=129, n_ctx=32, n_tgt=32,
        batch_size=8, epochs=800, lr=3e-3, loss="mse", seed=0,
    )
    plain = train(
        TrainConfig(model_kwargs={"hidden": 64, "layers": 1, "width": 128,
                                  "n_fourier": 0}, **common), verbose=False)
    fourier = train(
        TrainConfig(model_kwargs={"hidden": 64, "layers": 1, "width": 128,
                                  "n_fourier": 8}, **common), verbose=False)
    assert (
        fourier["history"][-1]["train_loss"] < plain["history"][-1]["train_loss"]
    ), f"plain {plain['history'][-1]['train_loss']:.4g} vs fourier {fourier['history'][-1]['train_loss']:.4g}"


def test_training_reduces_held_out_error():
    cfg = TrainConfig(n_train=256, n_val=64, n_fine=257, epochs=60, seed=0)
    out = train(cfg, verbose=False)
    assert out["history"][-1]["val_mse"] < out["history"][0]["val_mse"]


def test_evaluate_reports_every_metric():
    cfg = TrainConfig(n_train=32, n_val=32, n_fine=129, epochs=1)
    out = train(cfg, verbose=False)
    assert set(out["final"]) == {
        "mse",
        "integral_l2",
        "integral_l1",
        "integral_l4",
        "integral_linf",
        "fine_mse",
        "fine_integral_l2",
        "fine_integral_l1",
        "fine_integral_l4",
        "fine_integral_linf",
    }
    assert all(np.isfinite(v) for v in out["final"].values())


def test_optional_test_split_is_reported_separately():
    cfg = TrainConfig(
        n_train=16,
        n_val=8,
        n_test=8,
        n_fine=65,
        n_ctx=16,
        n_tgt=16,
        epochs=1,
    )
    out = train(cfg, verbose=False)
    assert "test" in out
    assert set(out["test"]) == set(out["final"])


def test_missingness_sweep_keeps_truth_fixed_and_reports_fine_grid():
    data = to_tensors(
        make_dataset(8, n_fine=65, n_ctx=12, n_tgt=12, missing_rate=0.0, rng=0)
    )
    model = build_model("gru_query", d=1, hidden=8, layers=1, width=16)
    out = evaluate_missingness(model, data, (0.0, 0.5), seed=0)
    assert set(out) == {"0.0", "0.5"}
    assert "fine_integral_l2" in out["0.5"]
    assert all(np.isfinite(v) for row in out.values() for v in row.values())
