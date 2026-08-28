#!/usr/bin/env python
"""Train one model to reconstruct a path at query times from irregular context.

One config, one fit. Generates synthetic paths, splits each into disjoint
context and target observations, trains the configured model under one loss,
then evaluates on the stored fine grid.

    python scripts/run_reconstruction.py --config configs/baseline_mse.yaml --out results/runs/test

Use this to check a single configuration. For the seed by architecture by
mechanism by loss grid over the same task, use `run_integral_study.py`.

Deliberately thin: it resolves a config into calls on `pathloss`, records
provenance, and writes results. All maths lives in `src/pathloss/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pathloss.provenance import load_config, run_metadata, utc_now




def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--seed", type=int, help="override config seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    args.out.mkdir(parents=True, exist_ok=True)

    # Provenance first, so even a crashed run is attributable.
    meta = run_metadata(args.config, cfg)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    from pathloss.train import TrainConfig, train

    data, model_cfg, loss_cfg, tr, evaluation = (
        cfg.get("data", {}),
        cfg.get("model", {}),
        cfg.get("loss", {}),
        cfg.get("train", {}),
        cfg.get("evaluation", {}),
    )
    tcfg = TrainConfig(
        seed=cfg.get("seed", 0),
        device=tr.get("device", "cpu"),
        generator=data.get("generator", "ornstein_uhlenbeck"),
        n_train=data.get("n_train", 512),
        n_val=data.get("n_val", 128),
        n_test=data.get("n_test", 0),
        n_fine=data.get("n_fine", 513),
        n_ctx=data.get("n_ctx", 64),
        n_tgt=data.get("n_tgt", 64),
        mode=data.get("sampling", {}).get("mode", "clustered"),
        density_bias=data.get("sampling", {}).get("density_bias", 3.0),
        context_mode=data.get("sampling", {}).get("context_mode"),
        target_mode=data.get("sampling", {}).get("target_mode"),
        context_density_bias=data.get("sampling", {}).get("context_density_bias"),
        target_density_bias=data.get("sampling", {}).get("target_density_bias"),
        noise=data.get("noise", 0.0),
        missing_rate=data.get("missing_rate", 0.0),
        d=data.get("d", 1),
        model=model_cfg.get("kind", "gru_query"),
        model_kwargs={
            k: v for k, v in model_cfg.items()
            if k in {"hidden", "layers", "width", "n_fourier", "step_size"}
        },
        loss=loss_cfg.get("kind", "mse"),
        epochs=tr.get("epochs", 200),
        batch_size=tr.get("batch_size", 64),
        lr=tr.get("lr", 1.0e-3),
        eval_missingness_rates=tuple(evaluation.get("missingness_rates", ())),
    )

    out = train(tcfg)

    (args.out / "history.json").write_text(json.dumps(out["history"], indent=2))
    (args.out / "final.json").write_text(json.dumps(out["final"], indent=2))
    if "test" in out:
        (args.out / "test.json").write_text(json.dumps(out["test"], indent=2))
    if out["robustness"]:
        (args.out / "robustness.json").write_text(
            json.dumps(out["robustness"], indent=2)
        )
    meta["finished"] = utc_now()
    meta["final"] = out["final"]
    if "test" in out:
        meta["test"] = out["test"]
    meta["robustness"] = out["robustness"]
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{cfg['name']} -> {args.out}")
    for k, v in out["final"].items():
        print(f"  {k:14s} {v:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
