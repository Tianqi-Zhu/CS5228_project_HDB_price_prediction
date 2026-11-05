"""Fit one v4 temporal arm on orig/seed42 with the predeclared fixed-7 B refit.

Protocol, fixed before the run and not selected afterwards:

* Stage A trains on A/fit with the same optimiser/scheduler/batch protocol as the
  verified fixed-7 diagnostic (AdamW lr 1e-3, wd 1e-4, clip 1.0, batch 512, batch-order
  seed `seed*1000 + epoch`, cosine T_max 20, max 20 epochs, patience 5). The monitor is
  the A/holdout RMSE.
* **7 B refit epochs**, predeclared: the block reference's recorded inner selection.
  This is a fixed-epoch diagnostic, not the outcome of inner model selection, and no
  epoch is chosen using the outer validation population.
* The B scaler is the **saved fit-only** scaler from the verified diagnostic, loaded
  from disk and checked equal to a fresh B/fit fit. Nothing is fitted on validation.
* No macro branch.

One process, two threads.

Run:
    .venv/bin/python -m experiments.temporal_fusion_v4.main --arm padding_fixed_48
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
REPORTS = ROOT / "reports" / "temporal_fusion_v4"
PACKAGE = Path(__file__).resolve().parent

REFERENCE_RUN = "spatial-probe-perfeature-none-s42-h1"
BLOCK_RUN = "macro-probe-block-none-s42"
RUNS = ROOT / "reports" / "neural_fusion_v3" / "runs"

FIXED_B_EPOCHS = 7
SEED = 42
SPLIT = "orig"
MAX_EPOCHS = 20
PATIENCE = 5
BATCH_SIZE = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint() -> dict:
    return {str(p.relative_to(PACKAGE)): sha256_file(p)
            for p in sorted(PACKAGE.rglob("*.py"))}


def build_representations(population, which: str):
    from experiments.temporal_fusion_v4.adapter import legacy_48_step, monthly_12_step

    arrays = population.arrays
    if which == "legacy_48_step":
        rep = legacy_48_step(arrays.sequence, arrays.sequence_mask, arrays.sequence_order)
    else:
        rep = monthly_12_step(arrays.sequence, arrays.sequence_mask, arrays.sequence_order,
                              population.samples["origin_month_index"].to_numpy())
    return rep


def fit_stage(model, tensors, config, epochs: int, monitor=None) -> dict:
    """v2's `_epoch` at the byte level, plus an optional holdout curve."""
    from neural_fusion_v2.training import _epoch, metrics

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                          T_max=max(config.epochs, 1))
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    curve = []
    for epoch in range(1, epochs + 1):
        train_loss = _epoch(model, tensors, config, optimiser, scheduler, epoch)
        row = {"epoch": epoch, "train_mse_ksgd2": float(train_loss)}
        if monitor is not None:
            from neural_fusion_v2.training import predict
            predicted = monitor["tensors"].to_sgd(predict(model, monitor["tensors"]))
            rmse = metrics(monitor["actual"], predicted)["rmse_sgd"]
            row["holdout_rmse_sgd"] = float(rmse)
            if rmse < best - 1e-6:
                best, best_epoch, stale = rmse, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        curve.append(row)
        if monitor is not None and stale >= PATIENCE:
            break
    if monitor is not None and best_state is not None:
        model.load_state_dict(best_state)
    return {"curve": curve, "best_epoch": best_epoch, "best_inner_rmse": best}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=["padding_fixed_48", "monthly_12"])
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    from neural_fusion_v2 import contract as C
    from neural_fusion_v2.adapter import build_tensors
    from neural_fusion_v2.preprocessing import ScalerState
    from neural_fusion_v2.settings import TRAIN
    from neural_fusion_v2.training import FitConfig, metrics, predict, set_seed
    from neural_fusion_v3.main import shapes_of
    from experiments.temporal_fusion_v4.models import (assert_init_comparable, build_arm,
                                                       parameter_count)

    run_id = args.run_id or f"temporal-{args.arm}-none-s42-h1"
    run_dir = REPORTS / "runs" / run_id
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite {run_dir}")
    started = time.perf_counter()

    manifest = C.load_manifest(V2_DIR)
    encoding_a = C.load_encoding(V2_DIR, manifest, SPLIT, "A")
    encoding_b = C.load_encoding(V2_DIR, manifest, SPLIT, "B")
    a_fit = C.load_population(V2_DIR, manifest, SPLIT, "A", "fit")
    a_hold = C.load_population(V2_DIR, manifest, SPLIT, "A", "holdout")
    b_fit = C.load_population(V2_DIR, manifest, SPLIT, "B", "fit")
    b_val = C.load_population(V2_DIR, manifest, SPLIT, "B", "validation")
    print(f"{run_id}")
    print(f"  A/fit {len(a_fit):,}  A/holdout {len(a_hold):,}  "
          f"B/fit {len(b_fit):,}  B/validation {len(b_val):,}")

    # scalers: A is fit on A/fit; B is the SAVED fit-only state from the reference run
    state_a = ScalerState.fit(a_fit.arrays, population="A/fit")
    saved_path = (RUNS / REFERENCE_RUN / "preprocessing" / SPLIT
                  / f"lstm_fusion__per_feature_seed{SEED}_stageB.json")
    state_b = ScalerState.from_dict(json.loads(saved_path.read_text()))
    fresh_b = ScalerState.fit(b_fit.arrays, population="B/fit")
    if json.dumps(state_b.to_dict(), sort_keys=True, default=str) != json.dumps(
            fresh_b.to_dict(), sort_keys=True, default=str):
        raise SystemExit("the saved B scaler does not equal a fresh B/fit fit")
    print(f"  B scaler reused from {saved_path.relative_to(ROOT)} (verified fit-only)")

    shapes = shapes_of(a_fit, True, encoding_a)
    config = FitConfig(control=args.arm, seed=SEED, epochs=MAX_EPOCHS,
                       patience=PATIENCE, batch_size=BATCH_SIZE)
    columns = state_a.static_columns

    # ---------------------------------------------------------------- stage A
    rep_a_fit = build_representations(a_fit, "legacy_48_step" if args.arm == "padding_fixed_48"
                                      else "monthly_12_step")
    rep_a_hold = build_representations(a_hold, "legacy_48_step" if args.arm == "padding_fixed_48"
                                       else "monthly_12_step")
    channels = rep_a_fit.values.shape[2]

    if args.arm == "padding_fixed_48":
        parity = assert_init_comparable(shapes, columns, channels, SEED)
        print(f"  init parity: shared={parity['shared_tensors']} "
              f"differing={len(parity['differing'])} "
              f"only_in_arm={parity['only_in_arm']} "
              f"only_in_reference={parity['only_in_reference']} "
              f"comparable={parity['comparable']}")
        if not parity["comparable"]:
            raise SystemExit(f"init parity failed: {parity['differing']}")
    else:
        parity = {"comparable": None,
                  "note": "monthly_12 changes the recurrent input width, so the LSTM "
                          "cannot inherit v2's weights; recorded as a residual "
                          "difference between the arms"}

    model_a = build_arm(args.arm, shapes, columns, channels, seed=SEED)
    n_params_a = parameter_count(model_a)
    print(f"  parameters (stage A model): {n_params_a:,}")

    def tensors_for(population, state, rep):
        base = build_tensors(population, state, use_sequence=False)
        return _Repacked(base, rep)

    t_a_fit = tensors_for(a_fit, state_a, rep_a_fit)
    t_a_hold = tensors_for(a_hold, state_a, rep_a_hold)
    stage_a_started = time.perf_counter()
    stage_a = fit_stage(model_a, t_a_fit, config, MAX_EPOCHS,
                        monitor={"tensors": t_a_hold, "actual": a_hold.actual_sgd})
    stage_a_seconds = time.perf_counter() - stage_a_started
    print(f"  stage A: {len(stage_a['curve'])} epochs, best epoch {stage_a['best_epoch']} "
          f"(holdout {stage_a['best_inner_rmse']:.4f}), {stage_a_seconds:.1f}s")
    for row in stage_a["curve"]:
        print(f"    epoch {row['epoch']}: train {row['train_mse_ksgd2']:.6f} "
              f"holdout {row['holdout_rmse_sgd']:.4f}")

    # ---------------------------------------------------------------- stage B
    rep_b_fit = build_representations(b_fit, "legacy_48_step" if args.arm == "padding_fixed_48"
                                      else "monthly_12_step")
    rep_b_val = build_representations(b_val, "legacy_48_step" if args.arm == "padding_fixed_48"
                                      else "monthly_12_step")
    t_b_fit = tensors_for(b_fit, state_b, rep_b_fit)
    t_b_val = tensors_for(b_val, state_b, rep_b_val)

    # B/fit has its own categorical vocabulary (the third column's cardinality differs:
    # 19 at A/fit, 20 at B/fit), so the B model must be built from B's shapes or its
    # embeddings index out of range. v3's runner does the same.
    shapes_b = shapes_of(b_fit, True, encoding_b)
    if (shapes_b.numeric_dim, shapes_b.history_dim, shapes_b.context_dim,
            shapes_b.sequence_channels) != (shapes.numeric_dim, shapes.history_dim,
                                            shapes.context_dim, shapes.sequence_channels):
        raise SystemExit(f"A/B feature shapes disagree: {shapes} vs {shapes_b}")
    print(f"  stage-B cardinalities {shapes_b.cardinalities} "
          f"(stage-A {shapes.cardinalities})")
    model_b = build_arm(args.arm, shapes_b, state_b.static_columns, channels, seed=SEED)
    n_params_b = parameter_count(model_b)
    stage_b_started = time.perf_counter()
    stage_b = fit_stage(model_b, t_b_fit, config, FIXED_B_EPOCHS, monitor=None)
    stage_b_seconds = time.perf_counter() - stage_b_started
    print(f"  stage B: {FIXED_B_EPOCHS} epochs (predeclared), {stage_b_seconds:.1f}s")

    predicted = t_b_val.to_sgd(predict(model_b, t_b_val))
    if not np.isfinite(predicted).all():
        raise SystemExit("non-finite predictions")
    frame = pd.DataFrame({
        "SOURCE_ROW_ID": b_val.samples["SOURCE_ROW_ID"].to_numpy(),
        "split": b_val.samples["split_id"].to_numpy(),
        "stage": "B", "population": "validation",
        "cutoff_month": b_val.samples["origin_month_index"].to_numpy(),
        "horizon_months": b_val.samples["horizon_months"].to_numpy(),
        "target_month": b_val.samples["target_month_index"].to_numpy(),
        "actual_rent_sgd": b_val.actual_sgd,
        "predicted_rent_sgd": np.asarray(predicted, dtype=np.float64),
        "model": args.arm, "seed": SEED,
    })
    overall = metrics(frame["actual_rent_sgd"].to_numpy(),
                      frame["predicted_rent_sgd"].to_numpy())
    print(f"  RMSE {overall['rmse_sgd']:.6f}  MAE {overall['mae_sgd']:.6f}  "
          f"bias {overall['bias_sgd']:+.4f}  n={overall['n']:,}")

    # ---------------------------------------------------------------- save
    for name in ("predictions", "checkpoints", "metrics", "preprocessing"):
        (run_dir / name / SPLIT).mkdir(parents=True, exist_ok=True)
    tag = f"{args.arm}_seed{SEED}"
    frame.to_csv(run_dir / "predictions" / SPLIT / f"{tag}.csv", index=False)
    state_a.save(run_dir / "preprocessing" / SPLIT / f"{tag}_stageA.json")
    state_b.save(run_dir / "preprocessing" / SPLIT / f"{tag}_stageB.json")

    checkpoint = {
        "arm": args.arm, "control": "temporal_fusion_v4", "macro_mode": "none",
        "split": SPLIT, "seed": SEED, "uses_sequence": True,
        "sequence_representation": ("legacy_48_step" if args.arm == "padding_fixed_48"
                                    else "monthly_12_step"),
        "temporal_channels": channels,
        "state_dict": model_b.state_dict(), "shapes": shapes_b.__dict__,
        "numeric_columns": list(columns),
        "stage_a_state": state_a.to_dict(), "stage_b_state": state_b.to_dict(),
        "refit_epochs": FIXED_B_EPOCHS,
        "epochs_provenance": ("predeclared as the block reference's recorded inner "
                              "early-stopping selection; no outer score used"),
        "stage_a_curve": stage_a["curve"], "stage_b_curve": stage_b["curve"],
        "stage_a_shapes": shapes.__dict__, "stage_b_shapes": shapes_b.__dict__,
        "n_parameters_stage_a": n_params_a, "n_parameters_stage_b": n_params_b,
        "init_parity": parity,
        "description": model_b.describe(),
        "provenance": {
            "contract_id": manifest["contract_id"],
            "b_scaler_reused_from": str(saved_path.relative_to(ROOT)),
            "derived_from_runs": [REFERENCE_RUN, BLOCK_RUN],
            "code_fingerprint": code_fingerprint(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "platform": platform.platform(), "threads": torch.get_num_threads(),
        },
        "metrics": overall,
    }
    torch.save(checkpoint, run_dir / "checkpoints" / SPLIT / f"{tag}.pt")

    summary = {
        "run_id": run_id, "arm": args.arm, "split": SPLIT, "seed": SEED,
        "contract_id": manifest["contract_id"],
        "fixed_b_epochs": FIXED_B_EPOCHS,
        "epochs_predeclared": True,
        "n_parameters_stage_a": n_params_a, "n_parameters_stage_b": n_params_b,
        "init_parity": parity,
        "stage_a": stage_a, "stage_b": stage_b,
        "stage_a_seconds": stage_a_seconds, "stage_b_seconds": stage_b_seconds,
        "total_seconds": time.perf_counter() - started,
        "overall": overall,
        "by_horizon_band": __import__("neural_fusion_v2.training",
                                      fromlist=["band_report"]).band_report(frame),
        "by_month": __import__("neural_fusion_v2.training",
                               fromlist=["monthly_report"]).monthly_report(
            frame).to_dict(orient="records"),
        "representation": {
            "kind": checkpoint["sequence_representation"],
            "channels": channels,
            "level_names": _level_names(b_val),
        },
        "comparisons_note": ("fixed-epoch diagnostics; the reference fixed-7 per_feature "
                             "run is 474.491228121 SGD and the block/7 run is 475.598687709"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "run_id": run_id, "arm": args.arm, "split": SPLIT, "seed": SEED,
        "contract_id": manifest["contract_id"], "contract_dir": str(V2_DIR.relative_to(ROOT)),
        "fixed_b_epochs": FIXED_B_EPOCHS, "command": " ".join(sys.argv),
        "code_fingerprint": code_fingerprint(),
    }, indent=2, default=str))

    # ---------------------------------------------------------------- reload check
    stored = torch.load(run_dir / "checkpoints" / SPLIT / f"{tag}.pt",
                        map_location="cpu", weights_only=False)
    from neural_fusion_v3.models import Shapes
    recorded = dict(stored["shapes"])
    recorded["cardinalities"] = tuple(recorded["cardinalities"])
    rebuilt = build_arm(stored["arm"], Shapes(**recorded),
                        list(stored["numeric_columns"]), stored["temporal_channels"],
                        seed=stored["seed"])
    rebuilt.load_state_dict(stored["state_dict"], strict=True)
    restored_b = ScalerState.from_dict(stored["stage_b_state"])
    rep_reload = build_representations(b_val, "legacy_48_step"
                                       if stored["sequence_representation"] == "legacy_48_step"
                                       else "monthly_12_step")
    t_reload = _Repacked(build_tensors(b_val, restored_b, use_sequence=False), rep_reload)
    rebuilt_sgd = t_reload.to_sgd(predict(rebuilt, t_reload))
    delta = float(np.max(np.abs(np.asarray(rebuilt_sgd) - predicted)))
    print(f"  reload max |delta| = {delta:.3e} SGD")
    (run_dir / "checkpoints" / SPLIT / f"{tag}.reload.json").write_text(json.dumps({
        "max_abs_delta_sgd": delta, "reproduced": delta < 1e-4,
        "rmse_recomputed": metrics(frame["actual_rent_sgd"].to_numpy(),
                                   np.asarray(rebuilt_sgd))["rmse_sgd"]}, indent=2))
    print(f"saved {run_dir}")
    return 0


def _level_names(population) -> list:
    from experiments.temporal_fusion_v4.adapter import level_names

    order = population.arrays.sequence_order
    return level_names(order, population.arrays.n_levels, population.arrays.n_channels)


class _Repacked:
    """`Tensors` view whose sequence slot is the v4 representation.

    `Tensors.batches()` is reused unchanged; only the sequence arrays are substituted, so
    batching, targets and the parity signature stay v2's.
    """

    def __init__(self, base, rep):
        self._base = base
        self._values = torch.from_numpy(rep.values)
        self._mask = torch.from_numpy(rep.mask)

    def __len__(self):
        return len(self._base)

    def __getattr__(self, item):
        # only called for names not found normally; `_base` must not recurse
        if item.startswith("_"):
            raise AttributeError(item)
        return getattr(self._base, item)

    def subset(self, index):
        sub = self._base.subset(index)
        return _Repacked(sub, _RepSlice(self._values[index], self._mask[index]))

    def batches(self, batch_size, shuffle=False, seed=0, with_target=False):
        # Default False to match `Tensors.batches`, so shared callers such as
        # `training.predict` get a feature dict rather than a (features, target) tuple.
        total = len(self)
        order = np.arange(total)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, total, batch_size):
            index = order[start:start + batch_size]
            features = dict(next(iter(self._base.subset(index).batches(
                len(index), shuffle=False, with_target=False))))
            features["sequence"] = self._values[index]
            features["sequence_mask"] = self._mask[index]
            if with_target:
                # v2 stores `target` as a numpy array; batches must yield tensors
                yield features, torch.from_numpy(self._base.target[index])
            else:
                yield features

    def to_sgd(self, predicted):
        return self._base.to_sgd(predicted)

    @property
    def actual_sgd(self):
        return self._base.actual_sgd

    def parity_signature(self):
        return self._base.parity_signature()


class _RepSlice:
    def __init__(self, values, mask):
        self.values = values
        self.mask = mask


if __name__ == "__main__":
    raise SystemExit(main())
