"""Direction-2 arms: does predicting `actual - forecast_anchor` beat `actual - anchor`?

Both arms receive the **same full 33 new features** plus the v2.1 base block. They differ only
in the regression **target**:

* ``orig_resid``  — target `actual - anchor_rent_sgd` (the contract's backward anchor)
* ``fcst_resid``  — target `actual - forecast_anchor` (recent level projected forward along
  the causal six-month slope, damped by `damp`, horizon applied once)

Predictions are converted back to rent as `anchor + predicted_residual` before scoring, so
the two arms are compared on the same final scale, which is what actually matters.

## The confound this script makes visible

A better anchor makes the residual task easier *before any model is involved*. The forecast
anchor's own residual RMSE is ~666 SGD on holdout versus ~916 for the original anchor. So a
lower `fcst_resid` score is not evidence that the model learned more; it may only reflect a
better starting point.

`anchor_only` rows are therefore written for both anchors: the prediction is the anchor with
**zero** residual, no fitting at all. Any arm must beat its own anchor-only baseline to have
demonstrated anything. Both are reported side by side.

## Protocol limits, stated up front

* Tree-internal. The frozen FT v4 pipeline uses a raw-SGD residual with its own scalers and
  cannot consume this NPZ; no claim here transfers to it.
* The contract's base block already contains `anchor_rent_sgd` as a **feature**. It is kept
  in both arms unchanged so the only difference is the target. That means the model can see
  the anchor it is being asked to improve on, which is intentional (it can learn a correction)
  but also means the base block is not anchor-free.
* No scaler is applied: LightGBM takes raw features, and the log transform on per-sqm levels
  already happened inside the frozen feature matrix. `scalers` in the manifest says exactly
  this rather than pretending a scaler exists.
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

FEATURES = ROOT / "reports" / "market_dynamics_v5" / "features_origA.npz"
TARGETS = ROOT / "reports" / "market_dynamics_v5" / "direction2_targets.npz"
V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
OUT_DIR = ROOT / "reports" / "market_dynamics_v5" / "direction2"

CATEGORICALS = ("TOWN_CLEAN", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FURNISHED_CLEAN")
SEED = 42
PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
          "min_child_samples": 30, "reg_lambda": 3.0, "n_jobs": 2, "verbosity": -1}


def base_frame() -> dict:
    from neural_fusion_v2 import contract as C

    manifest = C.load_manifest(V2_DIR)
    out = {}
    for population in ("fit", "holdout"):
        pop = C.load_population(V2_DIR, manifest, "orig", "A", population)
        arrays = pop.arrays
        frame = pd.concat([
            pd.DataFrame(arrays.static, columns=list(arrays.static_order)),
            pd.DataFrame(arrays.context, columns=list(arrays.context_order)),
            pd.DataFrame(arrays.history, columns=list(arrays.history_order)),
        ], axis=1)
        frame.insert(0, "SOURCE_ROW_ID", arrays.row_id)
        out[population] = frame
    return out


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.asarray(predicted, float) - np.asarray(actual, float)
    return {"rmse_sgd": float(np.sqrt(np.mean(error ** 2))),
            "mae_sgd": float(np.mean(np.abs(error))),
            "bias_sgd": float(np.mean(error)),
            "error_sd_sgd": float(np.std(error)),
            "n_rows": int(len(error))}


def prepare_categoricals(fit: pd.DataFrame, hold: pd.DataFrame):
    """Consistent category dtype; an unseen holdout code becomes an explicit category."""
    report = {}
    for column in CATEGORICALS:
        if column not in fit.columns:
            continue
        fit_codes = fit[column].astype("int64")
        hold_codes = hold[column].astype("int64")
        categories = sorted(set(fit_codes.unique()))
        unseen = sorted(set(hold_codes.unique()) - set(categories))
        if unseen:
            code = max(categories) + 1
            hold_codes = hold_codes.where(~hold_codes.isin(unseen), code)
            categories = categories + [code]
            report[column] = {"codes": [int(c) for c in unseen], "mapped_to": int(code),
                              "n_rows": int((hold_codes == code).sum())}
        fit[column] = pd.Categorical(fit_codes, categories=categories)
        hold[column] = pd.Categorical(hold_codes, categories=categories)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=TARGETS,
                        help="direction-2 target npz (defaults to the original run)")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    try:
        import lightgbm
        from lightgbm import LGBMRegressor
    except Exception as error:                                   # noqa: BLE001
        print(f"BLOCKED: LightGBM unavailable: {error!r}")
        return 3

    frozen = np.load(FEATURES, allow_pickle=True)
    targets = np.load(args.targets, allow_pickle=True)
    print(f"targets: {args.targets.resolve().relative_to(ROOT)}")
    feature_names = [str(x) for x in frozen["feature_names"]]
    print(f"new features: {len(feature_names)} (identical in both arms)")
    print(f"anchor level {str(targets['level'][0])!r}, "
          f"damp {float(targets['damp'][0])}, "
          f"horizon applied once: {bool(targets['horizon_multiplier_applied_once'][0])}")

    base = base_frame()
    fit = base["fit"].copy()
    hold = base["holdout"].copy()

    # align frozen features and direction-2 targets to the base frames by row id
    for label, frame, ids_key in (("fit", fit, "fit__row_id"),
                                  ("holdout", hold, "holdout__row_id")):
        ids = targets[ids_key]
        index = {rid: i for i, rid in enumerate(ids)}
        if len(index) != len(ids):
            raise AssertionError(f"{label}: duplicate row ids in the targets npz")
        order = np.asarray([index[rid] for rid in frame["SOURCE_ROW_ID"]])
        feature_ids = frozen[f"{label}__row_id"]
        feature_index = {rid: i for i, rid in enumerate(feature_ids)}
        feature_order = np.asarray([feature_index[rid] for rid in frame["SOURCE_ROW_ID"]])
        new_block = pd.DataFrame(frozen[f"{label}__X"][feature_order], columns=feature_names)
        # the direction-2 npz carries the same rows; assert rather than assume
        if not np.array_equal(targets[f"{label}__row_id"][order],
                              frame["SOURCE_ROW_ID"].to_numpy()):
            raise AssertionError(f"{label}: target row ids do not align")
        frame[feature_names] = new_block.to_numpy()
        frame["__actual"] = targets[f"{label}__actual_sgd"][order]
        frame["__orig_anchor"] = targets[f"{label}__orig_anchor"][order]
        frame["__fcst_anchor"] = targets[f"{label}__fcst_anchor"][order]
        frame["__origin"] = targets[f"{label}__origin_month_index"][order]
        frame["__target_month"] = targets[f"{label}__target_month_index"][order]
        frame["__horizon"] = targets[f"{label}__horizon"][order]
        frame["__shift"] = targets[f"{label}__shift"][order]
        frame["__level_psm"] = targets[f"{label}__level_psm"][order]
        frame["__slope"] = targets[f"{label}__slope_per_month"][order]
        frame["__basis"] = targets[f"{label}__basis_code"][order]
        if f"{label}__trend_valid" in targets.files:
            frame["__trend_valid"] = targets[f"{label}__trend_valid"][order]
        else:
            frame["__trend_valid"] = True
        if not np.array_equal(frame["__horizon"].to_numpy(),
                              frame["horizon_months"].to_numpy()):
            raise AssertionError(f"{label}: horizon disagrees with the base block")
        if not np.allclose(frame["__actual"].to_numpy(),
                           targets[f"{label}__actual_sgd"][order], rtol=0, atol=0):
            raise AssertionError(f"{label}: actual disagrees with the target npz")

    unseen = prepare_categoricals(fit, hold)
    if unseen:
        print(f"holdout categories unseen at fit: {unseen}")

    drop = ["SOURCE_ROW_ID", "__actual", "__orig_anchor", "__fcst_anchor", "__origin",
            "__target_month", "__horizon", "__shift", "__level_psm", "__slope", "__basis",
            "__trend_valid"]
    X_fit = fit.drop(columns=drop)
    X_hold = hold.drop(columns=drop)
    print(f"design matrix: {X_fit.shape[1]} columns "
          f"({len(feature_names)} new + {X_fit.shape[1] - len(feature_names)} base)")

    args.out.mkdir(parents=True, exist_ok=True)
    metrics, prediction_frames = [], []

    arms = {
        "orig_resid": ("__orig_anchor", "__actual"),
        "fcst_resid": ("__fcst_anchor", "__actual"),
    }

    trained_rows = None
    for arm, (anchor_col, actual_col) in arms.items():
        anchor_fit = fit[anchor_col].to_numpy(float)
        anchor_hold = hold[anchor_col].to_numpy(float)
        y_fit = fit[actual_col].to_numpy(float) - anchor_fit
        y_hold = hold[actual_col].to_numpy(float) - anchor_hold
        usable_fit = np.isfinite(y_fit)
        usable_hold = np.isfinite(y_hold)
        # Protocol parity: every arm must train and score on the SAME rows. With the
        # no-correction fallback the forecast anchor exists for every row, so a gap here
        # means the fallback failed and the comparison would be confounded again.
        if not usable_fit.all():
            raise AssertionError(
                f"{arm}: {int((~usable_fit).sum())} fit rows unresolvable; "
                "the same-id protocol requires every row to train")
        if not usable_hold.all():
            raise AssertionError(
                f"{arm}: {int((~usable_hold).sum())} holdout rows unresolvable")
        if trained_rows is not None and len(y_fit) != trained_rows:
            raise AssertionError(f"{arm}: {len(y_fit)} train rows != {trained_rows}")
        trained_rows = len(y_fit)

        # anchor-only baseline: zero residual, no fitting. This is what an arm must beat.
        anchor_only_hold = anchor_hold.copy()
        anchor_only_metrics = score(hold[actual_col].to_numpy(float)[usable_hold],
                                    anchor_only_hold[usable_hold])

        model = LGBMRegressor(**dict(PARAMS, random_state=SEED))
        started = time.perf_counter()
        model.fit(X_fit.loc[usable_fit], y_fit[usable_fit],
                  categorical_feature=[c for c in CATEGORICALS if c in X_fit.columns])
        seconds = time.perf_counter() - started
        residual = np.asarray(model.predict(X_hold), float)
        predicted = anchor_hold + residual
        if not np.isfinite(predicted[usable_hold]).all():
            raise ValueError(f"{arm}: non-finite predictions")

        row = {"arm": arm, "trees": PARAMS["n_estimators"], "seed": SEED,
               "n_features": int(X_fit.shape[1]),
               "n_train": int(usable_fit.sum()), "n_holdout_scored": int(usable_hold.sum()),
               **score(hold[actual_col].to_numpy(float)[usable_hold],
                       predicted[usable_hold]),
               "anchor_only_rmse_sgd": anchor_only_metrics["rmse_sgd"],
               "anchor_only_mae_sgd": anchor_only_metrics["mae_sgd"],
               "gain_vs_anchor_only_sgd":
                   anchor_only_metrics["rmse_sgd"]
                   - score(hold[actual_col].to_numpy(float)[usable_hold],
                           predicted[usable_hold])["rmse_sgd"],
               "seconds": round(seconds, 1)}
        metrics.append(row)
        print(f"\n{arm}: train {row['n_train']:,} rows, scored {row['n_holdout_scored']:,}")
        print(f"  model      RMSE {row['rmse_sgd']:8.3f}  MAE {row['mae_sgd']:8.3f}  "
              f"bias {row['bias_sgd']:+8.2f}")
        print(f"  anchor-only RMSE {anchor_only_metrics['rmse_sgd']:8.3f}  "
              f"MAE {anchor_only_metrics['mae_sgd']:8.3f}")
        print(f"  gain vs anchor-only {row['gain_vs_anchor_only_sgd']:+.3f} SGD "
              f"({seconds:.1f}s)")

        # persist the fitted model so a reviewer can reload without refitting
        model_path = args.out / f"model_{arm}.txt"
        model.booster_.save_model(str(model_path))

        # Predictions for BOTH populations. The holdout rows are what the metrics score; the
        # fit rows are emitted so every row's origin / horizon / anchor / shift is auditable
        # and so the 6,246 unusable fit rows are visible as NaN rather than merely counted.
        residual_fit = np.asarray(model.predict(X_fit), float)
        for population, frame_src, anchor_v, pred_resid, scored_mask in (
                ("holdout", hold, anchor_hold, residual, usable_hold),
                ("fit", fit, anchor_fit, residual_fit, usable_fit)):
            prediction_frames.append(pd.DataFrame({
                "SOURCE_ROW_ID": frame_src["SOURCE_ROW_ID"].to_numpy(),
                "population": population,
                "arm": arm,
                "origin_month_index": frame_src["__origin"].to_numpy(),
                "target_month_index": frame_src["__target_month"].to_numpy(),
                "horizon_months": frame_src["__horizon"].to_numpy(),
                "actual_rent_sgd": frame_src["__actual"].to_numpy(),
                "anchor_sgd": anchor_v,
                "shift_sgd": frame_src["__shift"].to_numpy(),
                "level_psm_sgd": frame_src["__level_psm"].to_numpy(),
                "slope_per_month": frame_src["__slope"].to_numpy(),
                "basis_code": frame_src["__basis"].to_numpy(),
                "trend_valid": frame_src["__trend_valid"].to_numpy(),
                "predicted_residual_sgd": pred_resid,
                "predicted_rent_sgd": anchor_v + pred_resid,
                "scored": scored_mask,
            }))

    pd.DataFrame(metrics).to_csv(args.out / "direction2_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.out / "direction2_predictions.csv", index=False)

    manifest = {
        "generated": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "seed": SEED, "params": PARAMS,
        "lightgbm": lightgbm.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "python": platform.python_version(),
        "features_npz_sha256": hashlib.sha256(FEATURES.read_bytes()).hexdigest(),
        "targets_npz_sha256": hashlib.sha256(args.targets.read_bytes()).hexdigest(),
        "targets_npz": str(args.targets.resolve().relative_to(ROOT)),
        "scalers": "none — LightGBM consumes raw features; per-sqm levels are already "
                   "log-transformed inside the frozen feature matrix",
        "anchor_level": str(targets["level"][0]),
        "damp": float(targets["damp"][0]),
        "horizon_rule": "exp(slope * h * damp), h applied once inside the exponent",
        "unseen_categories": json.dumps(unseen, sort_keys=True),
        "train_rows_per_arm": trained_rows,
        "fit_row_id_sha256": hashlib.sha256(
            fit["SOURCE_ROW_ID"].to_numpy().tobytes()).hexdigest(),
        "holdout_row_id_sha256": hashlib.sha256(
            hold["SOURCE_ROW_ID"].to_numpy().tobytes()).hexdigest(),
        "trend_valid_rows": {
            "fit": int(fit["__trend_valid"].sum()),
            "holdout": int(hold["__trend_valid"].sum()),
        },
        "arms": {arm: {"target": f"actual - {anchor}",
                       "model_file": f"model_{arm}.txt"}
                 for arm, (anchor, _) in arms.items()},
        "interpretation_note": (
            "A better anchor makes the residual task easier before any model is involved. "
            "Compare each arm against its OWN anchor-only baseline, not against the other "
            "arm, or the anchor improvement is mistaken for model learning."),
        "scope_note": (
            "Tree-internal only; the frozen FT v4 pipeline uses a raw-SGD residual protocol "
            "and cannot consume this NPZ. The base block already contains anchor_rent_sgd "
            "as a feature, identical in both arms."),
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n=== arm vs its own anchor-only baseline")
    for row in metrics:
        print(f"  {row['arm']:11s} model {row['rmse_sgd']:8.3f}  "
              f"anchor-only {row['anchor_only_rmse_sgd']:8.3f}  "
              f"gain {row['gain_vs_anchor_only_sgd']:+8.3f}")
    print(f"\nwrote {args.out.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
