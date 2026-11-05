"""LightGBM base vs +global-mean on the shared frozen table.

Both arms read `reports/global_mean_comparison_v6/data/shared_table_origA.npz`, so they see
byte-identical numeric and categorical inputs. The only difference is whether the 57th
numeric column (`grm_latest_log`) is included.

Protocol, fixed by the plan:
* target `(actual - anchor)/1000`; predictions reconstruct as `(residual + anchor)*1000`
* `anchor_ksgd` is an **output offset**, never a learned column
* one LightGBM budget: 300 trees, lr 0.05, 31 leaves, min_child 30, lambda 3, 2 threads,
  seed 42 — no sweep, no HPO
* categorical columns are passed natively by their mapped codes (reserved 0 included),
  the same codes the Transformer receives; holdout unknowns are never remapped using
  future categories

The horizon breakdown and per-origin breakdown are written because the plan asks for
shift/horizon behaviour, and because the single-origin holdout makes the per-origin table
the only place the identification limit is visible.
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

DATA = ROOT / "reports" / "global_mean_comparison_v6" / "data" / "shared_table_origA.npz"
OUT = ROOT / "reports" / "global_mean_comparison_v6" / "tree"

PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
          "min_child_samples": 30, "reg_lambda": 3.0, "n_jobs": 2, "verbosity": -1}
SEED = 42
NEW_FEATURE = "grm_latest_log"


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    err = np.asarray(predicted, float) - np.asarray(actual, float)
    return {"rmse_sgd": float(np.sqrt(np.mean(err ** 2))),
            "mae_sgd": float(np.mean(np.abs(err))),
            "bias_sgd": float(np.mean(err)),
            "error_sd_sgd": float(np.std(err)),
            "n_rows": int(len(err))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    try:
        import lightgbm
        from lightgbm import LGBMRegressor
    except Exception as error:                                   # noqa: BLE001
        print(f"BLOCKED: LightGBM unavailable: {error!r}")
        return 3

    manifest_path = DATA.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    table = np.load(DATA, allow_pickle=True)
    numeric_names = list(manifest["numeric_order"])
    categorical_names = list(manifest["categorical_order"])
    base_numeric = len(numeric_names) - 1
    print(f"table: {manifest['npz_sha256'][:16]}…  "
          f"{len(numeric_names)} numeric ({base_numeric} base + 1 new) "
          f"+ {len(categorical_names)} categorical")
    print(f"lightgbm {lightgbm.__version__}, numpy {np.__version__}, "
          f"pandas {pd.__version__}, python {platform.python_version()}")

    frames = {}
    for pop in ("fit", "holdout"):
        frame = pd.DataFrame(table[f"{pop}__numeric"], columns=numeric_names)
        for j, name in enumerate(categorical_names):
            frame[name] = table[f"{pop}__categorical"][:, j]
        frame["__anchor"] = table[f"{pop}__anchor_ksgd"]
        frame["__target"] = table[f"{pop}__target"]
        frame["__actual"] = table[f"{pop}__actual_sgd"]
        frame["__row_id"] = table[f"{pop}__row_id"]
        frame["__horizon"] = table[f"{pop}__horizon"]
        frame["__origin"] = table[f"{pop}__origin"]
        frames[pop] = frame

    # identical scored rows across arms is a precondition, so assert it once
    if not np.array_equal(frames["fit"]["__row_id"], table["fit__row_id"]):
        raise AssertionError("fit row order changed while building frames")
    print(f"fit {len(frames['fit']):,} rows, holdout {len(frames['holdout']):,} rows")

    args.out.mkdir(parents=True, exist_ok=True)
    rows, preds = [], []
    for arm, use_new in (("base", False), ("global_mean", True)):
        learned = numeric_names if use_new else numeric_names[:base_numeric]
        columns = learned + categorical_names
        fit_x = frames["fit"][columns]
        hold_x = frames["holdout"][columns]

        # The reserved code 0 means "unknown / not present". Categories come from fit only;
        # holdout values outside that set would indicate a drifted mapping, so surface it
        # rather than silently coercing.
        cats = list(categorical_names)
        unseen = {}
        for name in cats:
            fit_codes = sorted(set(fit_x[name].to_numpy()))
            hold_codes = set(hold_x[name].to_numpy())
            extra = sorted(hold_codes - set(fit_codes))
            if extra:
                unseen[name] = {"codes": [int(c) for c in extra],
                                "n_rows": int(np.isin(hold_x[name].to_numpy(),
                                                      extra).sum())}
            all_codes = sorted(set(fit_codes) | set(hold_codes))
            fit_x[name] = pd.Categorical(fit_x[name], categories=all_codes)
            hold_x[name] = pd.Categorical(hold_x[name], categories=all_codes)
        if unseen:
            print(f"  {arm}: holdout codes absent from fit: {unseen}")

        model = LGBMRegressor(**dict(PARAMS, random_state=SEED))
        started = time.perf_counter()
        model.fit(fit_x, frames["fit"]["__target"], categorical_feature=cats)
        seconds = time.perf_counter() - started

        residual = np.asarray(model.predict(hold_x), dtype=float)
        anchor = frames["holdout"]["__anchor"].to_numpy(float)
        predicted = (residual + anchor) * 1000.0
        actual = frames["holdout"]["__actual"].to_numpy(float)
        if not np.isfinite(predicted).all():
            raise ValueError(f"{arm}: non-finite predictions")

        metrics = {"arm": arm, "uses_new_feature": use_new, "trees": PARAMS["n_estimators"],
                   "seed": SEED, "n_features": len(columns),
                   "n_numeric": len(learned), "n_categorical": len(cats),
                   "n_train": int(len(fit_x)), "n_scored": int(len(hold_x)),
                   **score(actual, predicted), "seconds": round(seconds, 2),
                   "unseen_holdout_codes": json.dumps(unseen, sort_keys=True)}
        rows.append(metrics)
        print(f"  {arm:12s} feats {len(columns):3d}  RMSE {metrics['rmse_sgd']:8.3f}  "
              f"MAE {metrics['mae_sgd']:8.3f}  bias {metrics['bias_sgd']:+8.2f}  "
              f"({seconds:.1f}s)")

        model.booster_.save_model(str(args.out / f"lgb_{arm}.txt"))
        preds.append(pd.DataFrame({
            "SOURCE_ROW_ID": frames["holdout"]["__row_id"].to_numpy(),
            "arm": arm, "origin": frames["holdout"]["__origin"].to_numpy(),
            "horizon": frames["holdout"]["__horizon"].to_numpy(),
            "actual_sgd": actual, "anchor_ksgd": anchor,
            "predicted_residual": residual, "predicted_sgd": predicted}))

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(args.out / "tree_metrics.csv", index=False)
    predictions = pd.concat(preds, ignore_index=True)
    predictions.to_csv(args.out / "tree_predictions.csv", index=False)

    base = rows[0]
    gain = {"delta_rmse_sgd": rows[1]["rmse_sgd"] - base["rmse_sgd"],
            "delta_mae_sgd": rows[1]["mae_sgd"] - base["mae_sgd"],
            "delta_bias_sgd": rows[1]["bias_sgd"] - base["bias_sgd"]}
    print(f"\ngain vs base: dRMSE {gain['delta_rmse_sgd']:+.3f}  "
          f"dMAE {gain['delta_mae_sgd']:+.3f}  dbias {gain['delta_bias_sgd']:+.3f} SGD")

    # horizon and origin breakdowns
    horizon_rows = []
    for h in sorted(predictions["horizon"].unique()):
        sub = predictions[predictions.horizon == h]
        entry = {"horizon": int(h), "n": int(len(sub) // 2)}
        for arm in ("base", "global_mean"):
            a = sub[sub.arm == arm]
            err = a.predicted_sgd.to_numpy() - a.actual_sgd.to_numpy()
            entry[f"{arm}_rmse"] = float(np.sqrt((err ** 2).mean()))
            entry[f"{arm}_bias"] = float(err.mean())
        entry["delta_rmse"] = entry["global_mean_rmse"] - entry["base_rmse"]
        horizon_rows.append(entry)
    horizon_frame = pd.DataFrame(horizon_rows)
    horizon_frame.to_csv(args.out / "tree_by_horizon.csv", index=False)
    print("\nhorizon  n     base    +mean    delta    base_bias  mean_bias")
    for r in horizon_rows:
        print(f"  {r['horizon']:2d} {r['n']:6d}  {r['base_rmse']:7.1f}  "
              f"{r['global_mean_rmse']:7.1f}  {r['delta_rmse']:+7.2f}   "
              f"{r['base_bias']:+8.2f}  {r['global_mean_bias']:+8.2f}")

    origin_rows = []
    for o in sorted(predictions["origin"].unique()):
        sub = predictions[predictions.origin == o]
        entry = {"origin": int(o), "n": int(len(sub) // 2)}
        for arm in ("base", "global_mean"):
            a = sub[sub.arm == arm]
            err = a.predicted_sgd.to_numpy() - a.actual_sgd.to_numpy()
            entry[f"{arm}_rmse"] = float(np.sqrt((err ** 2).mean()))
        entry["delta_rmse"] = entry["global_mean_rmse"] - entry["base_rmse"]
        origin_rows.append(entry)
    pd.DataFrame(origin_rows).to_csv(args.out / "tree_by_origin.csv", index=False)
    print(f"\nscored origins: {[r['origin'] for r in origin_rows]} "
          f"(each holds {origin_rows[0]['n']:,} rows)")

    run_manifest = {
        "generated": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "contract": manifest["contract"], "split": "orig", "stage": "A",
        "data_npz": str(DATA.resolve().relative_to(ROOT)),
        "data_npz_sha256": manifest["npz_sha256"],
        "new_feature": NEW_FEATURE,
        "new_feature_state": manifest["new_feature_state"],
        "numeric_order": numeric_names, "categorical_order": categorical_names,
        "anchor_role": "output offset only; excluded from learned columns",
        "target_definition": manifest["target_definition"],
        "reconstruction": manifest["reconstruction"],
        "params": PARAMS, "seed": SEED, "gain": gain,
        "lightgbm": lightgbm.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "python": platform.python_version(),
        "holdout_identification_note": (
            "The holdout has a single origin, so the new column is constant across all "
            "35,882 scored rows and can only act as a fixed offset there. Any gain is a "
            "property of that one origin, not evidence of temporal generalisation."),
        "scope_note": (
            "Inner-only, single seed, single window. <0.1 SGD is a tie. Outer B results are "
            "a different protocol and are not comparable to this table."),
    }
    (args.out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(f"\nwrote {args.out.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
