"""LightGBM base / recent / trend arms on the frozen orig/A features.

Three arms, identical rows, targets, horizons, folds and hyper-parameters; only the added
feature block changes:

* ``base``   — the v2.1 static + context + history scalars, i.e. what the frozen contract
               already gives a tree. Note the history block is *cumulative*: with
               `low = zeros` it averages from the panel start, so it is not a recent window.
* ``recent`` — base + the rolling 3/6/12-month per-sqm level features and the 3-vs-prior-3
               change, per level (global / flat_type / town_ft).
* ``trend``  — base + the explicit level, slope and residual decomposition: the recent
               level, the six-month log slope, and recent minus the cumulative mean,
               all on one shared basis.

The ``recent`` arm isolates "is there short-run level information beyond the cumulative
history", and ``trend`` isolates "is there a slope/residual component beyond the level".
Both are compared against the same ``base``, so any difference is attributable to the added
columns and to nothing else in the pipeline.

LightGBM settings are pinned to 300 trees because that is the budget this diagnostic was
authorised at. The frozen v3 macro controls used 400 trees on orig/B, so those numbers are
**not** directly comparable to these; a 400-tree bridge arm is available via ``--bridge``
to make the comparison honest when needed.

Missing values are passed through as NaN rather than imputed: LightGBM routes NaN down its
own branch, and imputing a recent-rent level would invent a market observation.
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
V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
OUT_DIR = ROOT / "reports" / "market_dynamics_v5"

CATEGORICALS = ("TOWN_CLEAN", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "FURNISHED_CLEAN")
SEED = 42

BASE_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
    "min_child_samples": 30, "reg_lambda": 3.0, "n_jobs": 2, "verbosity": -1,
}


def load_v21_scalars(split: str = "orig", stage: str = "A") -> pd.DataFrame:
    """v2.1 static + context + history scalars for orig/A, keyed to SOURCE_ROW_ID."""
    from neural_fusion_v2 import contract as C

    manifest = C.load_manifest(V2_DIR)
    frames = {}
    for population in ("fit", "holdout"):
        pop = C.load_population(V2_DIR, manifest, split, stage, population)
        arrays = pop.arrays
        blocks = {
            "static": pd.DataFrame(arrays.static, columns=list(arrays.static_order)),
            "context": pd.DataFrame(arrays.context, columns=list(arrays.context_order)),
            "history": pd.DataFrame(arrays.history, columns=list(arrays.history_order)),
        }
        frame = pd.concat([blocks["static"], blocks["context"], blocks["history"]], axis=1)
        frame.insert(0, "SOURCE_ROW_ID", arrays.row_id)
        frames[population] = frame
    return frames


def score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {"rmse_sgd": float(np.sqrt(np.mean(np.square(error)))),
            "mae_sgd": float(np.mean(np.abs(error))),
            "bias_sgd": float(np.mean(error)),
            "error_sd_sgd": float(np.std(error)),
            "n_rows": int(len(error))}


def arm_columns(feature_names: list[str], arm: str) -> list[str]:
    """Which frozen columns belong to each arm. `base` uses none of them."""
    if arm == "base":
        return []
    if arm == "recent":
        keep = []
        for name in feature_names:
            if name.startswith(("recent3_logpsm", "recent6_logpsm", "recent12_logpsm",
                                "recent3_over_prior3_log", "n3__", "n12__")):
                keep.append(name)
        return keep
    if arm == "trend":
        keep = []
        for name in feature_names:
            if name.startswith(("recent3_logpsm", "slope6_logpsm", "valid6__",
                                "ratio_source__", "slope_source__", "fallback__")):
                keep.append(name)
        return keep
    if arm == "global":
        # the global-bank arm from experiments/global_rent_mean_v6: all of its columns
        return [name for name in feature_names if name.startswith("grm_")]
    raise ValueError(f"unknown arm {arm!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--bridge", action="store_true",
                        help="also run 400 trees so the v3 macro controls are comparable")
    parser.add_argument("--arms", default="base,recent,trend")
    parser.add_argument("--features", type=Path, default=FEATURES,
                        help="frozen feature npz (defaults to the transaction-weighted set)")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "tree_arms")
    parser.add_argument("--split", default="orig", help="v2.1 split id")
    parser.add_argument("--stage", default="A", help="v2.1 stage")
    args = parser.parse_args(argv)

    try:
        import lightgbm
        from lightgbm import LGBMRegressor
    except Exception as error:                                   # noqa: BLE001
        print(f"BLOCKED: LightGBM unavailable: {error!r}")
        return 3

    frozen = np.load(args.features, allow_pickle=True)
    print(f"features: {args.features.resolve().relative_to(ROOT)}")
    feature_names = [str(n) for n in frozen["feature_names"]]
    print(f"frozen features: {len(feature_names)} per row, "
          f"npz sha256 {hashlib.sha256(args.features.read_bytes()).hexdigest()[:16]}…")
    print(f"lightgbm {lightgbm.__version__}, numpy {np.__version__}, "
          f"pandas {pd.__version__}, python {platform.python_version()}")

    scalars = load_v21_scalars(args.split, args.stage)
    fit_base = scalars["fit"]
    hold_base = scalars["holdout"]

    # align the frozen matrix to the scalar frame by row id, not by position
    fit_ids = frozen["fit__row_id"]
    hold_ids = frozen["holdout__row_id"]
    fit_index = {rid: i for i, rid in enumerate(fit_ids)}
    hold_index = {rid: i for i, rid in enumerate(hold_ids)}
    if len(fit_index) != len(fit_ids) or len(hold_index) != len(hold_ids):
        raise AssertionError("duplicate SOURCE_ROW_ID in the frozen features")
    order_fit = np.asarray([fit_index[rid] for rid in fit_base["SOURCE_ROW_ID"]])
    order_hold = np.asarray([hold_index[rid] for rid in hold_base["SOURCE_ROW_ID"]])

    X_fit_new = pd.DataFrame(frozen["fit__X"][order_fit], columns=feature_names)
    X_hold_new = pd.DataFrame(frozen["holdout__X"][order_hold], columns=feature_names)

    # The frozen NPZ carries its own copy of the target and horizon. They must agree with
    # the contract's populations row-for-row, or the two sources have drifted and nothing
    # downstream is trustworthy. Compared exactly, not approximately.
    from neural_fusion_v2 import contract as C
    manifest_v2 = C.load_manifest(V2_DIR)
    for label, population, order in (("fit", "fit", order_fit),
                                     ("holdout", "holdout", order_hold)):
        contract_pop = C.load_population(V2_DIR, manifest_v2, args.split, args.stage,
                                        population)
        contract_ids = contract_pop.samples["SOURCE_ROW_ID"].to_numpy()
        if label == "fit":
            if not np.array_equal(contract_ids, fit_base["SOURCE_ROW_ID"].to_numpy()):
                raise AssertionError("fit row ids differ between contract and scalars")
        frozen_ids = frozen[f"{label}__row_id"]
        if not np.array_equal(frozen_ids[order], contract_ids):
            raise AssertionError(f"{label}: frozen row ids do not match the contract")
        if not np.allclose(frozen[f"{label}__actual_sgd"][order],
                           contract_pop.samples["actual_rent_sgd"].to_numpy(dtype=float),
                           rtol=0, atol=0):
            raise AssertionError(f"{label}: frozen target differs from the contract target")
        if not np.array_equal(frozen[f"{label}__horizon"][order],
                              contract_pop.samples["horizon_months"].to_numpy()):
            raise AssertionError(f"{label}: frozen horizon differs from the contract")
    print("targets and horizons verified identical to the v2.1 contract populations")

    y_fit = frozen["fit__actual_sgd"][order_fit].astype(float)
    y_hold = frozen["holdout__actual_sgd"][order_hold].astype(float)
    print(f"fit {len(y_fit):,} rows, holdout {len(y_hold):,} rows")

    grids = [(args.trees, "")]
    if args.bridge:
        grids.append((400, "_bridge400"))

    args.out.mkdir(parents=True, exist_ok=True)
    rows, predictions = [], []
    for trees, suffix in grids:
        params = dict(BASE_PARAMS, n_estimators=trees, random_state=SEED)
        for arm in args.arms.split(","):
            columns = arm_columns(feature_names, arm)
            frame_fit = pd.concat([fit_base, X_fit_new[columns]], axis=1) if columns \
                else fit_base.copy()
            frame_hold = pd.concat([hold_base, X_hold_new[columns]], axis=1) if columns \
                else hold_base.copy()
            categorical = [c for c in CATEGORICALS if c in frame_fit.columns]
            unseen_report = {}
            for column in categorical:
                fit_codes = frame_fit[column].astype("int64")
                hold_codes = frame_hold[column].astype("int64")
                categories = sorted(set(fit_codes.unique()))
                # A holdout category the fit stage never saw has no learned split. Map it to
                # an explicit unseen code rather than letting pandas silently coerce it to
                # NaN (which LightGBM would also treat as its own missing branch).
                unseen = sorted(set(hold_codes.unique()) - set(categories))
                if unseen:
                    unseen_code = max(categories) + 1
                    hold_codes = hold_codes.where(~hold_codes.isin(unseen), unseen_code)
                    categories = categories + [unseen_code]
                    unseen_report[column] = {
                        "codes": [int(code) for code in unseen],
                        "mapped_to": int(unseen_code),
                        "n_rows": int(hold_codes.isin([unseen_code]).sum())}
                frame_fit[column] = pd.Categorical(fit_codes, categories=categories)
                frame_hold[column] = pd.Categorical(hold_codes, categories=categories)
            if unseen_report:
                print(f"    holdout categories unseen at fit: {unseen_report}")

            model = LGBMRegressor(**params)
            started = time.perf_counter()
            model.fit(frame_fit.drop(columns=["SOURCE_ROW_ID"]), y_fit,
                      categorical_feature=categorical)
            seconds = time.perf_counter() - started
            predicted = np.asarray(
                model.predict(frame_hold.drop(columns=["SOURCE_ROW_ID"])), dtype=float)
            if not np.isfinite(predicted).all():
                raise ValueError(f"{arm}{suffix}: non-finite predictions")

            metrics = {"arm": arm, "suffix": suffix or "primary", "trees": trees,
                       "seed": SEED, "n_features_added": len(columns),
                       "unseen_categories": json.dumps(unseen_report, sort_keys=True),
                       **score(y_hold, predicted), "seconds": round(seconds, 1)}
            rows.append(metrics)
            print(f"  {arm+suffix:18s} trees {trees}  +{len(columns):3d} feats  "
                  f"RMSE {metrics['rmse_sgd']:8.3f}  MAE {metrics['mae_sgd']:8.3f}  "
                  f"bias {metrics['bias_sgd']:+8.2f}  ({seconds:.1f}s)")

            predictions.append(pd.DataFrame({
                "SOURCE_ROW_ID": hold_base["SOURCE_ROW_ID"].to_numpy(),
                "arm": arm, "suffix": suffix or "primary", "trees": trees,
                "actual_rent_sgd": y_hold, "predicted_rent_sgd": predicted}))

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(args.out / "tree_arms_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        args.out / "tree_arms_predictions.csv", index=False)

    # paired deltas against base at the same tree count
    print("\npaired delta vs base (same trees, same rows):")
    deltas = []
    for trees in {row["trees"] for row in rows}:
        base = next(r for r in rows if r["arm"] == "base" and r["trees"] == trees)
        for arm in ("recent", "trend"):
            other = next((r for r in rows if r["arm"] == arm and r["trees"] == trees), None)
            if other is None:
                continue
            delta = other["rmse_sgd"] - base["rmse_sgd"]
            deltas.append({"trees": trees, "arm": arm, "delta_rmse_sgd": delta,
                           "delta_mae_sgd": other["mae_sgd"] - base["mae_sgd"]})
            print(f"  {arm:8s} @{trees}: RMSE {delta:+.3f} SGD  "
                  f"MAE {other['mae_sgd'] - base['mae_sgd']:+.3f} SGD")

    manifest = {
        "generated": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "seed": SEED, "params": BASE_PARAMS, "trees": [g[0] for g in grids],
        "split": args.split, "stage": args.stage,
        "features_npz_sha256": hashlib.sha256(args.features.read_bytes()).hexdigest(),
        "features_npz": str(args.features.resolve().relative_to(ROOT)),
        "feature_names": feature_names,
        "lightgbm": lightgbm.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "python": platform.python_version(),
        "note": ("300 trees as authorised. The frozen v3 macro controls used 400 trees on "
                 "orig/B, so their RMSE is not comparable to these arms; --bridge adds a "
                 "400-tree run on the same rows. Paired deltas vs base are the comparison "
                 "of interest. Single seed, exploratory."),
        "deltas": deltas,
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {args.out.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
