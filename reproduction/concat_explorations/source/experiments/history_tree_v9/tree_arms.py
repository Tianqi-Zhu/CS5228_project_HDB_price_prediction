"""LightGBM base vs +history on the legal v8 rev2 rows.

Two arms, identical rows, targets, folds and hyper-parameters. The only difference is
whether the 50 history columns are appended:

* ``base``    — v8 house (21 numeric + 3 native categorical) + 2 calendar + 6 stats per
                market channel (90), i.e. exactly the v8 StatsMarket definition, using
                calendar windows.
* ``history`` — base + the 50 history columns.

Both use the **direct** target `MONTHLY_RENT/1000`, no anchor and no residual transform, so
nothing here is comparable to the older raw-SGD or anchor-offset tree scores. That is
deliberate: the plan forbids standing in an old number for this baseline.

The market statistics are recomputed here from the v8 `market_values` window block using the
same six suffixes and the same calendar windows as the v8 data module, so the tree sees the
same quantity the neural side sees. They are **not** "last three observations".
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
for candidate in (ROOT, ROOT / "experiments"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE_NPZ = ROOT / "reports" / "target_asof_v8" / "data" / "orig_dataset.npz"
BASE_SCHEMA = ROOT / "reports" / "target_asof_v8" / "data" / "schema.json"
HIST_NPZ = ROOT / "reports" / "history_fusion_v9" / "data" / "orig_history.npz"
OUT = ROOT / "reports" / "history_fusion_v9" / "data" / "tree"

PARAMS = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
          "min_child_samples": 30, "reg_lambda": 3.0, "n_jobs": 2, "verbosity": -1}
SEED = 42
STATS_SUFFIXES = ["last_valid", "mean3", "mean12", "recent3_minus_prior3", "n_valid",
                  "months_since_last_valid"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def market_stats(values: np.ndarray, observed: np.ndarray, step_valid: np.ndarray) -> np.ndarray:
    """Six calendar-window statistics per channel, from the v8 window block.

    `mean3` is the mean over the last three CALENDAR steps of the window (t-3..t-1), not the
    last three observed values. A window with no observation in a slot yields NaN, never an
    average of older data — that would stretch the window across the gap.
    """
    n_windows, n_steps, n_channels = values.shape
    out = np.full((n_windows, n_channels * len(STATS_SUFFIXES)), np.nan)
    for w in range(n_windows):
        for c in range(n_channels):
            series = values[w, :, c]
            obs = observed[w, :, c] & step_valid[w]
            base = c * len(STATS_SUFFIXES)
            valid_positions = np.flatnonzero(obs)
            if valid_positions.size == 0:
                out[w, base + STATS_SUFFIXES.index("n_valid")] = 0.0
                continue
            last = valid_positions[-1]
            out[w, base + 0] = series[last]
            out[w, base + 4] = float(obs.sum())
            out[w, base + 5] = float(n_steps - 1 - last)

            def cal_mean(lo, hi):
                block = slice(max(0, lo), min(n_steps, hi))
                m = obs[block]
                return float(series[block][m].mean()) if m.any() else np.nan

            out[w, base + 1] = cal_mean(n_steps - 3, n_steps)
            out[w, base + 2] = cal_mean(0, n_steps)
            recent = cal_mean(n_steps - 3, n_steps)
            prior = cal_mean(n_steps - 6, n_steps - 3)
            if np.isfinite(recent) and np.isfinite(prior):
                out[w, base + 3] = recent - prior
    return out


def score(actual, predicted):
    err = np.asarray(predicted, float) - np.asarray(actual, float)
    return {"rmse_sgd": float(np.sqrt(np.mean(err ** 2))),
            "mae_sgd": float(np.mean(np.abs(err))),
            "bias_sgd": float(np.mean(err)),
            "error_sd_sgd": float(np.std(err)),
            "n_rows": int(len(err))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    try:
        import lightgbm
        from lightgbm import LGBMRegressor
    except Exception as error:                                   # noqa: BLE001
        print(f"BLOCKED: LightGBM unavailable: {error!r}")
        return 3

    base = np.load(BASE_NPZ, allow_pickle=True)
    schema = json.loads(BASE_SCHEMA.read_text())
    hist = np.load(HIST_NPZ, allow_pickle=True)
    channel_names = schema["market_channel_order"]
    house_names = schema["house_numeric_order"]
    cat_names = schema["house_categorical_order"]
    hist_names = [str(n) for n in hist["feature_names"]]
    print(f"lightgbm {lightgbm.__version__}, numpy {np.__version__}, "
          f"pandas {pd.__version__}, python {platform.python_version()}")
    print(f"base rows: train {len(base['train__row_id']):,} / "
          f"validation {len(base['validation__row_id']):,}; "
          f"history columns {len(hist_names)}")

    # ---- align history to base by row id, and refuse to proceed on any mismatch
    for population in ("train", "validation"):
        base_ids = base[f"{population}__row_id"]
        hist_ids = hist[f"{population}__row_id"]
        if not np.array_equal(base_ids, hist_ids):
            raise AssertionError(f"{population}: history row ids are not identical to base")
        if not np.array_equal(hist[f"{population}__actual_sgd"],
                              base[f"{population}__actual_sgd"].astype(np.float64)):
            raise AssertionError(f"{population}: actual rent disagrees between packages")
        if not np.array_equal(hist[f"{population}__target_month"],
                              base[f"{population}__target_month"]):
            raise AssertionError(f"{population}: target month disagrees")
    print("row ids, targets and target months are identical across the two packages")

    # ---- build the shared design matrix
    frames = {}
    for population in ("train", "validation"):
        n = len(base[f"{population}__row_id"])
        values = base["market_values"]
        observed = base["market_observed"]
        step_valid = base["market_step_valid"]
        stats = market_stats(values, observed, step_valid)
        window_ids = base[f"{population}__window_id"]
        market_block = stats[window_ids]

        frame = pd.DataFrame(base[f"{population}__house_numeric"], columns=house_names)
        for j, name in enumerate(cat_names):
            frame[name] = base[f"{population}__house_cat"][:, j]
        cal = base[f"{population}__calendar"]
        frame["cal_sin"] = cal[:, 0]
        frame["cal_cos"] = cal[:, 1]
        for j, name in enumerate(f"{channel_names[i]}__{s}" for i in range(len(channel_names))
                                 for s in STATS_SUFFIXES):
            frame[name] = market_block[:, j]
        frame["__actual"] = base[f"{population}__actual_sgd"].astype(np.float64)
        frame["__row_id"] = base[f"{population}__row_id"]
        frame["__target_month"] = base[f"{population}__target_month"]
        hist_block = pd.DataFrame(hist[f"{population}__history_numeric"], columns=hist_names)
        frame = pd.concat([frame, hist_block], axis=1)
        frames[population] = frame

    base_features = ([c for c in house_names] + list(cat_names)
                     + ["cal_sin", "cal_cos"]
                     + [f"{ch}__{s}" for ch in channel_names for s in STATS_SUFFIXES])
    print(f"base design: {len(base_features)} columns "
          f"({len(house_names)} house + {len(cat_names)} cat + 2 calendar "
          f"+ {len(channel_names) * len(STATS_SUFFIXES)} market stats)")
    print(f"history arm adds {len(hist_names)} columns -> "
          f"{len(base_features) + len(hist_names)} total")

    args.out.mkdir(parents=True, exist_ok=True)
    rows, preds = [], []
    for arm in ("base", "history"):
        columns = list(base_features)
        if arm == "history":
            columns += hist_names
        fit_x = frames["train"][columns].copy()
        hold_x = frames["validation"][columns].copy()
        cats = list(cat_names)
        unseen_report = {}
        for name in cats:
            f_codes = fit_x[name].astype("int64")
            h_codes = hold_x[name].astype("int64")
            categories = sorted(set(f_codes.unique()))
            extra = sorted(set(h_codes.unique()) - set(categories))
            if extra:
                code = max(categories) + 1
                h_codes = h_codes.where(~h_codes.isin(extra), code)
                categories = categories + [code]
                unseen_report[name] = {"codes": [int(x) for x in extra],
                                       "n_rows": int((h_codes == code).sum())}
            fit_x[name] = pd.Categorical(f_codes, categories=categories)
            hold_x[name] = pd.Categorical(h_codes, categories=categories)
        if unseen_report:
            print(f"  {arm}: holdout codes unseen at fit: {unseen_report}")

        y_fit = frames["train"]["__actual"].to_numpy(float) / 1000.0
        y_hold = frames["validation"]["__actual"].to_numpy(float) / 1000.0
        model = LGBMRegressor(**dict(PARAMS, random_state=SEED))
        started = time.perf_counter()
        model.fit(fit_x, y_fit, categorical_feature=cats)
        seconds = time.perf_counter() - started
        residual = np.asarray(model.predict(hold_x), dtype=float)
        predicted = residual * 1000.0

        metrics = {"arm": arm, "n_features": len(columns), "trees": PARAMS["n_estimators"],
                   "seed": SEED, "n_train": int(len(fit_x)), "n_scored": int(len(hold_x)),
                   **score(frames["validation"]["__actual"].to_numpy(float), predicted),
                   "seconds": round(seconds, 2),
                   "unseen_holdout_codes": json.dumps(unseen_report, sort_keys=True)}
        rows.append(metrics)
        print(f"  {arm:8s} feats {len(columns):3d}  RMSE {metrics['rmse_sgd']:8.3f}  "
              f"MAE {metrics['mae_sgd']:8.3f}  bias {metrics['bias_sgd']:+8.2f}  "
              f"({seconds:.1f}s)")

        model.booster_.save_model(str(args.out / f"lgb_{arm}.txt"))
        gain = model.booster_.feature_importance(importance_type="gain")
        pd.DataFrame({"feature": columns, "gain": gain}).sort_values(
            "gain", ascending=False).to_csv(args.out / f"gain_{arm}.csv", index=False)

        preds.append(pd.DataFrame({
            "SOURCE_ROW_ID": frames["validation"]["__row_id"].to_numpy(),
            "arm": arm,
            "target_month": frames["validation"]["__target_month"].to_numpy(),
            "actual_sgd": frames["validation"]["__actual"].to_numpy(),
            "predicted_sgd": predicted}))

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(args.out / "tree_metrics.csv", index=False)
    predictions = pd.concat(preds, ignore_index=True)
    predictions.to_csv(args.out / "tree_predictions.csv", index=False)

    gain_row = {"delta_rmse_sgd": rows[1]["rmse_sgd"] - rows[0]["rmse_sgd"],
                "delta_mae_sgd": rows[1]["mae_sgd"] - rows[0]["mae_sgd"],
                "delta_bias_sgd": rows[1]["bias_sgd"] - rows[0]["bias_sgd"]}
    print(f"\ngain vs base: dRMSE {gain_row['delta_rmse_sgd']:+.3f}  "
          f"dMAE {gain_row['delta_mae_sgd']:+.3f}  "
          f"dbias {gain_row['delta_bias_sgd']:+.3f} SGD")

    # ---- by month and by history support
    by_month = []
    for month in sorted(predictions["target_month"].unique()):
        sub = predictions[predictions.target_month == month]
        entry = {"target_month": int(month), "n": int(len(sub) // 2)}
        for arm in ("base", "history"):
            a = sub[sub.arm == arm]
            err = a.predicted_sgd.to_numpy() - a.actual_sgd.to_numpy()
            entry[f"{arm}_rmse"] = float(np.sqrt((err ** 2).mean()))
            entry[f"{arm}_bias"] = float(err.mean())
        entry["delta_rmse"] = entry["history_rmse"] - entry["base_rmse"]
        by_month.append(entry)
    pd.DataFrame(by_month).to_csv(args.out / "by_month.csv", index=False)
    print("\nmonth  n     base    +hist   delta")
    for r in by_month:
        print(f"  {r['target_month']:3d} {r['n']:6d}  {r['base_rmse']:7.1f}  "
              f"{r['history_rmse']:7.1f}  {r['delta_rmse']:+7.2f}")

    # support buckets from the finest grouping that has history
    count_idx = hist_names.index("hist_town_street_block_ft__count")
    counts = hist["validation__history_raw"][:, count_idx]
    buckets = [(0, 0, "n=0"), (1, 1, "n=1"), (2, 4, "n=2-4"), (5, 9, "n=5-9"),
               (10, 10**9, "n>=10")]
    support_rows = []
    for lo, hi, label in buckets:
        mask = (counts >= lo) & (counts <= hi)
        if not mask.any():
            continue
        entry = {"bucket": label, "n": int(mask.sum())}
        for arm in ("base", "history"):
            a = predictions[(predictions.arm == arm)]
            err = a.predicted_sgd.to_numpy()[mask] - a.actual_sgd.to_numpy()[mask]
            entry[f"{arm}_rmse"] = float(np.sqrt((err ** 2).mean()))
        entry["delta_rmse"] = entry["history_rmse"] - entry["base_rmse"]
        support_rows.append(entry)
    pd.DataFrame(support_rows).to_csv(args.out / "by_support.csv", index=False)
    print("\nblock-level support  n     base    +hist   delta")
    for r in support_rows:
        print(f"  {r['bucket']:18s} {r['n']:6d}  {r['base_rmse']:7.1f}  "
              f"{r['history_rmse']:7.1f}  {r['delta_rmse']:+7.2f}")

    manifest = {
        "generated": pd.Timestamp.now(tz="Asia/Singapore").isoformat(),
        "split": "orig", "fold": "orig/A-derived",
        "target": "MONTHLY_RENT/1000 direct; no anchor, no residual transform",
        "params": PARAMS, "seed": SEED, "gain": gain_row,
        "base_dataset": {"path": str(BASE_NPZ.relative_to(ROOT)),
                         "sha256": sha256(BASE_NPZ)},
        "history_dataset": {"path": str(HIST_NPZ.relative_to(ROOT)),
                            "sha256": sha256(HIST_NPZ)},
        "base_features": base_features,
        "history_features": hist_names,
        "market_stats_note": ("six calendar-window statistics per channel recomputed from "
                              "the v8 window block; mean3 is the last three CALENDAR steps, "
                              "not the last three observations"),
        "lightgbm": lightgbm.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "python": platform.python_version(),
        "scope_note": ("Single fold, single seed, two arms only. Not comparable to older "
                       "raw-SGD or anchor-offset tree scores; those use a different target "
                       "and are excluded by design."),
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {args.out.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
