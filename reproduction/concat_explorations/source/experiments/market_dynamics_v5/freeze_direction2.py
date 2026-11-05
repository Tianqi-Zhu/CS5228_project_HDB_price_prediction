"""Freeze the direction-2 residual targets for orig/A: original anchor vs causal forecast anchor.

## What direction 2 actually is

Not "a different group of columns". The claim under test is:

> predict the **residual** `actual - anchor`, where `anchor` is a *forward* extrapolation of
> the causal recent level along its causal slope, instead of predicting `actual` directly.

Two residual targets are built, differing only in the anchor:

* ``orig``  — `anchor = anchor_rent_sgd`, the contract's own anchor. This is the tree
  protocol's existing baseline: the anchor is a *backward* reference rent, so predicting
  `actual - anchor` asks the model to learn a rent **change** rather than a rent level.
* ``fcst``  — `anchor = psm_recent3 * FLOOR_AREA_SQM * exp(slope6 * h * DAMP)`. This is the
  direction-2 anchor: the recent level is projected forward along the causal six-month
  slope, damped because a six-month slope extrapolated 12 months out is not credible.

`h` is the sample's own horizon in months, and it is applied **once**, inside the exponent.
The level and the slope inside the forecast come from `level_and_slope`, i.e. one shared
basis — see that function for why.

## Causal guarantees

`psm_recent3` and `slope6` read only months `<= origin` (established by `panel.py` and
asserted here). `h` is not future information: it is the horizon of the sample, known at
prediction time, not the realised outcome. Nothing reads a label after `origin`.

## Honesty about the comparison

* These features/targets are **tree-internal only**. The frozen FT v4 pipeline works with a
  raw-SGD residual protocol on its own scalers and cannot consume this NPZ as-is.
* `orig`-anchor arms are *not* equivalent to any earlier tree arm: base already contains
  `anchor_rent_sgd` as a feature and predicts raw `actual_rent_sgd`. Here base still holds
  that column, and the *target* changes instead. That difference is exactly the thing being
  varied, and it is recorded per arm.

## Output

`direction2_targets.npz` with, per population: the residual targets and anchors, plus the
full audit table (origin, horizon, anchor, shift, level, slope, basis provenance) so a
reviewer can recompute any row without trusting this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.market_dynamics_v5 import panel as P  # noqa: E402

TRAIN_CSV = ROOT / "data" / "processed" / "train_spatial_features.csv"
V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
OUT_DIR = ROOT / "reports" / "market_dynamics_v5"
OUT_NPZ = OUT_DIR / "direction2_targets.npz"

DAMP = 0.5          # pre-declared damping for the slope extrapolation
LEVEL = "town_ft"   # the level whose (level, slope) pair anchors the forecast


def month_index(frame: pd.DataFrame) -> np.ndarray:
    months = pd.PeriodIndex(frame["RENT_APPROVAL_DATE"], freq="M")
    first = months.min()
    return np.asarray([(m - first).n for m in months], dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--damp", type=float, default=DAMP)
    parser.add_argument("--level", default=LEVEL, choices=list(P.LEVELS))
    parser.add_argument("--out", type=Path, default=OUT_NPZ)
    args = parser.parse_args(argv)

    from neural_fusion_v2 import contract as C

    frame = pd.read_csv(TRAIN_CSV)
    frame["__month"] = month_index(frame)
    manifest = C.load_manifest(V2_DIR)

    fit = C.load_population(V2_DIR, manifest, "orig", "A", "fit")
    hold = C.load_population(V2_DIR, manifest, "orig", "A", "holdout")
    fit_end = int(max(fit.samples.origin_month_index.max(),
                      hold.samples.origin_month_index.max()))
    observed = frame[frame["__month"] <= fit_end].copy()

    town_ft = list(zip(observed["TOWN_CLEAN"].astype(str),
                       observed["FLAT_TYPE_CLEAN"].astype(str)))
    keys = {"global": [("all",)] * len(observed),
            "flat_type": [(t,) for t in observed["FLAT_TYPE_CLEAN"].astype(str)],
            "town_ft": town_ft}
    parents = {}
    for town, flat in keys["town_ft"]:
        parents[("town_ft", (town, flat))] = {"flat_type": (flat,), "global": ("all",)}
    for flat in keys["flat_type"]:
        parents[("flat_type", flat)] = {"flat_type": flat, "global": ("all",)}

    panel = P.build_panel(observed, level_keys=keys, rent_col="MONTHLY_RENT",
                          area_col="FLOOR_AREA_SQM", month_col="__month",
                          fit_end=fit_end, parents=parents)
    print(f"panel fit_end {fit_end}, rows {len(observed):,}; "
          f"anchor level {args.level!r}, damp {args.damp}")

    spec = P.FeatureSpec()
    names = spec.names()
    idx = {name: j for j, name in enumerate(names)}
    level_slope_columns = [f"recent3_logpsm__{args.level}",
                           f"slope6_logpsm__{args.level}"]

    payload = {"feature_names": np.array(names), "level": np.array([args.level]),
               "damp": np.array([args.damp]), "horizon_multiplier_applied_once":
               np.array([True])}
    diagnostics = {"damp": args.damp, "anchor_level": args.level,
                   "horizon_rule": "exp(slope * h * damp), h applied once",
                   "populations": {}}

    for label, population in (("fit", fit), ("holdout", hold)):
        samples = population.samples.copy()
        merged = samples.merge(
            frame[["SOURCE_ROW_ID", "TOWN_CLEAN", "FLAT_TYPE_CLEAN", "FLOOR_AREA_SQM"]],
            on="SOURCE_ROW_ID", how="left", validate="one_to_one")
        if merged[["TOWN_CLEAN", "FLAT_TYPE_CLEAN", "FLOOR_AREA_SQM"]].isna().any().any():
            raise AssertionError(f"{label}: rows missing keys or floor area")

        n = len(merged)
        level_psm = np.full(n, np.nan)
        slope = np.full(n, np.nan)
        basis_code = np.full(n, -1.0)
        valid_months = np.zeros(n)
        level_n = np.zeros(n)
        for i, record in enumerate(merged.itertuples(index=False)):
            row_keys = {"global": ("all",),
                        "flat_type": (str(record.FLAT_TYPE_CLEAN),),
                        "town_ft": (str(record.TOWN_CLEAN),
                                    str(record.FLAT_TYPE_CLEAN))}
            lv, sl, basis, vm, ln = P.level_and_slope(
                panel, args.level, row_keys[args.level],
                int(record.origin_month_index))
            if lv is not None:
                level_psm[i] = lv
                slope[i] = sl
                basis_code[i] = P.SOURCE_CODE.get(basis, P.FALLBACK_MISSING)
                valid_months[i] = vm
                level_n[i] = ln

        h = merged["horizon_months"].to_numpy(dtype=float)
        area = merged["FLOOR_AREA_SQM"].to_numpy(dtype=float)
        actual = merged["actual_rent_sgd"].to_numpy(dtype=float)

        orig_anchor = population.arrays.context[
            :, list(population.arrays.context_order).index("anchor_rent_sgd")
        ].astype(float)

        # the causal forward anchor; h appears once, inside the single exponent
        multiplier = np.exp(slope * h * args.damp)
        forecast = level_psm * area * multiplier
        trend_valid = np.isfinite(level_psm) & np.isfinite(slope) & np.isfinite(forecast)

        # Rows without a causal trend get NO correction: the forecast anchor falls back to
        # that row's own original anchor, exactly as if the trend term were absent. The
        # alternative, dropping them, would train the two arms on different row sets and
        # confound the protocol, which is what the earlier run did (52,310 vs 46,064).
        fcst_anchor = np.where(trend_valid, forecast, orig_anchor)
        shift = fcst_anchor - orig_anchor
        if not np.isfinite(fcst_anchor).all():
            raise AssertionError(f"{label}: fallback left a non-finite anchor")
        payload[f"{label}__row_id"] = merged["SOURCE_ROW_ID"].to_numpy()
        payload[f"{label}__actual_sgd"] = actual
        payload[f"{label}__orig_anchor"] = orig_anchor
        payload[f"{label}__fcst_anchor"] = fcst_anchor
        payload[f"{label}__residual_orig"] = actual - orig_anchor
        payload[f"{label}__residual_fcst"] = actual - fcst_anchor
        payload[f"{label}__origin_month_index"] = merged["origin_month_index"].to_numpy()
        payload[f"{label}__target_month_index"] = merged["target_month_index"].to_numpy()
        payload[f"{label}__horizon"] = merged["horizon_months"].to_numpy()
        payload[f"{label}__shift"] = shift
        payload[f"{label}__trend_valid"] = trend_valid
        payload[f"{label}__level_psm"] = level_psm
        payload[f"{label}__slope_per_month"] = slope
        payload[f"{label}__basis_code"] = basis_code
        payload[f"{label}__valid_months"] = valid_months
        payload[f"{label}__level_n"] = level_n
        payload[f"{label}__floor_area"] = area

        # residual target scale: if these are tiny relative to actual, the arm is trivial
        stats = {
            "rows": int(n), "trend_valid": int(trend_valid.sum()),
            "trend_valid_pct": round(100 * float(trend_valid.mean()), 2),
            "trend_fallback_to_orig_anchor": int((~trend_valid).sum()),
            "orig_residual_rmse_sgd": float(np.sqrt(np.mean(
                (actual - orig_anchor) ** 2))),
            "fcst_residual_rmse_sgd": float(np.sqrt(np.mean(
                (actual - fcst_anchor) ** 2))),
            "actual_mean_sgd": float(actual.mean()),
            "fcst_anchor_mean_sgd": float(np.nanmean(fcst_anchor)),
            "orig_anchor_mean_sgd": float(orig_anchor.mean()),
            "shift_mean_sgd": float(np.nanmean(shift)),
            "shift_sd_sgd": float(np.nanstd(shift)),
            "slope_mean_per_month": float(np.nanmean(slope)),
            "slope_sd_per_month": float(np.nanstd(slope)),
            "slope_p05_p95": [float(x) for x in np.nanpercentile(slope, [5, 95])],
            "basis_counts": {str(int(k)): int(v) for k, v in
                             zip(*np.unique(basis_code, return_counts=True))},
        }
        diagnostics["populations"][label] = stats
        print(f"\n{label}: {n:,} rows, trend valid {stats['trend_valid_pct']}% "
              f"({stats['trend_fallback_to_orig_anchor']:,} rows fall back to orig anchor)")
        print(f"  residual RMSE: orig {stats['orig_residual_rmse_sgd']:.2f} SGD, "
              f"fcst {stats['fcst_residual_rmse_sgd']:.2f} SGD")
        print(f"  anchors: actual {stats['actual_mean_sgd']:.1f}, "
              f"orig {stats['orig_anchor_mean_sgd']:.1f}, "
              f"fcst {stats['fcst_anchor_mean_sgd']:.1f} SGD")
        print(f"  shift  mean {stats['shift_mean_sgd']:+.1f} SD "
              f"{stats['shift_sd_sgd']:.1f} SGD")
        print(f"  slope  mean {stats['slope_mean_per_month']:+.5f} "
              f"SD {stats['slope_sd_per_month']:.5f} per month, "
              f"p05/p95 {stats['slope_p05_p95'][0]:+.4f}/{stats['slope_p05_p95'][1]:+.4f}")
        print(f"  basis {stats['basis_counts']} "
              f"(0=global 1=flat_type 2=town_ft -1=missing)")

    # direction-2 causality: the forecast anchor must depend on nothing after `origin`.
    # Structurally guaranteed by panel.py; re-asserted on the realised values.
    for label in ("fit", "holdout"):
        origins = payload[f"{label}__origin_month_index"]
        if int(origins.max()) > fit_end:
            raise AssertionError(f"{label}: an origin exceeds the panel fit_end")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    diagnostics["npz_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    diagnostics["npz"] = str(args.out.resolve().relative_to(ROOT))
    for label in ("fit", "holdout"):
        ids = payload[f"{label}__row_id"]
        diagnostics[f"{label}_row_id_sha256"] = hashlib.sha256(
            ids.tobytes()).hexdigest()
        diagnostics[f"{label}_n_rows"] = int(len(ids))
    # parity: both arms must train on identical rows, so the fallback must leave every row
    # usable. Assert rather than describe.
    for label in ("fit", "holdout"):
        anchor = payload[f"{label}__fcst_anchor"]
        if not np.isfinite(anchor).all():
            raise AssertionError(f"{label}: some rows have no forecast anchor after fallback")
    if diagnostics["populations"]["fit"]["rows"] != diagnostics["fit_n_rows"]:
        raise AssertionError("fit population size changed during freezing")
    diagnostics["scope_note"] = (
        "Tree-internal protocol only. The frozen FT v4 pipeline uses a raw-SGD residual "
        "with its own scalers and cannot consume this NPZ. The orig-anchor arm is not "
        "equivalent to earlier tree arms because the target, not the feature set, changes.")
    manifest_path = args.out.parent / "direction2_manifest.json"
    manifest_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"\nwrote {args.out.resolve().relative_to(ROOT)}  "
          f"sha256 {diagnostics['npz_sha256'][:16]}…")
    print(f"wrote {manifest_path.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
