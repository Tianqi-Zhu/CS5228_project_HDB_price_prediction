"""Freeze the global-rent-mean feature bank for a frozen v2.1 population.

Builds `M_t` from **all** training transactions up to the stage's origin bound, then emits
the per-row feature vector at each row's own origin. The output layout matches the frozen
v5 feature npz, so the existing tree runner consumes it unchanged with `--arms base,global`.

The point of the contrast: `base` is the frozen v2.1 static+context+history block, whose
global rent entry is a **cumulative** average. This bank adds the monthly nominal level and
its short-horizon shape. Only this one block is added, so a difference is attributable to it.

Nothing here extrapolates a slope forward, and no lag is chosen by correlating over the full
timeline: the windows are fixed at 1/3/6/12 months, declared before the run.
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

from experiments.global_rent_mean_v6 import series as S  # noqa: E402

TRAIN_CSV = ROOT / "data" / "processed" / "train_spatial_features.csv"
V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
OUT_DIR = ROOT / "reports" / "global_rent_mean_v6"


def month_index(frame: pd.DataFrame) -> np.ndarray:
    months = pd.PeriodIndex(frame["RENT_APPROVAL_DATE"], freq="M")
    first = months.min()
    return np.asarray([(m - first).n for m in months], dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="orig")
    parser.add_argument("--stage", default="A")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "features_origA_grm.npz")
    args = parser.parse_args(argv)

    from neural_fusion_v2 import contract as C

    frame = pd.read_csv(TRAIN_CSV)
    frame["__month"] = month_index(frame)
    manifest = C.load_manifest(V2_DIR)
    fit = C.load_population(V2_DIR, manifest, args.split, args.stage, "fit")
    holdout = C.load_population(V2_DIR, manifest, args.split, args.stage, "holdout")

    fit_end = int(max(fit.samples.origin_month_index.max(),
                      holdout.samples.origin_month_index.max()))
    observed = frame[frame["__month"] <= fit_end]
    means, counts = S.build_series(observed, rent_col="MONTHLY_RENT",
                                   month_col="__month", fit_end=fit_end)
    names = S.feature_names()
    print(f"{args.split}/{args.stage}: panel fit_end {fit_end}, "
          f"{len(observed):,} transactions across {int((counts > 0).sum())} observed months")
    print(f"monthly mean rent: {np.nanmin(means):.1f} .. {np.nanmax(means):.1f} SGD "
          f"(first {means[0]:.1f}, last {means[fit_end]:.1f})")
    empty = int((counts == 0).sum())
    print(f"months with no transactions (missing, not zero): {empty}")

    payload = {"feature_names": np.array(names)}
    diagnostics = {"split": args.split, "stage": args.stage,
                   "fit_end_month_index": fit_end,
                   "observed_months": int((counts > 0).sum()),
                   "empty_months": empty,
                   "feature_names": names,
                   "definitions": {
                       "grm_latest_log": "log of the most recent observed monthly mean "
                                         "at or before the origin",
                       "grm_latest_age": "months between the origin and that observation",
                       "grm_dlog_1": "log change of the monthly mean vs the previous month",
                       "grm_mean{3,6,12}_log": "log of the EQUAL-MONTH-WEIGHT mean of "
                                               "monthly means over the trailing window "
                                               "(not a pooled transaction mean)",
                       "grm_valid{3,6,12}": "number of observed months in that window",
                       "grm_log_last3_over_prior3": "log(mean of last 3 monthly means) - "
                                                    "log(mean of the 3 before it)",
                       "grm_count_latest": "transactions in the origin month",
                       "grm_count_mean12": "mean monthly transaction count over 12 months",
                   },
                   "caveat": ("M_t is raw nominal rent over all transactions, so it moves "
                              "with flat-type/area/location mix. This is a raw market "
                              "proxy, not a composition-adjusted index. A whole-timeline "
                              "correlation is motivation, not evidence of per-dwelling "
                              "prediction gain."),
                   "windows_declared_before_run": list(S.WINDOWS),
                   "monthly_means_sgd": [None if not np.isfinite(v) else float(v)
                                         for v in means],
                   "monthly_counts": [int(c) for c in counts]}

    for label, population in (("fit", fit), ("holdout", holdout)):
        samples = population.samples
        n = len(samples)
        matrix = np.full((n, len(names)), np.nan)
        origins = samples.origin_month_index.to_numpy()
        for i in range(n):
            matrix[i] = S.features_at(means, counts, int(origins[i]))
        payload[f"{label}__X"] = matrix
        payload[f"{label}__row_id"] = samples.SOURCE_ROW_ID.to_numpy()
        payload[f"{label}__actual_sgd"] = samples.actual_rent_sgd.to_numpy(float)
        payload[f"{label}__horizon"] = samples.horizon_months.to_numpy()
        payload[f"{label}__target_month_index"] = samples.target_month_index.to_numpy()
        payload[f"{label}__origin_month_index"] = origins

        idx = {name: j for j, name in enumerate(names)}
        complete = np.isfinite(matrix).all(axis=1)
        latest_age = matrix[:, idx["grm_latest_age"]]
        diagnostics[label] = {
            "rows": int(n),
            "fully_finite_rows": int(complete.sum()),
            "fully_finite_pct": round(100 * float(complete.mean()), 2),
            "latest_age_max": float(np.nanmax(latest_age)),
            "mean12_finite_pct": round(100 * float(
                np.isfinite(matrix[:, idx["grm_mean12_log"]]).mean()), 2),
            "origin_min": int(origins.min()), "origin_max": int(origins.max()),
        }
        print(f"\n{label}: {n:,} rows, fully finite {diagnostics[label]['fully_finite_pct']}%, "
              f"mean12 finite {diagnostics[label]['mean12_finite_pct']}%, "
              f"oldest latest-level age {diagnostics[label]['latest_age_max']:.0f} months")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    diagnostics["npz_sha256"] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    diagnostics["npz"] = str(args.out.resolve().relative_to(ROOT))
    diagnostics["train_csv_sha256"] = hashlib.sha256(TRAIN_CSV.read_bytes()).hexdigest()
    # One manifest per output stem. A shared name let the early/A run overwrite the orig
    # metadata, so a reader of the orig NPZ could find early's fit_end and month series.
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(f"\nwrote {args.out.resolve().relative_to(ROOT)}  "
          f"sha256 {diagnostics['npz_sha256'][:16]}…")
    print(f"wrote {manifest_path.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
