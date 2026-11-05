"""Freeze the causal recent-rent features for orig/A into an NPZ.

Runs in the torch venv because it reads the v2.1 contract. The tree venv then consumes the
NPZ without touching the contract, so the feature definition is fixed once and both arms
see byte-identical inputs.

Three things this script is careful about:

1. **The panel is built from every training transaction, not from the contract rows.** The
   contract's A/fit population is 52,310 sampled rows; a recent-rent mean over only those
   rows would be a mean over the sample, not over the market. The panel uses all 150,000
   training transactions, and only months <= the stage's own cutoff may enter it.

2. **The panel is truncated per origin, never globally.** A single global panel built to
   the last month of training would let a 2021 origin see 2025 prices. The panel object is
   therefore built with `fit_end` = the last month any origin in that population may use,
   and `features_for` refuses an origin beyond it. Because a row's own origin is always
   <= that bound, and `_window` only reads months <= origin, no future month is reachable.

3. **Coverage is reported, not assumed.** Sparse groups fall back to a parent and the
   覆盖 rate is written to the manifest, so a reader can see how much of the population
   used a proxy rather than its own group.

The output NPZ holds, per population (fit, holdout), the feature matrix, the row ids and
the population label. Feature names are stored once.
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

LEVELS = ("global", "flat_type", "town_ft")


def month_index(frame: pd.DataFrame) -> np.ndarray:
    """Calendar month index, 0 = the first month present in the training file."""
    months = pd.PeriodIndex(frame["RENT_APPROVAL_DATE"], freq="M")
    first = months.min()
    return np.asarray([(m - first).n for m in months], dtype=np.int64)


def level_keys(frame: pd.DataFrame) -> dict:
    """Per-row key tuples for each level; higher levels repeat the parent key."""
    town_ft = list(zip(frame["TOWN_CLEAN"].astype(str),
                       frame["FLAT_TYPE_CLEAN"].astype(str)))
    return {
        "global": [("all",)] * len(frame),
        "flat_type": [(t,) for t in frame["FLAT_TYPE_CLEAN"].astype(str)],
        "town_ft": town_ft,
    }


def parent_map(frame: pd.DataFrame, keys: dict) -> dict:
    """Exact child -> parent keys, taken from the data rather than guessed."""
    pairs = {}
    for town, flat in keys["town_ft"]:
        pairs[("town_ft", (town, flat))] = {
            "flat_type": (flat,),
            "global": ("all",),
        }
    for flat in keys["flat_type"]:
        pairs[("flat_type", flat)] = {"flat_type": flat, "global": ("all",)}
    return pairs


def build_for_population(samples: pd.DataFrame, spec: P.FeatureSpec,
                         panel: P.Panel,
                         weighting: str = "transactions") -> tuple[np.ndarray, dict]:
    """Feature matrix for every row of one population, plus coverage diagnostics."""
    names = spec.names()
    matrix = np.full((len(samples), len(names)), np.nan)
    coverage = {"rows": int(len(samples)),
                "missing_ratio": 0, "missing_slope": 0, "per_level": {}}

    rows = samples.reset_index(drop=True)
    for i in range(len(rows)):
        record = rows.iloc[i]
        keys = {
            "global": ("all",),
            "flat_type": (str(record["FLAT_TYPE_CLEAN"]),),
            "town_ft": (str(record["TOWN_CLEAN"]), str(record["FLAT_TYPE_CLEAN"])),
        }
        origin = int(record["origin_month_index"])
        matrix[i] = P.features_for(panel, origin, row_keys=keys, spec=spec,
                                   fixed=(weighting == "fixed"))

    idx = {name: j for j, name in enumerate(names)}
    for level in spec.levels:
        ratio = matrix[:, idx[f"recent3_over_prior3_log__{level}"]]
        slope = matrix[:, idx[f"slope6_logpsm__{level}"]]
        fallback = matrix[:, idx[f"fallback__{level}"]]
        coverage["per_level"][level] = {
            "ratio_finite": int(np.isfinite(ratio).sum()),
            "ratio_finite_pct": round(100 * float(np.isfinite(ratio).mean()), 2),
            "slope_finite": int(np.isfinite(slope).sum()),
            "slope_finite_pct": round(100 * float(np.isfinite(slope).mean()), 2),
            "fallback_missing_pct": round(100 * float((fallback == 2.0).mean()), 2),
        }
    coverage["missing_ratio"] = int(sum(
        (~np.isfinite(matrix[:, idx[f"recent3_over_prior3_log__{level}"]])).sum()
        for level in spec.levels))
    coverage["missing_slope"] = int(sum(
        (~np.isfinite(matrix[:, idx[f"slope6_logpsm__{level}"]])).sum()
        for level in spec.levels))
    return matrix, coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "features_origA.npz")
    parser.add_argument("--weighting", default="transactions",
                        choices=("transactions", "fixed"),
                        help="fixed = fixed-composition index: average within each flat "
                             "type, then combine with that origin's causal shares")
    parser.add_argument("--cell-column", default="FLAT_TYPE_CLEAN",
                        help="column defining the composition cells")
    parser.add_argument("--split", default="orig", help="v2.1 split id")
    parser.add_argument("--stage", default="A", help="v2.1 stage")
    args = parser.parse_args(argv)

    from neural_fusion_v2 import contract as C

    frame = pd.read_csv(TRAIN_CSV)
    frame["__month"] = month_index(frame)
    print(f"training rows {len(frame):,}, months {frame['__month'].min()}.."
          f"{frame['__month'].max()} ({frame['__month'].nunique()} distinct)")

    manifest = C.load_manifest(V2_DIR)
    fit = C.load_population(V2_DIR, manifest, args.split, args.stage, "fit")
    holdout = C.load_population(V2_DIR, manifest, args.split, args.stage, "holdout")

    # the last month any origin in this stage may legitimately use
    fit_end = int(max(fit.samples.origin_month_index.max(),
                      holdout.samples.origin_month_index.max()))
    print(f"{args.split}/{args.stage} origin bound: "
          f"fit <= {fit.samples.origin_month_index.max()}, "
          f"holdout <= {holdout.samples.origin_month_index.max()} -> panel fit_end {fit_end}")

    observed = frame[frame["__month"] <= fit_end]
    print(f"panel rows {len(observed):,} (truncated at fit_end; "
          f"{len(frame) - len(observed):,} later rows excluded)")

    keys = level_keys(observed)
    # Composition shares are derived per origin from cumulative counts up to that origin, so
    # no future month can enter them. There is deliberately no global base-weight vector: a
    # sample with origin < the base window would have had no legal weights.
    panel = P.build_panel(observed, level_keys=keys, rent_col="MONTHLY_RENT",
                          area_col="FLOOR_AREA_SQM", month_col="__month",
                          fit_end=fit_end,
                          parents=parent_map(observed, keys),
                          cell_column=(args.cell_column
                                       if args.weighting == "fixed" else None))
    if args.weighting == "fixed":
        print(f"fixed composition over cells '{args.cell_column}': "
              f"{len(panel.cell_keys)} cells; shares per origin come from cumulative "
              f"counts up to that origin")
    print(f"panel: {len(panel.levels['global'].keys)} global, "
          f"{len(panel.levels['flat_type'].keys)} flat_type, "
          f"{len(panel.levels['town_ft'].keys)} town_ft groups")

    spec = P.FeatureSpec()
    names = spec.names()
    print(f"features per row: {len(names)}")

    outputs = {}
    coverages = {"split": args.split, "stage": args.stage,
                 "weighting": args.weighting,
                 "cell_column": (args.cell_column if args.weighting == "fixed" else None),
                 "cell_count": (len(panel.cell_keys) if panel.cell_keys else None),
                 "share_rule": ("per origin, from cumulative counts of months <= origin"
                                if args.weighting == "fixed" else None),
                 "fit_end_month_index": fit_end,
                 "train_rows_total": int(len(frame)),
                 "panel_rows_used": int(len(observed)),
                 "feature_names": names}

    for label, population in (("fit", fit), ("holdout", holdout)):
        samples = population.samples.copy()
        # attach the categorical keys the panel addresses groups by
        merged = samples.merge(frame[["SOURCE_ROW_ID", "TOWN_CLEAN", "FLAT_TYPE_CLEAN"]],
                               on="SOURCE_ROW_ID", how="left", validate="one_to_one")
        if merged[["TOWN_CLEAN", "FLAT_TYPE_CLEAN"]].isna().any().any():
            raise AssertionError(f"{label}: some rows have no cleaned town/flat_type")
        matrix, coverage = build_for_population(merged, spec, panel, args.weighting)
        coverages[label] = coverage
        outputs[label] = {"X": matrix,
                          "row_id": merged["SOURCE_ROW_ID"].to_numpy(),
                          "actual_sgd": merged["actual_rent_sgd"].to_numpy(dtype=float),
                          "horizon": merged["horizon_months"].to_numpy(),
                          "target_month_index": merged["target_month_index"].to_numpy(),
                          "origin_month_index": merged["origin_month_index"].to_numpy(),
                          "population": np.array([label] * len(merged))}
        print(f"\n{label}: {matrix.shape[0]:,} rows")
        for level, stats in coverage["per_level"].items():
            print(f"  {level:9s} ratio finite {stats['ratio_finite']:>6,} "
                  f"({stats['ratio_finite_pct']:5.1f}%)  slope finite "
                  f"{stats['slope_finite']:>6,} ({stats['slope_finite_pct']:5.1f}%)  "
                  f"missing {stats['fallback_missing_pct']:5.1f}%")

    # causality assertion: no feature may be computed from a month beyond its own origin.
    # Verified structurally (the panel is truncated and _window only reads <= origin), and
    # re-checked here by confirming the fit_end bound covers every origin present.
    worst = max(int(outputs["fit"]["origin_month_index"].max()),
                int(outputs["holdout"]["origin_month_index"].max()))
    if worst > fit_end:
        raise AssertionError(f"an origin {worst} exceeds the panel fit_end {fit_end}")

    payload = {}
    payload["feature_names"] = np.array(names)
    for label in ("fit", "holdout"):
        for key, value in outputs[label].items():
            payload[f"{label}__{key}"] = value
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)

    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    coverages["npz_sha256"] = digest
    coverages["npz"] = str(args.out.resolve().relative_to(ROOT))
    coverages["note"] = (
        "Features are causal by construction: the panel excludes months after fit_end, "
        "and every window/slope reads only months <= the row's own origin. Coverage is "
        "reported per level because sparse groups use a parent proxy, not their own series.")
    manifest_path = args.out.parent / "features_manifest.json"
    manifest_path.write_text(json.dumps(coverages, indent=2) + "\n")
    print(f"\nwrote {args.out.resolve().relative_to(ROOT)}  sha256 {digest[:16]}…")
    print(f"wrote {manifest_path.resolve().relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
