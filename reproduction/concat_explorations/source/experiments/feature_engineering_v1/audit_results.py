"""Audit the result CSVs for duplicate keys and inconsistent scores.

A row's identity is (model, split_id, seed, parameters, feature_groups).  Rows
that share that key but disagree on a metric are *conflicts*: they are reported,
never silently deduplicated down to the better number.  A second file
(``metrics_duplicates.csv``) lists every conflicting group with the spread.

Run: ``./run.sh audit_results.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

KEY = ["model", "split_id", "seed", "parameters", "feature_groups"]
FILES = [
    "metrics_ablation.csv",
    "metrics_splits.csv",
    "metrics_focused.csv",
    "metrics_catboost.csv",
    "metrics_smoothing.csv",
    "metrics_seeds.csv",
]


def main() -> int:
    base = config.REPORT_DIR
    frames = []
    for name in FILES:
        path = base / name
        if not path.exists():
            print(f"MISSING {name}")
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = name
        frames.append(frame)

    if not frames:
        print("no result files found")
        return 1

    allrows = pd.concat(frames, ignore_index=True)
    print(f"total metric rows: {len(allrows)}")

    # ---- 1. exact duplicate rows (same key AND same scores) --------------
    metric_cols = ["rmse_sgd", "mae_sgd", "bias_sgd", "n_rows"]
    exact = allrows.duplicated(subset=KEY + metric_cols, keep="first")
    print(f"exact duplicate rows (identical key and scores): {int(exact.sum())}")
    if exact.any():
        dupes = allrows[exact].groupby(KEY + metric_cols, dropna=False).size()
        print(dupes.head(20).to_string())

    # ---- 2. conflicting scores under one key ----------------------------
    grouped = allrows.groupby(KEY, dropna=False)
    conflicts = grouped["rmse_sgd"].agg(["count", "nunique", "min", "max"])
    conflicts = conflicts[conflicts["nunique"] > 1]
    print(f"\nkeys with inconsistent rmse: {len(conflicts)}")
    if len(conflicts):
        print(conflicts.to_string())

    # ---- 3. same model across files, different split coverage -----------
    coverage = allrows.pivot_table(index="model", columns="split_id",
                                   values="rmse_sgd", aggfunc="nunique")
    inconsistent_across_splits = coverage[(coverage.fillna(0) > 1).any(axis=1)]
    print(f"\nmodel/split cells whose score itself varies internally: {len(inconsistent_across_splits)}")
    if len(inconsistent_across_splits):
        print(inconsistent_across_splits.to_string())

    # ---- 4. write the conflict report -----------------------------------
    rows = []
    for key, group in allrows.groupby(KEY, dropna=False):
        if group["rmse_sgd"].nunique() > 1:
            for entry in group.itertuples():
                rows.append({
                    "model": key[0], "split_id": key[1], "seed": key[2],
                    "feature_groups": key[4], "source_file": entry.source_file,
                    "rmse_sgd": entry.rmse_sgd, "mae_sgd": entry.mae_sgd,
                    "bias_sgd": entry.bias_sgd, "n_rows": entry.n_rows,
                    "n_features": getattr(entry, "n_features", None),
                    "conflict": "same key, different score",
                })
    report = pd.DataFrame(rows)
    out = base / "metrics_duplicates.csv"
    if report.empty:
        # Write an explicit empty-with-header file so the absence is documented.
        pd.DataFrame(columns=["model", "split_id", "seed", "feature_groups", "source_file",
                              "rmse_sgd", "mae_sgd", "bias_sgd", "n_rows", "n_features",
                              "conflict"]).to_csv(out, index=False)
        print(f"\nno conflicting keys -> wrote empty {out.name}")
    else:
        report.to_csv(out, index=False)
        print(f"\nwrote {out.name} with {len(report)} rows over "
              f"{report.groupby(['model','split_id','seed']).ngroups} conflicting keys")

    # ---- 5. seed determinism statement --------------------------------
    seeds = allrows[allrows["source_file"] == "metrics_seeds.csv"]
    if len(seeds):
        spread = seeds.groupby("model")["rmse_sgd"].agg(["count", "nunique"])
        identical = spread[spread["nunique"] == 1]
        print(f"\nmodels whose 3 seeds give bit-identical RMSE: {len(identical)}/{len(spread)}")
        print(identical.to_string())
        print("NOTE: identical scores across seeds indicate a deterministic fit "
              "(no row/column subsampling), not independent robustness evidence.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
