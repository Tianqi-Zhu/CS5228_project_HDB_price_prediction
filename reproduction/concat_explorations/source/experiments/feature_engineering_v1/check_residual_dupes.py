"""Confirm the residual exact duplicates are only cross-file repetition."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

KEY = ["model", "split_id", "seed", "parameters", "feature_groups"]
METRICS = ["rmse_sgd", "mae_sgd", "bias_sgd", "n_rows"]
FILES = ["metrics_ablation.csv", "metrics_splits.csv", "metrics_focused.csv",
         "metrics_catboost.csv", "metrics_smoothing.csv", "metrics_seeds.csv"]

base = config.REPORT_DIR
frames = []
for name in FILES:
    f = pd.read_csv(base / name)
    f["source_file"] = name
    frames.append(f)
all_rows = pd.concat(frames, ignore_index=True)

dupes = all_rows[all_rows.duplicated(subset=KEY + METRICS, keep=False)]
print(f"rows participating in an exact-duplicate group: {len(dupes)}")
print(f"\ndistinct keys: {dupes.groupby(KEY, dropna=False).ngroups}\n")

print("each duplicate group and which file(s) it came from:")
for key, group in dupes.groupby(KEY, dropna=False):
    files = sorted(group["source_file"].unique())
    spread = group["rmse_sgd"].nunique()
    flag = "CROSS-FILE-SAME-RUN" if len(files) > 1 and spread == 1 else "WITHIN-FILE"
    print(f"  {key[0]:28s} {key[1]:8s} seed={key[2]:<3} files={files} rmse={group['rmse_sgd'].iloc[0]:.6f} -> {flag}")

within = [k for k, g in dupes.groupby(KEY, dropna=False) if len(g["source_file"].unique()) == 1]
print(f"\nwithin-file duplicate groups remaining: {len(within)}")
