"""Summarise the result CSVs for REVIEW.md (run via run.sh)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def main() -> None:
    base = config.REPORT_DIR
    pd.set_option("display.width", 200)

    abl = pd.read_csv(base / "metrics_ablation.csv")
    print("=== ABLATION (seed 42, main split) ===")
    sub = abl[(abl.seed == 42) & (abl.split_id == "main")].drop_duplicates("model")
    print(sub[["model", "rmse_sgd", "mae_sgd", "bias_sgd", "n_features"]].to_string(index=False))

    print("\n=== ABLATION delta vs baseline (seed 42, main) ===")
    base_rmse = sub.loc[sub.model == "lgb_spatial_baseline", "rmse_sgd"].iloc[0]
    for row in sub.itertuples():
        print(f"{row.model:32s} {row.rmse_sgd:8.3f}  {row.rmse_sgd - base_rmse:+8.3f}")

    print("\n=== SPLITS (seed 42) ===")
    sp = pd.read_csv(base / "metrics_splits.csv")
    print(sp[sp.seed == 42].pivot_table(index="model", columns="split_id",
                                        values="rmse_sgd").round(3).to_string())
    print("\nrows per split:")
    print(sp[sp.seed == 42].groupby("split_id")["n_rows"].first().to_string())
    print("\nmonths per split (from manifest):")
    import json
    manifest = json.loads((base / "run_manifest.json").read_text())
    for split_id, spec in manifest["splits"].items():
        print(f"  {split_id:8s} cutoff={spec['cutoff_month']} "
              f"val={spec['val_start']}..{spec['val_end']} "
              f"({spec['n_validation_months_expected']} months)")

    print("\n=== FOCUSED (seed 42) ===")
    fo = pd.read_csv(base / "metrics_focused.csv")
    print(fo[fo.seed == 42].pivot_table(index="model", columns="split_id",
                                        values="rmse_sgd").round(3).to_string())
    print("\nfocused rows per split:")
    print(fo[fo.seed == 42].groupby("split_id")["n_rows"].first().to_string())

    print("\n=== CATBOOST ===")
    print(pd.read_csv(base / "metrics_catboost.csv")[
        ["model", "rmse_sgd", "mae_sgd", "bias_sgd", "n_features"]].to_string(index=False))

    print("\n=== SMOOTHING ===")
    print(pd.read_csv(base / "metrics_smoothing.csv")[
        ["model", "prior_strength_rows", "rmse_sgd", "mae_sgd", "bias_sgd"]].to_string(index=False))

    print("\n=== SEEDS ===")
    seeds = pd.read_csv(base / "metrics_seeds.csv")
    print(seeds.groupby("model")["rmse_sgd"].agg(["count", "nunique", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
