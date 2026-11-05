"""Summarize completed tree-cover ablations against matching frozen predictions."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd

from analysis.compare_worldcover_tree_cover import VARIANTS, parent_paths, prepare, reference
from analysis.prepare_worldcover_tree_cover import locations


def main():
    prepare()
    cfg, out, _ = locations()
    parent_out, parent_path = parent_paths(cfg)
    parent = joblib.load(parent_path)
    rows = []
    for stage, entry in parent.items():
        b = entry["bundle"]
        baseline = reference(stage, parent_out, b)
        weights = np.asarray(b.get("query_weights", np.ones(len(b["X_query"]))))
        for variant in VARIANTS:
            path = out / f"predictions_{stage}_{variant}.csv"
            if variant == "baseline":
                predicted = baseline.predicted.to_numpy()
            elif path.exists():
                frame = pd.read_csv(path).set_index("row_id")
                if not frame.index.is_unique or len(frame) != len(baseline):
                    raise ValueError(f"Prediction rows differ: {path}")
                frame = frame.loc[baseline.index]
                np.testing.assert_array_equal(frame.actual, baseline.actual)
                predicted = frame.predicted.to_numpy()
            else:
                continue
            if not np.isfinite(predicted).all():
                raise ValueError("Incomplete predictions")
            error = predicted - baseline.actual.to_numpy()
            rows.append({"stage": stage, "variant": variant, "rows": len(error), "weight": float(weights.sum()),
                         "weighted_squared_sum": float(np.sum(weights*error**2)), "weighted_absolute_sum": float(np.sum(weights*np.abs(error))),
                         "rmse_sgd": float(np.sqrt(np.average(error**2, weights=weights))),
                         "mae_sgd": float(np.average(np.abs(error), weights=weights)), "bias_sgd": float(np.average(error, weights=weights))})
    scores = pd.DataFrame(rows)
    for stage, subset in scores.groupby("stage"):
        mask = scores.stage.eq(stage)
        base = subset.loc[subset.variant.eq("baseline"), "rmse_sgd"].iloc[0]
        scores.loc[mask, "delta_rmse_sgd"] = scores.loc[mask, "rmse_sgd"] - base
    scores.to_csv(out / "scores.csv", index=False)
    pooled = []
    inner = scores.loc[scores.stage.str.startswith("inner_")]
    for variant, subset in inner.groupby("variant"):
        if len(subset) == 2:
            pooled.append({"variant": variant, "rows": int(subset.rows.sum()),
                           "rmse_sgd": float(np.sqrt(subset.weighted_squared_sum.sum()/subset.weight.sum())),
                           "mae_sgd": float(subset.weighted_absolute_sum.sum()/subset.weight.sum())})
    pd.DataFrame(pooled).to_csv(out / "inner_pooled_scores.csv", index=False)
    completed = scores.loc[~scores.variant.eq("baseline")]
    print(completed[["stage", "variant", "rows", "rmse_sgd", "delta_rmse_sgd"]].to_string(index=False))
    print("Smoke runs are excluded. Screening scores are population-weighted; outer scores use every query equally.")


if __name__ == "__main__":
    main()
