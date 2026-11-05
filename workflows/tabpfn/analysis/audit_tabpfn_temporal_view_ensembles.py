"""Independently verify saved temporal views, weights and reported metrics."""
from __future__ import annotations

import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import first: the runner isolates the pandas version required by old caches.
from analysis import compare_tabpfn_temporal_view_ensembles as run
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.experiments.temporal_feature_views import LEVEL_COLUMNS


def audit():
    run.verify_sources()
    run.prior.check_sources()  # Recheck inherited train/test and feature sources, too.
    rows = []
    for origin in run.ORIGINS:
        original = run.load_bundle(origin)
        for member in run.NEW:
            changed = run.load_prepared(origin, member)
            for key in ("y_train", "y_query", "train_baseline", "query_baseline"):
                np.testing.assert_array_equal(original[key], changed[key])
            for split in ("train", "query"):
                x, z = original[f"X_{split}"], changed[f"X_{split}"]
                pd.testing.assert_index_equal(x.index, z.index)
                same = [c for c in x if c in z]
                pd.testing.assert_frame_equal(x[same], z[same])
                if member == "relative_history":
                    for column in LEVEL_COLUMNS:
                        np.testing.assert_allclose(z[column + "_REL_BASE"],
                            x[column].to_numpy(float) - original[f"{split}_baseline"], rtol=0, atol=0, equal_nan=True)
                else:
                    pd.testing.assert_frame_equal(z[x.columns], x)
                if member in ("coe_origin", "stocks_origin") and split == "query":
                    # All ordinary macro values are frozen across the entire
                    # horizon. Only the explicit horizon interactions may vary.
                    for column in z.filter(like="MACRO_"):
                        if not column.endswith("_X_H"):
                            assert z[column].nunique(dropna=False) == 1, column
                rows.append(dict(origin=origin, member=member, split=split, rows=len(z), columns=z.shape[1],
                                 original_surviving_columns_exact=True, targets_and_baselines_exact=True))
    pool = pd.concat([run.aligned(o) for o in run.ORIGINS[:-1]], ignore_index=True)
    chosen = json.loads((run.OUT / "selection.json").read_text())
    solver = []
    for name, spec in chosen["models"].items():
        w = np.asarray(spec["weights"]); p = pool[spec["members"]].to_numpy()
        e = p - pool.actual.to_numpy()[:, None]; v = pool.weight.to_numpy(copy=True); v /= v.sum()
        # Independent numerical optimizer with a scaled objective.
        scale = np.mean(np.sum(v[:, None] * e**2, axis=0))
        objective = lambda z: np.sum(v * (e @ z)**2) / scale
        reference = minimize(objective, np.full(len(w), 1 / len(w)), method="SLSQP",
                             jac=lambda z: 2 * e.T @ (v * (e @ z)) / scale,
                             bounds=[(0., 1.)] * len(w), constraints={"type": "eq", "fun": lambda z: z.sum() - 1},
                             options={"ftol": 1e-13, "maxiter": 2000})
        assert reference.success, reference.message
        assert abs(objective(w) - reference.fun) < 1e-9
        solver.append(dict(model=name, independent_slsqp_agrees=True, normalized_objective_difference=float(objective(w) - reference.fun)))
    scores = pd.read_csv(run.OUT / "comparison.csv")
    predictions = pd.read_csv(run.OUT / "predictions_all_origins.csv", float_precision="round_trip")
    for score in scores.itertuples(index=False):
        group = predictions.loc[predictions.origin.eq(score.origin)]
        error = group[score.model].to_numpy() - group.actual.to_numpy()
        rmse = np.sqrt(np.average(error**2, weights=group.weight))
        mae = np.average(np.abs(error), weights=group.weight)
        assert abs(rmse - score.rmse_sgd) < 1e-8
        assert abs(mae - score.mae_sgd) < 1e-8
    run.write_json(run.OUT / "independent_audit.json", dict(verified=True, feature_checks=rows,
        weight_checks=solver, independently_recomputed_metric_rows=len(scores), inherited_source_hashes_verified=True,
        unchanged_submission_hash_verified=True))
    print(f"VERIFIED {len(rows)} feature matrices, {len(solver)} weight solutions and {len(scores)} metric rows")


if __name__ == "__main__": audit()
