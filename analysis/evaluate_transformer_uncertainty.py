"""Rank 9: chronological residual-quantile intervals for T0 and TabPFN.

The intervals are empirical diagnostics. A time shift means ordinary split
conformal finite-sample coverage guarantees do not apply to the later period.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.config import REPORTS
from analysis.data import sha256

SOURCE = REPORTS / "spatial_temporal_transformer"
TEACHER = REPORTS / "tabpfn_abcde_ensemble/predictions_all_origins.csv"
OUT = REPORTS / "transformer_ranked_experiments/09_uncertainty_calibration"
ORIGIN = {"early_2022": "2022-09", "early_2023": "2023-07", "outer": "2024-06"}
LEVELS = (0.80, 0.90)


def weighted_quantile(values, weights, level):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values)
    vals, w = values[order], weights[order]
    if len(vals) == 0 or not np.isfinite(w).all() or np.sum(w) <= 0:
        raise ValueError("Empty or invalid calibration weights")
    return float(vals[min(np.searchsorted(np.cumsum(w), level*np.sum(w)), len(vals)-1)])


def group(horizon):
    h = np.asarray(horizon)
    return np.select([h <= 3, h <= 6, h <= 10], [0, 1, 2], default=3)


def frame(teacher, fold, seed):
    name = f"calibration/T0/{fold}" if fold != "outer" else f"outer/T0/seed{seed}"
    data = pd.read_csv(SOURCE / name / "predictions.csv")
    teach = teacher.loc[teacher.origin.eq(ORIGIN[fold])].set_index("row_id")
    aligned = teach.reindex(data.row_id)
    if aligned.ABCDE_equal.isna().any() or not np.allclose(data.actual, aligned.actual):
        raise ValueError(f"Prediction row or target mismatch in {fold}")
    data["teacher"] = aligned.ABCDE_equal.to_numpy(float)
    data["horizon_group"] = group(data.horizon)
    return data


def calibrate(data, center, level, method):
    residual = np.abs(data.actual.to_numpy()-data[center].to_numpy())
    weights = data.weight.to_numpy()
    global_q = weighted_quantile(residual, weights, level)
    values = np.full(4, global_q)
    if method == "horizon":
        groups = data.horizon_group.to_numpy()
        for k in range(4):
            mask = groups == k
            if mask.sum() >= 30:
                local_q = weighted_quantile(residual[mask], weights[mask], level)
                share = float(np.sum(weights[mask])/(np.sum(weights[mask])+500.))
                values[k] = share*local_q + (1-share)*global_q
    return values


def evaluate(data, center, q, level, split, method, seed):
    y = data.actual.to_numpy()
    p = data[center].to_numpy()
    weight = data.weight.to_numpy()
    width = q[data.horizon_group.to_numpy()]
    miss = np.maximum(np.abs(y-p)-width, 0.)
    covered = (np.abs(y-p) <= width).astype(float)
    alpha = 1.-level
    record = {"split": split, "center": center, "method": method,
        "level": level, "seed": seed, "rows": len(data),
        "coverage": float(np.average(covered, weights=weight)),
        "mean_width_sgd": float(np.average(2*width, weights=weight)),
        "interval_score_sgd": float(np.average(2*width+2*miss/alpha, weights=weight)),
        "rmse_sgd": float(np.sqrt(np.average((p-y)**2, weights=weight)))}
    return record


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [TEACHER]
    sources += [SOURCE / f"{name}/predictions.csv" for name in (
        "calibration/T0/early_2022", "calibration/T0/early_2023",
        "outer/T0/seed42", "outer/T0/seed43", "outer/T0/seed44")]
    before = {str(p): sha256(p) for p in sources}
    teacher = pd.read_csv(TEACHER)
    first = frame(teacher, "early_2022", 42)
    second = frame(teacher, "early_2023", 42)
    validation = []
    for center in ("predicted", "teacher"):
        for method in ("global", "horizon"):
            for level in LEVELS:
                q = calibrate(first, center, level, method)
                validation.append(evaluate(second, center, q, level,
                                           "early_2023", method, 42))
    val = pd.DataFrame(validation)
    val.to_csv(OUT / "earlier_validation.csv", index=False)
    selection = {}
    for center in ("predicted", "teacher"):
        subset = val[val.center.eq(center)]
        scores = subset.groupby("method").interval_score_sgd.mean().sort_values()
        selection[center] = str(scores.index[0])
    calibration = pd.concat([first, second], ignore_index=True)
    records = []
    q_table = []
    for center in ("predicted", "teacher"):
        method = selection[center]
        for level in LEVELS:
            q = calibrate(calibration, center, level, method)
            q_table.append({"center": center, "method": method, "level": level,
                            **{f"horizon_group_{k}_halfwidth_sgd": float(v)
                               for k, v in enumerate(q)}})
            for seed in (42, 43, 44):
                outer = frame(teacher, "outer", seed)
                records.append(evaluate(outer, center, q, level, "outer", method, seed))
                if level == 0.90:
                    result = outer[["row_id", "month", "actual", "weight", "horizon"]].copy()
                    result["center"] = outer[center]
                    result["lower_90"] = outer[center]-q[outer.horizon_group.to_numpy()]
                    result["upper_90"] = outer[center]+q[outer.horizon_group.to_numpy()]
                    result.to_csv(OUT / f"outer_{center}_seed{seed}_90.csv", index=False,
                                  float_format="%.10f")
    pd.DataFrame(records).to_csv(OUT / "outer_scores.csv", index=False)
    pd.DataFrame(q_table).to_csv(OUT / "calibration_halfwidths.csv", index=False)
    if {str(p): sha256(Path(p)) for p in before} != before:
        raise ValueError("Source artifact changed")
    (OUT / "verification.json").write_text(json.dumps({
        "source_hashes": before,
        "selected_method": selection,
        "selection_basis": "earlier-2023 mean 80/90 percent interval score",
        "calibration_rows": [len(first), len(second)],
        "outer_labels_used_for_selection": False,
        "coverage_guarantee_under_time_shift": False,
    }, indent=2), encoding="utf-8")
    print("SELECTED", selection, flush=True)
    print(val.to_string(index=False), flush=True)
    print(pd.DataFrame(records).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
