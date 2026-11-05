"""Rank 8: chronology-safe, lightweight TabPFN-to-T0 teacher distillation.

The available teacher forecasts cover sampled queries in two historical origins,
not all 121k outer-origin fitting cases. This tests a frozen-backbone student
correction; it does not claim to be full end-to-end transformer distillation.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.evaluate_transformer_tabpfn_correction import (
    PREPARED, SOURCE, TEACHER, fold_frame,
)

OUT = REPORTS / "transformer_ranked_experiments/08_tabpfn_distillation"


def student_features(frame):
    """Teacher-free features available to the student at inference."""
    x = pd.DataFrame({
        "scratch_prediction": frame.predicted,
        "area": frame.FLOOR_AREA_SQM,
        "age": frame.FLAT_AGE_YEARS,
        "latitude": frame.LATITUDE,
        "longitude": frame.LONGITUDE,
        "cbd": frame.DISTANCE_TO_CBD_KM,
        "horizon": frame.TEMP_FORECAST_HORIZON,
    })
    for kind in ("3-room", "4-room", "5-room", "executive"):
        x[f"type_{kind}"] = frame.FLAT_TYPE_CLEAN.eq(kind).astype(float)
    return x.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)


def weighted_rmse(y, p, w):
    return float(np.sqrt(np.average((np.asarray(p)-np.asarray(y))**2,
                                    weights=np.asarray(w))))


def fit_student(frame, alpha):
    x = student_features(frame)
    scaler = StandardScaler().fit(x)
    target = frame.teacher.to_numpy()-frame.predicted.to_numpy()
    model = Ridge(alpha=alpha).fit(scaler.transform(x), target,
                                   sample_weight=frame.weight.to_numpy())
    return scaler, model


def student_predict(pair, frame):
    scaler, model = pair
    return frame.predicted.to_numpy()+model.predict(scaler.transform(student_features(frame)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / "prepared.json").read_text())
    if sha256(PREPARED) != manifest["sha256"]:
        raise ValueError("Original prepared cache changed")
    paths = [TEACHER, PREPARED]
    paths += [SOURCE / f"{name}/predictions.csv" for name in (
        "calibration/T0/early_2022", "calibration/T0/early_2023",
        "outer/T0/seed42", "outer/T0/seed43", "outer/T0/seed44")]
    before = {str(p): sha256(p) for p in paths}
    prepared = joblib.load(PREPARED)
    teacher = pd.read_csv(TEACHER)
    first = fold_frame(prepared, teacher, "early_2022")
    second = fold_frame(prepared, teacher, "early_2023")
    selection = []
    w = second.weight.to_numpy()
    selection.append({"alpha": "none", "teacher_mimic_rmse_sgd":
        weighted_rmse(second.teacher, second.predicted, w),
        "actual_rmse_sgd": weighted_rmse(second.actual, second.predicted, w)})
    for alpha in (10., 100., 1000., 10000.):
        prediction = student_predict(fit_student(first, alpha), second)
        selection.append({"alpha": str(alpha),
                          "teacher_mimic_rmse_sgd": weighted_rmse(second.teacher, prediction, w),
                          "actual_rmse_sgd": weighted_rmse(second.actual, prediction, w)})
    grid = pd.DataFrame(selection).sort_values("teacher_mimic_rmse_sgd")
    grid.to_csv(OUT / "earlier_selection.csv", index=False)
    chosen = grid.iloc[0].alpha
    combined = pd.concat([first, second], ignore_index=True)
    model = None if chosen == "none" else fit_student(combined, float(chosen))
    scores = []
    for seed in (42, 43, 44):
        outer = fold_frame(prepared, teacher, "outer", seed)
        pred = outer.predicted.to_numpy() if model is None else student_predict(model, outer)
        outer["distilled_prediction"] = pred
        outer.to_csv(OUT / f"outer_seed{seed}.csv", index=False, float_format="%.10f")
        weights = outer.weight.to_numpy()
        scores.append({"seed": seed,
            "student_rmse_sgd": weighted_rmse(outer.actual, pred, weights),
            "scratch_rmse_sgd": weighted_rmse(outer.actual, outer.predicted, weights),
            "teacher_rmse_sgd": weighted_rmse(outer.actual, outer.teacher, weights),
            "teacher_mimic_rmse_sgd": weighted_rmse(outer.teacher, pred, weights),
            "mean_student_shift_sgd": float(np.mean(pred-outer.predicted))})
    pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)
    if {str(p): sha256(Path(p)) for p in before} != before:
        raise ValueError("Source artifact changed")
    (OUT / "verification.json").write_text(json.dumps({
        "source_hashes": before, "chosen_alpha": chosen,
        "selection_target": "historical teacher forecasts only",
        "first_teacher_rows": len(first), "second_teacher_rows": len(second),
        "outer_origin_fitting_rows": len(prepared["folds"]["outer_d0"]["X_train"]),
        "scope": "frozen T0 backbone with ridge teacher imitation head",
        "outer_labels_used_for_selection": False,
    }, indent=2), encoding="utf-8")
    print(grid.to_string(index=False), flush=True)
    print(pd.DataFrame(scores).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
