"""Rank 7: chronological TabPFN ensemble correction using scratch-model signals."""
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


TEACHER = REPORTS / "tabpfn_abcde_ensemble/predictions_all_origins.csv"
SOURCE = REPORTS / "spatial_temporal_transformer"
PREPARED = ROOT / ".cache/spatial_temporal_transformer/prepared.joblib"
OUT = REPORTS / "transformer_ranked_experiments/07_tabpfn_correction"
ORIGINS = {"early_2022": "2022-09", "early_2023": "2023-07", "outer": "2024-06"}


def fold_frame(prepared, teacher, fold, seed=42):
    name = f"calibration/T0/{fold}" if fold != "outer" else f"outer/T0/seed{seed}"
    original = pd.read_csv(SOURCE / name / "predictions.csv")
    teachers = teacher[teacher.origin.eq(ORIGINS[fold])].set_index("row_id")
    aligned = teachers.reindex(original.row_id)
    if aligned.ABCDE_equal.isna().any() or not np.allclose(aligned.actual, original.actual):
        raise ValueError("Teacher and independent transformer forecasts misaligned")
    features = prepared["folds"][f"{fold}_d0"]["X_query"]
    if not np.array_equal(features.index, original.row_id.to_numpy()):
        raise ValueError("Forecast row IDs misaligned")
    result = original.copy()
    result["teacher"] = aligned.ABCDE_equal.to_numpy()
    for col in ("FLOOR_AREA_SQM", "FLAT_AGE_YEARS", "LATITUDE", "LONGITUDE",
                "DISTANCE_TO_CBD_KM", "TEMP_FORECAST_HORIZON"):
        result[col] = pd.to_numeric(features[col], errors="coerce").to_numpy()
    result["FLAT_TYPE_CLEAN"] = features.FLAT_TYPE_CLEAN.astype(str).to_numpy()
    return result


def design_matrix(frame):
    base = pd.DataFrame({
        "scratch_minus_teacher": frame.predicted-frame.teacher,
        "area": frame.FLOOR_AREA_SQM, "age": frame.FLAT_AGE_YEARS,
        "latitude": frame.LATITUDE, "longitude": frame.LONGITUDE,
        "cbd": frame.DISTANCE_TO_CBD_KM,
        "horizon": frame.TEMP_FORECAST_HORIZON,
    })
    for kind in ("3-room", "4-room", "5-room", "executive"):
        base[f"type_{kind}"] = frame.FLAT_TYPE_CLEAN.eq(kind).astype(float)
    return base.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)


def fit(first, mode, alpha):
    y = first.actual.to_numpy()-first.teacher.to_numpy()
    if mode == "blend":
        delta = first.predicted.to_numpy()-first.teacher.to_numpy()
        weight = first.weight.to_numpy(float)
        coefficient = float(np.dot(weight*delta, y)/(np.dot(weight*delta, delta)+alpha))
        return float(np.clip(coefficient, -.25, 1.))
    x = design_matrix(first)
    scale = StandardScaler().fit(x)
    model = Ridge(alpha=alpha, fit_intercept=False).fit(
        scale.transform(x), y, sample_weight=first.weight.to_numpy(float))
    return scale, model


def correction(model, frame, mode):
    if mode == "blend":
        return model*(frame.predicted.to_numpy()-frame.teacher.to_numpy())
    scaler, ridge = model
    return ridge.predict(scaler.transform(design_matrix(frame)))


def score(frame, corr=None):
    predicted = frame.teacher.to_numpy()
    if corr is not None:
        predicted = predicted+corr
    return float(np.sqrt(np.average((predicted-frame.actual.to_numpy())**2,
                                    weights=frame.weight.to_numpy())))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / "prepared.json").read_text())
    if sha256(PREPARED) != manifest["sha256"]:
        raise ValueError("Original prepared fold changed")
    before = {str(p): sha256(p) for p in (TEACHER, PREPARED,
              *(SOURCE / f"{name}/predictions.csv" for name in
                ("calibration/T0/early_2022", "calibration/T0/early_2023",
                 "outer/T0/seed42", "outer/T0/seed43", "outer/T0/seed44")))}
    prepared = joblib.load(PREPARED)
    teacher = pd.read_csv(TEACHER)
    first, second = (fold_frame(prepared, teacher, fold) for fold in ("early_2022", "early_2023"))
    selection = [{"mode": "none", "alpha": 0., "earlier_rmse_sgd": score(second),
                  "earlier_delta_sgd": 0.}]
    for mode in ("blend", "ridge"):
        for alpha in ((0., 1e4, 1e5) if mode == "blend" else (10., 100., 1000.)):
            model = fit(first, mode, alpha)
            value = score(second, correction(model, second, mode))
            selection.append({"mode": mode, "alpha": alpha, "earlier_rmse_sgd": value,
                              "earlier_delta_sgd": value-score(second)})
    grid = pd.DataFrame(selection).sort_values(["earlier_rmse_sgd", "mode", "alpha"])
    grid.to_csv(OUT / "earlier_selection.csv", index=False)
    mode, alpha = str(grid.iloc[0]["mode"]), float(grid.iloc[0]["alpha"])
    calibration = pd.concat([first, second], ignore_index=True)
    model = None if mode == "none" else fit(calibration, mode, alpha)
    result = []
    for seed in (42,43,44):
        outer = fold_frame(prepared, teacher, "outer", seed)
        corr = np.zeros(len(outer)) if model is None else correction(model, outer, mode)
        outer["corrected"] = outer.teacher+corr
        outer["correction_sgd"] = corr
        outer.to_csv(OUT / f"outer_seed{seed}.csv", index=False, float_format="%.10f")
        result.append({"seed": seed, "teacher_rmse_sgd": score(outer),
                       "corrected_rmse_sgd": score(outer, corr),
                       "delta_rmse_sgd": score(outer, corr)-score(outer),
                       "scratch_rmse_sgd": float(np.sqrt(np.mean((outer.predicted-outer.actual)**2))),
                       "mean_correction_sgd": float(corr.mean())})
    pd.DataFrame(result).to_csv(OUT / "scores.csv", index=False)
    if {str(p): sha256(Path(p)) for p in before} != before:
        raise ValueError("Original teacher or transformer artifact changed")
    (OUT / "verification.json").write_text(json.dumps({
        "source_hashes": before, "chosen_mode": mode, "chosen_alpha": alpha,
        "teacher_earlier_predictions_are_chronological": True,
        "teacher_fit_details": "see source TabPFN ABCDE report",
        "outer_labels_used_for_selection": False}, indent=2), encoding="utf-8")
    print(pd.DataFrame(result).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
