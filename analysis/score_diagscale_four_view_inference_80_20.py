"""Score the saved four-view DiagScale adapter with four inference views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import torch
from tabpfn.constants import ModelVersion

from analysis.check_diagscale_categorical_alignment_80_20 import category_positions
from analysis.compare_tabpfn_diagscale_80_20 import BASE, REFERENCE
from analysis.compare_tabpfn_v3_80_20 import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import REPORTS, ROOT
from analysis.evaluation import metrics
from analysis.tune_tabpfn_diagscale_80_20 import (
    SavedDiagScaleRegressor, inner_bundle, predict_batches, weighted_rmse,
)


TRAINED = REPORTS / "tabpfn_diagscale_categorical_alignment_4views_80_20"
OUT = REPORTS / "tabpfn_diagscale_categorical_alignment_4views_inference_80_20"
ADAPTER = ROOT / ".cache/tabpfn_diagscale_categorical_alignment_4views_80_20/adapter.pt"


def estimator(x: pd.DataFrame, y: np.ndarray, *, cache_on_device: bool):
    cat = category_positions(x)
    if cat != json.loads((TRAINED / "calibration.json").read_text())["categorical_indices"]:
        raise AssertionError("Categorical schema changed")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = SavedDiagScaleRegressor.create_default_for_version(
        version=ModelVersion.V3_5_FAST, adapter_path=str(ADAPTER),
        model_path=str(CHECKPOINT), categorical_features_indices=cat,
        device="cuda", n_estimators=4, auto_scale_n_estimators=False,
        random_state=42, fit_mode="fit_with_cache",
        keep_cache_on_device=cache_on_device,
        inference_config={"SUBSAMPLE_SAMPLES": None}, n_preprocessing_jobs=1,
    )
    model.fit(x, y)
    if model.n_train_samples_ != len(x):
        raise AssertionError("Inference context was truncated")
    return model, started


def score_calibration(*, cache_on_device: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = inner_bundle()
    x, q = bundle["X_fit"], bundle["X_cal"]
    if category_positions(x) != category_positions(q):
        raise AssertionError("Calibration schema changed")
    model, started = estimator(x, bundle["y_fit"], cache_on_device=cache_on_device)
    predicted = predict_batches(model, q, bundle["cal_baseline"], progress_label="calibration 4-view")
    previous = pd.read_csv(TRAINED / "calibration_predictions.csv")
    np.testing.assert_array_equal(previous.row_id.to_numpy(), q.index.to_numpy())
    result = previous.rename(columns={"predicted": "trained_4_infer_1"}).copy()
    result["trained_4_infer_4"] = predicted
    result.to_csv(OUT / "calibration_predictions.csv", index=False)
    write_json(OUT / "calibration.json", {
        "train_views": 4, "checkpoint_views": 4, "deployment_views": 4,
        "context_rows": len(x), "calibration_rows": len(q),
        "cache_on_device": cache_on_device,
        "rmse_weighted": weighted_rmse(bundle["y_cal_actual"], predicted, bundle["cal_weight"]),
        "rmse_unweighted": float(np.sqrt(np.mean((bundle["y_cal_actual"] - predicted) ** 2))),
        "fit_predict_seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "outer_labels_used_for_selection": False,
    })
    print(json.dumps(json.loads((OUT / "calibration.json").read_text())), flush=True)


def score_outer(*, cache_on_device: bool):
    OUT.mkdir(parents=True, exist_ok=True)
    base = joblib.load(BASE)
    reference = pd.read_csv(REFERENCE)
    previous = pd.read_csv(TRAINED / "predictions.csv")
    np.testing.assert_array_equal(reference.row_id.to_numpy(), previous.row_id.to_numpy())
    np.testing.assert_array_equal(reference.row_id.to_numpy(), base["X_query"].index.to_numpy())
    x, q = base["X_train"], base["X_query"]
    if category_positions(x) != category_positions(q):
        raise AssertionError("Outer schema changed")
    y = np.asarray(base["y_train"], float) - np.asarray(base["train_baseline"], float)
    model, started = estimator(x, y, cache_on_device=cache_on_device)
    predicted = predict_batches(model, q, np.asarray(base["query_baseline"], float),
                                progress_label="outer 4-view")
    result = previous.rename(columns={"diagscale_aligned": "trained_4_infer_1"}).copy()
    result["trained_4_infer_4"] = predicted
    result.to_csv(OUT / "predictions.csv", index=False, float_format="%.8f")
    labels = ("frozen", "trained_4_infer_1", "trained_4_infer_4")
    pd.DataFrame([{"model": label, "n": len(result), **metrics(result.actual, result[label])}
                  for label in labels]).to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame([{"month": month, "model": label, "n": len(part), **metrics(part.actual, part[label])}
                  for month, part in result.groupby("month", sort=True)
                  for label in labels]).to_csv(OUT / "monthly_scores.csv", index=False)
    write_json(OUT / "timing.json", {
        "train_views": 4, "checkpoint_views": 4, "deployment_views": 4,
        "context_rows": len(x), "validation_rows": len(q),
        "cache_on_device": cache_on_device,
        "fit_predict_seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "outer_labels_used_for_selection": False,
    })
    print(pd.read_csv(OUT / "scores.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibration", "outer"))
    parser.add_argument("--cpu-cache", action="store_true")
    args = parser.parse_args()
    if args.phase == "calibration":
        score_calibration(cache_on_device=not args.cpu_cache)
    else:
        score_outer(cache_on_device=not args.cpu_cache)
