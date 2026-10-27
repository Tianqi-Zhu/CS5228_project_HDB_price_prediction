"""Fit the best measured Fast month setup on all labeled rows and predict Kaggle test.

March 2025 test rows use a February history origin; later test rows use a
March origin. Hidden test rents never enter history features. Forecasts beyond
the validated ten-month temporal-model range hold its ten-month level flat.
"""

from __future__ import annotations

import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.compare_tabpfn_calendar_80_20 import frozen_settings
from analysis.compare_tabpfn_v3_80_20 import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.experiments.aligned_temporal_features import (
    FEATURES, HISTORY_COLUMNS, LocalTemporalModel, episode_origins, origin_history,
)
from analysis.experiments.calendar_encoding import MONTH_COLUMN, encoded_features
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET, spatial_frame
from analysis.experiments.tabpfn_adaptation_features import ENHANCED, backbone_inputs, enhanced_feature_pair


OUT = REPORTS / "tabpfn_fast_final_submission"
CACHE = ROOT / ".cache/tabpfn_fast_final_submission"
FEATURE_PATH = CACHE / "features.joblib"
SUBMISSION = OUT / "submission_tabpfn_fast_150k.csv"
BATCH = 200
MAX_FITTED_HORIZON = 10
FIRST_TEST_MONTH = pd.Period("2025-03", "M")


def capped_temporal_features(temporal, blind, origin):
    """Preserve actual month/horizon, but plateau the local forecast after month 10."""
    periods = pd.PeriodIndex(blind[MONTH], freq="M")
    horizon = periods.asi8 - origin.ordinal
    if (horizon < 1).any():
        raise ValueError("A test feature uses a label from its own or a later month")
    clipped = np.minimum(horizon, MAX_FITTED_HORIZON)
    projected = blind.copy()
    projected[MONTH] = pd.PeriodIndex.from_ordinals(origin.ordinal + clipped, freq="M")
    features = temporal.features(projected, origin).reindex(blind.index)
    features["TEMP_FORECAST_HORIZON"] = horizon.astype(float)
    features["TEMP_FLAT_LAST_OBSERVATION_AGE"] += (horizon - clipped)
    if not np.isfinite(features.TEMP_LOCAL_FORECAST.to_numpy(float)).all():
        raise AssertionError("Test temporal baseline contains missing values")
    return features


def prepare_features():
    manifest_path = OUT / "feature_manifest.json"
    if FEATURE_PATH.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256(FEATURE_PATH) != manifest["sha256"]:
            raise ValueError("Saved final feature bundle changed")
        return joblib.load(FEATURE_PATH)
    raw_train = pd.read_csv(DATA / "train.csv")
    raw_test = pd.read_csv(DATA / "test.csv")
    train = spatial_frame(raw_train)
    test = spatial_frame(raw_test)
    if len(train) != 150000 or len(test) != 50000 or train[MONTH].max() != FIRST_TEST_MONTH:
        raise AssertionError("Training/test rows or dates changed")
    early = train[train[MONTH] < FIRST_TEST_MONTH]
    march = test[test[MONTH] == FIRST_TEST_MONTH]
    later = test[test[MONTH] > FIRST_TEST_MONTH]
    if len(march) == 0 or len(later) == 0:
        raise AssertionError("Expected March overlap and later test months")
    print("Building strictly prior static/history features for 150,000 context and 50,000 test rows", flush=True)
    fit_features, later_features = enhanced_feature_pair(train, later)
    _, march_features = enhanced_feature_pair(early, march)
    query_features = pd.concat((march_features, later_features)).sort_index()
    if not query_features.index.equals(test.index):
        raise AssertionError("Query row order changed")
    x, q = backbone_inputs(fit_features, query_features, ENHANCED)
    for column in FEATURES:
        x[column] = np.nan
        q[column] = np.nan
    assignments = episode_origins(train[MONTH].astype(str).to_numpy(), train.index.to_numpy(), train[MONTH].min())
    temporal = LocalTemporalModel(train)
    origins = sorted(assignments.forecast_origin.dropna().unique())
    for index, origin in enumerate(origins, 1):
        ids = assignments.index[assignments.forecast_origin.eq(origin)]
        past = train.loc[train[MONTH] <= origin]
        blind = train.loc[ids].drop(columns=TARGET)
        x.loc[ids, HISTORY_COLUMNS] = origin_history(past, blind).reindex(ids).to_numpy()
        x.loc[ids, FEATURES] = temporal.features(blind, origin).reindex(ids).to_numpy()
        if index % 6 == 0 or index == len(origins):
            print(f"  Rebuilt aligned context histories: {index}/{len(origins)} origins", flush=True)
    query_origins = {str(FIRST_TEST_MONTH - 1): march.index, str(FIRST_TEST_MONTH): later.index}
    for origin_text, ids in query_origins.items():
        origin = pd.Period(origin_text, "M")
        past = train.loc[train[MONTH] <= origin]
        blind = test.loc[ids]
        q.loc[ids, HISTORY_COLUMNS] = origin_history(past, blind).reindex(ids).to_numpy()
        q.loc[ids, FEATURES] = capped_temporal_features(temporal, blind, origin)[FEATURES].reindex(ids).to_numpy()
        print(f"  Prepared {len(ids):,} test rows at history origin {origin_text}", flush=True)
    x = encoded_features(x, train[MONTH].astype(str).to_numpy(), "categorical")
    q = encoded_features(q, test[MONTH].astype(str).to_numpy(), "categorical")
    if list(x.columns) != list(q.columns) or len(x.columns) != 77:
        raise AssertionError("Final feature schema does not match categorical-month 80/20 run")
    if x[MONTH_COLUMN].dtype != q[MONTH_COLUMN].dtype or q[MONTH_COLUMN].isna().any():
        raise AssertionError("Month categories differ between context and test")
    y = train[TARGET].to_numpy(float)
    train_baseline = np.where(np.isfinite(x.TEMP_LOCAL_FORECAST.to_numpy(float)), x.TEMP_LOCAL_FORECAST.to_numpy(float), 0)
    query_baseline = q.TEMP_LOCAL_FORECAST.to_numpy(float)
    if not np.isfinite(y).all() or not np.isfinite(query_baseline).all():
        raise AssertionError("Nonfinite label or temporal baseline")
    print(f"  {int((train_baseline == 0).sum()):,} context rows use zero additive baseline", flush=True)
    bundle = {"X_train": x, "X_query": q, "y_train": y,
              "train_baseline": train_baseline, "query_baseline": query_baseline,
              "train_month": train[MONTH].astype(str).to_numpy(),
              "query_month": test[MONTH].astype(str).to_numpy(),
              "query_origin": np.where(test[MONTH].eq(FIRST_TEST_MONTH), "2025-02", "2025-03")}
    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = FEATURE_PATH.with_suffix(".part.joblib")
    joblib.dump(bundle, temp, compress=3)
    temp.replace(FEATURE_PATH)
    write_json(manifest_path, {"artifact": FEATURE_PATH.relative_to(ROOT).as_posix(),
                               "sha256": sha256(FEATURE_PATH), "context_rows": len(x),
                               "test_rows": len(q), "features": len(x.columns),
                               "march_test_origin": "2025-02", "later_test_origin": "2025-03",
                               "long_horizon_policy": "local forecast held at month 10; actual horizon, age and calendar month retained",
                               "beyond_ten_month_rows": int((q.TEMP_FORECAST_HORIZON > MAX_FITTED_HORIZON).sum()),
                               "query_baseline_finite": True,
                               "train_categorical_month_complete": bool(not x[MONTH_COLUMN].isna().any()),
                               "query_categorical_month_complete": True})
    return bundle


def predict(bundle):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    progress_path = CACHE / "progress.json"
    prediction_path = CACHE / "predictions.npy"
    feature_hash = sha256(FEATURE_PATH)
    q = bundle["X_query"]
    if progress_path.exists() and prediction_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["rows"] != len(q) or progress["feature_sha256"] != feature_hash:
            raise ValueError("Saved inference progress uses different features")
        start = progress["next_row"]
        values = np.lib.format.open_memmap(prediction_path, mode="r+")
        if len(values) != len(q) or not np.isfinite(values[:start]).all():
            raise AssertionError("Incomplete or corrupted saved progress")
    else:
        values = np.lib.format.open_memmap(prediction_path, mode="w+", dtype="float64", shape=(len(q),))
        values[:] = np.nan
        values.flush()
        start = 0
    if start == len(q):
        return np.asarray(values).copy(), json.loads((OUT / "timing.json").read_text(encoding="utf-8"))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT), device="cuda",
        n_estimators=1, random_state=42, fit_mode="fit_with_cache", keep_cache_on_device=True,
        inference_config={"SUBSAMPLE_SAMPLES": None})
    started = time.perf_counter()
    try:
        model.fit(bundle["X_train"], bundle["y_train"] - bundle["train_baseline"])
        if model.n_train_samples_ != len(bundle["X_train"]) or model.inference_config_.SUBSAMPLE_SAMPLES is not None:
            raise AssertionError("Full training context was subsampled")
        indices = model.ensemble_preprocessor_.subsample_row_indices
        if indices is not None and any(len(i) != len(bundle["X_train"]) or len(np.unique(i)) != len(bundle["X_train"]) for i in indices):
            raise AssertionError("Estimator context differs from all labeled rows")
        fitted = time.perf_counter()
        print(f"Fast backbone fitted on all {len(bundle['X_train']):,} labels; predicting {len(q):,} test rows", flush=True)
        for offset in range(start, len(q), BATCH):
            chunk = q.iloc[offset:offset + BATCH]
            residual = np.asarray(model.predict(chunk, output_type="mean"), float)
            reconstructed = residual + bundle["query_baseline"][offset:offset + len(chunk)]
            if not np.isfinite(reconstructed).all():
                raise AssertionError(f"Nonfinite prediction at test offset {offset}")
            values[offset:offset + len(chunk)] = reconstructed
            values.flush()
            write_json(progress_path, {"feature_sha256": feature_hash,
                                       "rows": len(q), "next_row": offset + len(chunk)})
            if offset % 4000 == 0 or offset + BATCH >= len(q):
                print(f"  Predicted {min(offset + BATCH, len(q)):,}/{len(q):,}", flush=True)
        timing = {"fit_seconds": fitted - started, "predict_seconds": time.perf_counter() - fitted,
                  "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                  "context_rows": len(bundle["X_train"]), "test_rows": len(q),
                  "n_estimators": 1, "batch_size": BATCH, "subsampled": False}
        write_json(OUT / "timing.json", timing)
        return np.asarray(values).copy(), timing
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    import torch

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    source = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
              for p in (DATA / "train.csv", DATA / "test.csv", DATA / "example-submission.csv")}
    code = [Path(__file__), ROOT / "analysis/experiments/aligned_temporal_features.py",
            ROOT / "analysis/experiments/tabpfn_adaptation_features.py",
            ROOT / "analysis/experiments/recent_history.py",
            ROOT / "analysis/experiments/calendar_encoding.py",
            ROOT / "analysis/experiments/spatiotemporal_hybrid.py"]
    settings = {"model": "tabpfn-3.5-fast", "checkpoint_sha256": sha256(CHECKPOINT),
                "source_sha256": source, "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code},
                "month_encoding": "categorical; lowest measured 80/20 RMSE (455.81357012 SGD)",
                "context": "all 150000 labeled training rows", "batch": BATCH, "n_estimators": 1,
                "temporal_training_horizons": [1, 10], "test_horizon_extrapolation": "plateau forecast after horizon 10",
                "test_origin_policy": {"2025-03": "2025-02", "2025-04 through 2026-07": "2025-03"},
                "seed": 42, "gpu": torch.cuda.get_device_name(),
                "versions": {p: version(p) for p in ("tabpfn", "torch", "numpy", "pandas", "scikit-learn")}}
    frozen_settings(OUT / "settings.json", settings)
    bundle = prepare_features()
    prediction, timing = predict(bundle)
    example = pd.read_csv(DATA / "example-submission.csv")
    if len(prediction) != len(example) or not np.array_equal(example.Id.to_numpy(), np.arange(len(prediction))):
        raise AssertionError("Example submission row IDs differ from test row order")
    if not np.isfinite(prediction).all() or (prediction <= 0).any():
        raise AssertionError("Predicted rents must be positive and finite")
    submission = pd.DataFrame({"Id": example.Id.astype(int), "Predicted": prediction})
    submission.to_csv(SUBMISSION, index=False, float_format="%.6f")
    saved = pd.read_csv(SUBMISSION)
    if list(saved.columns) != ["Id", "Predicted"] or not np.array_equal(saved.Id.to_numpy(), example.Id.to_numpy()):
        raise AssertionError("Saved Kaggle CSV schema or row order changed")
    if np.max(np.abs(saved.Predicted.to_numpy() - prediction)) > 0.000001:
        raise AssertionError("Saved Kaggle CSV rounded unexpectedly")
    monthly = pd.DataFrame({"month": bundle["query_month"], "predicted": prediction,
                            "horizon": bundle["X_query"].TEMP_FORECAST_HORIZON.to_numpy()})
    summary = monthly.groupby("month", sort=True).agg(rows=("predicted", "size"),
                                                     mean_prediction=("predicted", "mean"),
                                                     min_prediction=("predicted", "min"),
                                                     max_prediction=("predicted", "max"),
                                                     max_horizon=("horizon", "max"))
    summary.to_csv(OUT / "test_month_summary.csv", float_format="%.4f")
    write_json(OUT / "verification.json", {"submission": SUBMISSION.relative_to(ROOT).as_posix(),
                                           "submission_sha256": sha256(SUBMISSION),
                                           "rows": len(saved), "columns": list(saved.columns),
                                           "row_order_matches_example": True, "all_positive_and_finite": True,
                                           "prediction_range_sgd": [float(prediction.min()), float(prediction.max())],
                                           "test_labels_available": False,
                                           "test_rmse_available": False,
                                           "peak_vram_gib": timing["peak_vram_gib"]})
    report = ["# TabPFN-3.5-Fast Kaggle test submission", "",
              f"Ready-to-upload CSV: `{SUBMISSION.relative_to(ROOT).as_posix()}` ({len(saved):,} rows; `Id,Predicted`).", "",
              "The frozen checkpoint is fitted as a default regressor on residual rent targets using **all 150,000** labeled training rows as context. The feature set matches the 77-column categorical-month 80/20 setup (S$455.81 development validation RMSE). The failed embedding MLP is omitted. The supplied test set has no rent labels, so no test RMSE or other test loss is measurable.", "",
              "March 2025 test features use rental histories through February 2025. April 2025-July 2026 test features use histories through March 2025. March labeled training rows are available to the full TabPFN context, but never feed into March test history features. Training context rows use the same pseudo-origin aligned rent histories and local temporal residual baseline as the validation setup.", "",
              "The local temporal model was fitted and validated only for horizons 1-10. For later test months, its level and uncertainty are held at the horizon-10 forecast; the actual month, calendar category, forecast horizon, and history age remain accurate. This is an extrapolation that was not validated by the ten-month 80/20 experiment. See `test_month_summary.csv` and `feature_manifest.json`.", "",
              f"All {len(saved):,} CSV rows were reloaded, checked finite and positive, and checked against the sample submission's exact column names and row order. Prediction range: S${prediction.min():,.2f}-S${prediction.max():,.2f}; peak GPU memory: {timing['peak_vram_gib']:.2f} GiB.", "",
              "Reproduce with `.venv/Scripts/python.exe analysis/train_tabpfn_fast_submission.py`. Model inference progress is resumable under ignored `.cache/tabpfn_fast_final_submission/`.", ""]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved upload-ready {SUBMISSION} ({len(saved):,} predictions)", flush=True)


if __name__ == "__main__":
    main()
