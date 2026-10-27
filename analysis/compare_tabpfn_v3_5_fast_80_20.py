"""Evaluate TabPFN-3.5-Fast on the existing chronological 80/20 holdout.

Uses every training row as context and reuses hash-checked TabPFN-3/CatBoost
controls from compare_tabpfn_v3_80_20.py. No validation labels enter prediction.
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

from analysis.compare_tabpfn_v3_80_20 import markdown_table, write_json
from analysis.config import REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics


OUTPUT = REPORTS / "tabpfn_v3_5_fast_80_20"
REFERENCE = REPORTS / "tabpfn_v3_80_20"
CHECKPOINT = ROOT / ".cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors"
MODEL = "tabpfn_3_5_fast_full_context"


def predict_all_context(x, y, query, *, keep_cache_on_device):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT), device="cuda",
        n_estimators=1, random_state=42, fit_mode="fit_with_cache",
        keep_cache_on_device=keep_cache_on_device,
        inference_config={"SUBSAMPLE_SAMPLES": None},
    )
    started = time.perf_counter()
    try:
        model.fit(x, y)
        torch.cuda.synchronize()
        fitted = time.perf_counter()
        if model.n_train_samples_ != len(x) or model.inference_config_.SUBSAMPLE_SAMPLES is not None:
            raise AssertionError("SDK changed the requested full context")
        indices = model.ensemble_preprocessor_.subsample_row_indices
        if indices is not None:
            for subset in indices:
                if len(subset) != len(x) or len(np.unique(subset)) != len(x):
                    raise AssertionError("SDK subsampled the requested context")
        print(f"Fit complete: all {len(x):,} rows retained; predicting {len(query):,} rows", flush=True)
        batches = []
        for offset in range(0, len(query), 200):
            batches.append(np.asarray(model.predict(query.iloc[offset:offset + 200], output_type="mean"), float))
            if offset % 4000 == 0 or offset + 200 >= len(query):
                print(f"  Predicted {min(offset + 200, len(query)):,}/{len(query):,}", flush=True)
        torch.cuda.synchronize()
        values = np.concatenate(batches)
        if len(values) != len(query) or not np.isfinite(values).all():
            raise AssertionError("Incomplete or nonfinite predictions")
        return values, {
            "fit_seconds": fitted - started,
            "predict_seconds": time.perf_counter() - fitted,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "context_rows": len(x), "subsampling": False,
            "keep_cache_on_device": keep_cache_on_device,
            "architecture": type(model.models_[0]).__name__,
        }
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    import torch

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original = json.loads((REFERENCE / "settings.json").read_text(encoding="utf-8"))
    source_hashes = {name: sha256(ROOT / name) for name in original["source_sha256"]}
    if source_hashes != original["source_sha256"]:
        raise ValueError("Reference source data changed")
    audit = json.loads((REFERENCE / "feature_manifest.json").read_text(encoding="utf-8"))
    bundle_path = ROOT / audit["artifact"]
    if sha256(bundle_path) != audit["sha256"]:
        raise ValueError("Saved feature bundle changed")
    control_paths = {
        "tabpfn_3_full_context": REFERENCE / "predictions_tabpfn_full_context.csv",
        "catboost_matched": REFERENCE / "predictions_catboost_matched.csv",
    }
    settings = {
        "model_version": "v3.5-fast", "origin": audit["origin"],
        "n_estimators": 1, "context_rows": audit["train_rows"], "subsampling": False,
        "batch_size": 200, "seed": 42, "fit_mode": "fit_with_cache",
        "checkpoint": CHECKPOINT.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": sha256(CHECKPOINT), "source_sha256": source_hashes,
        "feature_bundle_sha256": audit["sha256"],
        "reference_predictions_sha256": {name: sha256(path) for name, path in control_paths.items()},
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in
                        (Path(__file__), ROOT / "analysis/compare_tabpfn_v3_80_20.py")},
        "versions": {key: version(key) for key in ("tabpfn", "torch", "pandas", "numpy")},
        "gpu": torch.cuda.get_device_name(),
    }
    settings_path = OUTPUT / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing output uses different code, data, or settings")
    write_json(settings_path, settings)
    write_json(OUTPUT / "status.json", {"status": "loading_features"})
    bundle = joblib.load(bundle_path)  # Trusted, hash-checked project artifact.
    x, y = bundle["X_train"], np.asarray(bundle["y_train"], float)
    query, actual = bundle["X_query"], np.asarray(bundle["y_query"], float)
    if len(x) != 121_005 or len(query) != 28_995 or len(x.columns) != 68:
        raise AssertionError("Feature bundle no longer matches the recorded 80/20 experiment")
    if pd.PeriodIndex(bundle["train_month"], freq="M").max() >= pd.PeriodIndex(bundle["query_month"], freq="M").min():
        raise AssertionError("Training and validation months overlap")
    results = pd.DataFrame({"row_id": query.index.to_numpy(), "month": bundle["query_month"], "actual": actual})
    for name, path in control_paths.items():
        saved = pd.read_csv(path)
        np.testing.assert_array_equal(saved.row_id.to_numpy(), results.row_id.to_numpy())
        np.testing.assert_array_equal(saved.actual.to_numpy(), actual)
        results[name] = saved.predicted.to_numpy(float)
    prediction_path = OUTPUT / "predictions.csv"
    timing_path = OUTPUT / "timing.json"
    if prediction_path.exists() and timing_path.exists():
        saved = pd.read_csv(prediction_path)
        np.testing.assert_array_equal(saved.row_id.to_numpy(), results.row_id.to_numpy())
        np.testing.assert_array_equal(saved.actual.to_numpy(), actual)
        predicted = saved.predicted.to_numpy(float)
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    else:
        write_json(OUTPUT / "status.json", {"status": "fitting", "context_rows": len(x)})
        retry_with_cpu_cache = False
        try:
            predicted, timing = predict_all_context(x, y, query, keep_cache_on_device=True)
        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print("GPU cache exceeded memory; retrying the same complete context with CPU cache", flush=True)
            write_json(OUTPUT / "gpu_cache_retry.json", {"reason": str(exc), "context_rows": len(x)})
            retry_with_cpu_cache = True
        if retry_with_cpu_cache:
            gc.collect()
            predicted, timing = predict_all_context(x, y, query, keep_cache_on_device=False)
        pd.DataFrame({"row_id": results.row_id, "actual": actual, "predicted": predicted}).to_csv(
            prediction_path, index=False, float_format="%.8f")
        write_json(timing_path, timing)
    results[MODEL] = predicted
    names = (MODEL, *control_paths)
    scores = pd.DataFrame([{"model": name, "n": len(results), **metrics(actual, results[name])} for name in names])
    monthly = pd.DataFrame([
        {"model": name, "month": month, "n": len(part), **metrics(part.actual, part[name])}
        for month, part in results.groupby("month", sort=True) for name in names
    ])
    results.to_csv(OUTPUT / "validation_predictions.csv", index=False, float_format="%.8f")
    scores.to_csv(OUTPUT / "scores.csv", index=False, float_format="%.8f")
    monthly.to_csv(OUTPUT / "monthly_scores.csv", index=False, float_format="%.8f")
    timings = [{"model": MODEL, **timing}]
    for name, filename in (("tabpfn_3_full_context", "tabpfn_full_context"), ("catboost_matched", "catboost_matched")):
        timings.append({"model": name, **json.loads((REFERENCE / f"timing_{filename}.json").read_text(encoding="utf-8"))})
    pd.DataFrame(timings).to_csv(OUTPUT / "timings.csv", index=False)
    report = ["# TabPFN-3.5-Fast full-context 80/20 experiment", "",
              "Context: January 2021-May 2024, 121,005 rows (80.67%). "
              "Validation: June 2024-March 2025, 28,995 rows (19.33%). Whole months remain together.", "",
              "TabPFN-3.5-Fast uses its separate Fast checkpoint, one estimator, seed 42, "
              "and every context row. Row subsampling is disabled and checked after fitting. "
              "All models share the exact saved 68 features and validation row IDs. "
              "Training histories use earlier months; validation histories freeze at May 2024. "
              "The TabPFN-3 and CatBoost predictions are reused after source, feature and row-label checks.", "",
              markdown_table(scores), "", "## Runtime", "", markdown_table(pd.DataFrame(timings)), "",
              "GPU figures are peak PyTorch allocation. Control timings come from the previous run on the same GPU. "
              "This is a ten-month development holdout; the unlabelled test spans 17 months.", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_v3_5_fast_80_20.py`."]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUTPUT / "status.json", {"status": "complete", "context_rows": len(x), "validation_rows": len(query)})
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
