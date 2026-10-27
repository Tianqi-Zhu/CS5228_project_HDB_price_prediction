"""Fit and compare origin-aligned temporal inputs and baseline-residual targets."""

from __future__ import annotations

import argparse
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
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.aligned_temporal_features import (
    FEATURES, HISTORY_COLUMNS, LocalTemporalModel, episode_origins, origin_history,
)
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET, spatial_frame


OUT = REPORTS / "tabpfn_v3_5_fast_aligned_temporal_80_20"
ARTIFACT = ROOT / ".cache/tabpfn_v3_features/holdout_80_20_2024-06_aligned_temporal.joblib"
BASE_REPORT = REPORTS / "tabpfn_v3_80_20"
ORIGIN = pd.Period("2024-06", "M")
MODELS = {"tabpfn_aligned_features": "raw", "tabpfn_aligned_residual": "residual"}


def prepare(audit):
    path = ROOT / audit["artifact"]
    if sha256(path) != audit["sha256"]:
        raise ValueError("The original feature bundle changed")
    base = joblib.load(path)  # Trusted project-generated artifact.
    raw = pd.read_csv(DATA / "train.csv")
    frame = spatial_frame(raw)
    np.testing.assert_array_equal(frame[TARGET].to_numpy(), raw[TARGET].to_numpy())
    fitting = frame.loc[frame[MONTH] < ORIGIN]
    np.testing.assert_array_equal(fitting.index.to_numpy(), base["X_train"].index.to_numpy())
    temporal = LocalTemporalModel(fitting)
    assignments = {
        "train": episode_origins(base["train_month"], base["X_train"].index, fitting[MONTH].min()),
        "query": episode_origins(base["query_month"], base["X_query"].index, fitting[MONTH].min(), ORIGIN),
    }
    matrices = {"train": base["X_train"].copy(), "query": base["X_query"].copy()}
    for matrix in matrices.values():
        for column in FEATURES:
            matrix[column] = np.nan
    origins = sorted(set(assignments["train"].forecast_origin.dropna()).union(assignments["query"].forecast_origin.dropna()))
    for i, origin in enumerate(origins, 1):
        past = fitting.loc[fitting[MONTH] <= origin]
        temporal.fit_at(origin)
        for split, matrix in matrices.items():
            ids = assignments[split].index[assignments[split].forecast_origin.eq(origin)]
            if not len(ids):
                continue
            blind = frame.loc[ids].drop(columns=TARGET)
            aligned = origin_history(past, blind).reindex(ids)
            matrix.loc[ids, HISTORY_COLUMNS] = aligned.to_numpy()
            features = temporal.features(blind, origin).reindex(ids)
            matrix.loc[ids, FEATURES] = features.to_numpy()
        if i % 5 == 0 or i == len(origins):
            print(f"Aligned histories and fitted temporal forecasts: {i}/{len(origins)} origins (through {origin})", flush=True)
    static = [c for c in base["X_train"] if c not in HISTORY_COLUMNS]
    for split, key in (("train", "X_train"), ("query", "X_query")):
        pd.testing.assert_frame_equal(matrices[split][static], base[key][static])
    # At the actual validation origin the historical information cutoff is unchanged.
    pd.testing.assert_frame_equal(matrices["query"][base["X_query"].columns], base["X_query"],
                                  check_exact=False, rtol=1e-10, atol=1e-8)
    bundle = {**base, "X_train": matrices["train"], "X_query": matrices["query"],
              "train_temporal_origins": assignments["train"], "query_temporal_origins": assignments["query"]}
    for split, key in (("train", "X_train"), ("query", "X_query")):
        values = bundle[key].TEMP_LOCAL_FORECAST.to_numpy(float)
        # With no historical labels (the first context month), use the neutral
        # additive baseline zero. Keep those transactions in the full context.
        baseline = np.where(np.isfinite(values), values, 0.0)
        bundle[f"{split}_baseline"] = baseline
        exported = assignments[split].join(bundle[key][FEATURES])
        exported["residual_baseline_sgd"] = baseline
        exported.to_csv(OUT / f"temporal_features_{split}.csv", index=True, index_label="row_id", float_format="%.8f")
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT, compress=3)
    diagnostics = [{"origin_index": index, **params} for index, params in temporal.parameters.items()]
    write_json(OUT / "temporal_model_audit.json", {"selection": "last 12 observed target months; equal horizon weights; conservative 1% rule",
                                                 "max_horizon": 10, "parameters": diagnostics})
    manifest = {
        "artifact": ARTIFACT.relative_to(ROOT).as_posix(), "sha256": sha256(ARTIFACT),
        "base_feature_artifact_sha256": audit["sha256"], "context_rows": len(bundle["X_train"]),
        "validation_rows": len(bundle["X_query"]), "feature_count": len(bundle["X_train"].columns),
        "aligned_history_columns": HISTORY_COLUMNS, "added_features": FEATURES,
        "context_baseline_zero_warmup_rows": int((bundle["train_baseline"] == 0).sum()),
        "training_missing_fractions": bundle["X_train"][FEATURES].isna().mean().to_dict(),
        "validation_missing_fractions": bundle["X_query"][FEATURES].isna().mean().to_dict(),
    }
    write_json(OUT / "feature_manifest.json", manifest)
    return bundle, manifest


def predict(x, y, query, *, batch_size, keep_cache_on_device):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT), device="cuda", n_estimators=1,
        random_state=42, fit_mode="fit_with_cache", keep_cache_on_device=keep_cache_on_device,
        inference_config={"SUBSAMPLE_SAMPLES": None},
    )
    started = time.perf_counter()
    try:
        model.fit(x, y)
        torch.cuda.synchronize()
        fitted = time.perf_counter()
        if model.n_train_samples_ != len(x) or model.inference_config_.SUBSAMPLE_SAMPLES is not None:
            raise AssertionError("Requested context was subsampled")
        indices = model.ensemble_preprocessor_.subsample_row_indices
        if indices is not None and any(len(v) != len(x) or len(np.unique(v)) != len(x) for v in indices):
            raise AssertionError("Actual estimator context differs from the requested rows")
        print(f"Fit complete with {len(x):,} context rows; checking prediction batching", flush=True)
        probe_n = min(batch_size, len(query))
        reference = np.concatenate([np.asarray(model.predict(query.iloc[i:min(i + 200, probe_n)], output_type="mean"), float)
                                    for i in range(0, probe_n, 200)])
        large = np.asarray(model.predict(query.iloc[:probe_n], output_type="mean"), float)
        agreement = bool(np.allclose(reference, large, rtol=1e-5, atol=0.02))
        difference = float(np.max(np.abs(reference - large)))
        effective = batch_size if agreement else 200
        parts = [large if agreement else reference]
        print(f"Batch check: maximum difference S${difference:.6f}; using {effective}-row batches", flush=True)
        for offset in range(probe_n, len(query), effective):
            parts.append(np.asarray(model.predict(query.iloc[offset:offset + effective], output_type="mean"), float))
            if (offset - probe_n) % (effective * 4) == 0 or offset + effective >= len(query):
                print(f"  Predicted {min(offset + effective, len(query)):,}/{len(query):,}", flush=True)
        torch.cuda.synchronize()
        values = np.concatenate(parts)
        if len(values) != len(query) or not np.isfinite(values).all():
            raise AssertionError("Incomplete or nonfinite predictions")
        return values, {"fit_seconds": fitted - started, "predict_seconds": time.perf_counter() - fitted,
                        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                        "context_rows": len(x), "subsampling": False, "keep_cache_on_device": keep_cache_on_device,
                        "batch_size": effective, "batch_agreement": agreement, "batch_max_difference_sgd": difference}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((BASE_REPORT / "settings.json").read_text(encoding="utf-8"))
    source_hashes = {name: sha256(ROOT / name) for name in prior["source_sha256"]}
    if source_hashes != prior["source_sha256"]:
        raise ValueError("Underlying data differs from the earlier experiments")
    audit = json.loads((BASE_REPORT / "feature_manifest.json").read_text(encoding="utf-8"))
    controls = {
        "previous_temporal_levels": REPORTS / "tabpfn_v3_5_fast_temporal_levels_80_20/predictions.csv",
        "original_fast": REPORTS / "tabpfn_v3_5_fast_80_20/predictions.csv",
    }
    code = [Path(__file__), ROOT / "analysis/experiments/aligned_temporal_features.py", ROOT / "analysis/data.py",
            ROOT / "analysis/experiments/spatiotemporal_hybrid.py", ROOT / "analysis/experiments/recent_history.py",
            ROOT / "analysis/experiments/tabpfn_adaptation_features.py"]
    settings = {"model_version": "v3.5-fast", "n_estimators": 1, "seed": 42,
                "context_rows": 121005, "subsampling": False, "requested_batch_size": 1000,
                "models": MODELS, "maximum_forecast_horizon": 10, "economic_features": False,
                "checkpoint_sha256": sha256(CHECKPOINT), "source_sha256": source_hashes,
                "base_feature_artifact_sha256": audit["sha256"],
                "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code},
                "reference_predictions_sha256": {name: sha256(p) for name, p in controls.items()},
                "versions": {v: version(v) for v in ("tabpfn", "torch", "numpy", "pandas")},
                "gpu": torch.cuda.get_device_name()}
    settings_path = OUT / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Output directory has different code, data, or model settings")
    write_json(settings_path, settings)
    write_json(OUT / "status.json", {"status": "preparing"})
    if ARTIFACT.exists() and (OUT / "feature_manifest.json").exists():
        manifest = json.loads((OUT / "feature_manifest.json").read_text(encoding="utf-8"))
        if sha256(ARTIFACT) != manifest["sha256"]:
            raise ValueError("Aligned feature artifact changed")
        bundle = joblib.load(ARTIFACT)
    else:
        bundle, manifest = prepare(audit)
    if args.prepare_only:
        write_json(OUT / "status.json", {"status": "features_ready"})
        print(json.dumps(manifest, indent=2), flush=True)
        return
    x, query = bundle["X_train"], bundle["X_query"]
    y, actual = np.asarray(bundle["y_train"], float), np.asarray(bundle["y_query"], float)
    results = pd.DataFrame({"row_id": query.index.to_numpy(), "month": bundle["query_month"], "actual": actual})
    for name, path in controls.items():
        saved = pd.read_csv(path)
        np.testing.assert_array_equal(saved.row_id.to_numpy(), results.row_id.to_numpy())
        np.testing.assert_array_equal(saved.actual.to_numpy(), actual)
        results[name] = saved.predicted.to_numpy(float)
    timings = []
    for name, target_mode in MODELS.items():
        path, timing_path = OUT / f"predictions_{name}.csv", OUT / f"timing_{name}.json"
        if path.exists() and timing_path.exists():
            saved = pd.read_csv(path)
            np.testing.assert_array_equal(saved.row_id.to_numpy(), results.row_id.to_numpy())
            np.testing.assert_array_equal(saved.actual.to_numpy(), actual)
            values = saved.predicted.to_numpy(float)
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
        else:
            target = y if target_mode == "raw" else y - bundle["train_baseline"]
            if target_mode == "residual":
                np.testing.assert_allclose(target + bundle["train_baseline"], y, atol=1e-10)
            attempts, completed = [], False
            for batch, cache in ((1000, True), (500, True), (200, True), (200, False)):
                write_json(OUT / "status.json", {"status": "fitting", "model": name, "batch_size": batch})
                try:
                    values, timing = predict(x, target, query, batch_size=batch, keep_cache_on_device=cache)
                    completed = True
                    break
                except Exception as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    attempts.append({"batch_size": batch, "gpu_cache": cache, "error": str(exc)})
                    gc.collect()
            if not completed:
                raise RuntimeError("Full context exceeded memory in every inference configuration")
            if target_mode == "residual":
                values = values + bundle["query_baseline"]
            timing.update({"target_mode": target_mode, "memory_retries": attempts})
            pd.DataFrame({"row_id": results.row_id, "actual": actual, "predicted": values}).to_csv(
                path, index=False, float_format="%.8f")
            write_json(timing_path, timing)
        if not np.isfinite(values).all():
            raise AssertionError("Final rent predictions are nonfinite")
        results[name] = values
        timings.append({"model": name, **{k: v for k, v in timing.items() if k != "memory_retries"}})
        print(f"{name}: {metrics(actual, values)}", flush=True)
    names = (*MODELS, *controls)
    scores = pd.DataFrame([{"model": name, "n": len(results), **metrics(actual, results[name])} for name in names])
    monthly = pd.DataFrame([{"model": name, "month": month, "n": len(part), **metrics(part.actual, part[name])}
                            for month, part in results.groupby("month", sort=True) for name in names])
    results.to_csv(OUT / "validation_predictions.csv", index=False, float_format="%.8f")
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    monthly.to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.8f")
    pd.DataFrame(timings).to_csv(OUT / "timings.csv", index=False)
    selected = scores.sort_values("rmse_sgd").iloc[0]
    write_json(OUT / "selection.json", {"criterion": "development holdout RMSE", "model": selected.model,
                                        "rmse_sgd": float(selected.rmse_sgd)})
    report = ["# Improved temporal training and usage with TabPFN-3.5-Fast", "",
              "The chronological 80/20 split is January 2021-May 2024 context (121,005 rows) and "
              "June 2024-March 2025 validation (28,995 rows). All source context rows remain included.", "",
              "All 37 rental-history inputs are regenerated at the same earlier origin as the temporal forecasts. "
              "Historical context rows cover 1-10 month horizons, matching the evaluation window. The previous "
              "version combined stale temporal forecasts with preceding-transaction-month local histories; this "
              "experiment aligns those information sets. Static inputs and rental labels are preserved.", "",
              "A pooled local model selects flat-type trend damping, planning-area shrinkage, and local-premium "
              "decay from rolling forecasts whose target labels are already observed at each origin. It uses the "
              "last twelve observed target months, gives available horizons equal weight, and prefers a simpler "
              "candidate within 1% of the best error. Forecast error and local pooling weight are also inputs. "
              "No economic observations are used.", "",
              "Both new variants use the same 77 inputs. The feature variant predicts raw rent. The residual "
              "variant predicts rent minus the local temporal baseline and adds that baseline back exactly once. "
              "Rows without any earlier history use additive baseline zero and keep missing temporal inputs.", "",
              markdown_table(scores), "", "## Runtime and batch checks", "", markdown_table(pd.DataFrame(timings)), "",
              "The larger prediction batch is checked against 200-row batches on the same fitted model without "
              "using labels. If numerical agreement fails, inference uses 200-row batches. Memory retries retain "
              "all context rows. All models use one estimator, seed 42, and the same Fast checkpoint.", "",
              "This is a repeatedly examined ten-month development holdout. Temporal parameters use earlier "
              "data, but model design has evolved using development results. The supplied test spans seventeen "
              "months and has no rent labels. The new method should be confirmed at another forecast origin.", "",
              f"Model-ready data: `{ARTIFACT.relative_to(ROOT).as_posix()}`. Origin metadata and new numeric "
              "inputs are in `temporal_features_train.csv` and `temporal_features_query.csv`.", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_aligned_temporal_80_20.py`."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "status.json", {"status": "complete", "context_rows": len(x), "validation_rows": len(query),
                                     "feature_count": len(x.columns)})
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
