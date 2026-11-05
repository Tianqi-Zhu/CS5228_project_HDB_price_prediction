"""Inference-only spatial/temporal context retrieval for the saved TabPFN adapter.

The 77 model features and residual target are loaded verbatim from the winning
categorical-month artifacts. Retrieval metadata never enters the model input.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
import torch
from tabpfn.constants import ModelVersion

from analysis.check_diagscale_categorical_alignment_80_20 import category_positions
from analysis.compare_tabpfn_diagscale_80_20 import BASE, REFERENCE
from analysis.compare_tabpfn_v3_80_20 import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.tabpfn_context_sampler import (
    SamplerConfig, metadata_from_features, select_context,
)
from analysis.tune_tabpfn_diagscale_80_20 import (
    SavedDiagScaleRegressor, inner_bundle, predict_batches, weighted_rmse,
)


OUT = REPORTS / "tabpfn_context_sampling"
ADAPTER_DIR = REPORTS / "tabpfn_diagscale_categorical_alignment_80_20"
ADAPTER = ROOT / ".cache/tabpfn_diagscale_categorical_alignment_80_20/adapter.pt"
STRATEGIES = ("S0", "S1", "S2", "S3", "S4")


def _months(features: pd.DataFrame) -> np.ndarray:
    start = pd.Period("2021-01", "M")
    return np.asarray([str(start + int(offset)) for offset in features.APPROVAL_MONTH_INDEX], dtype=str)


def _sample_query(q: pd.DataFrame, limit: int | None):
    if limit is None or limit >= len(q):
        return np.arange(len(q))
    if limit < 1:
        raise ValueError("Query limit must be positive")
    return np.unique(np.linspace(0, len(q) - 1, num=limit, dtype=int))


def _load_stage(stage: str, query_limit: int | None):
    if stage == "calibration":
        bundle = inner_bundle()
        x, q = bundle["X_fit"], bundle["X_cal"]
        y = np.asarray(bundle["y_fit"], float)
        baseline = np.asarray(bundle["cal_baseline"], float)
        actual = np.asarray(bundle["y_cal_actual"], float)
        weights = np.asarray(bundle["cal_weight"], float)
        baseline_predictions = pd.read_csv(ADAPTER_DIR / "calibration_predictions.csv")
    else:
        bundle = joblib.load(BASE)
        x, q = bundle["X_train"], bundle["X_query"]
        y = np.asarray(bundle["y_train"], float) - np.asarray(bundle["train_baseline"], float)
        baseline = np.asarray(bundle["query_baseline"], float)
        actual = np.asarray(bundle["y_query"], float)
        weights = np.ones(len(q), float)
        baseline_predictions = pd.read_csv(ADAPTER_DIR / "predictions.csv")
        reference = pd.read_csv(REFERENCE)
        np.testing.assert_array_equal(reference.row_id.to_numpy(), q.index.to_numpy())
        np.testing.assert_allclose(reference.actual.to_numpy(float), actual)
        baseline_predictions = baseline_predictions.rename(columns={"diagscale_aligned": "predicted"})
    positions = _sample_query(q, query_limit)
    q, baseline, actual, weights = q.iloc[positions], baseline[positions], actual[positions], weights[positions]
    if list(x.columns) != list(q.columns) or category_positions(x) != category_positions(q):
        raise AssertionError("Winning model feature schema changed")
    np.testing.assert_array_equal(baseline_predictions.row_id.to_numpy()[positions], q.index.to_numpy())
    np.testing.assert_allclose(baseline_predictions.actual.to_numpy(float)[positions], actual)
    baseline_prediction = baseline_predictions.predicted.to_numpy(float)[positions]
    return x, y, q, baseline, actual, weights, baseline_prediction


def _metadata(x: pd.DataFrame, q: pd.DataFrame):
    raw = pd.read_csv(DATA / "train.csv", usecols=lambda name: name != "MONTHLY_RENT")
    duplicates = pd.util.hash_pandas_object(raw, index=False).to_numpy(dtype=np.uint64)
    if max(x.index.max(), q.index.max()) >= len(duplicates):
        raise AssertionError("Feature row IDs do not map to raw training rows")
    median_age = float(np.nanmedian(x.BUILDING_AGE_FROM_COMPLETION.to_numpy(float)))
    combined = pd.concat((x, q), axis=0)
    months = np.concatenate((_months(x), _months(q)))
    return metadata_from_features(
        combined, months, duplicate_groups=duplicates[combined.index.to_numpy()],
        building_age_imputation=median_age,
    )


def _predict(x: pd.DataFrame, y: np.ndarray, q: pd.DataFrame, baseline: np.ndarray):
    cat = category_positions(x)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = SavedDiagScaleRegressor.create_default_for_version(
        version=ModelVersion.V3_5_FAST, adapter_path=str(ADAPTER),
        model_path=str(CHECKPOINT), categorical_features_indices=cat,
        device="cuda", n_estimators=1, auto_scale_n_estimators=False,
        random_state=42, fit_mode="fit_with_cache", keep_cache_on_device=True,
        inference_config={"SUBSAMPLE_SAMPLES": None}, n_preprocessing_jobs=1,
    )
    model.fit(x, y)
    if model.n_train_samples_ != len(x) or model.inference_config_.SUBSAMPLE_SAMPLES is not None:
        raise AssertionError("The selected context was truncated")
    result = predict_batches(model, q, baseline, progress_label="sampled context")
    timing = {"fit_predict_seconds": time.perf_counter() - start,
              "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}
    del model
    torch.cuda.empty_cache()
    return result, timing


def run(stage: str, strategies: list[str], config: SamplerConfig, *, seed: int,
        query_limit: int | None, information_policy: str, dry_run: bool):
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    if not ADAPTER.exists() or sha256(ADAPTER) != json.loads(
            (ADAPTER_DIR / "calibration.json").read_text())["adapter_sha256"]:
        raise ValueError("Saved winning DiagScale adapter is missing or changed")
    expected_bundle = json.loads((REPORTS / "tabpfn_v3_5_fast_calendar_80_20/feature_manifest_categorical.json").read_text())
    if sha256(BASE) != expected_bundle["sha256"]:
        raise ValueError("Winning 77-feature artifact changed")
    x, y, q, baseline, actual, weights, full_prediction = _load_stage(stage, query_limit)
    metadata = _metadata(x, q)
    label = "full" if query_limit is None else str(len(q))
    run_dir = OUT / f"phase_a_{stage}_budget{config.context_size}_queries{label}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "stage": stage, "strategies": strategies, "information_policy": information_policy,
        "sampler_config": asdict(config), "seed": seed, "query_limit": query_limit,
        "model_feature_columns": list(x.columns),
        "model_categorical_indices": category_positions(x),
        "model_feature_artifact": str(BASE.relative_to(ROOT)).replace("\\", "/"),
        "model_feature_sha256": sha256(BASE), "adapter_sha256": sha256(ADAPTER),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "target": "y_train - train_baseline; prediction + query_baseline",
        "context_rows_available": len(x), "query_rows": len(q),
        "model_feature_schema_identical": True,
        "retrieval_building_age_imputation": metadata.building_age_imputation,
        "source_train_sha256": sha256(DATA / "train.csv"),
        "versions": {name: version(name) for name in
                     ("tabpfn", "torch", "numpy", "pandas", "scikit-learn")},
        "gpu": torch.cuda.get_device_name(),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 1024 ** 3,
    }
    settings_path = run_dir / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Existing run directory uses different settings")
    write_json(settings_path, settings)
    records = []
    for strategy in strategies:
        context_file = run_dir / f"context_ids_{strategy}.npy"
        audit_file = run_dir / f"selection_{strategy}.json"
        prediction_file = run_dir / f"predictions_{strategy}.csv"
        if strategy == "S0":
            selection = select_context(q.index.to_numpy(), x.index.to_numpy(), metadata,
                                       strategy=strategy, information_policy=information_policy,
                                       config=config, seed=seed)
            if len(selection.row_ids) != len(x):
                raise AssertionError("S0 failed to preserve full existing context")
            prediction = full_prediction
            timing = {"fit_predict_seconds": 0.0, "peak_vram_gib": None,
                      "source": "saved corrected full-context one-view predictions"}
        else:
            selection = select_context(q.index.to_numpy(), x.index.to_numpy(), metadata,
                                       strategy=strategy, information_policy=information_policy,
                                       config=config, seed=seed)
            if dry_run:
                prediction = np.full(len(q), np.nan)
                timing = {"fit_predict_seconds": None, "peak_vram_gib": None}
            elif prediction_file.exists():
                saved = pd.read_csv(prediction_file)
                np.testing.assert_array_equal(saved.row_id.to_numpy(), q.index.to_numpy())
                prediction = saved.predicted.to_numpy(float)
                timing = json.loads((run_dir / f"timing_{strategy}.json").read_text())
            else:
                positions = x.index.get_indexer(selection.row_ids)
                if (positions < 0).any():
                    raise AssertionError("Selected context ID is missing from model features")
                prediction, timing = _predict(x.iloc[positions], y[positions], q, baseline)
        np.save(context_file, selection.row_ids)
        write_json(audit_file, selection.audit)
        write_json(run_dir / f"timing_{strategy}.json", timing)
        if not dry_run:
            pd.DataFrame({"row_id": q.index, "month": _months(q), "actual": actual,
                          "predicted": prediction, "weight": weights}).to_csv(
                              prediction_file, index=False, float_format="%.8f")
            score = {"weighted_rmse_sgd": weighted_rmse(actual, prediction, weights),
                     **metrics(actual, prediction)}
        else:
            score = {}
        record = {"run_id": f"{stage}_{strategy}_budget{config.context_size}_seed{seed}",
                  "sampler_train": "existing_random", "sampler_inference": strategy,
                  "feature_pipeline_id": "calendar_80_20_categorical_77_residual",
                  "train_views": 2, "val_views": 2, "inference_views": 1,
                  "context_size": len(selection.row_ids), "query_size": len(q),
                  "learning_rate": .01, "seed": seed,
                  "retrieval_seconds": selection.audit["retrieval_seconds"],
                  "peak_vram_gib": timing["peak_vram_gib"],
                  "inference_seconds": timing["fit_predict_seconds"], **score}
        records.append(record)
        pd.DataFrame(records).to_csv(run_dir / "scores.csv", index=False)
        print(json.dumps({"strategy": strategy, "context": len(selection.row_ids),
                          "query": len(q), **score}), flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "outer"), default="calibration")
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    parser.add_argument("--context-budget", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tau", type=float, default=6.0)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--information-policy", choices=("historical_only", "available_labeled_pool"),
                        default="historical_only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.stage, args.strategies,
        SamplerConfig(context_size=args.context_budget, tau_months=args.tau),
        seed=args.seed, query_limit=args.query_limit,
        information_policy=args.information_policy, dry_run=args.dry_run)
