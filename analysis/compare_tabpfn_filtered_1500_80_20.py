"""Evaluate the <=S$1,500 training-context filter with the retained 77 features.

The chronological June 2024 origin and all 28,995 validation labels stay fixed.
All target-derived histories and temporal baselines are rebuilt from the filtered
context. Static inputs reuse the hash-checked original 77-feature bundle.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.config import ROOT, REPORTS
from analysis.data import prepare_main, sha256
from analysis.evaluation import metrics
from analysis.experiments.aligned_temporal_features import (
    FEATURES, HISTORY_COLUMNS, LocalTemporalModel, episode_origins, origin_history,
)
from analysis.experiments.gpu_capacity import wait_for_gpu
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT


ORIGIN = pd.Period("2024-06", "M")
NAME = "tabpfn_filtered_1500_80_20"
OUT = REPORTS / NAME
CACHE = ROOT / ".cache" / NAME
SOURCE = ROOT / "data/train.csv"
REMOVED = ROOT / "data/processed/whole_flat_rent_le_1500/removed_monthly_rent_le_1500.csv"
BASE = ROOT / ".cache/tabpfn_v3_features/calendar_80_20_categorical.joblib"
BASE_MANIFEST = REPORTS / "tabpfn_v3_5_fast_calendar_80_20/feature_manifest_categorical.json"
CONTROL = REPORTS / "tabpfn_feature_diversified_ensemble_80_20/predictions_2024-06_baseline.csv"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def checked_base() -> dict:
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    if sha256(BASE) != manifest["sha256"] or manifest["features"] != 77:
        raise ValueError("Reference 77-feature bundle changed")
    return joblib.load(BASE)


def prepare() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    bundle_path = CACHE / "features.joblib"
    manifest_path = OUT / "feature_manifest.json"
    if bundle_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256(bundle_path) != manifest["feature_sha256"]:
            raise ValueError("Existing filtered feature bundle changed")
        print("PREPARE reused verified bundle", flush=True)
        return

    base = checked_base()
    raw = pd.read_csv(SOURCE)
    removed = pd.read_csv(REMOVED)
    excluded = removed.SOURCE_CSV_LINE.to_numpy(int) - 2
    if len(raw) != 150000 or len(np.unique(excluded)) != len(excluded):
        raise ValueError("Original source or filter index changed")
    if not raw.iloc[excluded][list(raw)].reset_index(drop=True).equals(
        removed[list(raw)].reset_index(drop=True)
    ):
        raise ValueError("Removed rows do not match the source")
    x_all = base["X_train"]
    q = base["X_query"].copy()
    keep = ~x_all.index.isin(excluded)
    x = x_all.loc[keep].copy()
    train_ids = x.index.to_numpy()
    query_ids = q.index.to_numpy()
    if len(x) != 117560 or len(q) != 28995 or len(x.columns) != 77:
        raise AssertionError("Expected filtered context / full validation shape changed")
    if set(HISTORY_COLUMNS + FEATURES) - set(x.columns):
        raise AssertionError("Target-derived feature definitions changed")

    # Only columns needed to regenerate causal target-derived inputs are built.
    frame = prepare_main(raw)
    locations = pd.concat([x_all.PLANNING_AREA, q.PLANNING_AREA]).sort_index()
    frame["PLANNING_AREA"] = locations.astype(str).to_numpy()
    frame[MONTH] = pd.PeriodIndex(frame.RENT_APPROVAL_DATE, freq="M")
    fitting = frame.loc[train_ids]
    if fitting[MONTH].max() >= ORIGIN or frame.loc[query_ids, MONTH].min() < ORIGIN:
        raise AssertionError("Chronological split changed")
    temporal = LocalTemporalModel(fitting)
    assignments = {
        "train": episode_origins(base["train_month"][keep], train_ids, fitting[MONTH].min()),
        "query": episode_origins(base["query_month"], query_ids, fitting[MONTH].min(), ORIGIN),
    }
    matrices = {"train": x, "query": q}
    target_columns = HISTORY_COLUMNS + FEATURES
    for matrix in matrices.values():
        for column in target_columns:
            matrix[column] = np.nan
    origins = sorted(set(assignments["train"].forecast_origin.dropna()).union(
        assignments["query"].forecast_origin.dropna()))
    for number, origin in enumerate(origins, 1):
        past = fitting.loc[fitting[MONTH] <= origin]
        temporal.fit_at(origin)
        for split, matrix in matrices.items():
            ids = assignments[split].index[assignments[split].forecast_origin.eq(origin)]
            if not len(ids):
                continue
            blind = frame.loc[ids].drop(columns=TARGET)
            history = origin_history(past, blind).reindex(ids)
            matrix.loc[ids, HISTORY_COLUMNS] = history.to_numpy()
            temporal_values = temporal.features(blind, origin).reindex(ids)
            matrix.loc[ids, FEATURES] = temporal_values.to_numpy()
        if number % 5 == 0 or number == len(origins):
            print(f"PREPARE aligned origins {number}/{len(origins)} through {origin}", flush=True)

    train_baseline = np.nan_to_num(x.TEMP_LOCAL_FORECAST.to_numpy(float), nan=0.0)
    query_baseline = q.TEMP_LOCAL_FORECAST.to_numpy(float)
    if not np.isfinite(query_baseline).all():
        raise AssertionError("Validation baseline missing")
    actual = raw.loc[query_ids, TARGET].to_numpy(float)
    target = raw.loc[train_ids, TARGET].to_numpy(float)
    original_static = [name for name in x_all if name not in target_columns]
    for column in original_static:
        pd.testing.assert_series_equal(x[column], x_all.loc[train_ids, column])
        pd.testing.assert_series_equal(q[column], base["X_query"][column])
    filtered = {
        "X_train": x, "X_query": q, "y_train": target, "y_query": actual,
        "train_baseline": train_baseline, "query_baseline": query_baseline,
        "train_month": base["train_month"][keep], "query_month": base["query_month"],
    }
    joblib.dump(filtered, bundle_path, compress=3)
    write_json(manifest_path, {
        "source_sha256": sha256(SOURCE), "removed_sha256": sha256(REMOVED),
        "reference_features_sha256": sha256(BASE), "feature_sha256": sha256(bundle_path),
        "checkpoint_sha256": sha256(CHECKPOINT), "origin": str(ORIGIN),
        "context_rows": len(x), "validation_rows": len(q), "removed_from_context": int((~keep).sum()),
        "removed_from_validation": 0, "features": list(x.columns),
        "target_derived_columns_rebuilt": target_columns,
        "context_last_month": str(fitting[MONTH].max()),
        "validation_first_month": str(frame.loc[query_ids, MONTH].min()),
    })
    write_json(OUT / "status.json", {"stage": "prepared", "context_rows": len(x), "validation_rows": len(q)})
    print(f"PREPARE complete {len(x):,} context / {len(q):,} validation", flush=True)


def run() -> None:
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    prediction_path = OUT / "predictions.csv"
    if prediction_path.exists():
        print("RUN existing predictions preserved", flush=True)
        return
    if sha256(CACHE / "features.joblib") != json.loads((OUT / "feature_manifest.json").read_text())["feature_sha256"]:
        raise ValueError("Prepared features changed")
    wait_for_gpu(min_free_gib=10, max_utilization=35, stable_samples=2, poll_seconds=15,
                 status_path=OUT / "gpu_gate.json")
    bundle = joblib.load(CACHE / "features.joblib")
    x, query = bundle["X_train"], bundle["X_query"]
    target = bundle["y_train"] - bundle["train_baseline"]
    category_positions = [i for i, name in enumerate(x) if isinstance(x[name].dtype, pd.CategoricalDtype)]
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    write_json(OUT / "status.json", {"stage": "fitting", "context_rows": len(x)})
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT), device="cuda", n_estimators=2,
        random_state=42, fit_mode="fit_with_cache", keep_cache_on_device=True,
        n_preprocessing_jobs=1, categorical_features_indices=category_positions,
        inference_config={"SUBSAMPLE_SAMPLES": None},
    )
    try:
        model.fit(x, target)
        if model.n_train_samples_ != len(x) or model.n_estimators_ != 2:
            raise AssertionError("TabPFN did not retain the full filtered context")
        indices = model.ensemble_preprocessor_.subsample_row_indices
        if indices is not None and any(len(v) != len(x) for v in indices):
            raise AssertionError("TabPFN subsampled context")
        fitted = time.perf_counter()
        print(f"RUN fitted {len(x):,} rows; predicting {len(query):,}", flush=True)
        values = np.empty(len(query), dtype=float)
        for offset in range(0, len(query), 200):
            stop = min(offset + 200, len(query))
            torch.cuda.empty_cache()
            values[offset:stop] = np.asarray(model.predict(query.iloc[offset:stop], output_type="mean"), float)
            if offset % 1000 == 0 or stop == len(query):
                print(f"RUN predicted {stop:,}/{len(query):,}", flush=True)
        torch.cuda.synchronize()
        if not np.isfinite(values).all():
            raise AssertionError("Nonfinite predictions")
        predicted = values + bundle["query_baseline"]
        pd.DataFrame({"row_id": query.index.to_numpy(), "month": bundle["query_month"],
                      "actual": bundle["y_query"], "predicted": predicted}).to_csv(
            prediction_path, index=False, float_format="%.10f")
        write_json(OUT / "timing.json", {"fit_seconds": fitted - started,
                   "predict_seconds": time.perf_counter() - fitted,
                   "peak_pytorch_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "context_rows": len(x), "validation_rows": len(query), "n_estimators": 2,
                   "subsampling": False, "prediction_sha256": sha256(prediction_path)})
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(OUT / "status.json", {"stage": "predicted", "validation_rows": len(query)})


def summarize() -> None:
    current = pd.read_csv(OUT / "predictions.csv")
    reference = pd.read_csv(CONTROL)
    np.testing.assert_array_equal(current.row_id.to_numpy(), reference.row_id.to_numpy())
    np.testing.assert_array_equal(current.actual.to_numpy(), reference.actual.to_numpy())
    current["reference"] = reference.predicted.to_numpy(float)
    removed = pd.read_csv(REMOVED)
    current["would_filter_validation"] = current.row_id.isin(removed.SOURCE_CSV_LINE.to_numpy(int) - 2)
    rows = []
    for group, part in [("all", current), ("low_rent_validation", current[current.would_filter_validation]),
                        ("other_validation", current[~current.would_filter_validation])]:
        for name in ("reference", "predicted"):
            rows.append({"group": group, "model": "unfiltered_77" if name == "reference" else "filtered_1500_77",
                         "n": len(part), **metrics(part.actual.to_numpy(float), part[name].to_numpy(float))})
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    current.to_csv(OUT / "comparison_predictions.csv", index=False, float_format="%.10f")
    timing = json.loads((OUT / "timing.json").read_text(encoding="utf-8"))
    all_scores = scores[scores.group.eq("all")]
    original_rmse, filtered_rmse = all_scores.rmse_sgd.to_numpy(float)
    lines = ["# TabPFN on <=S$1,500-filtered context", "",
             f"Chronological origin: June 2024. Filtered context: {timing['context_rows']:,} through May 2024; "
             "all 28,995 June 2024-March 2025 validation rows retained. The same 77 static, spatial, "
             "causal rent-history and local temporal forecast features as the strongest retained frozen "
             "TabPFN baseline were used. Every target-derived history and baseline was rebuilt using only "
             "the filtered pre-origin context. Frozen TabPFN-3.5 Fast weights, two estimators, full context, "
             "seed 42, no subsampling; fit computes a context cache rather than updating neural weights.", "",
             "| Validation group | Model | Rows | RMSE (S$) | MAE (S$) | Bias (S$) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in scores.itertuples():
        lines.append(f"| {row.group} | {row.model} | {row.n:,} | {row.rmse_sgd:.3f} | {row.mae_sgd:.3f} | {row.bias_sgd:.3f} |")
    lines += ["", f"Filtered minus unfiltered all-row RMSE: **S${filtered_rmse - original_rmse:+.3f}** "
              "(negative favors filtering). The reference is the saved native-default two-view 77-feature "
              "frozen baseline on the identical validation rows. These development labels have been examined "
              "in earlier experiments; this is a controlled development comparison, not an untouched test.", "",
              "The price rule does not establish that any removed record is a bedroom-only lease. "
              "Validation includes rows that would meet the rule, preserving a realistic population metric.", "",
              "Artifacts: [scores](scores.csv), [paired predictions](comparison_predictions.csv), "
              "[feature manifest](feature_manifest.json), [GPU timing](timing.json).", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_filtered_1500_80_20.py --stage all`."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(OUT / "status.json", {"stage": "complete", "rmse_sgd": filtered_rmse,
               "reference_rmse_sgd": original_rmse})
    print(scores.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "summarize", "all"), default="all")
    stage = parser.parse_args().stage
    if stage in ("prepare", "all"):
        prepare()
    if stage in ("run", "all"):
        run()
    if stage in ("summarize", "all"):
        summarize()


if __name__ == "__main__":
    main()
