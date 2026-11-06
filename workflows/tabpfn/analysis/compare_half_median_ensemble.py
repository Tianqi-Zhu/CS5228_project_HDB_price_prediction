"""Controlled training-only half-median ablation of zch + anniversary TabPFN.

Run: .venv/Scripts/python.exe analysis/compare_half_median_ensemble.py --stage all
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.compare_tabpfn_anniversary_history_80_20 import source_bundle, CHECKPOINT, BASE
from analysis.config import ROOT, DATA, REPORTS
from analysis.data import prepare_main, normalized_address, sha256
from analysis.evaluation import metrics
from analysis.experiments.aligned_temporal_features import (
    FEATURES, HISTORY_COLUMNS, LocalTemporalModel, episode_origins, origin_history,
)
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET
from analysis.experiments.tabpfn_anniversary_history import AnniversaryHistory, FEATURES as ANNIVERSARY
from analysis.filter_rent_below_half_median import build_audit, FLAG

NAME = "half_median_zch_tabpfn_ensemble"
OUT = REPORTS / NAME
CACHE = ROOT / ".cache" / NAME
ORIGIN = pd.Period("2024-06", "M")
CONTROL = REPORTS / "zch_anniversary_tabpfn_blend/predictions_june.csv"


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def filter_context(raw: pd.DataFrame, train_ids) -> pd.DataFrame:
    """Estimate cutoffs exclusively from supplied training IDs, preserving IDs."""
    fitting = raw.loc[train_ids]
    if fitting.RENT_APPROVAL_DATE.ge(str(ORIGIN)).any():
        raise ValueError("Validation rows entered the training filter")
    audit = build_audit(fitting)
    audit["SOURCE_CSV_LINE"] = audit.index + 2
    return audit


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "features.joblib"
    if path.exists():
        manifest = json.loads((OUT / "feature_manifest.json").read_text())
        if sha256(path) != manifest["feature_sha256"] or sha256(DATA / "train.csv") != manifest["source_sha256"]:
            raise ValueError("Cached features/source changed")
        print("PREPARE reused verified features", flush=True)
        return
    base = source_bundle()
    raw = pd.read_csv(DATA / "train.csv")
    audit = filter_context(raw, base["X_train"].index)
    train_ids = audit.index[~audit[FLAG]]
    query_ids = base["X_query"].index
    assert set(train_ids).isdisjoint(query_ids)
    assert len(train_ids) + int(audit[FLAG].sum()) == len(base["X_train"])
    audit.loc[audit[FLAG]].to_csv(OUT / "removed_training_rows.csv", index=False)
    pd.DataFrame({"row_id": train_ids}).to_csv(OUT / "training_row_ids.csv", index=False)
    x, q = base["X_train"].loc[train_ids].copy(), base["X_query"].copy()
    frame = prepare_main(raw)
    locations = pd.concat([base["X_train"].PLANNING_AREA, q.PLANNING_AREA]).sort_index()
    frame["PLANNING_AREA"] = locations.astype(str).to_numpy()
    frame[MONTH] = pd.PeriodIndex(frame.RENT_APPROVAL_DATE, freq="M")
    fitting = frame.loc[train_ids]
    assert fitting[MONTH].max() < ORIGIN <= frame.loc[query_ids, MONTH].min()
    temporal = LocalTemporalModel(fitting)
    assignments = {
        "train": episode_origins(raw.loc[train_ids, "RENT_APPROVAL_DATE"], train_ids, fitting[MONTH].min()),
        "query": episode_origins(base["query_month"], query_ids, fitting[MONTH].min(), ORIGIN),
    }
    matrices = {"train": x, "query": q}
    target_columns = HISTORY_COLUMNS + FEATURES
    for matrix in matrices.values():
        for col in target_columns:
            matrix[col] = np.nan
    origins = sorted(set(assignments["train"].forecast_origin.dropna()).union(assignments["query"].forecast_origin.dropna()))
    print(f"PREPARE {len(train_ids):,} retained / {int(audit[FLAG].sum()):,} removed; rebuilding {len(origins)} origins", flush=True)
    for number, origin in enumerate(origins, 1):
        past = fitting.loc[fitting[MONTH] <= origin]
        temporal.fit_at(origin)
        for split, matrix in matrices.items():
            ids = assignments[split].index[assignments[split].forecast_origin.eq(origin)]
            if not len(ids):
                continue
            blind = frame.loc[ids].drop(columns=TARGET)
            matrix.loc[ids, HISTORY_COLUMNS] = origin_history(past, blind).reindex(ids).to_numpy()
            matrix.loc[ids, FEATURES] = temporal.features(blind, origin).reindex(ids).to_numpy()
        if number % 5 == 0 or number == len(origins):
            print(f"PREPARE origins {number}/{len(origins)} through {origin}", flush=True)
    for col in [c for c in base["X_train"] if c not in target_columns]:
        pd.testing.assert_series_equal(x[col], base["X_train"].loc[train_ids, col])
        pd.testing.assert_series_equal(q[col], base["X_query"][col])
    # Refit category vocabularies after removal, using training rows only.
    for col in x.select_dtypes(include="category"):
        levels = pd.Index(x[col].dropna().astype(str).unique())
        for matrix in (x, q):
            values = matrix[col].astype("string")
            matrix[col] = pd.Categorical(values.where(values.isin(levels)), categories=levels)
    addresses = normalized_address(prepare_main(raw))
    history = AnniversaryHistory().fit(addresses.loc[train_ids])
    x = pd.concat([x, history.transform(addresses.loc[train_ids])[ANNIVERSARY[:10]]], axis=1)
    q = pd.concat([q, history.transform(addresses.loc[query_ids].drop(columns=TARGET))[ANNIVERSARY[:10]]], axis=1)
    assert len(x.columns) == 87 and list(x) == list(q)
    query_baseline = q.TEMP_LOCAL_FORECAST.to_numpy(float)
    assert np.isfinite(query_baseline).all()
    bundle = dict(X_train=x, X_query=q, y_train=raw.loc[train_ids, TARGET].to_numpy(float),
                  y_query=raw.loc[query_ids, TARGET].to_numpy(float),
                  train_baseline=np.nan_to_num(x.TEMP_LOCAL_FORECAST.to_numpy(float), nan=0.0),
                  query_baseline=query_baseline, query_month=base["query_month"])
    joblib.dump(bundle, path, compress=3)
    write_json(OUT / "feature_manifest.json", dict(
        source_sha256=sha256(DATA / "train.csv"), reference_feature_sha256=sha256(BASE),
        feature_sha256=sha256(path), checkpoint_sha256=sha256(CHECKPOINT),
        context_rows=len(train_ids), removed_training_rows=int(audit[FLAG].sum()),
        original_context_rows=len(audit), validation_rows=len(q), removed_validation_rows=0,
        training_ids_sha256=sha256(OUT / "training_row_ids.csv"),
        filter="rent < half year x town x flat-type median, estimated from training partition only",
        origin=str(ORIGIN), features=list(x), rebuilt_columns=target_columns + ANNIVERSARY[:10],
        removed_by_year=audit.loc[audit[FLAG]].groupby("YEAR").size().to_dict(),
    ))
    print("PREPARE complete", flush=True)


def tabpfn() -> None:
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    from analysis.experiments.gpu_capacity import wait_for_gpu

    output = OUT / "predictions_tabpfn.csv"
    if output.exists():
        print("TABPFN preserving existing predictions", flush=True)
        return
    manifest = json.loads((OUT / "feature_manifest.json").read_text())
    if sha256(CACHE / "features.joblib") != manifest["feature_sha256"]:
        raise ValueError("Prepared features changed")
    wait_for_gpu(min_free_gib=10, max_utilization=35, stable_samples=2, poll_seconds=10, status_path=OUT / "gpu_gate.json")
    base = joblib.load(CACHE / "features.joblib")
    x, q = base["X_train"], base["X_query"]
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    categorical = [i for i, col in enumerate(x) if isinstance(x[col].dtype, pd.CategoricalDtype)]
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT), device="cuda", n_estimators=2,
        random_state=42, fit_mode="fit_with_cache", keep_cache_on_device=True,
        categorical_features_indices=categorical, inference_config={"SUBSAMPLE_SAMPLES": None},
        n_preprocessing_jobs=1,
    )
    started = time.perf_counter()
    partial = CACHE / "tabpfn_partial.npy"
    values = np.load(partial) if partial.exists() else np.full(len(q), np.nan)
    assert values.shape == (len(q),)
    try:
        model.fit(x, base["y_train"] - base["train_baseline"])
        assert model.n_train_samples_ == len(x) and model.n_estimators_ == 2
        assert model.inference_config_.SUBSAMPLE_SAMPLES is None
        fitted = time.perf_counter()
        print(f"TABPFN fitted {len(x):,} rows", flush=True)
        for start in range(0, len(q), 200):
            stop = min(start + 200, len(q))
            if np.isfinite(values[start:stop]).all():
                continue
            torch.cuda.empty_cache()
            values[start:stop] = np.asarray(model.predict(q.iloc[start:stop], output_type="mean"), float) + base["query_baseline"][start:stop]
            np.save(partial, values)
            if start % 2000 == 0 or stop == len(q):
                print(f"TABPFN {stop:,}/{len(q):,}", flush=True)
        assert np.isfinite(values).all()
        pd.DataFrame(dict(row_id=q.index, month=base["query_month"], actual=base["y_query"], predicted=values)).to_csv(output, index=False, float_format="%.17g")
        write_json(OUT / "tabpfn_timing.json", dict(fit_seconds=fitted-started, predict_seconds=time.perf_counter()-fitted,
                   peak_vram_gib=torch.cuda.max_memory_allocated()/1024**3,
                   feature_sha256=manifest["feature_sha256"], prediction_sha256=sha256(output), context_rows=len(x), subsampling=False))
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def summarize() -> None:
    original = pd.read_csv(CONTROL).sort_values("row_id").reset_index(drop=True)
    current = original[["row_id", "month", "actual"]].copy()
    for name in ["zch", "tabpfn"]:
        part = pd.read_csv(OUT / f"predictions_{name}.csv").sort_values("row_id").reset_index(drop=True)
        np.testing.assert_array_equal(part.row_id, original.row_id)
        np.testing.assert_array_equal(part.actual, original.actual)
        np.testing.assert_array_equal(part.month, original.month)
        current[f"filtered_{name}"] = part.predicted.to_numpy(float)
    current["original_zch"] = original.zch_original
    current["original_tabpfn"] = original.tabpfn87
    current["original_equal_blend"] = (current.original_zch + current.original_tabpfn) / 2
    current["filtered_equal_blend"] = (current.filtered_zch + current.filtered_tabpfn) / 2
    current["only_zch_filtered_blend"] = (current.filtered_zch + current.original_tabpfn) / 2
    current["only_tabpfn_filtered_blend"] = (current.original_zch + current.filtered_tabpfn) / 2
    models = [c for c in current if c not in ["row_id", "month", "actual"]]
    scores = pd.DataFrame([dict(model=c, rows=len(current), **metrics(current.actual, current[c])) for c in models])
    monthly = pd.DataFrame([dict(month=m, model=c, rows=len(part), **metrics(part.actual, part[c]))
                            for m, part in current.groupby("month") for c in models])
    # Diagnostic only: identify cheap validation rows using training-only cutoffs.
    train_audit = filter_context(pd.read_csv(DATA / "train.csv"), source_bundle()["X_train"].index)
    lookup = train_audit.groupby(["YEAR", "NORMALIZED_TOWN", "NORMALIZED_FLAT_TYPE"]).GROUP_MEDIAN_RENT.first()
    raw = pd.read_csv(DATA / "train.csv").loc[current.row_id]
    keys = pd.MultiIndex.from_arrays([raw.RENT_APPROVAL_DATE.str[:4].astype(int), raw.TOWN.str.lower().str.strip(),
                                     raw.FLAT_TYPE.str.strip().str.lower().str.replace(r"\s+", "-", regex=True)], names=lookup.index.names)
    cutoff = lookup.reindex(keys).to_numpy(float) / 2
    supported = np.isfinite(cutoff)
    low = supported & (raw.MONTHLY_RENT.to_numpy(float) < cutoff)
    subsets = [("below_training_cutoff", low), ("other_with_training_cutoff", supported & ~low), ("no_same_year_training_cutoff", ~supported)]
    subgroup = pd.DataFrame([dict(group=name, model=c, rows=int(mask.sum()), **metrics(current.actual[mask], current.loc[mask, c]))
                            for name, mask in subsets if mask.any() for c in ["original_equal_blend", "filtered_equal_blend"]])
    # Paired uncertainty, clustered by validation month; descriptive development evidence.
    errors = (current.filtered_equal_blend-current.actual)**2 - (current.original_equal_blend-current.actual)**2
    block = pd.DataFrame(dict(month=current.month, delta_sse=errors)).groupby("month").agg(total=("delta_sse", "sum"), n=("delta_sse", "size"))
    rng = np.random.default_rng(42)
    draws = rng.integers(len(block), size=(10000, len(block)))
    bootstrap = block.total.to_numpy()[draws].sum(axis=1) / block.n.to_numpy()[draws].sum(axis=1)
    interval = np.quantile(bootstrap, [.025, .975])
    original_rmse = float(scores.set_index("model").loc["original_equal_blend", "rmse_sgd"])
    filtered_rmse = float(scores.set_index("model").loc["filtered_equal_blend", "rmse_sgd"])
    wide = monthly.pivot(index="month", columns="model", values="rmse_sgd")
    wins = int((wide.filtered_equal_blend < wide.original_equal_blend).sum())
    for filename, frame in [("scores", scores), ("monthly_scores", monthly), ("subgroup_scores", subgroup), ("comparison_predictions", current)]:
        frame.to_csv(OUT / f"{filename}.csv", index=False, float_format="%.10f")
    summary = dict(original_rmse_sgd=original_rmse, filtered_rmse_sgd=filtered_rmse,
                   delta_rmse_sgd=filtered_rmse-original_rmse, months_improved=wins, validation_rows=len(current),
                   month_cluster_bootstrap_delta_mse_95pct=interval.tolist(),
                   original_prediction_sha256=sha256(CONTROL), filtered_predictions_sha256=sha256(OUT / "comparison_predictions.csv"))
    write_json(OUT / "summary.json", summary)
    manifest = json.loads((OUT / "feature_manifest.json").read_text())
    lines = ["# Half-median training filter: latest zch + TabPFN ensemble", "",
             f"Original equal-blend RMSE **S${original_rmse:.4f}**; filtered equal-blend RMSE **S${filtered_rmse:.4f}**; change **S${filtered_rmse-original_rmse:+.4f}** (negative favors filtering). Improved in **{wins}/10 months**.", "",
             f"Training: January 2021-May 2024, {manifest['original_context_rows']:,} original rows, {manifest['removed_training_rows']:,} removed, {manifest['context_rows']:,} retained. Validation: every original {len(current):,} June 2024-March 2025 row, unchanged.", "",
             "Cutoffs are estimated once from each year x town x normalized flat-type group within the training partition only. Thus 2024 medians use January-May only; the prior whole-dataset flag file is not used. No validation rent influences the filter. All training rent histories, local temporal residual baselines, market normalizers and anniversary cohorts are rebuilt after removal. Static TabPFN columns are verified unchanged; category vocabularies are refit on retained rows.", "",
             "The latest ensemble remains 50% zch (seven fixed published tree weights) and 50% anniversary87 TabPFN. Tree rounds are fixed to their original earlier-screen values; there is no validation early stopping. TabPFN retains the same checkpoint, 87 columns, two estimator views, full retained context and seed 42. No blend weights or hyperparameters are tuned for this comparison.", "",
             "| Model | Rows | RMSE (SGD) | MAE (SGD) | Bias (SGD) |", "|---|---:|---:|---:|---:|"]
    for row in scores.itertuples():
        lines.append(f"| {row.model} | {row.rows:,} | {row.rmse_sgd:.4f} | {row.mae_sgd:.4f} | {row.bias_sgd:+.4f} |")
    lines += ["", "The two partly filtered blends are diagnostic combinations of the same predictions, without additional training or weight fitting.", "",
              "| Month | Original RMSE | Filtered RMSE | Change |", "|---|---:|---:|---:|"]
    for month, row in wide.iterrows():
        lines.append(f"| {month} | {row.original_equal_blend:.3f} | {row.filtered_equal_blend:.3f} | {row.filtered_equal_blend-row.original_equal_blend:+.3f} |")
    lines += ["", f"Paired month-cluster bootstrap (10,000 draws, seed 42): filtered-minus-original MSE 95% descriptive interval [{interval[0]:+.2f}, {interval[1]:+.2f}] SGD squared.", "",
              "These validation months have been used for development repeatedly. This is a matched ablation, not an untouched test or evidence of Kaggle improvement. Low-rent flags identify statistical outliers, not verified incorrect labels. The year-based filter is retrospective within the training partition, not an online chronological cleaning rule.", "",
              "`subgroup_scores.csv` separates validation rows with rents below the available training-only cutoff, other supported rows, and unsupported groups (including 2025, since no 2025 labels occur in training). These diagnostic groups do not change the primary all-row score.", "",
              "Original training data, source code and submissions are preserved. No replacement full-data submission is generated by this evaluation.", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/compare_half_median_ensemble.py --stage all`.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(scores.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "tabpfn", "trees", "summarize", "all"], default="all")
    args = parser.parse_args()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.pop("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", None)
    if args.stage in ["prepare", "all"]:
        prepare()
    if args.stage in ["tabpfn", "all"]:
        tabpfn()
    if args.stage in ["trees", "all"]:
        subprocess.run([sys.executable, str(ROOT / "analysis/train_half_median_zch_worker.py")], check=True, cwd=ROOT)
    if args.stage in ["summarize", "all"]:
        summarize()


if __name__ == "__main__":
    main()
