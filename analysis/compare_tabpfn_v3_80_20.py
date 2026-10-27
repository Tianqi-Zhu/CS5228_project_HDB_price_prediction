"""Chronological 80/20 evaluation with the full training pool as TabPFN context.

Run from the repository root with .venv/Scripts/python.exe. The closest whole
month boundary to 80% of labelled rows is June 2024. Validation histories stay
frozen at that origin. A recent-year context and identically featured CatBoost
control are evaluated on the same validation transactions.
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

from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.dynamic_market_index import TREE_SETTINGS
from analysis.experiments.spatiotemporal_hybrid import MONTH, SOURCES, spatial_frame
from analysis.prepare_tabpfn_v3_features import make_bundle


OUTPUT = REPORTS / "tabpfn_v3_80_20"
BUNDLE = ROOT / ".cache/tabpfn_v3_features/holdout_80_20_2024-06.joblib"
ORIGIN = pd.Period("2024-06", freq="M")
CHECKPOINT = ROOT / ".cache/tabpfn/tabpfn-v3-regressor-v3_default.ckpt"
MODELS = ("tabpfn_full_context", "tabpfn_recent_year", "catboost_matched")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def markdown_table(frame):
    def display(value):
        if pd.isna(value):
            return ""
        return f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
    rows = ["| " + " | ".join(frame.columns) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |"]
    rows.extend("| " + " | ".join(display(value) for value in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def tabpfn_predict(x, y, query, *, keep_cache_on_device):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3, model_path=str(CHECKPOINT), device="cuda",
        n_estimators=1, random_state=42, fit_mode="fit_with_cache",
        keep_cache_on_device=keep_cache_on_device,
        inference_config={"SUBSAMPLE_SAMPLES": None},
    )
    started = time.perf_counter()
    try:
        model.fit(x, y)
        torch.cuda.synchronize()
        fitted = time.perf_counter()
        if model.n_train_samples_ != len(x):
            raise AssertionError("SDK fit row count differs from the requested context")
        indices = model.ensemble_preprocessor_.subsample_row_indices
        if indices is not None:
            for subset in indices:
                if len(subset) != len(x) or len(np.unique(subset)) != len(x):
                    raise AssertionError("SDK subsampled the requested context")
        print(f"  Fit complete: all {len(x):,} context rows retained", flush=True)
        predicted = []
        for offset in range(0, len(query), 200):
            predicted.append(np.asarray(model.predict(query.iloc[offset:offset + 200], output_type="mean"), float))
            if offset % 4000 == 0 or offset + 200 >= len(query):
                print(f"  Predicted {min(offset + 200, len(query)):,}/{len(query):,}", flush=True)
        torch.cuda.synchronize()
        values = np.concatenate(predicted)
        if len(values) != len(query) or not np.isfinite(values).all():
            raise AssertionError("Incomplete or nonfinite predictions")
        return values, {
            "fit_seconds": fitted - started,
            "predict_seconds": time.perf_counter() - fitted,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
            "context_rows": len(x), "subsampling": False,
            "keep_cache_on_device": keep_cache_on_device,
        }
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def full_context_predict(x, y, query):
    try:
        return tabpfn_predict(x, y, query, keep_cache_on_device=True)
    except Exception as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print("GPU cache exceeded memory; retrying the same complete context with CPU cache", flush=True)
        write_json(OUTPUT / "gpu_cache_retry.json", {"reason": str(exc), "context_rows": len(x)})
    gc.collect()
    return tabpfn_predict(x, y, query, keep_cache_on_device=False)


def main():
    import torch
    from catboost import CatBoostRegressor

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    code_paths = [Path(__file__), ROOT / "analysis/prepare_tabpfn_v3_features.py",
                  ROOT / "analysis/experiments/tabpfn_adaptation_features.py",
                  ROOT / "analysis/experiments/recent_history.py",
                  ROOT / "analysis/experiments/spatiotemporal_hybrid.py", ROOT / "analysis/data.py"]
    settings = {
        "model_version": "v3", "origin": str(ORIGIN), "n_estimators": 1,
        "context_policy": "all training rows; no subsampling", "batch_size": 200,
        "seed": 42, "fit_mode": "fit_with_cache", "checkpoint_sha256": sha256(CHECKPOINT),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in SOURCES},
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code_paths},
        "versions": {key: version(key) for key in ("tabpfn", "torch", "pandas", "catboost")},
        "gpu": torch.cuda.get_device_name(), "catboost_settings": TREE_SETTINGS["catboost"],
    }
    settings_path = OUTPUT / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing results have different code, data, or settings")
    write_json(settings_path, settings)
    write_json(OUTPUT / "status.json", {"status": "preparing"})
    manifest_path = OUTPUT / "feature_manifest.json"
    if BUNDLE.exists() and manifest_path.exists():
        audit = json.loads(manifest_path.read_text(encoding="utf-8"))
        if sha256(BUNDLE) != audit["sha256"]:
            raise ValueError("Saved feature bundle changed")
        bundle = joblib.load(BUNDLE)  # Trusted project-generated artifact.
    else:
        frame = spatial_frame(pd.read_csv(DATA / "train.csv"))
        fitting = frame.loc[frame[MONTH] < ORIGIN].copy()
        query = frame.loc[frame[MONTH] >= ORIGIN].copy()
        # Check that the chosen complete-month boundary is closest to 80% by rows.
        counts = frame.groupby(MONTH).size().sort_index().cumsum().iloc[:-1]
        closest = (counts - 0.8 * len(frame)).abs().idxmin() + 1
        if closest != ORIGIN:
            raise AssertionError("Dataset no longer supports the recorded 80/20 origin")
        print(f"Preparing {len(fitting):,} context / {len(query):,} validation rows", flush=True)
        bundle, audit = make_bundle(fitting, query, str(ORIGIN), has_query_labels=True)
        BUNDLE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, BUNDLE, compress=3)
        audit.update({"artifact": BUNDLE.relative_to(ROOT).as_posix(), "sha256": sha256(BUNDLE),
                      "train_fraction": len(fitting) / len(frame)})
        write_json(manifest_path, audit)
    x, y = bundle["X_train"], np.asarray(bundle["y_train"], float)
    query, actual = bundle["X_query"], np.asarray(bundle["y_query"], float)
    results = pd.DataFrame({"row_id": query.index.to_numpy(), "month": bundle["query_month"], "actual": actual})
    timing_rows = []
    for name in MODELS:
        path = OUTPUT / f"predictions_{name}.csv"
        timing_path = OUTPUT / f"timing_{name}.json"
        write_json(OUTPUT / "status.json", {"status": "running", "model": name})
        if path.exists() and timing_path.exists():
            saved = pd.read_csv(path)
            np.testing.assert_array_equal(saved.row_id.to_numpy(), results.row_id.to_numpy())
            np.testing.assert_array_equal(saved.actual.to_numpy(), actual)
            values = saved.predicted.to_numpy(float)
            timing = json.loads(timing_path.read_text(encoding="utf-8"))
        elif name.startswith("tabpfn"):
            keep = np.ones(len(x), dtype=bool)
            if name == "tabpfn_recent_year":
                keep = pd.PeriodIndex(bundle["train_month"], freq="M") >= ORIGIN - 12
            print(f"Running {name}: {int(keep.sum()):,} context rows", flush=True)
            values, timing = full_context_predict(x.iloc[np.flatnonzero(keep)], y[keep], query)
        else:
            print("Running CatBoost with identical features and all training rows", flush=True)
            train_tree, query_tree = x.copy(), query.copy()
            categories = [c for c in x if isinstance(x[c].dtype, pd.CategoricalDtype)]
            for c in categories:
                train_tree[c] = x[c].astype("string").fillna("__missing__").astype(str)
                query_tree[c] = query[c].astype("string").fillna("__missing__").astype(str)
            started = time.perf_counter()
            model = CatBoostRegressor(**TREE_SETTINGS["catboost"])
            model.fit(train_tree, y, cat_features=categories)
            fitted = time.perf_counter()
            values = model.predict(query_tree)
            timing = {"fit_seconds": fitted - started, "predict_seconds": time.perf_counter() - fitted,
                      "training_rows": len(x)}
        results[name] = values
        pd.DataFrame({"row_id": results.row_id, "actual": actual, "predicted": values}).to_csv(path, index=False, float_format="%.8f")
        write_json(timing_path, timing)
        timing_rows.append({"model": name, **timing})
        print(f"{name}: {metrics(actual, values)}", flush=True)
    scores = pd.DataFrame([{"model": name, "n": len(results), **metrics(actual, results[name])} for name in MODELS])
    monthly = pd.DataFrame([
        {"model": name, "month": month, "n": len(part), **metrics(part.actual, part[name])}
        for month, part in results.groupby("month", sort=True) for name in MODELS
    ])
    results.to_csv(OUTPUT / "validation_predictions.csv", index=False, float_format="%.8f")
    scores.to_csv(OUTPUT / "scores.csv", index=False, float_format="%.8f")
    monthly.to_csv(OUTPUT / "monthly_scores.csv", index=False, float_format="%.8f")
    pd.DataFrame(timing_rows).to_csv(OUTPUT / "timings.csv", index=False)
    report = ["# TabPFN-3 chronological 80/20 experiment", "",
              f"Context: January 2021–May 2024, {len(x):,} rows ({audit['train_fraction']:.2%}). "
              f"Validation: June 2024–March 2025, {len(query):,} rows. Whole months remain together.", "",
              "All models use the same 68 feature columns. Training histories use preceding months only; "
              "validation histories freeze at May 2024. TabPFN uses one estimator with no row subsampling. "
              "The recent-year control uses June 2023–May 2024 as direct context, while retaining the same "
              "pre-origin historical features. CatBoost fits all context rows. Validation labels are never fed back.", "",
              markdown_table(scores), "", "## Runtime", "",
              markdown_table(pd.DataFrame(timing_rows)), "",
              "This is a ten-month development holdout, shorter than the supplied test's 17-month horizon. "
              "Its months have been examined in previous project experiments, so it is not an untouched final test.", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_v3_80_20.py`."]
    (OUTPUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUTPUT / "status.json", {"status": "complete", "validation_rows": len(query), "context_rows": len(x)})
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
