"""Ablate learned temporal forecasts while retaining causal levels and trend."""

from __future__ import annotations

import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.compare_tabpfn_v3_80_20 import markdown_table, write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT, predict_all_context
from analysis.config import REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.tabpfn_temporal_features import FEATURES


SOURCE = REPORTS / "tabpfn_v3_5_fast_temporal_80_20"
OUT = REPORTS / "tabpfn_v3_5_fast_temporal_levels_80_20"
ARTIFACT = ROOT / ".cache/tabpfn_v3_features/holdout_80_20_2024-06_temporal_levels.joblib"
REMOVED = ["TEMP_FLAT_MODEL_FORECAST", "TEMP_FLAT_MODEL_DELTA"]
ADDED = [name for name in FEATURES if name not in REMOVED]
MODEL = "tabpfn_fast_temporal_levels"


def main():
    import torch

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((SOURCE / "settings.json").read_text(encoding="utf-8"))
    audit = json.loads((SOURCE / "feature_manifest.json").read_text(encoding="utf-8"))
    source = ROOT / audit["artifact"]
    if sha256(source) != audit["sha256"]:
        raise ValueError("Saved temporal feature bundle changed")
    hashes = {name: sha256(ROOT / name) for name in prior["source_sha256"]}
    if hashes != prior["source_sha256"]:
        raise ValueError("Underlying source data changed")
    controls = {
        "tabpfn_fast_original": REPORTS / "tabpfn_v3_5_fast_80_20/predictions.csv",
        "tabpfn_fast_temporal_full": SOURCE / "predictions.csv",
        "catboost_original_68_features": REPORTS / "tabpfn_v3_80_20/predictions_catboost_matched.csv",
    }
    settings = {
        "model_version": "v3.5-fast", "n_estimators": 1, "seed": 42,
        "batch_size": 200, "context_rows": 121_005, "subsampling": False,
        "fit_mode": "fit_with_cache", "checkpoint_sha256": sha256(CHECKPOINT),
        "source_sha256": hashes, "source_enriched_bundle_sha256": audit["sha256"],
        "added_features": ADDED, "removed_forecast_features": REMOVED,
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in (
            Path(__file__), ROOT / "analysis/compare_tabpfn_v3_5_fast_80_20.py",
            ROOT / "analysis/compare_tabpfn_v3_80_20.py")},
        "reference_predictions_sha256": {name: sha256(path) for name, path in controls.items()},
        "versions": {name: version(name) for name in ("tabpfn", "torch", "pandas", "numpy")},
        "gpu": torch.cuda.get_device_name(),
    }
    settings_path = OUT / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing output has different settings")
    write_json(settings_path, settings)
    bundle = joblib.load(source)  # Trusted, hash-checked project-generated artifact.
    bundle["X_train"] = bundle["X_train"].drop(columns=REMOVED)
    bundle["X_query"] = bundle["X_query"].drop(columns=REMOVED)
    x, query = bundle["X_train"], bundle["X_query"]
    y, actual = np.asarray(bundle["y_train"], float), np.asarray(bundle["y_query"], float)
    if len(x) != 121_005 or len(query) != 28_995 or len(x.columns) != 74:
        raise AssertionError("The ablation does not match the original split")
    if not ARTIFACT.exists():
        joblib.dump(bundle, ARTIFACT, compress=3)
    write_json(OUT / "feature_manifest.json", {
        "artifact": ARTIFACT.relative_to(ROOT).as_posix(), "sha256": sha256(ARTIFACT),
        "source_enriched_bundle_sha256": audit["sha256"], "feature_count": len(x.columns),
        "added_features": ADDED, "context_rows": len(x), "validation_rows": len(query),
    })
    for partition in ("train", "validation"):
        columns = ["row_id", "month", "forecast_origin", "flat_type", *ADDED]
        pd.read_csv(SOURCE / f"temporal_features_{partition}.csv")[columns].to_csv(
            OUT / f"temporal_features_{partition}.csv", index=False, float_format="%.8f")
    results = pd.DataFrame({"row_id": query.index.to_numpy(), "month": bundle["query_month"], "actual": actual})
    for name, path in controls.items():
        saved = pd.read_csv(path)
        np.testing.assert_array_equal(saved.row_id.to_numpy(), results.row_id.to_numpy())
        np.testing.assert_array_equal(saved.actual.to_numpy(), actual)
        results[name] = saved.predicted.to_numpy(float)
    prediction_path, timing_path = OUT / "predictions.csv", OUT / "timing.json"
    if prediction_path.exists() and timing_path.exists():
        saved = pd.read_csv(prediction_path)
        np.testing.assert_array_equal(saved.row_id.to_numpy(), results.row_id.to_numpy())
        np.testing.assert_array_equal(saved.actual.to_numpy(), actual)
        predicted = saved.predicted.to_numpy(float)
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    else:
        write_json(OUT / "status.json", {"status": "fitting", "feature_count": len(x.columns)})
        retry = False
        try:
            predicted, timing = predict_all_context(x, y, query, keep_cache_on_device=True)
        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise
            retry = True
            write_json(OUT / "gpu_cache_retry.json", {"reason": str(exc)})
        if retry:
            gc.collect()
            predicted, timing = predict_all_context(x, y, query, keep_cache_on_device=False)
        pd.DataFrame({"row_id": results.row_id, "actual": actual, "predicted": predicted}).to_csv(
            prediction_path, index=False, float_format="%.8f")
        write_json(timing_path, timing)
    results[MODEL] = predicted
    names = (MODEL, *controls)
    if not np.isfinite(results[list(names)].to_numpy()).all():
        raise AssertionError("A model returned nonfinite predictions")
    scores = pd.DataFrame([{"model": name, "n": len(results), **metrics(actual, results[name])} for name in names])
    monthly = pd.DataFrame([
        {"model": name, "month": month, "n": len(part), **metrics(part.actual, part[name])}
        for month, part in results.groupby("month", sort=True) for name in names
    ])
    results.to_csv(OUT / "validation_predictions.csv", index=False, float_format="%.8f")
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    monthly.to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.8f")
    timings = [{"model": MODEL, **timing}]
    for name, directory in (("tabpfn_fast_original", REPORTS / "tabpfn_v3_5_fast_80_20"),
                            ("tabpfn_fast_temporal_full", SOURCE)):
        timings.append({"model": name, **json.loads((directory / "timing.json").read_text(encoding="utf-8"))})
    pd.DataFrame(timings).to_csv(OUT / "timings.csv", index=False)
    report = ["# TabPFN-3.5-Fast temporal feature ablation", "",
              "The same chronological 80/20 split uses January 2021-May 2024 as context (121,005 rows) "
              "and June 2024-March 2025 as validation (28,995 rows). Every context row is retained. "
              "All Fast models use one estimator, seed 42, the same checkpoint, and 200-row prediction batches.", "",
              "The levels/trend variant adds six predictors to the original 68: " + ", ".join(f"`{c}`" for c in ADDED) + ". "
              "Its causal forecast origins match the full temporal-feature experiment. All temporal values use "
              "only rent observations available at their own origin; no COE inputs are included. The learned "
              "forecast and its delta are removed to diagnose the eight-feature run's higher RMSE.", "",
              markdown_table(scores), "", "## Runtime", "", markdown_table(pd.DataFrame(timings)), "",
              "This is a development ablation selected after observing the full feature run. It is not an "
              "independent final test. The holdout covers ten months; the supplied unlabelled test covers seventeen. "
              "The CatBoost reference uses the original 68 features.", "",
              f"Enriched model-ready data: `{ARTIFACT.relative_to(ROOT).as_posix()}`. "
              "Added training and validation columns are exported separately as `temporal_features_train.csv` "
              "and `temporal_features_validation.csv`, with source row IDs and origin metadata.", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_fast_temporal_levels_80_20.py`."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "status.json", {"status": "complete", "feature_count": len(x.columns),
                                     "context_rows": len(x), "validation_rows": len(query)})
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
