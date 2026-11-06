"""Add causal flat-type forecasts and rerun TabPFN-3.5-Fast's 80/20 holdout."""

from __future__ import annotations

import argparse
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
from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.flat_type_economic_forecast import rental_panel
from analysis.experiments.tabpfn_temporal_features import (
    CausalForecastBank, FEATURES, MAX_HORIZON, MIN_MODEL_MONTHS,
    attach_temporal_features, transaction_origins,
)


REFERENCE = REPORTS / "tabpfn_v3_5_fast_80_20"
BASE_PREPARATION = REPORTS / "tabpfn_v3_80_20/feature_manifest.json"
OUT = REPORTS / "tabpfn_v3_5_fast_temporal_80_20"
ARTIFACT = ROOT / ".cache/tabpfn_v3_features/holdout_80_20_2024-06_temporal.joblib"
ORIGIN = pd.Period("2024-06", "M")
MODEL = "tabpfn_fast_temporal"


def prepare(base_audit):
    source = ROOT / base_audit["artifact"]
    if sha256(source) != base_audit["sha256"]:
        raise ValueError("Base feature artifact changed")
    bundle = joblib.load(source)  # Trusted project-generated artifact.
    raw = pd.read_csv(DATA / "train.csv")
    fitting = raw.loc[raw.RENT_APPROVAL_DATE < str(ORIGIN)].copy()
    np.testing.assert_array_equal(bundle["X_train"].index.to_numpy(), fitting.index.to_numpy())
    np.testing.assert_array_equal(bundle["y_train"], fitting.MONTHLY_RENT.to_numpy(float))
    # Query labels are excluded from the temporal model's panel at construction.
    panel = rental_panel(fitting)
    training_origins = transaction_origins(bundle["train_month"], bundle["X_train"].index, panel.month.min())
    query_origins = transaction_origins(bundle["query_month"], bundle["X_query"].index, panel.month.min(),
                                        validation_origin=ORIGIN)
    origins = sorted(set(training_origins.forecast_origin.dropna()).union(query_origins.forecast_origin.dropna()))
    forecaster = CausalForecastBank(panel)
    frames, audits = [], []
    for i, origin in enumerate(origins, 1):
        frame, audit = forecaster.at_origin(origin)
        frames.append(frame)
        audits.append(audit)
        if i % 5 == 0 or i == len(origins):
            print(f"Prepared causal temporal forecasts: {i}/{len(origins)} origins (through {origin})", flush=True)
    bank = pd.concat(frames, ignore_index=True)
    x_train, train_export = attach_temporal_features(bundle["X_train"], training_origins, bank)
    x_query, query_export = attach_temporal_features(bundle["X_query"], query_origins, bank)
    if len(x_train) != 121_005 or len(x_query) != 28_995:
        raise AssertionError("The 80/20 context or validation row counts changed")
    if list(x_train.columns[-len(FEATURES):]) != FEATURES or list(x_train.columns) != list(x_query.columns):
        raise AssertionError("Temporal feature order differs across partitions")
    for assignments in (training_origins, query_origins):
        valid = assignments.dropna(subset="forecast_origin")
        if not (valid.forecast_origin < valid.month).all():
            raise AssertionError("A forecast origin includes its target month")
    bundle = {**bundle, "X_train": x_train, "X_query": x_query,
              "train_temporal_origins": training_origins, "query_temporal_origins": query_origins}
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACT, compress=3)
    train_export.to_csv(OUT / "temporal_features_train.csv", index=True, index_label="row_id", float_format="%.8f")
    query_export.to_csv(OUT / "temporal_features_validation.csv", index=True, index_label="row_id", float_format="%.8f")
    bank.to_csv(OUT / "forecast_bank.csv", index=False, float_format="%.8f")
    write_json(OUT / "temporal_model_audit.json", {"economic_features": False, "origins": audits})
    audit = {
        "artifact": ARTIFACT.relative_to(ROOT).as_posix(), "sha256": sha256(ARTIFACT),
        "base_artifact_sha256": base_audit["sha256"], "context_rows": len(x_train),
        "validation_rows": len(x_query), "base_feature_count": 68, "feature_count": len(x_train.columns),
        "added_features": FEATURES, "maximum_training_forecast_horizon": MAX_HORIZON,
        "minimum_model_history_months": MIN_MODEL_MONTHS,
        "training_forecast_missing_fraction": x_train[FEATURES].isna().mean().to_dict(),
        "validation_forecast_missing_fraction": x_query[FEATURES].isna().mean().to_dict(),
        "first_train_month": str(min(bundle["train_month"])), "last_train_month": str(max(bundle["train_month"])),
        "first_validation_month": str(min(bundle["query_month"])), "last_validation_month": str(max(bundle["query_month"])),
    }
    write_json(OUT / "feature_manifest.json", audit)
    return bundle, audit


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((REFERENCE / "settings.json").read_text(encoding="utf-8"))
    source_hashes = {name: sha256(ROOT / name) for name in prior["source_sha256"]}
    if source_hashes != prior["source_sha256"]:
        raise ValueError("Reference source data changed")
    base_audit = json.loads(BASE_PREPARATION.read_text(encoding="utf-8"))
    code_paths = [Path(__file__), ROOT / "analysis/experiments/tabpfn_temporal_features.py",
                  ROOT / "analysis/experiments/flat_type_economic_forecast.py",
                  ROOT / "analysis/data.py",
                  ROOT / "analysis/compare_tabpfn_v3_5_fast_80_20.py",
                  ROOT / "analysis/compare_tabpfn_v3_80_20.py"]
    controls = {
        "tabpfn_fast_original": REFERENCE / "predictions.csv",
        "catboost_original_68_features": REPORTS / "tabpfn_v3_80_20/predictions_catboost_matched.csv",
    }
    settings = {
        "model_version": "v3.5-fast", "n_estimators": 1, "seed": 42,
        "context_rows": 121_005, "subsampling": False, "batch_size": 200,
        "fit_mode": "fit_with_cache", "checkpoint_sha256": sha256(CHECKPOINT),
        "base_feature_artifact_sha256": base_audit["sha256"], "source_sha256": source_hashes,
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code_paths},
        "reference_predictions_sha256": {name: sha256(path) for name, path in controls.items()},
        "added_features": FEATURES, "economic_features": False,
        "training_origin_policy": "deterministic row-ID sampling of earlier origins, 1-17 month horizons; no target-based sampling",
        "temporal_tuning_policy": "each origin uses only its available labels for fitting and two earlier six-month tuning blocks",
        "versions": {name: version(name) for name in ("tabpfn", "torch", "pandas", "numpy", "scikit-learn")},
        "gpu": torch.cuda.get_device_name(),
    }
    settings_path = OUT / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing output uses different code, data, or settings")
    write_json(settings_path, settings)
    write_json(OUT / "status.json", {"status": "preparing_temporal_features"})
    if ARTIFACT.exists() and (OUT / "feature_manifest.json").exists():
        feature_audit = json.loads((OUT / "feature_manifest.json").read_text(encoding="utf-8"))
        if sha256(ARTIFACT) != feature_audit["sha256"]:
            raise ValueError("Enriched feature artifact changed")
        bundle = joblib.load(ARTIFACT)
    else:
        bundle, feature_audit = prepare(base_audit)
    if args.prepare_only:
        write_json(OUT / "status.json", {"status": "features_ready", "feature_count": len(bundle["X_train"].columns)})
        print(json.dumps(feature_audit, indent=2), flush=True)
        return
    x, y = bundle["X_train"], np.asarray(bundle["y_train"], float)
    query, actual = bundle["X_query"], np.asarray(bundle["y_query"], float)
    results = pd.DataFrame({"row_id": query.index.to_numpy(), "month": bundle["query_month"], "actual": actual})
    for name, path in controls.items():
        saved = pd.read_csv(path)
        np.testing.assert_array_equal(saved.row_id.to_numpy(), results.row_id.to_numpy())
        np.testing.assert_array_equal(saved.actual.to_numpy(), actual)
        results[name] = saved.predicted.to_numpy(float)
    path, timing_path = OUT / "predictions.csv", OUT / "timing.json"
    if path.exists() and timing_path.exists():
        saved = pd.read_csv(path)
        np.testing.assert_array_equal(saved.row_id.to_numpy(), results.row_id.to_numpy())
        np.testing.assert_array_equal(saved.actual.to_numpy(), actual)
        predicted = saved.predicted.to_numpy(float)
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    else:
        write_json(OUT / "status.json", {"status": "fitting", "context_rows": len(x), "feature_count": len(x.columns)})
        retry = False
        try:
            predicted, timing = predict_all_context(x, y, query, keep_cache_on_device=True)
        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise
            retry = True
            write_json(OUT / "gpu_cache_retry.json", {"reason": str(exc), "context_rows": len(x)})
        if retry:
            gc.collect()
            predicted, timing = predict_all_context(x, y, query, keep_cache_on_device=False)
        pd.DataFrame({"row_id": results.row_id, "actual": actual, "predicted": predicted}).to_csv(
            path, index=False, float_format="%.8f")
        write_json(timing_path, timing)
    results[MODEL] = predicted
    if not np.isfinite(results[[MODEL, *controls]].to_numpy(float)).all():
        raise AssertionError("Prediction output contains nonfinite values")
    names = (MODEL, *controls)
    scores = pd.DataFrame([{"model": name, "n": len(results), **metrics(actual, results[name])} for name in names])
    monthly = pd.DataFrame([
        {"model": name, "month": month, "n": len(part), **metrics(part.actual, part[name])}
        for month, part in results.groupby("month", sort=True) for name in names
    ])
    results.to_csv(OUT / "validation_predictions.csv", index=False, float_format="%.8f")
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    monthly.to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.8f")
    timings = [{"model": MODEL, **timing},
               {"model": "tabpfn_fast_original", **json.loads((REFERENCE / "timing.json").read_text(encoding="utf-8"))}]
    pd.DataFrame(timings).to_csv(OUT / "timings.csv", index=False)
    report = ["# TabPFN-3.5-Fast with temporal forecast features", "",
              "Context: January 2021-May 2024, 121,005 rows (80.67%). Validation: June 2024-March 2025, 28,995 rows. "
              "One estimator, seed 42, raw monthly rent target, 200-row prediction batches, all context rows retained.", "",
              "Eight numeric inputs are added to the original 68 features: " + ", ".join(f"`{name}`" for name in FEATURES) + ".", "",
              "Context transactions receive deterministic forecasts from earlier origins spanning 1-17 months. "
              "Each temporal model and its regularization are fit/selected using only outcomes available at its own origin. "
              "The learned forecaster requires 18 history months; earlier rows retain missing learned forecasts. "
              "Simple levels use whatever strictly earlier rental history is available. Validation forecasts all originate "
              "at May 2024. COE is excluded following the standalone economic-model ablation.", "",
              "The original 68 inputs retain their causal definitions, including preceding-month histories on context rows. "
              "The new forecast columns carry their separate origin and horizon. They are predictor inputs; the model "
              "still predicts raw rent, so no forecast is added to its output a second time.", "",
              markdown_table(scores), "", "## Runtime", "", markdown_table(pd.DataFrame(timings)), "",
              "The CatBoost reference uses the original 68 features; the direct feature ablation is against "
              "`tabpfn_fast_original` on exactly the same validation rows. Baseline predictions are hash-checked. "
              "All scores are development results on a ten-month horizon, shorter than the unlabelled test's 17 months.", "",
              f"Model-ready enriched data: `{ARTIFACT.relative_to(ROOT).as_posix()}`. "
              "The `temporal_features_train.csv` and `temporal_features_validation.csv` exports contain source row IDs "
              "and only the added features plus origin metadata. Original rental labels remain separate in the joblib bundle.", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_fast_temporal_80_20.py`."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "status.json", {"status": "complete", "context_rows": len(x), "validation_rows": len(query),
                                     "feature_count": len(x.columns)})
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
