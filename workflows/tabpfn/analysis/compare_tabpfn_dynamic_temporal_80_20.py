"""Evaluate composition-adjusted comprehensive temporal inputs with TabPFN Fast."""

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
from analysis.config import AUX, DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.aligned_temporal_features import FEATURES as PREVIOUS_TEMPORAL
from analysis.experiments.comprehensive_temporal import ComprehensiveTemporalModel, FEATURES
from analysis.experiments.flat_type_economic_forecast import coe_monthly
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET, spatial_frame


SOURCE = REPORTS / "tabpfn_v3_5_fast_aligned_temporal_80_20"
OUT = REPORTS / "tabpfn_v3_5_fast_dynamic_temporal_80_20"
ARTIFACT = ROOT / ".cache/tabpfn_v3_features/holdout_80_20_2024-06_dynamic_temporal.joblib"
ORIGIN = pd.Period("2024-06", "M")
MODEL = "tabpfn_dynamic_temporal_residual"


def prepare(source_audit):
    source = ROOT / source_audit["artifact"]
    if sha256(source) != source_audit["sha256"]:
        raise ValueError("Aligned source artifact changed")
    original = joblib.load(source)  # Trusted, hash-checked project artifact.
    frame = spatial_frame(pd.read_csv(DATA / "train.csv"))
    fitting = frame.loc[frame[MONTH] < ORIGIN]
    np.testing.assert_array_equal(fitting.index.to_numpy(), original["X_train"].index.to_numpy())
    np.testing.assert_array_equal(fitting[TARGET].to_numpy(float), original["y_train"])
    coe = coe_monthly(pd.read_csv(AUX / "sg-coe-prices.csv"))
    model = ComprehensiveTemporalModel(fitting, coe)
    matrices = {"train": original["X_train"].drop(columns=PREVIOUS_TEMPORAL),
                "query": original["X_query"].drop(columns=PREVIOUS_TEMPORAL)}
    assignments = {split: original[f"{split}_temporal_origins"] for split in matrices}
    baseline = {split: original[f"{split}_baseline"].copy() for split in matrices}
    for split, matrix in matrices.items():
        for name in FEATURES:
            matrix[name] = np.nan
        months = pd.PeriodIndex(original[f"{split}_month"], freq="M")
        matrix["DYN_TARGET_MONTH_SIN"] = np.sin(2 * np.pi * months.month / 12)
        matrix["DYN_TARGET_MONTH_COS"] = np.cos(2 * np.pi * months.month / 12)
    origins = sorted(set(assignments["train"].forecast_origin.dropna()).union(assignments["query"].forecast_origin.dropna()))
    for i, origin in enumerate(origins, 1):
        for split, matrix in matrices.items():
            ids = assignments[split].index[assignments[split].forecast_origin.eq(origin)]
            if not len(ids):
                continue
            positions = matrix.index.get_indexer(ids)
            query = frame.loc[ids].drop(columns=TARGET)
            values = model.features(query, origin, original[f"{split}_baseline"][positions])
            matrix.loc[ids, FEATURES] = values.reindex(ids).to_numpy(float)
            forecasts = values.DYN_BASELINE_FORECAST.to_numpy(float)
            baseline[split][positions] = np.where(np.isfinite(forecasts), forecasts, original[f"{split}_baseline"][positions])
        if i % 5 == 0 or i == len(origins):
            print(f"Prepared comprehensive temporal features: {i}/{len(origins)} origins (through {origin})", flush=True)
    for split, matrix in matrices.items():
        original_columns = [c for c in original[f"X_{split}"] if c not in PREVIOUS_TEMPORAL]
        pd.testing.assert_frame_equal(matrix[original_columns], original[f"X_{split}"][original_columns])
        if not np.isfinite(baseline[split]).all() or np.isinf(matrix.select_dtypes(include="number").to_numpy(float)).any():
            raise AssertionError("Baseline or input features contain infinities")
        export = assignments[split].join(matrix[FEATURES])
        export["residual_baseline_sgd"] = baseline[split]
        export.to_csv(OUT / f"temporal_features_{split}.csv", index=True, index_label="row_id", float_format="%.8f")
    bundle = {**original, "X_train": matrices["train"], "X_query": matrices["query"],
              "train_baseline": baseline["train"], "query_baseline": baseline["query"]}
    joblib.dump(bundle, ARTIFACT, compress=3)
    write_json(OUT / "temporal_model_audit.json", {
        "reference_end": str(model.reference_end), "reference_training_rows": int((fitting[MONTH] <= model.reference_end).sum()),
        "selection_guard": "correction must improve earlier rolling forecast RMSE by at least 1%; simpler mode preferred within 1%",
        "choices": [{"origin_index": k, **v} for k, v in model.choices.items()],
        "final_origin_choice": model.choices.get(ORIGIN.ordinal - 1 - model.first.ordinal),
    })
    manifest = {
        "artifact": ARTIFACT.relative_to(ROOT).as_posix(), "sha256": sha256(ARTIFACT),
        "source_aligned_artifact_sha256": source_audit["sha256"], "feature_count": len(matrices["train"].columns),
        "context_rows": len(matrices["train"]), "validation_rows": len(matrices["query"]),
        "temporal_features": FEATURES, "residual_training_target": "y_train - train_baseline",
        "rent_reconstruction": "predicted_residual + query_baseline",
        "training_missing_fractions": matrices["train"][FEATURES].isna().mean().to_dict(),
        "validation_missing_fractions": matrices["query"][FEATURES].isna().mean().to_dict(),
        "final_economic_enabled": model.choices[ORIGIN.ordinal - 1 - model.first.ordinal]["economic"],
    }
    write_json(OUT / "feature_manifest.json", manifest)
    # These diagnostics are computed only after feature fitting and selection finish.
    diagnostics = pd.DataFrame([
        {"baseline": name, **metrics(bundle["y_query"], values)}
        for name, values in (("previous_local", original["query_baseline"]), ("dynamic_composition_adjusted", baseline["query"]))
    ])
    diagnostics.to_csv(OUT / "baseline_diagnostics.csv", index=False, float_format="%.8f")
    return bundle, manifest


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    OUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((SOURCE / "settings.json").read_text(encoding="utf-8"))
    source_hashes = {name: sha256(ROOT / name) for name in prior["source_sha256"]}
    if source_hashes != prior["source_sha256"]:
        raise ValueError("Source data changed")
    source_hashes["data/auxiliary/sg-coe-prices.csv"] = sha256(AUX / "sg-coe-prices.csv")
    source_audit = json.loads((SOURCE / "feature_manifest.json").read_text(encoding="utf-8"))
    controls = {
        "previous_best_residual": SOURCE / "predictions_tabpfn_aligned_residual.csv",
        "previous_aligned_features": SOURCE / "predictions_tabpfn_aligned_features.csv",
        "original_fast": REPORTS / "tabpfn_v3_5_fast_80_20/predictions.csv",
    }
    code = [Path(__file__), ROOT / "analysis/experiments/comprehensive_temporal.py",
            ROOT / "analysis/experiments/aligned_temporal_features.py", ROOT / "analysis/experiments/flat_type_economic_forecast.py",
            ROOT / "analysis/compare_tabpfn_v3_5_fast_80_20.py", ROOT / "analysis/data.py"]
    settings = {
        "model_version": "v3.5-fast", "n_estimators": 1, "seed": 42, "context_rows": 121005,
        "subsampling": False, "batch_size": 200, "target": "residual around composition-adjusted dynamic baseline",
        "checkpoint_sha256": sha256(CHECKPOINT), "source_sha256": source_hashes,
        "source_aligned_artifact_sha256": source_audit["sha256"],
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code},
        "reference_predictions_sha256": {name: sha256(path) for name, path in controls.items()},
        "versions": {name: version(name) for name in ("tabpfn", "torch", "numpy", "pandas", "scikit-learn")},
        "gpu": torch.cuda.get_device_name(),
    }
    settings_path = OUT / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Existing output uses different settings")
    write_json(settings_path, settings)
    write_json(OUT / "status.json", {"status": "preparing"})
    if ARTIFACT.exists() and (OUT / "feature_manifest.json").exists():
        manifest = json.loads((OUT / "feature_manifest.json").read_text(encoding="utf-8"))
        if sha256(ARTIFACT) != manifest["sha256"]:
            raise ValueError("Comprehensive temporal artifact changed")
        bundle = joblib.load(ARTIFACT)
    else:
        bundle, manifest = prepare(source_audit)
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
    prediction_path, timing_path = OUT / "predictions.csv", OUT / "timing.json"
    if prediction_path.exists() and timing_path.exists():
        saved = pd.read_csv(prediction_path)
        np.testing.assert_array_equal(saved.row_id.to_numpy(), results.row_id.to_numpy())
        np.testing.assert_array_equal(saved.actual.to_numpy(), actual)
        predicted = saved.predicted.to_numpy(float)
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    else:
        target = y - bundle["train_baseline"]
        np.testing.assert_allclose(target + bundle["train_baseline"], y, atol=1e-10)
        retry = False
        write_json(OUT / "status.json", {"status": "fitting", "feature_count": len(x.columns)})
        try:
            residual, timing = predict_all_context(x, target, query, keep_cache_on_device=True)
        except Exception as exc:
            if "out of memory" not in str(exc).lower():
                raise
            retry = True
            write_json(OUT / "gpu_cache_retry.json", {"reason": str(exc)})
        if retry:
            gc.collect()
            residual, timing = predict_all_context(x, target, query, keep_cache_on_device=False)
        predicted = residual + bundle["query_baseline"]
        pd.DataFrame({"row_id": results.row_id, "actual": actual, "predicted": predicted}).to_csv(
            prediction_path, index=False, float_format="%.8f")
        write_json(timing_path, timing)
    results[MODEL] = predicted
    names = (MODEL, *controls)
    if not np.isfinite(results[list(names)].to_numpy()).all():
        raise AssertionError("Nonfinite rent predictions")
    scores = pd.DataFrame([{"model": name, "n": len(results), **metrics(actual, results[name])} for name in names])
    monthly = pd.DataFrame([{"model": name, "month": month, "n": len(part), **metrics(part.actual, part[name])}
                            for month, part in results.groupby("month", sort=True) for name in names])
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    monthly.to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.8f")
    results.to_csv(OUT / "validation_predictions.csv", index=False, float_format="%.8f")
    report = ["# Comprehensive temporal model with TabPFN-3.5-Fast", "",
              "The same chronological 80/20 split uses 121,005 context rows (January 2021-May 2024) "
              "and 28,995 validation rows (June 2024-March 2025). All context rows, aligned history columns, "
              "property inputs, and source labels are retained.", "",
              "A property reference fitted on the first twelve months removes changes in transaction mix from "
              "the monthly market index. It is used only at origins where that reference is already available. "
              "Earlier origins use the previous causal baseline. The market model pools planning-area and "
              "flat-type information and learns damped local trends with sparse-group shrinkage.", "",
              "Additional direct dynamic regressions consider 3/12-month levels and trends, annual changes, "
              "acceleration/regime shifts, annual Fourier seasonality, rental-market volatility, observation "
              "counts, activity momentum, common/type-specific movement, and delayed category-A COE momentum. "
              "Future COE values are never inputs. Regularization is scaled to distinct market months, all "
              "coefficients including the intercept are penalized, and corrections are conservatively shrunk. "
              "Each origin tests structural-only and COE variants against its simple baseline using earlier "
              "forecast origins; a new correction must improve error by at least 1% to be used.", "",
              f"Final-origin economic correction selected: `{manifest['final_economic_enabled']}` "
              "(`null` means no learned correction). Twenty temporal outputs replace the previous nine, "
              f"giving {len(x.columns)} total inputs. TabPFN predicts a residual around the resulting baseline "
              "and reconstructs rent by adding that baseline exactly once.", "",
              markdown_table(scores), "", "## Runtime", "", markdown_table(pd.DataFrame([timing])), "",
              "These are development results on a repeatedly examined ten-month holdout, not an untouched "
              "test. The supplied test has a seventeen-month horizon. The model covers observable temporal "
              "signals in this dataset; it cannot account for unobserved economic drivers or future shocks.", "",
              f"Model-ready data: `{ARTIFACT.relative_to(ROOT).as_posix()}`. Fit with `y_train - train_baseline` "
              "and reconstruct with `predicted_residual + query_baseline`. Component choices at every origin "
              "are recorded in `temporal_model_audit.json`.", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/compare_tabpfn_dynamic_temporal_80_20.py`."]
    (OUT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    best = scores.sort_values("rmse_sgd").iloc[0]
    write_json(OUT / "selection.json", {"criterion": "development holdout RMSE", "model": best.model,
                                        "rmse_sgd": float(best.rmse_sgd)})
    write_json(OUT / "status.json", {"status": "complete", "feature_count": len(x.columns)})
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
