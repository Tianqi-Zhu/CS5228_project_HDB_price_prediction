"""Score TabPFN-3 on the saved, point-in-time rental feature bundles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.config import REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.tabpfn_comparison import fit_tabpfn


PREPARED = REPORTS / "tabpfn_v3_features" / "manifest.json"
REFERENCE = REPORTS / "tabpfn_adaptation" / "validation_predictions.csv"
FOLDS = ("2023-01", "2023-07", "2024-04")


def score_fold(args, origin: str, manifest: dict, reference: pd.DataFrame):
    audit = next(row for row in manifest["splits"] if row["origin"] == origin)
    path = ROOT / audit["artifact"]
    if sha256(path) != audit["sha256"]:
        raise ValueError(f"Prepared feature hash mismatch: {path}")
    bundle = joblib.load(path)  # Trusted local artifact from prepare_tabpfn_v3_features.py.
    x_train, x_query = bundle["X_train"], bundle["X_query"]
    if list(x_train.columns) != manifest["feature_order"] or list(x_query.columns) != manifest["feature_order"]:
        raise AssertionError("Model input column order differs from the prepared manifest")
    if len(x_train) != audit["train_rows"] or len(x_query) != audit["query_rows"]:
        raise AssertionError("Prepared row counts differ from the manifest")
    if not all(isinstance(x_train[name].dtype, pd.CategoricalDtype)
               and x_train[name].dtype == x_query[name].dtype for name in audit["categorical_columns"]):
        raise AssertionError("Categorical columns or category levels changed")
    control = reference.loc[reference.validation_start.eq(origin)].set_index("row_id").loc[x_query.index]
    actual = np.asarray(bundle["y_query"], float)
    np.testing.assert_array_equal(actual, control["MONTHLY_RENT"].to_numpy(float))
    prediction, timing = fit_tabpfn(args, x_train, bundle["y_train"], x_query)
    result = pd.DataFrame({"row_id": x_query.index.to_numpy(), "validation_start": origin,
                           "month": bundle["query_month"], "actual": actual,
                           "tabpfn": prediction,
                           "catboost": control["catboost"].to_numpy(float),
                           "tabpfn_3_5_recent_1x10k": control["pfn_recent"].to_numpy(float)})
    return result, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", choices=("v3", "v3.5"), default="v3")
    parser.add_argument("--output", type=Path, default=REPORTS / "tabpfn_v3_prepared")
    parser.add_argument("--fold", choices=FOLDS, help="Run only one fold for a smoke test")
    parser.add_argument("--context-rows", type=int, default=10_000)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-mode", choices=("fit_preprocessors", "fit_with_cache"), default="fit_with_cache")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.context_rows, args.n_estimators, args.batch_size) < 1:
        parser.error("Context rows, estimators, and batch size must be positive")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(ROOT / ".cache" / "tabpfn"))
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    prepared = json.loads(PREPARED.read_text(encoding="utf-8"))
    if prepared["status"] != "complete":
        raise RuntimeError("Feature preparation is incomplete")
    prior = json.loads((REPORTS / "tabpfn_adaptation" / "settings.json").read_text(encoding="utf-8"))
    reference_sources = {key.replace("\\", "/"): value
                         for key, value in prior["source_sha256"].items()}
    if prepared["source_sha256"] != reference_sources:
        raise RuntimeError("Reference predictions and prepared features use different source files")
    settings = {"model_version": args.model_version, "context_rows": args.context_rows,
                "n_estimators": args.n_estimators, "batch_size": args.batch_size,
                "seed": args.seed, "fit_mode": args.fit_mode,
                "prepared_manifest_sha256": sha256(PREPARED),
                "runner_sha256": sha256(Path(__file__)),
                "fit_helper_sha256": sha256(ROOT / "analysis/experiments/tabpfn_comparison.py")}
    settings_path = out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != settings:
        raise ValueError("Output directory has different settings; choose a new output")
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    reference = pd.read_csv(REFERENCE)
    rows, timings = [], []
    for origin in FOLDS if args.fold is None else (args.fold,):
        result_path = out / f"predictions_{origin}.csv"
        if args.resume and result_path.exists():
            result = pd.read_csv(result_path)
            if len(result) != next(a["query_rows"] for a in prepared["splits"] if a["origin"] == origin):
                raise ValueError(f"Saved fold {origin} has the wrong number of rows")
        else:
            print(f"TabPFN-{args.model_version}, fold {origin}", flush=True)
            result, timing = score_fold(args, origin, prepared, reference)
            result.to_csv(result_path, index=False, float_format="%.8f")
            timings.append({"validation_start": origin, **timing})
            (out / f"timing_{origin}.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
        rows.append(result)
        score = metrics(result.actual, result.tabpfn)
        print(f"{origin}: RMSE {score['rmse_sgd']:.3f}; CatBoost "
              f"{metrics(result.actual, result.catboost)['rmse_sgd']:.3f}", flush=True)
    pooled = pd.concat(rows, ignore_index=True)
    fold_scores = pd.DataFrame([
        {"validation_start": origin, "model": model, "n": len(part), **metrics(part.actual, part[model])}
        for origin, part in pooled.groupby("validation_start", sort=True)
        for model in ("tabpfn", "tabpfn_3_5_recent_1x10k", "catboost")
    ])
    pooled_scores = pd.DataFrame([
        {"model": model, "n": len(pooled), **metrics(pooled.actual, pooled[model])}
        for model in ("tabpfn", "tabpfn_3_5_recent_1x10k", "catboost")
    ])
    fold_scores.to_csv(out / "fold_scores.csv", index=False, float_format="%.8f")
    pooled_scores.to_csv(out / "pooled_scores.csv", index=False, float_format="%.8f")
    (out / "status.json").write_text(json.dumps({"status": "complete" if len(rows) == len(FOLDS) else "partial",
                                                  "completed_folds": list(pooled.validation_start.unique()),
                                                  "validation_rows": len(pooled)}, indent=2), encoding="utf-8")
    print(pooled_scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
