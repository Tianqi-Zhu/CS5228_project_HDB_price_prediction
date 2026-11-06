"""Matched frozen TabPFN test of the zch anniversary-history feature group.

Run ``--stage prepare`` on CPU, then ``--stage run`` when GPU memory is free.
The original 77-column model is rerun with identical inference settings. Both
candidate views retain its rows, residual target, temporal baseline and labels.
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

from analysis.config import DATA, REPORTS, ROOT
from analysis.data import normalized_address, prepare_main, sha256
from analysis.evaluation import metrics
from analysis.experiments.tabpfn_anniversary_history import AnniversaryHistory, FEATURES
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT


OUT = REPORTS / "tabpfn_anniversary_history_80_20"
CACHE = ROOT / ".cache/tabpfn_anniversary_history_80_20"
BASE = ROOT / ".cache/tabpfn_v3_features/calendar_80_20_categorical.joblib"
BASE_MANIFEST = REPORTS / "tabpfn_v3_5_fast_calendar_80_20/feature_manifest_categorical.json"
NAMES = ("baseline77", "anniversary87", "anniversary92")
EXPECTED_ROWS = (121005, 28995)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_bundle() -> dict:
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    if sha256(BASE) != manifest["sha256"] or not CHECKPOINT.exists():
        raise ValueError("Original TabPFN bundle or checkpoint is missing/changed")
    bundle = joblib.load(BASE)
    if (len(bundle["X_train"]), len(bundle["X_query"])) != EXPECTED_ROWS:
        raise ValueError("Unexpected chronological split")
    if len(bundle["X_train"].columns) != 77 or list(bundle["X_train"]) != list(bundle["X_query"]):
        raise ValueError("Expected the original, ordered 77-column matrices")
    if pd.PeriodIndex(bundle["train_month"], freq="M").max() >= pd.Period("2024-06", "M"):
        raise ValueError("Validation labels entered the fitting context")
    return bundle


def feature_path(name: str) -> Path:
    return CACHE / f"{name}.joblib"


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    bundle = source_bundle()
    raw = normalized_address(prepare_main(pd.read_csv(DATA / "train.csv", dtype={"BLOCK": "string"})))
    fitting = raw.loc[bundle["X_train"].index]
    query = raw.loc[bundle["X_query"].index]
    if not (fitting["RENT_APPROVAL_DATE"].max() < query["RENT_APPROVAL_DATE"].min()):
        raise ValueError("Source/query months overlap")
    np.testing.assert_array_equal(fitting["MONTHLY_RENT"], bundle["y_train"])
    np.testing.assert_array_equal(query["MONTHLY_RENT"], bundle["y_query"])
    encoder = AnniversaryHistory().fit(fitting)
    fit_added = encoder.transform(fitting)
    query_added = encoder.transform(query.drop(columns="MONTHLY_RENT"))
    if list(fit_added) != FEATURES or list(query_added) != FEATURES:
        raise AssertionError("Anniversary feature order changed")
    if (query_added.filter(like="_N") < 0).any().any():
        raise AssertionError("Negative cohort support")
    coverage = []
    for lag in (12, 24, 36):
        column = f"ANN_BLOCK_{lag}_R1_N"
        coverage.append({"lag_months": lag, "fitting_support_share": float(fit_added[column].gt(0).mean()),
                         "query_support_share": float(query_added[column].gt(0).mean())})
    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
    for name, columns in (("anniversary87", FEATURES[:10]), ("anniversary92", FEATURES)):
        path = feature_path(name)
        updated = {
            "X_train": pd.concat([bundle["X_train"], fit_added[columns]], axis=1),
            "X_query": pd.concat([bundle["X_query"], query_added[columns]], axis=1),
        }
        if len(updated["X_train"].columns) != int(name[-2:]):
            raise AssertionError("Unexpected feature count")
        if not updated["X_train"].iloc[:, :77].equals(bundle["X_train"]):
            raise AssertionError("Original features changed")
        if not updated["X_query"].iloc[:, :77].equals(bundle["X_query"]):
            raise AssertionError("Original query features changed")
        temporary = path.with_suffix(".part.joblib")
        joblib.dump(updated, temporary, compress=3)
        temporary.replace(path)
        write_json(OUT / f"features_{name}.json", {
            "artifact": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
            "added_columns": columns, "features": len(updated["X_train"].columns),
            "fitting_rows": len(updated["X_train"]), "query_rows": len(updated["X_query"]),
            "original_bundle_sha256": sha256(BASE), "rent_target_and_baseline_unchanged": True,
            "feature_pool_last_month": str(fitting["RENT_APPROVAL_DATE"].max()),
        })
        print(f"Prepared {name}: {len(columns)} anniversary fields", flush=True)


def matrices(name: str, base: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if name == "baseline77":
        return base["X_train"], base["X_query"]
    manifest = json.loads((OUT / f"features_{name}.json").read_text(encoding="utf-8"))
    path = feature_path(name)
    if sha256(path) != manifest["sha256"] or manifest["original_bundle_sha256"] != sha256(BASE):
        raise ValueError("Prepared feature artifact changed")
    saved = joblib.load(path)
    x, q = saved["X_train"], saved["X_query"]
    if not x.index.equals(base["X_train"].index) or not q.index.equals(base["X_query"].index):
        raise AssertionError("Candidate row IDs differ from the control")
    return x, q


def infer(name: str, base: dict) -> None:
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    output = OUT / f"predictions_{name}.csv"
    timing_path = OUT / f"timing_{name}.json"
    if output.exists() and timing_path.exists():
        if sha256(output) != json.loads(timing_path.read_text(encoding="utf-8"))["prediction_sha256"]:
            raise ValueError("Existing prediction changed")
        print(f"Reusing verified {name}", flush=True)
        return
    x, q = matrices(name, base)
    if name == "baseline77":
        original = pd.read_csv(REPORTS / "tabpfn_feature_diversified_ensemble_80_20/predictions_2024-06_baseline.csv")
        np.testing.assert_array_equal(original["row_id"], q.index)
        np.testing.assert_array_equal(original["actual"], base["y_query"])
    torch.set_num_threads(1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    categorical = [i for i, c in enumerate(x) if isinstance(x[c].dtype, pd.CategoricalDtype)]
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT), device="cuda", n_estimators=2,
        random_state=42, fit_mode="fit_with_cache", keep_cache_on_device=True,
        categorical_features_indices=categorical, inference_config={"SUBSAMPLE_SAMPLES": None},
        n_preprocessing_jobs=1,
    )
    partial = CACHE / f"partial_{name}.npy"
    values = np.load(partial) if partial.exists() else np.full(len(q), np.nan)
    if values.shape != (len(q),):
        raise ValueError("Partial prediction has a different shape")
    started = time.perf_counter()
    try:
        target = np.asarray(base["y_train"], float) - np.asarray(base["train_baseline"], float)
        model.fit(x, target)
        if model.n_train_samples_ != len(x) or model.n_estimators_ != 2 or model.inference_config_.SUBSAMPLE_SAMPLES is not None:
            raise AssertionError("TabPFN changed context or estimator count")
        fitted = time.perf_counter()
        if name == "baseline77":
            replay = np.asarray(model.predict(q.iloc[:200], output_type="mean"), float) + base["query_baseline"][:200]
            np.testing.assert_allclose(replay, original["predicted"].to_numpy(float)[:200], atol=1e-5, rtol=0)
        for start in range(0, len(q), 200):
            stop = min(start + 200, len(q))
            if np.isfinite(values[start:stop]).all():
                continue
            torch.cuda.empty_cache()
            values[start:stop] = (
                np.asarray(model.predict(q.iloc[start:stop], output_type="mean"), float)
                + np.asarray(base["query_baseline"])[start:stop]
            )
            np.save(partial, values)
            if start % 2000 == 0 or stop == len(q):
                print(f"{name}: {stop:,}/{len(q):,}", flush=True)
        if not np.isfinite(values).all():
            raise AssertionError("Incomplete predictions")
        prediction = pd.DataFrame({"row_id": q.index, "month": base["query_month"],
                                   "actual": base["y_query"], "predicted": values})
        prediction.to_csv(output, index=False, float_format="%.17g")
        write_json(timing_path, {"name": name, "features": len(x.columns),
                                 "context_rows": len(x), "query_rows": len(q), "n_estimators": 2,
                                 "checkpoint_sha256": sha256(CHECKPOINT), "original_bundle_sha256": sha256(BASE),
                                 "fit_seconds": fitted - started, "predict_seconds": time.perf_counter() - fitted,
                                 "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
                                 "prediction_sha256": sha256(output), "subsampling": False})
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def summarize() -> None:
    base = source_bundle()
    records = []
    for name in NAMES:
        path = OUT / f"predictions_{name}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        np.testing.assert_array_equal(frame["row_id"], base["X_query"].index)
        np.testing.assert_array_equal(frame["actual"], base["y_query"])
        records.append({"model": name, "features": int(name[-2:]), "rows": len(frame),
                        **metrics(frame["actual"], frame["predicted"])})
    if not records:
        return
    pd.DataFrame(records).to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    lines = ["# Matched TabPFN anniversary-history ablation", "",
             "Frozen TabPFN 3.5 Fast; two estimator views; 121,005 fitting rows and 28,995 June 2024–March 2025 query rows.",
             "The original 77 inputs, labels, residual target, temporal baseline, checkpoint, seed and inference budget are fixed.",
             "The candidates add 12/24-month (87 columns) or 12/24/36-month (92 columns) same-block/flat-type cohorts.",
             "Rent histories use fitting labels only and remain frozen at the May 2024 prediction origin.", "",
             "| Model | Features | RMSE (SGD) | MAE (SGD) | Bias (SGD) |",
             "|---|---:|---:|---:|---:|"]
    for row in records:
        lines.append(f"| {row['model']} | {row['features']} | {row['rmse_sgd']:.4f} | {row['mae_sgd']:.4f} | {row['bias_sgd']:+.4f} |")
    lines.append("")
    control = next((row for row in records if row["model"] == "baseline77"), None)
    if control is not None:
        for row in records:
            if row["model"] != "baseline77":
                lines.append(f"{row['model']} changes RMSE by {row['rmse_sgd'] - control['rmse_sgd']:+.4f} SGD versus the control.")
    lines += ["", "The holdout was previously examined in related experiments; these are development comparisons.",
              "Confirm the gain on other chronological folds before choosing a final configuration.",
              "Features and predictions are saved with source hashes for replay. No low-rent row filtering was applied.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "summarize"), required=True)
    args = parser.parse_args()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.pop("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", None)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "run":
        base = source_bundle()
        for name in NAMES:
            infer(name, base)
            summarize()
    else:
        summarize()


if __name__ == "__main__":
    main()
