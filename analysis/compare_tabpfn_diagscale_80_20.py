"""Chronological TabPFN-3.5-Fast DiagScale adaptation on the 80/20 rent split.

DiagScale follows Ning et al. (2026), adapted to the installed 3.5-Fast
attention scaling modules. The outer June 2024-March 2025 labels are used only
for final scoring; checkpoint selection uses a December 2023 origin.
"""

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

from analysis.compare_tabpfn_calendar_80_20 import frozen_settings
from analysis.compare_tabpfn_v3_80_20 import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.aligned_temporal_features import FEATURES, HISTORY_COLUMNS, LocalTemporalModel, origin_history
from analysis.experiments.calendar_encoding import MONTH_COLUMN, encoded_features, stratified_queries
from analysis.experiments.diagscale_tabpfn import DiagScale, FinetunedDiagScaleRegressor
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET, spatial_frame


OUT = REPORTS / "tabpfn_diagscale_80_20"
CACHE = ROOT / ".cache/tabpfn_diagscale_80_20"
BASE = ROOT / ".cache/tabpfn_v3_features/calendar_80_20_categorical.joblib"
REFERENCE = REPORTS / "tabpfn_v3_5_fast_calendar_80_20/predictions_outer_categorical.csv"
INNER_ORIGIN = pd.Period("2023-12", "M")
OUTER_ORIGIN = pd.Period("2024-06", "M")
MONTH_SAMPLE = 100
BATCH = 200


def inner_bundle(base, frame):
    path, manifest = CACHE / "inner_features.joblib", OUT / "inner_feature_manifest.json"
    if path.exists() and manifest.exists():
        if sha256(path) != json.loads(manifest.read_text(encoding="utf-8"))["sha256"]:
            raise ValueError("Saved inner feature bundle changed")
        return joblib.load(path)
    months = pd.PeriodIndex(base["train_month"], freq="M")
    context = np.flatnonzero(months < INNER_ORIGIN)
    possible = np.flatnonzero((months >= INNER_ORIGIN) & (months < OUTER_ORIGIN))
    sampled, weights = stratified_queries(months[possible], base["X_train"].index[possible],
                                           per_month=MONTH_SAMPLE, seed=42)
    selected = possible[sampled]
    x = base["X_train"].iloc[context].copy()
    q = base["X_train"].iloc[selected].copy()
    past = frame.loc[x.index]
    blind = frame.loc[q.index].drop(columns=TARGET)
    q.loc[:, HISTORY_COLUMNS] = origin_history(past, blind).reindex(q.index).to_numpy()
    q.loc[:, FEATURES] = LocalTemporalModel(past).features(blind, INNER_ORIGIN - 1).reindex(q.index).to_numpy()
    for column in x.select_dtypes(include="category"):
        if column == MONTH_COLUMN:
            continue
        known = pd.Index(x[column].dropna().astype(str).unique())
        for matrix in (x, q):
            values = matrix[column].astype("string")
            matrix[column] = pd.Categorical(values.where(values.isin(known)), categories=known)
    x = encoded_features(x, np.asarray(base["train_month"])[context], "categorical")
    q = encoded_features(q, np.asarray(base["train_month"])[selected], "categorical")
    y = np.asarray(base["y_train"], float)[context] - np.asarray(base["train_baseline"], float)[context]
    val_base = q.TEMP_LOCAL_FORECAST.to_numpy(float)
    val_actual = np.asarray(base["y_train"], float)[selected]
    if not np.isfinite(val_base).all() or months[context].max() >= INNER_ORIGIN:
        raise AssertionError("Calibration features are not causally available")
    if set(x.index) & set(q.index):
        raise AssertionError("Calibration labels entered context")
    bundle = {"X_fit": x, "y_fit": y, "X_cal": q,
              "y_cal_residual": val_actual - val_base, "y_cal_actual": val_actual,
              "cal_baseline": val_base, "cal_month": np.asarray(base["train_month"])[selected],
              "cal_weight": weights}
    joblib.dump(bundle, path, compress=3)
    write_json(manifest, {"artifact": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                          "fit_rows": len(x), "sampled_calibration_rows": len(q),
                          "last_fit_month": str(months[context].max()),
                          "calibration_months": [str(v) for v in months[selected].unique()],
                          "query_origin": str(INNER_ORIGIN - 1), "history_columns_rebuilt": len(HISTORY_COLUMNS),
                          "residual_baseline_finite": True, "context_query_disjoint": True})
    return bundle


def smoke_subset(inner):
    months = pd.PeriodIndex.from_ordinals(
        inner["X_fit"]["APPROVAL_MONTH_INDEX"].to_numpy(dtype=int) +
        pd.Period("2021-01", "M").ordinal, freq="M")
    ids = []
    rng = np.random.default_rng(42)
    for month in months.unique().sort_values():
        positions = np.flatnonzero(months == month)
        ids.extend(np.sort(rng.choice(positions, min(len(positions), 100), replace=False)))
    return np.asarray(sorted(ids), dtype=int)


def fit_adapter(inner, *, smoke):
    import torch
    from tabpfn.constants import ModelVersion

    if smoke:
        positions = smoke_subset(inner)
        x, y = inner["X_fit"].iloc[positions], inner["y_fit"][positions]
        q, z = inner["X_cal"].iloc[:100], inner["y_cal_residual"][:100]
        epochs, task_size = 1, 2048
    else:
        x, y = inner["X_fit"], inner["y_fit"]
        q, z = inner["X_cal"], inner["y_cal_residual"]
        epochs, task_size = 5, 4096
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = FinetunedDiagScaleRegressor(
        model_version=ModelVersion.V3_5_FAST, device="cuda", epochs=epochs,
        learning_rate=3e-2, weight_decay=.01, random_state=42,
        validation_split_ratio=0, early_stopping=not smoke, early_stopping_patience=2,
        n_finetune_ctx_plus_query_samples=task_size,
        finetune_ctx_query_split_ratio=.2, n_inference_subsample_samples=None,
        n_estimators_finetune=1, n_estimators_validation=1,
        n_estimators_final_inference=1, use_activation_checkpointing=True,
        ce_loss_weight=0, crps_loss_weight=0, mse_loss_weight=1,
        extra_regressor_kwargs={"model_path": str(CHECKPOINT),
                                "inference_config": {"SUBSAMPLE_SAMPLES": None},
                                "n_preprocessing_jobs": 1},
    )
    model.fit(x, y, X_val=q, y_val=z)
    architecture = model.finetuned_estimator_.model_
    trainable = [(name, p) for name, p in architecture.named_parameters() if p.requires_grad]
    if not trainable or any(not name.endswith(".delta") for name, _ in trainable):
        raise AssertionError("Backbone weights were unexpectedly trainable")
    adapters = [m for m in architecture.modules() if isinstance(m, DiagScale)]
    audit = {"fit_rows": len(x), "calibration_rows": len(q), "epochs_cap": epochs,
             "adapter_modules": len(adapters), "trainable_parameters": sum(p.numel() for _, p in trainable),
             "delta_l2": float(np.sqrt(sum(float(torch.sum(p.detach().float().square())) for _, p in trainable))),
             "fit_seconds": time.perf_counter() - started,
             "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
             "frozen_backbone": True, "finetune_task_size": task_size,
             "epoch_validation": model.diagscale_epochs_, "autocast_dtype": "bfloat16"}
    if smoke and audit["delta_l2"] <= 0:
        raise AssertionError("Smoke run did not update any DiagScale parameters")
    return model, audit


def outer_predictions(adapted, base):
    import torch
    from tabpfn.finetuning.train_util import clone_model_for_evaluation
    from tabpfn import TabPFNRegressor

    full_x, full_y = base["X_train"], np.asarray(base["y_train"], float) - np.asarray(base["train_baseline"], float)
    eval_model = clone_model_for_evaluation(
        adapted.finetuned_estimator_,
        {"device": "cuda", "n_estimators": 1, "random_state": 42,
         "fit_mode": "fit_with_cache", "keep_cache_on_device": True,
         "inference_config": {"SUBSAMPLE_SAMPLES": None}},
        TabPFNRegressor)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        eval_model.fit(full_x, full_y)
        if eval_model.n_train_samples_ != len(full_x) or eval_model.inference_config_.SUBSAMPLE_SAMPLES is not None:
            raise AssertionError("Adapted outer context was subsampled")
        print(f"Adapted outer model fitted on all {len(full_x):,} context rows", flush=True)
        q, baseline = base["X_query"], np.asarray(base["query_baseline"], float)
        partial = CACHE / "outer_predictions_partial.npy"
        completed = np.load(partial).tolist() if partial.exists() else []
        if len(completed) > len(q) or (len(completed) != len(q) and len(completed) % BATCH):
            raise ValueError("Invalid cached outer prediction prefix")
        results = [np.asarray(completed, float)]
        for offset in range(len(completed), len(q), BATCH):
            chunk = q.iloc[offset:offset + BATCH]
            predicted = np.asarray(eval_model.predict(chunk, output_type="mean"), float)
            results.append(predicted + baseline[offset:offset + len(chunk)])
            if offset % 4000 == 0 or offset + BATCH >= len(q):
                np.save(partial, np.concatenate(results))
                print(f"  Scored {min(offset + BATCH, len(q)):,}/{len(q):,}", flush=True)
        prediction = np.concatenate(results)
        if not np.isfinite(prediction).all():
            raise AssertionError("Adapted predictions are nonfinite")
        return prediction, {"outer_fit_predict_seconds": time.perf_counter() - started,
                            "outer_peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                            "context_rows": len(full_x), "validation_rows": len(q)}
    finally:
        del eval_model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    code = [Path(__file__), ROOT / "analysis/experiments/diagscale_tabpfn.py",
            ROOT / "analysis/experiments/aligned_temporal_features.py",
            ROOT / "analysis/experiments/calendar_encoding.py"]
    settings = {"model": "TabPFN-3.5-Fast adapted DiagScale", "backbone_sha256": sha256(CHECKPOINT),
                "base_bundle_sha256": sha256(BASE), "reference_sha256": sha256(REFERENCE),
                "train_csv_sha256": sha256(DATA / "train.csv"),
                "code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code},
                "paper_url": "https://arxiv.org/pdf/2610.12005",
                "reference_code_url": "https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models",
                "reference_commit": "45fb141b2acf15518e70885aa149c8e1898cd15c",
                "architecture_note": "paper tests TabPFN-3; this is an extension to installed TabPFN-3.5-Fast",
                "inner_origin": str(INNER_ORIGIN), "outer_origin": str(OUTER_ORIGIN),
                "inner_queries_per_month": MONTH_SAMPLE, "outer_batch_size": BATCH,
                "learning_rate": .03, "max_epochs": 5, "training_task_size": 4096,
                "training_query_fraction": .2, "validation_fallback": True,
                "n_estimators": 1, "month_encoding": "categorical", "seed": 42,
                "gpu": torch.cuda.get_device_name(),
                "versions": {p: version(p) for p in ("tabpfn", "torch", "numpy", "pandas", "scikit-learn")}}
    frozen_settings(OUT / ("smoke_settings.json" if args.smoke else "settings.json"), settings)
    base = joblib.load(BASE)
    reference = pd.read_csv(REFERENCE)
    np.testing.assert_array_equal(reference.row_id.to_numpy(), base["X_query"].index.to_numpy())
    np.testing.assert_array_equal(reference.actual.to_numpy(), base["y_query"])
    frame = spatial_frame(pd.read_csv(DATA / "train.csv"))
    inner = inner_bundle(base, frame)
    print(f"Inner fit {len(inner['X_fit']):,}; strict later calibration {len(inner['X_cal']):,}; outer 28,995", flush=True)
    adapted, audit = fit_adapter(inner, smoke=args.smoke)
    if args.smoke:
        write_json(OUT / "smoke.json", audit)
        print(audit, flush=True)
        return
    write_json(OUT / "training_audit.json", audit)
    from tabpfn.finetuning.train_util import clone_model_for_evaluation
    from tabpfn import TabPFNRegressor
    cal_model = clone_model_for_evaluation(
        adapted.finetuned_estimator_,
        {"device": "cuda", "n_estimators": 1, "random_state": 42,
         "fit_mode": "fit_with_cache", "keep_cache_on_device": True,
         "inference_config": {"SUBSAMPLE_SAMPLES": None}}, TabPFNRegressor)
    cal_model.fit(inner["X_fit"], inner["y_fit"])
    cal_predictions = np.concatenate([
        np.asarray(cal_model.predict(inner["X_cal"].iloc[i:i + BATCH], output_type="mean"), float)
        for i in range(0, len(inner["X_cal"]), BATCH)]) + inner["cal_baseline"]
    cal_rmse = float(np.sqrt(np.average((cal_predictions - inner["y_cal_actual"]) ** 2,
                                        weights=inner["cal_weight"])))
    del cal_model
    gc.collect()
    torch.cuda.empty_cache()
    write_json(OUT / "inner_selection.json", {"calibration_rmse_sgd": cal_rmse,
                                              "outer_labels_used": False,
                                              "model_selected_before_outer": True})
    if audit["delta_l2"] == 0:
        prediction = reference.predicted.to_numpy(float).copy()
        timing = {"outer_fit_predict_seconds": 0, "outer_peak_vram_gib": 0,
                  "context_rows": len(base["X_train"]), "validation_rows": len(base["X_query"]),
                  "frozen_fallback": True}
    else:
        prediction, timing = outer_predictions(adapted, base)
    result = pd.DataFrame({"row_id": reference.row_id, "month": base["query_month"],
                           "actual": reference.actual, "frozen": reference.predicted,
                           "diagscale": prediction})
    result.to_csv(OUT / "predictions.csv", index=False, float_format="%.8f")
    scores = pd.DataFrame([{"model": name, "n": len(result), **metrics(result.actual, result[name])}
                           for name in ("frozen", "diagscale")])
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    pd.DataFrame([{"month": month, "model": name, "n": len(part), **metrics(part.actual, part[name])}
                  for month, part in result.groupby("month", sort=True) for name in ("frozen", "diagscale")]
                 ).to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.8f")
    write_json(OUT / "timing.json", timing)
    frozen_rmse = float(scores.set_index("model").loc["frozen", "rmse_sgd"])
    adapted_rmse = float(scores.set_index("model").loc["diagscale", "rmse_sgd"])
    report = ["# DiagScale adaptation of TabPFN-3.5-Fast", "",
              "The exact categorical-month residual setup from the 80/20 experiment is used: 77 inputs, one estimator, 121,005 full outer context rows and 28,995 outer queries. All rental histories and temporal baselines have causal origins. The earlier December 2023 origin supplies a training-side checkpoint set; outer labels do not train or select the adapter.", "",
              f"Frozen RMSE: **S${frozen_rmse:.3f}**. Adapted RMSE: **S${adapted_rmse:.3f}** "
              f"(change {adapted_rmse - frozen_rmse:+.3f}). See `scores.csv` and `monthly_scores.csv`.", "",
              f"The optimizer targeted **{audit['trainable_parameters']:,}** zero-initialized diagonal parameters in {audit['adapter_modules']} attention scaling modules; all original weights were frozen. Selected adapter norm: {audit['delta_l2']:.4f}. The earlier calibration RMSE was S${cal_rmse:.3f}. Peak fine-tuning GPU allocation: {audit['peak_vram_gib']:.2f} GiB. BF16 autocast was needed because the installed default FP16 path generated NaN adapter gradients on this checkpoint.", "",
              "This ports the paper's query-key scaling mechanism to TabPFN-3.5-Fast. The paper's published results use TabPFN-3 on TabArena, so they are not a predicted gain for this checkpoint or rental task. The installed fine-tuning loop samples context/query tasks randomly from the pre-December training period; the checkpoint and final outer evaluation are strictly chronological. The outer holdout has already been examined in prior project experiments and remains a development comparison.", "",
              "Reference: [Ning et al., *Test-Time Compute for Tabular Foundation Models*](https://arxiv.org/pdf/2610.12005) and [released implementation](https://github.com/kanghui-learning/test-time-compute-for-tabular-foundation-models).", ""]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
