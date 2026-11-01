"""Tune DiagScale only on earlier chronological rental months, then score the outer holdout once.

Each candidate is a separate invocation. The later June 2024--March 2025 labels
never choose hyperparameters or checkpoints. Reuses the verified aligned
categorical-month feature bundle from compare_tabpfn_diagscale_80_20.py.
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
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from tabpfn.finetuning.train_util import clone_model_for_evaluation

from analysis.compare_tabpfn_diagscale_80_20 import BASE, CACHE as INNER_CACHE, REFERENCE
from analysis.compare_tabpfn_v3_80_20 import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.diagscale_tabpfn import DiagScale, DiagScaleTabPFNRegressor, FinetunedDiagScaleRegressor


OUT = REPORTS / "tabpfn_diagscale_tuning_80_20"
CACHE = ROOT / ".cache/tabpfn_diagscale_tuning_80_20"
BATCH = 200
CANDIDATES = {
    # Change loss first, then views, then learning rate. All use the same
    # chronological calibration rows and a longer checkpoint-search window.
    "loss_crps_mse_1view_lr03": {"lr": .03, "train_views": 1, "val_views": 1, "task_size": 2048},
    "loss_crps_mse_1view_lr01": {"lr": .01, "train_views": 1, "val_views": 1, "task_size": 2048},
    "loss_crps_mse_2view_lr03": {"lr": .03, "train_views": 2, "val_views": 2, "task_size": 1024},
    "loss_crps_mse_2view_lr01": {"lr": .01, "train_views": 2, "val_views": 2, "task_size": 1024},
}


def inner_bundle():
    artifact = INNER_CACHE / "inner_features.joblib"
    manifest = json.loads((REPORTS / "tabpfn_diagscale_80_20/inner_feature_manifest.json").read_text())
    if sha256(artifact) != manifest["sha256"]:
        raise ValueError("Inner feature cache does not match its saved manifest")
    return joblib.load(artifact)


def inference_estimator(finetuned, x, y):
    estimator = clone_model_for_evaluation(
        finetuned.finetuned_estimator_,
        {"device": "cuda", "n_estimators": 1, "auto_scale_n_estimators": False,
         "random_state": 42, "fit_mode": "fit_with_cache", "keep_cache_on_device": True,
         "inference_config": {"SUBSAMPLE_SAMPLES": None}}, TabPFNRegressor)
    estimator.fit(x, y)
    if estimator.n_train_samples_ != len(x) or estimator.inference_config_.SUBSAMPLE_SAMPLES is not None:
        raise AssertionError("Calibration context was subsampled")
    return estimator


def predict_batches(estimator, x, baseline, *, progress_label):
    values = []
    for offset in range(0, len(x), BATCH):
        batch = x.iloc[offset:offset + BATCH]
        residual = np.asarray(estimator.predict(batch, output_type="mean"), float)
        values.append(residual + baseline[offset:offset + len(batch)])
        if len(x) > 1000 and (offset % 4000 == 0 or offset + BATCH >= len(x)):
            print(f"  {progress_label} {min(offset + BATCH, len(x)):,}/{len(x):,}", flush=True)
    result = np.concatenate(values)
    if len(result) != len(x) or not np.isfinite(result).all():
        raise AssertionError("Incomplete/nonfinite predictions")
    return result


def weighted_rmse(actual, prediction, weights):
    return float(np.sqrt(np.average((np.asarray(actual) - prediction) ** 2, weights=weights)))


def fit_candidate(name):
    cfg = CANDIDATES[name]
    candidate_dir = OUT / name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    destination = candidate_dir / "calibration.json"
    if destination.exists():
        print(f"Existing completed candidate: {name}", flush=True)
        return
    inner = inner_bundle()
    x, y = inner["X_fit"], inner["y_fit"]
    q, z = inner["X_cal"], inner["y_cal_residual"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = FinetunedDiagScaleRegressor(
        model_version=ModelVersion.V3_5_FAST, device="cuda", epochs=8,
        learning_rate=cfg["lr"], weight_decay=.01, random_state=42,
        validation_split_ratio=0, early_stopping=True, early_stopping_patience=3,
        n_finetune_ctx_plus_query_samples=cfg["task_size"], finetune_ctx_query_split_ratio=.2,
        n_inference_subsample_samples=None,
        n_estimators_finetune=cfg["train_views"],
        n_estimators_validation=cfg["val_views"],
        n_estimators_final_inference=1, use_activation_checkpointing=True,
        ce_loss_weight=0, crps_loss_weight=1, mse_loss_weight=1,
        extra_regressor_kwargs={"model_path": str(CHECKPOINT),
                                "auto_scale_n_estimators": False,
                                "inference_config": {"SUBSAMPLE_SAMPLES": None},
                                "n_preprocessing_jobs": 1},
    )
    print(f"Training {name}: {len(x):,} fit; {len(q):,} chronological calibration", flush=True)
    model.fit(x, y, X_val=q, y_val=z)
    train_seconds = time.perf_counter() - started
    architecture = model.finetuned_estimator_.model_
    delta = {key: p.detach().float().cpu().clone()
             for key, p in architecture.named_parameters() if key.endswith(".delta")}
    trainable = {key: p for key, p in architecture.named_parameters() if p.requires_grad}
    if not trainable or set(trainable) - set(delta):
        raise AssertionError("Backbone parameters unexpectedly trainable")
    if not all(torch.isfinite(value).all() for value in delta.values()):
        raise AssertionError("Nonfinite DiagScale values")
    delta_norm = float(np.sqrt(sum(float(value.square().sum()) for value in delta.values())))
    adapter = CACHE / f"{name}_adapter.pt"
    torch.save(delta, adapter)
    evaluator = inference_estimator(model, x, y)
    predicted = predict_batches(evaluator, q, inner["cal_baseline"], progress_label="calibration")
    score = weighted_rmse(inner["y_cal_actual"], predicted, inner["cal_weight"])
    result = {"name": name, "config": cfg, "fit_rows": len(x), "calibration_rows": len(q),
              "calibration_rmse_sgd_weighted": score,
              "calibration_rmse_sgd_unweighted": float(np.sqrt(np.mean((inner["y_cal_actual"] - predicted) ** 2))),
              "delta_l2": delta_norm, "adapter_modules": sum(isinstance(m, DiagScale) for m in architecture.modules()),
              "trainable_parameters": sum(p.numel() for p in trainable.values()),
              "epoch_validation": model.diagscale_epochs_, "epochs_cap": 8, "patience": 3,
              "training_task_size": cfg["task_size"],
              "loss": {"crps": 1, "mse": 1, "ce": 0}, "fit_seconds": train_seconds,
              "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
              "adapter_sha256": sha256(adapter), "inner_feature_sha256": sha256(INNER_CACHE / "inner_features.joblib"),
              "code_sha256": sha256(Path(__file__)), "backbone_sha256": sha256(CHECKPOINT),
              "outer_labels_used": False}
    write_json(destination, result)
    pd.DataFrame({"row_id": q.index, "month": inner["cal_month"],
                  "actual": inner["y_cal_actual"], "predicted": predicted,
                  "weight": inner["cal_weight"]}).to_csv(candidate_dir / "calibration_predictions.csv", index=False)
    print(json.dumps({"name": name, "calibration_rmse": score, "delta_l2": delta_norm,
                      "fit_seconds": train_seconds}), flush=True)


class SavedDiagScaleRegressor(DiagScaleTabPFNRegressor):
    """Reload a selected tiny adapter immediately after every backbone load."""

    def __init__(self, *, adapter_path: str, **kwargs):
        self.adapter_path = adapter_path
        super().__init__(**kwargs)

    def _initialize_model_variables(self):
        size = super()._initialize_model_variables()
        delta = torch.load(self.adapter_path, map_location="cpu", weights_only=True)
        for model in self.models_:
            named = dict(model.named_parameters())
            if set(delta) - set(named):
                raise AssertionError("Adapter parameter names do not match backbone")
            with torch.no_grad():
                for key, value in delta.items():
                    named[key].copy_(value.to(device=named[key].device, dtype=named[key].dtype))
        return size


def score_best():
    scores, unavailable = [], []
    for name in CANDIDATES:
        path = OUT / name / "calibration.json"
        if not path.exists():
            unavailable.append(name)
        else:
            scores.append(json.loads(path.read_text(encoding="utf-8")))
    if not scores:
        raise FileNotFoundError("No completed calibration candidates")
    selected = min(scores, key=lambda row: row["calibration_rmse_sgd_weighted"])
    name = selected["name"]
    adapter = CACHE / f"{name}_adapter.pt"
    if sha256(adapter) != selected["adapter_sha256"]:
        raise ValueError("Selected adapter differs from its calibration artifact")
    write_json(OUT / "selection.json", {"selected": name, "metric": "weighted earlier-month RMSE",
                                         "candidate_scores": {row["name"]: row["calibration_rmse_sgd_weighted"]
                                                              for row in scores}, "unavailable_candidates": unavailable,
                                         "outer_labels_used": False})
    base = joblib.load(BASE)
    reference = pd.read_csv(REFERENCE)
    np.testing.assert_array_equal(reference.row_id.to_numpy(), base["X_query"].index.to_numpy())
    y = np.asarray(base["y_train"], float) - np.asarray(base["train_baseline"], float)
    estimator = SavedDiagScaleRegressor.create_default_for_version(
        version=ModelVersion.V3_5_FAST, adapter_path=str(adapter), model_path=str(CHECKPOINT),
        device="cuda", n_estimators=1, auto_scale_n_estimators=False, random_state=42,
        fit_mode="fit_with_cache", keep_cache_on_device=True,
        inference_config={"SUBSAMPLE_SAMPLES": None}, n_preprocessing_jobs=1)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    estimator.fit(base["X_train"], y)
    if estimator.n_train_samples_ != len(base["X_train"]):
        raise AssertionError("Outer context was truncated")
    print(f"Scoring selected {name} on full outer context", flush=True)
    predicted = predict_batches(estimator, base["X_query"], np.asarray(base["query_baseline"], float),
                                progress_label="outer")
    result = pd.DataFrame({"row_id": reference.row_id, "month": base["query_month"],
                           "actual": reference.actual, "frozen": reference.predicted,
                           "diagscale_tuned": predicted})
    result.to_csv(OUT / "predictions.csv", index=False, float_format="%.8f")
    pooled = pd.DataFrame([{"model": label, "n": len(result), **metrics(result.actual, result[label])}
                           for label in ("frozen", "diagscale_tuned")])
    pooled.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    pd.DataFrame([{"month": month, "model": label, "n": len(part), **metrics(part.actual, part[label])}
                  for month, part in result.groupby("month", sort=True)
                  for label in ("frozen", "diagscale_tuned")]).to_csv(OUT / "monthly_scores.csv", index=False)
    write_json(OUT / "timing.json", {"fit_predict_seconds": time.perf_counter() - started,
                                     "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                                     "outer_context_rows": len(base["X_train"]),
                                     "outer_validation_rows": len(base["X_query"])})
    print(pooled.to_string(index=False), flush=True)


def main():
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES)
    parser.add_argument("--score-best", action="store_true")
    args = parser.parse_args()
    if bool(args.candidate) == args.score_best:
        parser.error("Choose exactly one of --candidate and --score-best")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    if args.candidate:
        fit_candidate(args.candidate)
    else:
        score_best()


if __name__ == "__main__":
    main()
