"""Check DiagScale fine-tuning with explicit categorical indices after pandas-to-NumPy conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import torch
from tabpfn.constants import ModelVersion

from analysis.compare_tabpfn_diagscale_80_20 import BASE, REFERENCE
from analysis.compare_tabpfn_v3_80_20 import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.diagscale_tabpfn import DiagScale, FinetunedDiagScaleRegressor
from analysis.tune_tabpfn_diagscale_80_20 import (
    SavedDiagScaleRegressor, inference_estimator, inner_bundle,
    predict_batches, weighted_rmse,
)


OUT = REPORTS / "tabpfn_diagscale_categorical_alignment_80_20"
CACHE = ROOT / ".cache/tabpfn_diagscale_categorical_alignment_80_20"
PRIOR = REPORTS / "tabpfn_diagscale_tuning_80_20"


def category_positions(frame):
    return [i for i, name in enumerate(frame.columns) if isinstance(frame[name].dtype, pd.CategoricalDtype)]


def train_and_score(learning_rate: float = .01, *, views: int = 2, task_size: int = 1024):
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if views < 1 or task_size < 2:
        raise ValueError("views and task size must be positive")
    lr_suffix = "" if learning_rate == .01 else f"_lr{learning_rate:g}".replace(".", "")
    view_suffix = "" if views == 2 else f"_{views}views"
    task_suffix = "" if task_size == 1024 else f"_task{task_size}"
    suffix = lr_suffix + view_suffix + task_suffix
    out = OUT if not suffix else REPORTS / f"tabpfn_diagscale_categorical_alignment{suffix}_80_20"
    cache = CACHE if not suffix else ROOT / f".cache/tabpfn_diagscale_categorical_alignment{suffix}_80_20"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    inner = inner_bundle()
    x, y, q, z = inner["X_fit"], inner["y_fit"], inner["X_cal"], inner["y_cal_residual"]
    cat = category_positions(x)
    if cat != category_positions(q):
        raise AssertionError("Categorical columns differ between context and calibration")
    if not cat:
        raise AssertionError("No categorical columns to align")
    started = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = FinetunedDiagScaleRegressor(
        model_version=ModelVersion.V3_5_FAST, device="cuda", epochs=8,
        learning_rate=learning_rate, weight_decay=.01, random_state=42,
        validation_split_ratio=0, early_stopping=True, early_stopping_patience=3,
        n_finetune_ctx_plus_query_samples=task_size, finetune_ctx_query_split_ratio=.2,
        n_inference_subsample_samples=None,
        n_estimators_finetune=views, n_estimators_validation=views,
        n_estimators_final_inference=1, use_activation_checkpointing=True,
        ce_loss_weight=0, crps_loss_weight=1, mse_loss_weight=1,
        extra_regressor_kwargs={"model_path": str(CHECKPOINT),
                                "categorical_features_indices": cat,
                                "auto_scale_n_estimators": False,
                                "inference_config": {"SUBSAMPLE_SAMPLES": None},
                                "n_preprocessing_jobs": 1},
    )
    print(f"Aligned categories: {cat}; training {len(x):,} context rows; "
          f"{views} training/checkpoint views; task size {task_size}", flush=True)
    model.fit(x, y, X_val=q, y_val=z)
    training_seconds = time.perf_counter() - started
    architecture = model.finetuned_estimator_.model_
    delta = {key: p.detach().float().cpu().clone()
             for key, p in architecture.named_parameters() if key.endswith(".delta")}
    trainable = {key: p for key, p in architecture.named_parameters() if p.requires_grad}
    if not trainable or set(trainable) - set(delta):
        raise AssertionError("Backbone weights were unexpectedly trainable")
    adapter = cache / "adapter.pt"
    torch.save(delta, adapter)
    estimator = inference_estimator(model, x, y)
    predicted = predict_batches(estimator, q, inner["cal_baseline"], progress_label="calibration")
    score = weighted_rmse(inner["y_cal_actual"], predicted, inner["cal_weight"])
    frozen = json.loads((PRIOR / "frozen_calibration.json").read_text(encoding="utf-8"))
    prior = json.loads((PRIOR / "loss_crps_mse_2view_lr01/calibration.json").read_text(encoding="utf-8"))
    audit = {"categorical_indices": cat, "categorical_names": [x.columns[i] for i in cat],
             "fit_rows": len(x), "calibration_rows": len(q), "train_views": views,
             "checkpoint_views": views, "deployment_views": 1, "training_task_size": task_size,
             "learning_rate": learning_rate, "epochs_cap": 8, "patience": 3,
             "crps_weight": 1, "mse_weight": 1,
             "epoch_validation": model.diagscale_epochs_,
             "calibration_rmse_weighted": score,
             "calibration_rmse_unweighted": float(np.sqrt(np.mean((inner["y_cal_actual"] - predicted) ** 2))),
             "frozen_calibration_rmse_weighted": frozen["rmse_weighted"],
             "prior_missing_indices_calibration_rmse_weighted": prior["calibration_rmse_sgd_weighted"],
             "delta_l2": float(np.sqrt(sum(float(value.square().sum()) for value in delta.values()))),
             "trainable_parameters": sum(p.numel() for p in trainable.values()),
             "adapter_modules": sum(isinstance(m, DiagScale) for m in architecture.modules()),
             "fit_seconds": training_seconds,
             "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
             "adapter_sha256": sha256(adapter), "code_sha256": sha256(Path(__file__)),
             "outer_labels_used_for_selection": False}
    write_json(out / "calibration.json", audit)
    pd.DataFrame({"row_id": q.index, "month": inner["cal_month"], "actual": inner["y_cal_actual"],
                  "predicted": predicted, "weight": inner["cal_weight"]}).to_csv(
                      out / "calibration_predictions.csv", index=False)
    print(json.dumps({"calibration_rmse": score, "frozen_rmse": frozen["rmse_weighted"]}), flush=True)

    base = joblib.load(BASE)
    reference = pd.read_csv(REFERENCE)
    np.testing.assert_array_equal(reference.row_id.to_numpy(), base["X_query"].index.to_numpy())
    outer_cat = category_positions(base["X_train"])
    if outer_cat != cat or outer_cat != category_positions(base["X_query"]):
        raise AssertionError("Categorical schema changed between calibration and outer holdout")
    evaluator = SavedDiagScaleRegressor.create_default_for_version(
        version=ModelVersion.V3_5_FAST, adapter_path=str(adapter), model_path=str(CHECKPOINT),
        categorical_features_indices=cat, device="cuda", n_estimators=1,
        auto_scale_n_estimators=False, random_state=42, fit_mode="fit_with_cache",
        keep_cache_on_device=True, inference_config={"SUBSAMPLE_SAMPLES": None}, n_preprocessing_jobs=1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    outer_y = np.asarray(base["y_train"], float) - np.asarray(base["train_baseline"], float)
    evaluator.fit(base["X_train"], outer_y)
    if evaluator.n_train_samples_ != len(base["X_train"]):
        raise AssertionError("Outer context was truncated")
    predicted = predict_batches(evaluator, base["X_query"],
                                np.asarray(base["query_baseline"], float), progress_label="outer")
    result = pd.DataFrame({"row_id": reference.row_id, "month": base["query_month"],
                           "actual": reference.actual, "frozen": reference.predicted,
                           "diagscale_aligned": predicted})
    result.to_csv(out / "predictions.csv", index=False, float_format="%.8f")
    pd.DataFrame([{"model": label, "n": len(result), **metrics(result.actual, result[label])}
                  for label in ("frozen", "diagscale_aligned")]).to_csv(out / "scores.csv", index=False)
    pd.DataFrame([{"month": month, "model": label, "n": len(part), **metrics(part.actual, part[label])}
                  for month, part in result.groupby("month", sort=True)
                  for label in ("frozen", "diagscale_aligned")]).to_csv(out / "monthly_scores.csv", index=False)
    write_json(out / "timing.json", {"outer_fit_predict_seconds": time.perf_counter() - started,
                                     "outer_peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                                     "context_rows": len(base["X_train"]), "validation_rows": len(base["X_query"])})
    print(pd.read_csv(out / "scores.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--task-size", type=int, default=1024)
    args = parser.parse_args()
    train_and_score(args.learning_rate, views=args.views, task_size=args.task_size)
