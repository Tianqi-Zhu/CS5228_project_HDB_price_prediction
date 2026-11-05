"""Fine-tune TabPFN-3.5-Fast Q/K LoRA on the established chronological split."""

from __future__ import annotations

import argparse
import gc
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

from analysis.check_diagscale_categorical_alignment_80_20 import category_positions
from analysis.compare_tabpfn_diagscale_80_20 import BASE, REFERENCE
from analysis.compare_tabpfn_v3_80_20 import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.lora_qk_tabpfn import (
    FinetunedLoRAQKRegressor, LoRALinear, LoRAQKTabPFNRegressor,
)
from analysis.tune_tabpfn_diagscale_80_20 import (
    inference_estimator, inner_bundle, predict_batches, weighted_rmse,
)


OUT = REPORTS / "tabpfn_lora_qk_80_20"
CACHE = ROOT / ".cache/tabpfn_lora_qk_80_20"
ADAPTER = CACHE / "adapter_rank8_alpha8_lr0003.pt"
FROZEN = REPORTS / "tabpfn_diagscale_tuning_80_20/frozen_calibration.json"
DIAGSCALE = REPORTS / "tabpfn_diagscale_categorical_alignment_lr0003_80_20/calibration.json"


class SavedLoRAQKRegressor(LoRAQKTabPFNRegressor):
    """Apply saved adapter tensors immediately after loading the backbone."""

    def __init__(self, *, adapter_path: str, **kwargs):
        self.adapter_path = adapter_path
        super().__init__(**kwargs)

    def _initialize_model_variables(self):
        size = super()._initialize_model_variables()
        adapter = torch.load(self.adapter_path, map_location="cpu", weights_only=True)
        for model in self.models_:
            named = dict(model.named_parameters())
            # The checkpoint also contains a classification decoder. Its Q/K
            # adapters are present but frozen by the regression fine-tuner.
            expected = {name for name in named if name.endswith(".lora_a.weight")
                        or name.endswith(".lora_b.weight")}
            if set(adapter) - expected:
                raise AssertionError(
                    f"Saved Q/K LoRA names do not match checkpoint: "
                    f"missing={sorted(set(adapter) - expected)[:3]}")
            with torch.no_grad():
                for name, value in adapter.items():
                    named[name].copy_(value.to(device=named[name].device,
                                               dtype=named[name].dtype))
        return size


def train_and_calibrate(learning_rate: float):
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    inner = inner_bundle()
    x, y = inner["X_fit"], inner["y_fit"]
    q, z = inner["X_cal"], inner["y_cal_residual"]
    cat = category_positions(x)
    if cat != category_positions(q) or not cat:
        raise AssertionError("Context and calibration categorical indices differ")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = FinetunedLoRAQKRegressor(
        model_version=ModelVersion.V3_5_FAST, device="cuda", epochs=8,
        learning_rate=learning_rate, weight_decay=.01, random_state=42,
        validation_split_ratio=0, early_stopping=True, early_stopping_patience=3,
        n_finetune_ctx_plus_query_samples=1024, finetune_ctx_query_split_ratio=.2,
        n_inference_subsample_samples=None,
        n_estimators_finetune=2, n_estimators_validation=2,
        n_estimators_final_inference=1, use_activation_checkpointing=True,
        ce_loss_weight=0, crps_loss_weight=1, mse_loss_weight=1,
        extra_regressor_kwargs={"model_path": str(CHECKPOINT),
                                "categorical_features_indices": cat,
                                "auto_scale_n_estimators": False,
                                "inference_config": {"SUBSAMPLE_SAMPLES": None},
                                "n_preprocessing_jobs": 1},
    )
    print(f"Q/K LoRA: {len(x):,} fitting rows; {len(q):,} chronological "
          f"calibration rows; categorical indices {cat}", flush=True)
    model.fit(x, y, X_val=q, y_val=z)
    training_seconds = time.perf_counter() - started
    architecture = model.finetuned_estimator_.model_
    adapter = {name: parameter.detach().float().cpu().clone()
               for name, parameter in architecture.named_parameters()
               if parameter.requires_grad}
    if not adapter or any(not (name.endswith(".lora_a.weight") or
                               name.endswith(".lora_b.weight")) for name in adapter):
        raise AssertionError("Unexpected trainable parameters")
    if not all(torch.isfinite(value).all() for value in adapter.values()):
        raise AssertionError("Nonfinite LoRA matrices")
    torch.save(adapter, ADAPTER)
    evaluator = inference_estimator(model, x, y)
    predicted = predict_batches(evaluator, q, inner["cal_baseline"],
                                progress_label="calibration")
    score = weighted_rmse(inner["y_cal_actual"], predicted, inner["cal_weight"])
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    diagscale = json.loads(DIAGSCALE.read_text(encoding="utf-8"))
    audit = {"method": "Q/K LoRA", "rank": 8, "alpha": 8,
             "learning_rate": learning_rate, "train_views": 2, "checkpoint_views": 2,
             "one_view_calibration_views": 1,
             "finetuner_final_inference_views": 1,
             "epochs_cap": 8, "patience": 3,
             "task_size": 1024, "crps_weight": 1, "mse_weight": 1,
             "categorical_indices": cat,
             "fit_rows": len(x), "calibration_rows": len(q),
             "calibration_rmse_weighted": score,
             "calibration_rmse_unweighted": float(np.sqrt(np.mean(
                 (inner["y_cal_actual"] - predicted) ** 2))),
             "frozen_calibration_rmse_weighted": frozen["rmse_weighted"],
             "diagscale_lr0003_calibration_rmse_weighted": diagscale["calibration_rmse_weighted"],
             "epoch_validation": model.lora_epochs_,
             "attention_modules": sum(isinstance(m.q_projection, LoRALinear)
                                      for m in architecture.modules()
                                      if hasattr(m, "q_projection")),
             "trainable_parameters": sum(value.numel() for value in adapter.values()),
             "fit_seconds": training_seconds,
             "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
             "adapter_sha256": sha256(ADAPTER), "code_sha256": sha256(Path(__file__)),
             "inner_feature_sha256": sha256(ROOT / ".cache/tabpfn_diagscale_80_20/inner_features.joblib"),
             "backbone_sha256": sha256(CHECKPOINT),
             "outer_labels_used_for_selection": False}
    write_json(OUT / "calibration.json", audit)
    pd.DataFrame({"row_id": q.index, "month": inner["cal_month"],
                  "actual": inner["y_cal_actual"], "predicted": predicted,
                  "weight": inner["cal_weight"]}).to_csv(
                      OUT / "calibration_predictions.csv", index=False)
    print(json.dumps({"calibration_rmse": score,
                      "frozen_rmse": frozen["rmse_weighted"],
                      "diagscale_rmse": diagscale["calibration_rmse_weighted"]}), flush=True)


def score_outer(views: int = 1):
    base = joblib.load(BASE)
    reference = pd.read_csv(REFERENCE)
    np.testing.assert_array_equal(reference.row_id.to_numpy(),
                                  base["X_query"].index.to_numpy())
    cat = category_positions(base["X_train"])
    if cat != category_positions(base["X_query"]):
        raise AssertionError("Outer categorical schema changed")
    common = dict(version=ModelVersion.V3_5_FAST,
                  model_path=str(CHECKPOINT), categorical_features_indices=cat,
                  device="cuda", n_estimators=views, auto_scale_n_estimators=False,
                  random_state=42, fit_mode="fit_with_cache",
                  keep_cache_on_device=True,
                  inference_config={"SUBSAMPLE_SAMPLES": None},
                  n_preprocessing_jobs=1)
    started = time.perf_counter()
    outer_y = (np.asarray(base["y_train"], float) -
               np.asarray(base["train_baseline"], float))
    result = pd.DataFrame({"row_id": reference.row_id,
                           "month": base["query_month"],
                           "actual": reference.actual})
    if views == 1:
        result["frozen"] = reference.predicted
        runs = [("lora_qk", SavedLoRAQKRegressor,
                 {"adapter_path": str(ADAPTER)})]
    else:
        runs = [("frozen", LoRAQKTabPFNRegressor, {}),
                ("lora_qk", SavedLoRAQKRegressor,
                 {"adapter_path": str(ADAPTER)})]
    peaks = {}
    for label, cls, extra in runs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        estimator = cls.create_default_for_version(**common, **extra)
        estimator.fit(base["X_train"], outer_y)
        if estimator.n_train_samples_ != len(base["X_train"]):
            raise AssertionError("Outer context was truncated")
        result[label] = predict_batches(
            estimator, base["X_query"],
            np.asarray(base["query_baseline"], float),
            progress_label=f"outer {views} views {label}")
        peaks[label] = torch.cuda.max_memory_allocated() / 1024 ** 3
        del estimator
        gc.collect()
    suffix = "" if views == 1 else f"_{views}views"
    result.to_csv(OUT / f"predictions{suffix}.csv", index=False, float_format="%.8f")
    pd.DataFrame([{"model": label, "n": len(result),
                   **metrics(result.actual, result[label])}
                  for label in ("frozen", "lora_qk")]).to_csv(
                      OUT / f"scores{suffix}.csv", index=False)
    pd.DataFrame([{"month": month, "model": label, "n": len(part),
                   **metrics(part.actual, part[label])}
                  for month, part in result.groupby("month", sort=True)
                  for label in ("frozen", "lora_qk")]).to_csv(
                      OUT / f"monthly_scores{suffix}.csv", index=False)
    write_json(OUT / f"timing{suffix}.json", {
        "outer_fit_predict_seconds": time.perf_counter() - started,
        "outer_peak_vram_gib": peaks,
        "views": views,
        "context_rows": len(base["X_train"]),
        "validation_rows": len(base["X_query"])})
    print(pd.read_csv(OUT / f"scores{suffix}.csv").to_string(index=False), flush=True)


def score_calibration_two_views():
    """Compare trained and zero adapters with the two training-time views."""
    inner = inner_bundle()
    x, y, q = inner["X_fit"], inner["y_fit"], inner["X_cal"]
    cat = category_positions(x)
    common = dict(version=ModelVersion.V3_5_FAST,
                  model_path=str(CHECKPOINT), categorical_features_indices=cat,
                  device="cuda", n_estimators=2, auto_scale_n_estimators=False,
                  random_state=42, fit_mode="fit_with_cache",
                  keep_cache_on_device=True,
                  inference_config={"SUBSAMPLE_SAMPLES": None}, n_preprocessing_jobs=1)
    scores = {}
    for label, cls, extra in (
        ("frozen", LoRAQKTabPFNRegressor, {}),
        ("lora_qk", SavedLoRAQKRegressor, {"adapter_path": str(ADAPTER)}),
    ):
        estimator = cls.create_default_for_version(**common, **extra)
        estimator.fit(x, y)
        if estimator.n_train_samples_ != len(x):
            raise AssertionError("Calibration context was truncated")
        prediction = predict_batches(estimator, q, inner["cal_baseline"],
                                     progress_label=f"calibration 2 views {label}")
        scores[label] = weighted_rmse(inner["y_cal_actual"], prediction,
                                      inner["cal_weight"])
        pd.DataFrame({"row_id": q.index, "month": inner["cal_month"],
                      "actual": inner["y_cal_actual"], "predicted": prediction,
                      "weight": inner["cal_weight"]}).to_csv(
                          OUT / f"calibration_predictions_2views_{label}.csv", index=False)
    write_json(OUT / "calibration_2views.json", scores)
    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-only", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--calibration-two-views", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=.003)
    parser.add_argument("--outer-views", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if sum((args.outer_only, args.calibration_only, args.calibration_two_views)) > 1:
        parser.error("Choose at most one phase")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive")
    if args.learning_rate != .003:
        suffix = f"lr{args.learning_rate:g}".replace(".", "")
        OUT = REPORTS / f"tabpfn_lora_qk_{suffix}_80_20"
        CACHE = ROOT / f".cache/tabpfn_lora_qk_{suffix}_80_20"
        ADAPTER = CACHE / "adapter_rank8_alpha8.pt"
    if args.calibration_two_views:
        score_calibration_two_views()
    elif not args.outer_only:
        train_and_calibrate(args.learning_rate)
    if not args.calibration_only and not args.calibration_two_views:
        score_outer(args.outer_views)
