"""Refit selected DiagScale epochs on the whole chronological 80% train split."""

from __future__ import annotations

import gc
from importlib.metadata import version
import json
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

from analysis.compare_tabpfn_calendar_80_20 import frozen_settings
from analysis.compare_tabpfn_v3_80_20 import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.diagscale_tabpfn import DiagScale, FinetunedDiagScaleRegressor


OUT = REPORTS / "tabpfn_diagscale_80_20"
CACHE = ROOT / ".cache/tabpfn_diagscale_refit_80_20"
BASE = ROOT / ".cache/tabpfn_v3_features/calendar_80_20_categorical.joblib"
REFERENCE = REPORTS / "tabpfn_v3_5_fast_calendar_80_20/predictions_outer_categorical.csv"
BATCH = 200


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    first = json.loads((OUT / "training_audit.json").read_text(encoding="utf-8"))
    candidates = [row for row in first["epoch_validation"] if row["epoch"] >= 0]
    selected = min(candidates, key=lambda row: row["validation_mse"])
    epochs = selected["epoch"] + 1
    if epochs < 1:
        raise AssertionError("No positive training epoch was selected")
    settings = {"model": "TabPFN-3.5-Fast DiagScale full-train refit", "seed": 42,
                "selected_epochs": epochs, "selected_inner_mse": selected["validation_mse"],
                "base_bundle_sha256": sha256(BASE), "reference_sha256": sha256(REFERENCE),
                "backbone_sha256": sha256(CHECKPOINT), "learning_rate": .03,
                "weight_decay": .01, "task_size": 4096, "query_fraction": .2,
                "n_estimators": 1, "autocast_dtype": "bfloat16",
                "gpu": torch.cuda.get_device_name(),
                "versions": {p: version(p) for p in ("tabpfn", "torch", "numpy", "pandas")},
                "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                                (Path(__file__), ROOT / "analysis/experiments/diagscale_tabpfn.py")}}
    frozen_settings(OUT / "refit_settings.json", settings)
    base = joblib.load(BASE)
    reference = pd.read_csv(REFERENCE)
    np.testing.assert_array_equal(reference.row_id.to_numpy(), base["X_query"].index.to_numpy())
    np.testing.assert_array_equal(reference.actual.to_numpy(), base["y_query"])
    x = base["X_train"]
    y = np.asarray(base["y_train"], float) - np.asarray(base["train_baseline"], float)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    adapted = FinetunedDiagScaleRegressor(
        model_version=ModelVersion.V3_5_FAST, device="cuda", epochs=epochs,
        learning_rate=.03, weight_decay=.01, random_state=42,
        validation_split_ratio=0, early_stopping=False,
        n_finetune_ctx_plus_query_samples=4096, finetune_ctx_query_split_ratio=.2,
        n_inference_subsample_samples=None,
        n_estimators_finetune=1, n_estimators_validation=1,
        n_estimators_final_inference=1, use_activation_checkpointing=True,
        ce_loss_weight=0, crps_loss_weight=0, mse_loss_weight=1,
        extra_regressor_kwargs={"model_path": str(CHECKPOINT),
                                "inference_config": {"SUBSAMPLE_SAMPLES": None},
                                "n_preprocessing_jobs": 1},
    )
    print(f"Refitting {epochs} selected epochs on all {len(x):,} training rows", flush=True)
    adapted.fit(x, y)
    model = adapted.finetuned_estimator_.model_
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not name.endswith(".delta") for name, _ in trainable):
        raise AssertionError("Non-DiagScale parameters were trainable")
    norm = float(np.sqrt(sum(float(torch.sum(p.detach().float().square())) for _, p in trainable)))
    if not np.isfinite(norm) or norm <= 0:
        raise AssertionError("Refit adapter did not update")
    audit = {"train_rows": len(x), "selected_epochs": epochs,
             "inner_epoch_selection_mse": selected["validation_mse"],
             "adapter_modules": sum(isinstance(m, DiagScale) for m in model.modules()),
             "trainable_parameters": sum(p.numel() for _, p in trainable),
             "delta_l2": norm, "fit_seconds": time.perf_counter() - started,
             "peak_fit_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
             "outer_labels_used_for_selection_or_training": False}
    write_json(OUT / "refit_training_audit.json", audit)
    del adapted.finetuned_inference_regressor_
    gc.collect()
    torch.cuda.empty_cache()

    evaluator = clone_model_for_evaluation(
        adapted.finetuned_estimator_,
        {"device": "cuda", "n_estimators": 1, "random_state": 42,
         "fit_mode": "fit_with_cache", "keep_cache_on_device": True,
         "inference_config": {"SUBSAMPLE_SAMPLES": None}}, TabPFNRegressor)
    evaluator.fit(x, y)
    if evaluator.n_train_samples_ != len(x) or evaluator.inference_config_.SUBSAMPLE_SAMPLES is not None:
        raise AssertionError("Final outer context was subsampled")
    q = base["X_query"]
    baseline = np.asarray(base["query_baseline"], float)
    partial = CACHE / "outer_predictions_partial.npy"
    done = np.load(partial).tolist() if partial.exists() else []
    if len(done) > len(q) or (len(done) != len(q) and len(done) % BATCH):
        raise ValueError("Invalid saved prediction prefix")
    chunks = [np.asarray(done, float)]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for offset in range(len(done), len(q), BATCH):
        query = q.iloc[offset:offset + BATCH]
        pred = np.asarray(evaluator.predict(query, output_type="mean"), float)
        chunks.append(pred + baseline[offset:offset + len(query)])
        if offset % 4000 == 0 or offset + BATCH >= len(q):
            np.save(partial, np.concatenate(chunks))
            print(f"  Refit scored {min(offset + BATCH, len(q)):,}/{len(q):,}", flush=True)
    prediction = np.concatenate(chunks)
    if len(prediction) != len(q) or not np.isfinite(prediction).all():
        raise AssertionError("Incomplete or nonfinite outer predictions")
    write_json(OUT / "refit_timing.json", {"outer_predict_seconds": time.perf_counter() - started,
                                            "peak_predict_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3})
    result = pd.DataFrame({"row_id": reference.row_id, "month": base["query_month"],
                           "actual": reference.actual, "frozen": reference.predicted,
                           "diagscale_full_refit": prediction})
    result.to_csv(OUT / "refit_predictions.csv", index=False, float_format="%.8f")
    scores = pd.DataFrame([{"model": name, "n": len(result), **metrics(result.actual, result[name])}
                           for name in ("frozen", "diagscale_full_refit")])
    scores.to_csv(OUT / "refit_scores.csv", index=False, float_format="%.8f")
    pd.DataFrame([{"month": month, "model": name, "n": len(part), **metrics(part.actual, part[name])}
                  for month, part in result.groupby("month", sort=True)
                  for name in ("frozen", "diagscale_full_refit")]).to_csv(
                      OUT / "refit_monthly_scores.csv", index=False, float_format="%.8f")
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
