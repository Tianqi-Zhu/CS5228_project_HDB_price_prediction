"""Sequential, resumable component experiments on the established 77-feature split.

Run --smoke first, then --stage all. No existing artifact is overwritten.
Only earlier calibration labels select configurations, epochs or inference views.
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
import torch
from tabpfn.constants import ModelVersion

from analysis.check_diagscale_categorical_alignment_80_20 import category_positions
from analysis.compare_tabpfn_diagscale_80_20 import BASE, REFERENCE
from analysis.compare_tabpfn_v3_80_20 import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.tabpfn_component_adaptation import (
    ComponentRegressor, FinetunedComponentRegressor, frozen_digest,
)
from analysis.tune_tabpfn_diagscale_80_20 import (
    SavedDiagScaleRegressor, inner_bundle, weighted_rmse,
)


OUT = REPORTS / "tabpfn_components_80_20"
CACHE = ROOT / ".cache/tabpfn_components_80_20"
DIAG_ADAPTER = ROOT / ".cache/tabpfn_diagscale_categorical_alignment_80_20/adapter.pt"
DIAG_REPORT = REPORTS / "tabpfn_diagscale_categorical_alignment_80_20"
CANDIDATES = {
    "icl_rank2": {"component": "icl_lora", "rank": 2, "lr": .0003},
    "icl_rank4": {"component": "icl_lora", "rank": 4, "lr": .0003},
    "icl_last2_rank4": {"component": "icl_lora", "rank": 4, "last_blocks": 2, "lr": .0003},
    "regression_head": {"component": "head", "lr": .00003},
    "diag_crps_mse": {"component": "diagscale", "lr": .01},
    "diag_mse_only": {"component": "diagscale", "lr": .01, "crps": 0},
}
FAMILIES = {"icl": ["icl_rank2", "icl_rank4", "icl_last2_rank4"],
            "head": ["regression_head"], "loss_control": ["diag_crps_mse"],
            "loss_mse": ["diag_mse_only"]}


def immutable_json(path, value):
    value = json.loads(json.dumps(value))
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"Existing settings changed; use a new directory: {path}")
    write_json(path, value)


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((REPORTS / "tabpfn_v3_5_fast_calendar_80_20/feature_manifest_categorical.json").read_text())
    if sha256(BASE) != manifest["sha256"]:
        raise ValueError("Winning feature artifact changed")
    diag = json.loads((DIAG_REPORT / "calibration.json").read_text())
    if sha256(DIAG_ADAPTER) != diag["adapter_sha256"]:
        raise ValueError("Winning DiagScale checkpoint changed")
    base, inner = joblib.load(BASE), inner_bundle()
    x, q = base["X_train"], base["X_query"]
    assert len(x) == 121005 and len(q) == 28995 and len(x.columns) == 77
    assert list(x) == list(q) == list(inner["X_fit"]) == list(inner["X_cal"])
    for other in (q, inner["X_fit"], inner["X_cal"]):
        assert category_positions(other) == category_positions(x)
    assert set(inner["X_fit"].index).isdisjoint(inner["X_cal"].index)
    assert set(x.index).isdisjoint(q.index)
    assert pd.PeriodIndex(base["train_month"], freq="M").max() < pd.PeriodIndex(base["query_month"], freq="M").min()
    plan = {"feature_artifact_sha256": manifest["sha256"], "columns": list(x),
            "categorical_indices": category_positions(x), "checkpoint_sha256": sha256(CHECKPOINT),
            "diag_adapter_sha256": sha256(DIAG_ADAPTER), "candidates": CANDIDATES,
            "families": FAMILIES, "seed": 42, "train_views": 2, "deployment_views": 1,
            "epochs_cap": 8, "patience": 3, "task_rows": 1024, "query_fraction": .2,
            "checkpoint_selection": "minimum earlier population-weighted one-view rent RMSE, including zero updates",
            "family_selection": "minimum earlier weighted RMSE, ties use declared order",
            "view_selection": "minimum earlier weighted RMSE among 1,2,4 views of original winning DiagScale adapter",
            "outer_labels_used_for_selection": False,
            "target": "y_train - train_baseline; add query_baseline exactly once",
            "training_episodes": "unchanged upstream random mixed-month episodes within pre-December-2023 fitting data",
            "history_policy": "reuse original pseudo-origin training histories; calibration freezes at November 2023; outer at May 2024",
            "context_rows": len(x), "outer_rows": len(q), "fit_rows": len(inner["X_fit"]),
            "calibration_rows": len(inner["X_cal"]), "gpu": torch.cuda.get_device_name(),
            "gpu_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "versions": {p: version(p) for p in ("tabpfn", "torch", "numpy", "pandas", "scikit-learn")},
            "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in
                              (ROOT / "data/train.csv", ROOT / "data/test.csv", REFERENCE,
                               ROOT / ".cache/tabpfn_diagscale_80_20/inner_features.joblib")},
            "implementation_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in
                                      (Path(__file__), ROOT / "analysis/experiments/tabpfn_component_adaptation.py")},
            "existing_results_reused": {
                "all_attention_lora_two_views": "reports/tabpfn_lora_qk_lr00003_80_20/scores_2views.csv",
                "embedding_mlp": "reports/tabpfn_fast_embedding_head_80_20/scores.csv",
                "diagscale_one_view": "reports/tabpfn_diagscale_categorical_alignment_80_20/scores.csv",
                "four_train_four_deploy": "reports/tabpfn_diagscale_four_views_80_20/scores.csv"}}
    immutable_json(OUT / "plan.json", plan)
    return base, inner


def prediction_frame(x, month, actual, prediction, weights=None):
    result = pd.DataFrame({"row_id": x.index, "month": month, "actual": actual, "predicted": prediction})
    if weights is not None:
        result["weight"] = weights
    if not result.row_id.is_unique or not np.isfinite(result.predicted).all():
        raise AssertionError("Invalid row predictions")
    return result


def run_training(name, inner, *, smoke=False):
    cfg = CANDIDATES[name]
    directory = OUT / ("smoke" if smoke else "candidates") / name
    cache = CACHE / ("smoke" if smoke else "candidates") / name
    directory.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    adapter = cache / "adapter.pt"
    settings = {"config": cfg, "epochs": 1 if smoke else 8, "task_rows": 512 if smoke else 1024,
                "train_views": 2, "checkpoint_views": 1, "seed": 42, "smoke": smoke}
    immutable_json(directory / "settings.json", settings)
    if (directory / "calibration.json").exists():
        result = json.loads((directory / "calibration.json").read_text())
        if sha256(adapter) != result["adapter_sha256"]:
            raise ValueError("Saved component changed")
        print(f"REUSE training {name}", flush=True)
        return result
    x, q = inner["X_fit"], inner["X_cal"]
    y, z = inner["y_fit"], inner["y_cal_residual"]
    weights = inner["cal_weight"]
    if smoke:
        pos = np.random.default_rng(42).choice(len(x), 1024, replace=False)
        x, y = x.iloc[pos], np.asarray(y)[pos]
        q, z, weights = q.iloc[:24], np.asarray(z)[:24], np.asarray(weights)[:24]
    clear_gpu()
    torch.cuda.reset_peak_memory_stats()
    # Seed torch as well as TabPFN's NumPy/task sampler. New LoRA matrices
    # otherwise inherit an unrecorded process-global initialization state.
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = FinetunedComponentRegressor(
        model_version=ModelVersion.V3_5_FAST, device="cuda", epochs=settings["epochs"],
        learning_rate=cfg["lr"], weight_decay=.01, random_state=42,
        validation_split_ratio=0, early_stopping=True, early_stopping_patience=3,
        n_finetune_ctx_plus_query_samples=settings["task_rows"], finetune_ctx_query_split_ratio=.2,
        n_inference_subsample_samples=None, n_estimators_finetune=2,
        n_estimators_validation=1, n_estimators_final_inference=1, use_activation_checkpointing=True,
        ce_loss_weight=0, crps_loss_weight=cfg.get("crps", 1), mse_loss_weight=1,
        extra_regressor_kwargs={"model_path": str(CHECKPOINT), "categorical_features_indices": category_positions(x),
                                "auto_scale_n_estimators": False, "inference_config": {"SUBSAMPLE_SAMPLES": None},
                                "n_preprocessing_jobs": 1})
    model.component_config = cfg
    model.calibration_weights = weights
    model.calibration_X_train = x
    model.calibration_X_query = q
    model.component_checkpoint_path = adapter
    model.component_trace_path = directory / "epoch_trace.json"
    started = time.perf_counter()
    print(f"TRAIN {name}: {len(x)} fitting, {len(q)} calibration; {cfg}", flush=True)
    model.fit(x, y, X_val=q, y_val=z)
    architecture = model.finetuned_estimator_.model_
    after = frozen_digest(architecture)
    if after != model.frozen_before_:
        raise AssertionError("Frozen backbone was modified")
    state = torch.load(adapter, map_location="cpu", weights_only=True)
    # Reload the selected artifact in an independent inference model to verify
    # persistence, exact module selection and the reconstructed calibration score.
    del model
    clear_gpu()
    evaluator = make_estimator(cfg, adapter, category_positions(x), views=1)
    evaluator.fit(x, y)
    residual = np.concatenate([np.asarray(evaluator.predict(q.iloc[i:i+200]), float) for i in range(0, len(q), 200)])
    saved = np.load(adapter.with_suffix(".prediction.npy"))
    if not np.allclose(residual, saved, atol=.01, rtol=0):
        raise AssertionError(f"Selected checkpoint replay differs by {np.max(np.abs(residual-saved))}")
    trace = json.loads((directory / "epoch_trace.json").read_text())
    selected = min(trace, key=lambda r: r["weighted_rmse_sgd"])
    score = weighted_rmse(z, residual, weights)
    if abs(score - selected["weighted_rmse_sgd"]) > .01:
        raise AssertionError("Selected score did not replay")
    result = {"name": name, "config": cfg, "weighted_rmse_sgd": score,
              "initial_weighted_rmse_sgd": trace[0]["weighted_rmse_sgd"], "selected_epoch": selected["epoch"],
              "epochs_run": len(trace)-1, "fit_rows": len(x), "calibration_rows": len(q),
              "seconds": time.perf_counter()-started, "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3,
              "frozen_parameters_unchanged": True, "frozen_digest": after,
              "trainable_parameters": sum(t.numel() for t in state.values()),
              "trainable_names": list(state), "adapter_sha256": sha256(adapter),
              "max_checkpoint_replay_difference": float(np.max(np.abs(residual-saved))),
              "outer_labels_used_for_selection": False}
    if not smoke:
        frame = prediction_frame(q, inner["cal_month"], inner["y_cal_actual"], residual + inner["cal_baseline"], weights)
        frame.to_csv(directory / "calibration_predictions.csv", index=False, float_format="%.10f")
    write_json(directory / "calibration.json", result)
    print("CALIBRATION " + json.dumps(result), flush=True)
    del evaluator, state
    clear_gpu()
    return result


def make_estimator(config, adapter, cat, views=1):
    common = dict(version=ModelVersion.V3_5_FAST, model_path=str(CHECKPOINT),
                  categorical_features_indices=cat, device="cuda", n_estimators=views,
                  auto_scale_n_estimators=False, random_state=42, fit_mode="fit_with_cache",
                  keep_cache_on_device=True, inference_config={"SUBSAMPLE_SAMPLES": None}, n_preprocessing_jobs=1)
    if config is None:
        return SavedDiagScaleRegressor.create_default_for_version(adapter_path=str(adapter), **common)
    return ComponentRegressor.create_default_for_version(component_config=config, adapter_path=str(adapter), **common)


def predict_resumable(label, config, adapter, x, y, q, baseline, *, views, stage):
    directory = OUT / stage / label
    directory.mkdir(parents=True, exist_ok=True)
    immutable_json(directory / "settings.json", {"config": config, "adapter_sha256": sha256(adapter),
                   "views": views, "context_rows": len(x), "query_rows": len(q),
                   "query_ids_sha256": __import__("hashlib").sha256(q.index.to_numpy().tobytes()).hexdigest(),
                   "batch_rows": 200, "seed": 42})
    partial = directory / "partial.npy"
    done = np.load(partial) if partial.exists() else np.empty(0)
    if len(done) > len(q) or not np.isfinite(done).all():
        raise ValueError("Invalid partial predictions")
    if len(done) == len(q) and (directory / "timing.json").exists():
        print(f"REUSE {stage}/{label}", flush=True)
        return done
    clear_gpu()
    torch.cuda.reset_peak_memory_stats()
    model = make_estimator(config, adapter, category_positions(x), views)
    started = time.perf_counter()
    model.fit(x, y)
    assert model.n_train_samples_ == len(x)
    for start in range(len(done), len(q), 200):
        batch = q.iloc[start:start+200]
        prediction = np.asarray(model.predict(batch, output_type="mean"), float) + baseline[start:start+len(batch)]
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite prediction")
        done = np.concatenate([done, prediction])
        temporary = partial.with_suffix(".tmp.npy")
        np.save(temporary, done)
        temporary.replace(partial)
        if start % 2000 == 0 or len(done) == len(q):
            print(f"PREDICT {stage}/{label}: {len(done)}/{len(q)}", flush=True)
    write_json(directory / "timing.json", {"seconds_this_attempt": time.perf_counter()-started,
               "peak_vram_gib": torch.cuda.max_memory_allocated()/1024**3,
               "context_rows": len(x), "query_rows": len(q), "views": views})
    del model
    clear_gpu()
    return done


def views_calibration(inner):
    scores = []
    for views in (1, 2, 4):
        label = f"original_diag_{views}view"
        if views == 1:
            saved = pd.read_csv(DIAG_REPORT / "calibration_predictions.csv")
            np.testing.assert_array_equal(saved.row_id, inner["X_cal"].index)
            prediction = saved.predicted.to_numpy()
        else:
            prediction = predict_resumable(label, None, DIAG_ADAPTER, inner["X_fit"], inner["y_fit"],
                inner["X_cal"], inner["cal_baseline"], views=views, stage="view_calibration")
        frame = prediction_frame(inner["X_cal"], inner["cal_month"], inner["y_cal_actual"], prediction, inner["cal_weight"])
        frame.to_csv(OUT / f"view_calibration_{views}.csv", index=False, float_format="%.10f")
        scores.append({"views": views, "weighted_rmse_sgd": weighted_rmse(inner["y_cal_actual"], prediction, inner["cal_weight"])})
        print("VIEW CALIBRATION " + json.dumps(scores[-1]), flush=True)
    pd.DataFrame(scores).to_csv(OUT / "view_calibration_scores.csv", index=False)
    selected = min(scores, key=lambda row: row["weighted_rmse_sgd"])
    immutable_json(OUT / "view_selection.json", {"selected_views": selected["views"], "scores": scores,
                                               "outer_labels_used_for_selection": False})


def select_and_score(base):
    selected = {}
    for family, names in FAMILIES.items():
        records = [json.loads((OUT / "candidates" / name / "calibration.json").read_text()) for name in names]
        selected[family] = min(records, key=lambda r: r["weighted_rmse_sgd"])["name"]
    immutable_json(OUT / "component_selection.json", {"selected": selected, "outer_labels_used_for_selection": False})
    x, q = base["X_train"], base["X_query"]
    y = np.asarray(base["y_train"], float) - base["train_baseline"]
    jobs = [(name, CANDIDATES[name], CACHE / "candidates" / name / "adapter.pt", 1) for name in selected.values()]
    # Score all fixed view ablations for reporting; select the recommended view
    # only from view_selection.json, which exists before any of these scores.
    jobs += [(f"original_diag_{v}view", None, DIAG_ADAPTER, v) for v in (2, 4)]
    for label, config, adapter, views in jobs:
        prediction = predict_resumable(label, config, adapter, x, y, q, base["query_baseline"], views=views, stage="outer")
        frame = prediction_frame(q, base["query_month"], base["y_query"], prediction)
        frame.to_csv(OUT / "outer" / label / "predictions.csv", index=False, float_format="%.10f")
        print("OUTER " + json.dumps({"model": label, **metrics(frame.actual, frame.predicted)}), flush=True)


def summarize():
    rows, months = [], []
    sources = [("frozen_one_view", REFERENCE, "predicted"),
               ("winning_diag_one_view", DIAG_REPORT / "predictions.csv", "diagscale_aligned"),
               ("existing_all_qk_lora_two_views", REPORTS / "tabpfn_lora_qk_lr00003_80_20/predictions_2views.csv", "lora_qk"),
               ("existing_frozen_two_views", REPORTS / "tabpfn_lora_qk_lr00003_80_20/predictions_2views.csv", "frozen")]
    sources += [(p.parent.name, p, "predicted") for p in sorted((OUT / "outer").glob("*/predictions.csv"))]
    reference = pd.read_csv(REFERENCE)
    for label, path, column in sources:
        frame = pd.read_csv(path)
        np.testing.assert_array_equal(frame.row_id, reference.row_id)
        np.testing.assert_allclose(frame.actual, reference.actual, rtol=0, atol=1e-8)
        assert frame.row_id.is_unique and len(frame) == 28995 and np.isfinite(frame[column]).all()
        rows.append({"model": label, **metrics(frame.actual, frame[column])})
        for month, part in frame.groupby("month", sort=True):
            months.append({"model": label, "month": month, **metrics(part.actual, part[column])})
    scores = pd.DataFrame(rows)
    baseline = float(scores.set_index("model").loc["winning_diag_one_view", "rmse_sgd"])
    scores["delta_vs_winning_diag_sgd"] = scores.rmse_sgd - baseline
    scores.to_csv(OUT / "scores.csv", index=False, float_format="%.8f")
    pd.DataFrame(months).to_csv(OUT / "monthly_scores.csv", index=False, float_format="%.8f")
    calibration = [json.loads(p.read_text()) for p in sorted((OUT / "candidates").glob("*/calibration.json"))]
    if calibration:
        pd.DataFrame([{k: r[k] for k in ("name", "weighted_rmse_sgd", "initial_weighted_rmse_sgd", "selected_epoch",
                                         "epochs_run", "trainable_parameters", "seconds", "peak_vram_gib")}
                      for r in calibration]).to_csv(OUT / "calibration_scores.csv", index=False)
    print(scores.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "calibration", "views", "outer", "summarize"), default="all")
    parser.add_argument("--candidate", choices=list(CANDIDATES))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    torch.set_num_threads(1)
    base, inner = audit()
    if args.smoke:
        for name in ("icl_rank2", "regression_head", "diag_mse_only"):
            run_training(name, inner, smoke=True)
        return
    if args.stage in ("all", "calibration"):
        for name in ([args.candidate] if args.candidate else CANDIDATES):
            run_training(name, inner)
    if args.stage in ("all", "views"):
        views_calibration(inner)
    if args.stage in ("all", "outer"):
        select_and_score(base)
    if args.stage in ("all", "summarize"):
        summarize()


if __name__ == "__main__":
    main()
