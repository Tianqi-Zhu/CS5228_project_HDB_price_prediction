"""Fine-tune DiagScale with opt-in chronological context sampling.

The old random-episode runner is preserved. Within this runner all strategies
share exactly the same month-coherent supervised queries for each seed/epoch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from analysis.compare_tabpfn_context_sampling import _months
from analysis.compare_tabpfn_diagscale_80_20 import BASE, REFERENCE
from analysis.compare_tabpfn_v3_80_20 import write_json
from analysis.compare_tabpfn_v3_5_fast_80_20 import CHECKPOINT
from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.diagscale_tabpfn import DiagScale
from analysis.experiments.tabpfn_context_sampler import SamplerConfig, metadata_from_features
from analysis.experiments.tabpfn_sampled_finetuning import (
    EpisodeConfig, SampledContextDiagScaleRegressor,
)
from analysis.tune_tabpfn_diagscale_80_20 import (
    SavedDiagScaleRegressor, inference_estimator, inner_bundle,
    predict_batches, weighted_rmse,
)


OUT = REPORTS / "tabpfn_context_sampling"
CACHE = ROOT / ".cache/tabpfn_context_sampling"


def _metadata(x: pd.DataFrame):
    raw = pd.read_csv(DATA / "train.csv", usecols=lambda name: name != "MONTHLY_RENT")
    groups = pd.util.hash_pandas_object(raw, index=False).to_numpy(np.uint64)
    if x.index.max() >= len(groups):
        raise AssertionError("Fitting row IDs do not map to raw training rows")
    return metadata_from_features(x, _months(x), duplicate_groups=groups[x.index.to_numpy()])


def train(strategy: str, *, seed: int, epochs: int, episodes: int, patience: int,
          calibration_limit: int | None, smoke: bool):
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TABPFN_MAX_BATCHED_ESTIMATOR_ROWS", "0")
    if strategy not in ("S1", "S2", "S3", "S4"):
        raise ValueError("Use S1/S2/S3/S4 for the matched chronological episode study")
    if patience < 1:
        raise ValueError("Early-stopping patience must be positive")
    expected_bundle = json.loads((REPORTS / "tabpfn_v3_5_fast_calendar_80_20/feature_manifest_categorical.json").read_text())
    if sha256(BASE) != expected_bundle["sha256"]:
        raise ValueError("Winning feature artifact changed")
    bundle = inner_bundle()
    x, y = bundle["X_fit"], np.asarray(bundle["y_fit"], float)
    q, z = bundle["X_cal"], np.asarray(bundle["y_cal_residual"], float)
    if calibration_limit is not None:
        q, z = q.iloc[:calibration_limit], z[:calibration_limit]
    if category_positions(x) != category_positions(q):
        raise AssertionError("Fitting/calibration categorical schema differs")
    metadata = _metadata(x)
    episode_config = EpisodeConfig(episodes_per_epoch=episodes)
    sampler_config = SamplerConfig(context_size=episode_config.support_rows)
    label = f"phase_b_{strategy}_seed{seed}_episodes{episodes}_epochs{epochs}"
    if patience != 3:
        label += f"_patience{patience}"
    if smoke:
        label += "_smoke"
    out = OUT / label
    cache = CACHE / label
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    settings = {
        "strategy": strategy, "information_policy_training": "historical_only",
        "episode_construction": "same-month queries from 2022-01 onward; distinct within epoch",
        "sampler_config": asdict(sampler_config), "episode_config": asdict(episode_config),
        "seed": seed, "epochs": epochs, "early_stopping_patience": patience,
        "calibration_rows": len(q),
        "learning_rate": .01, "weight_decay": .01, "crps_weight": 1, "mse_weight": 1,
        "train_views": 2, "checkpoint_views": 2, "deployment_views": 1,
        "feature_artifact_sha256": sha256(BASE), "checkpoint_sha256": sha256(CHECKPOINT),
        "inner_feature_sha256": json.loads((REPORTS / "tabpfn_diagscale_80_20/inner_feature_manifest.json").read_text())["sha256"],
        "feature_columns": list(x.columns), "categorical_positions": category_positions(x),
        "target": "y_fit is rent minus saved baseline; reconstruct with saved query baseline",
        "gpu": torch.cuda.get_device_name(),
        "versions": {name: version(name) for name in
                     ("tabpfn", "torch", "numpy", "pandas", "scikit-learn")},
    }
    settings_path = out / "settings.json"
    if settings_path.exists():
        existing = json.loads(settings_path.read_text())
        existing.setdefault("early_stopping_patience", 3)
        if existing != settings:
            raise ValueError("Existing run has different settings")
    write_json(settings_path, settings)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = SampledContextDiagScaleRegressor(
        model_version=ModelVersion.V3_5_FAST, device="cuda", epochs=epochs,
        learning_rate=.01, weight_decay=.01, random_state=seed,
        validation_split_ratio=0, early_stopping=True, early_stopping_patience=patience,
        n_finetune_ctx_plus_query_samples=episode_config.task_rows,
        finetune_ctx_query_split_ratio=.2, n_inference_subsample_samples=None,
        n_estimators_finetune=2, n_estimators_validation=2,
        n_estimators_final_inference=1, use_activation_checkpointing=True,
        ce_loss_weight=0, crps_loss_weight=1, mse_loss_weight=1,
        extra_regressor_kwargs={
            "model_path": str(CHECKPOINT),
            "categorical_features_indices": category_positions(x),
            "auto_scale_n_estimators": False,
            "inference_config": {"SUBSAMPLE_SAMPLES": None},
            "n_preprocessing_jobs": 1,
        },
    )
    model.configure_context_sampling(metadata, strategy, sampler_config, episode_config)
    print(f"Training {strategy}: {len(x):,} fitting rows, {episodes} month-coherent episodes/epoch", flush=True)
    model.fit(x, y, X_val=q, y_val=z)
    fit_seconds = time.perf_counter() - started
    architecture = model.finetuned_estimator_.model_
    trainable = {name: p for name, p in architecture.named_parameters() if p.requires_grad}
    if not trainable or any(not name.endswith(".delta") for name in trainable):
        raise AssertionError("Non-DiagScale weights were trainable")
    adapter = {name: p.detach().float().cpu().clone() for name, p in trainable.items()}
    adapter_path = cache / "adapter.pt"
    torch.save(adapter, adapter_path)
    all_episodes = model.context_episode_audits_
    if len(all_episodes) == 0 or any(len(epoch["episodes"]) != episodes for epoch in all_episodes):
        raise AssertionError("Chronological episodes were not used at every epoch")
    write_json(out / "episodes.json", all_episodes)
    write_json(out / "training.json", {
        "fit_rows": len(x), "calibration_rows": len(q),
        "fit_seconds": fit_seconds, "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "adapter_modules": sum(isinstance(m, DiagScale) for m in architecture.modules()),
        "delta_l2": float(np.sqrt(sum(float(p.square().sum()) for p in adapter.values()))),
        "adapter_sha256": sha256(adapter_path),
        "epoch_validation": model.diagscale_epochs_,
        "episode_epochs_built": len(all_episodes),
        "mean_retrieval_seconds_per_episode": float(np.mean([
            entry["retrieval_seconds"] for epoch in all_episodes for entry in epoch["episodes"]
        ])),
        "outer_labels_used_for_selection": False,
    })
    # Score exactly the same one-view, full-context calibration task for every
    # trained adapter, regardless of which support sampler it trained with.
    estimator = inference_estimator(model, x, y)
    actual = np.asarray(bundle["y_cal_actual"], float)[:len(q)]
    baseline = np.asarray(bundle["cal_baseline"], float)[:len(q)]
    weights = np.asarray(bundle["cal_weight"], float)[:len(q)]
    predicted = predict_batches(estimator, q, baseline, progress_label=f"calibration {strategy}")
    pd.DataFrame({"row_id": q.index, "month": np.asarray(bundle["cal_month"])[:len(q)],
                  "actual": actual, "predicted": predicted, "weight": weights}).to_csv(
                      out / "calibration_predictions.csv", index=False, float_format="%.8f")
    write_json(out / "calibration.json", {
        "weighted_rmse_sgd": weighted_rmse(actual, predicted, weights),
        **metrics(actual, predicted), "context_rows": len(x), "query_rows": len(q),
        "deployment_views": 1,
    })
    print(json.dumps({"strategy": strategy, "calibration": json.loads(
        (out / "calibration.json").read_text())}), flush=True)
    return out


def score_outer(out: Path, *, query_limit: int | None, inference_strategy: str):
    settings = json.loads((out / "settings.json").read_text())
    audit = json.loads((out / "training.json").read_text())
    adapter_path = CACHE / out.name / "adapter.pt"
    if sha256(adapter_path) != audit["adapter_sha256"]:
        raise ValueError("Adapter checkpoint changed")
    base = joblib.load(BASE)
    reference = pd.read_csv(REFERENCE)
    x, q = base["X_train"], base["X_query"]
    positions = (np.arange(len(q)) if query_limit is None or query_limit >= len(q)
                 else np.unique(np.linspace(0, len(q) - 1, num=query_limit, dtype=int)))
    q = q.iloc[positions]
    np.testing.assert_array_equal(reference.row_id.to_numpy()[positions], q.index.to_numpy())
    if list(x.columns) != settings["feature_columns"] or category_positions(x) != settings["categorical_positions"]:
        raise AssertionError("Outer model feature schema changed")
    y = np.asarray(base["y_train"], float) - np.asarray(base["train_baseline"], float)
    if inference_strategy == "S0":
        context_x, context_y = x, y
    else:
        if inference_strategy not in ("S1", "S2", "S3", "S4"):
            raise ValueError("Unknown outer inference sampler")
        ids_path = (OUT / "phase_a_outer_budget10000_queriesfull_seed42"
                    / f"context_ids_{inference_strategy}.npy")
        selected_ids = np.load(ids_path)
        context_positions = x.index.get_indexer(selected_ids)
        if (context_positions < 0).any() or len(np.unique(selected_ids)) != len(selected_ids):
            raise AssertionError("Saved outer context IDs changed")
        context_x, context_y = x.iloc[context_positions], y[context_positions]
    baseline = np.asarray(base["query_baseline"], float)[positions]
    actual = np.asarray(base["y_query"], float)[positions]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = SavedDiagScaleRegressor.create_default_for_version(
        version=ModelVersion.V3_5_FAST, adapter_path=str(adapter_path),
        model_path=str(CHECKPOINT), categorical_features_indices=category_positions(x),
        device="cuda", n_estimators=1, auto_scale_n_estimators=False,
        random_state=settings["seed"], fit_mode="fit_with_cache",
        keep_cache_on_device=True, inference_config={"SUBSAMPLE_SAMPLES": None},
        n_preprocessing_jobs=1,
    )
    model.fit(context_x, context_y)
    if model.n_train_samples_ != len(context_x):
        raise AssertionError("Inference context was truncated")
    predicted = predict_batches(model, q, baseline, progress_label=f"outer {settings['strategy']}")
    label = "full" if query_limit is None else str(len(q))
    path = out / f"outer_predictions_{label}_{inference_strategy}.csv"
    pd.DataFrame({"row_id": q.index, "month": np.asarray(base["query_month"])[positions],
                  "actual": actual, "predicted": predicted}).to_csv(
                      path, index=False, float_format="%.8f")
    score = {"strategy": settings["strategy"], "sampler_train": settings["strategy"],
             "sampler_inference": inference_strategy, "query_rows": len(q),
             "context_rows": len(context_x),
             "train_views": 2, "checkpoint_views": 2, "deployment_views": 1,
             "learning_rate": .01, "seed": settings["seed"],
             "fit_predict_seconds": time.perf_counter() - started,
             "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
             **metrics(actual, predicted)}
    write_json(out / f"outer_score_{label}_{inference_strategy}.json", score)
    print(json.dumps(score), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("S1", "S2", "S3", "S4"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--calibration-limit", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--outer", action="store_true")
    parser.add_argument("--outer-query-limit", type=int)
    parser.add_argument("--inference-strategy", choices=("S0", "S1", "S2", "S3", "S4"),
                        default="S0")
    args = parser.parse_args()
    out = OUT / f"phase_b_{args.strategy}_seed{args.seed}_episodes{args.episodes}_epochs{args.epochs}"
    if args.patience != 3:
        out = OUT / (out.name + f"_patience{args.patience}")
    if args.smoke:
        out = OUT / (out.name + "_smoke")
    if not (out / "training.json").exists():
        train(args.strategy, seed=args.seed, epochs=args.epochs, episodes=args.episodes,
              patience=args.patience,
              calibration_limit=args.calibration_limit, smoke=args.smoke)
    if args.outer:
        score_outer(out, query_limit=args.outer_query_limit,
                    inference_strategy=args.inference_strategy)
