"""The eight-trial inner-only search, then the frozen top-two B evaluation.

Trial search never touches B. That is enforced here: `run_trials` takes only the A
populations, and `evaluate_final` is the single function that loads B, called after
`freeze_top_two` has written the registered candidates to disk.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .data import FeatureTensors, StageState, build_tensors, fit_stage_state
from .training import FitResult, TrainConfig, build_for, fit, metrics, predict

BASE = TrainConfig()

TRIALS = [
    {"id": 1, "pooling": "mean", "changes": {}},
    {"id": 2, "pooling": "cls", "changes": {}},
    {"id": 3, "pooling": "horizon_query", "changes": {}},
    {"id": 4, "pooling": "@best", "changes": {"learning_rate": 3e-4}},
    {"id": 5, "pooling": "@best", "changes": {"d_model": 96}},
    {"id": 6, "pooling": "@best", "changes": {"layers": 3}},
    {"id": 7, "pooling": "@best", "changes": {"dropout": 0.15, "weight_decay": 1e-3}},
    {"id": 8, "pooling": "@best", "changes": {"batch_size": 1024}},
]


def trial_config(entry: dict, best_pooling: str | None = None) -> TrainConfig:
    config = replace(BASE)
    pooling = entry["pooling"]
    if pooling == "@best":
        if best_pooling is None:
            raise ValueError(f"trial {entry['id']} depends on the best pooling from 1-3")
        pooling = best_pooling
    return replace(config, pooling=pooling, **entry["changes"])


def choose_best_pooling(results: list[dict]) -> dict:
    """Pick the pooling for trials 4-8 from trials 1-3, by eligible inner RMSE."""
    eligible = [r for r in results if r["id"] in (1, 2, 3) and r["selected_rmse_sgd"] is not None]
    if not eligible:
        raise RuntimeError("none of trials 1-3 produced an eligible epoch")
    eligible.sort(key=lambda r: (r["selected_rmse_sgd"], r["id"]))
    return {"trial_id": eligible[0]["id"], "pooling": eligible[0]["pooling"],
            "selected_rmse_sgd": eligible[0]["selected_rmse_sgd"],
            "decided_from": [r["id"] for r in eligible],
            "rule": "lowest eligible A/holdout RMSE among trials 1-3; ties -> earlier trial id"}


def run_trial(entry: dict, config: TrainConfig, state: StageState,
              a_fit: FeatureTensors, a_hold: FeatureTensors,
              run_dir: Path, log_path: Path) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model = build_for(state.numeric_names, state.cardinalities(),
                      context_dim=a_fit.query_context.shape[1], config=config)

    def log(record):
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"trial": entry["id"], **record.__dict__}) + "\n")

    result: FitResult = fit(a_fit, a_hold, config, model=model, log=log)

    torch.save({"config": config.to_dict(), "state_dict": result.model.state_dict(),
                "state": state.to_dict(), "pooling": config.pooling,
                "selected_epoch": result.selected_epoch,
                "description": result.model.describe()},
               run_dir / "checkpoint.pt")

    inner_predictions = a_hold.to_sgd(predict(result.model, a_hold))
    np.savez_compressed(run_dir / "inner_predictions.npz",
                        predicted_sgd=inner_predictions, actual_sgd=a_hold.actual_sgd,
                        row_id=a_hold.row_id, horizon=a_hold.horizon)

    summary = {
        "id": entry["id"], "pooling": config.pooling, "changes": entry["changes"],
        "selected_epoch": result.selected_epoch, "selected_rmse_sgd": result.selected_rmse,
        "inner_predictions": metrics(a_hold.actual_sgd, inner_predictions),
        "ineligible_best": result.ineligible_best, "epochs_run": result.epochs_run,
        "seconds": round(time.perf_counter() - started, 2),
        "parameters": result.model.describe()["parameters"],
        "config": config.to_dict(), "description": result.model.describe(),
        "curve": [r.__dict__ for r in result.curve],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def freeze_top_two(trials: list[dict], run_dir: Path) -> list[dict]:
    """Register the top two by eligible inner RMSE, before B is ever loaded."""
    scored = [t for t in trials if t["selected_rmse_sgd"] is not None]
    if len(scored) < 2:
        raise RuntimeError(f"only {len(scored)} trial(s) produced an eligible epoch")
    scored.sort(key=lambda t: (t["selected_rmse_sgd"], t["id"]))
    top = [{**t, "rank": rank} for rank, t in enumerate(scored[:2], start=1)]
    (run_dir / "registered_candidates.json").write_text(json.dumps({
        "registered_before_loading_B": True,
        "rule": "lowest eligible A/holdout RMSE; ties -> earlier trial id",
        "candidates": [{"rank": t["rank"], "trial_id": t["id"], "pooling": t["pooling"],
                        "selected_epoch": t["selected_epoch"],
                        "selected_rmse_sgd": t["selected_rmse_sgd"],
                        "config": t["config"]} for t in top],
    }, indent=2))
    return top


def evaluate_final(candidate: dict, b_state: StageState, b_fit: FeatureTensors,
                   b_val: FeatureTensors, run_dir: Path) -> dict:
    """Refit on B/fit for the inner-selected epoch count, then score B/validation once."""
    config = TrainConfig(**{k: v for k, v in candidate["config"].items()
                            if k in TrainConfig().__dict__})
    config.schedule_epochs = config.max_epochs          # identical schedule to stage A
    epochs = int(candidate["selected_epoch"])
    model = build_for(b_state.numeric_names, b_state.cardinalities(),
                      context_dim=b_fit.query_context.shape[1], config=config)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.schedule_length())
    from .training import _epoch

    curve = []
    for epoch in range(1, epochs + 1):
        train_mse = _epoch(model, b_fit, config, optimiser, epoch)
        curve.append({"epoch": epoch, "train_mse": train_mse})
        scheduler.step()

    predicted = b_val.to_sgd(predict(model, b_val))
    overall = metrics(b_val.actual_sgd, predicted)
    out_dir = run_dir / f"final_trial{candidate['trial_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_dict(), "state_dict": model.state_dict(),
                "state": b_state.to_dict(), "selected_epoch": epochs,
                "description": model.describe()}, out_dir / "checkpoint.pt")
    np.savez_compressed(out_dir / "validation_predictions.npz",
                        predicted_sgd=predicted, actual_sgd=b_val.actual_sgd,
                        row_id=b_val.row_id, horizon=b_val.horizon)
    summary = {"trial_id": candidate["trial_id"], "pooling": config.pooling,
               "rank": candidate["rank"], "epochs": epochs, "overall": overall,
               "curve": curve, "parameters": model.describe()["parameters"],
               "config": config.to_dict(), "description": model.describe()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
