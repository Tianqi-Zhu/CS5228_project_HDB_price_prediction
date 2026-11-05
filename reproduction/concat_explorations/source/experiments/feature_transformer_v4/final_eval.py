"""B/fit refit and B/validation evaluation on an explicit device.

`search.evaluate_final` is CPU-only and lives in a file whose hash is checked against the
run manifests, so this is a parallel implementation rather than an edit: same reference
class structure, same lr schedule, same epoch count taken from the inner selection.

The one thing that must not drift: **the epoch count is the inner-selected eligible epoch,
and the cosine `T_max` is the trial's full max-epoch budget**, exactly as in stage A. A
schedule shortened to the selected count would train B under a different curve than A and
make the two stages incomparable.
"""
from __future__ import annotations

import sys as _sys
import time
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import json
from pathlib import Path

import numpy as np
import torch

from .device import DeviceTensors, cpu_state_dict, predict_on
from .training import TrainConfig, build_for, metrics


def evaluate_final_on_device(candidate: dict, b_state, b_fit, b_val, run_dir: Path,
                             device: torch.device) -> dict:
    from .training import _epoch

    config = TrainConfig(**{k: v for k, v in candidate["config"].items()
                            if k in TrainConfig().__dict__})
    # keep the stage-A schedule: T_max stays the trial's max-epoch budget
    config.schedule_epochs = config.max_epochs
    epochs = int(candidate["selected_epoch"])

    started = time.perf_counter()
    model = build_for(b_state.numeric_names, b_state.cardinalities(),
                      context_dim=b_fit.query_context.shape[1], config=config)
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.schedule_length())

    dev_fit = DeviceTensors(b_fit, device)
    dev_val = DeviceTensors(b_val, device)
    curve = []
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        train_mse = _epoch(model, dev_fit, config, optimiser, epoch)
        scheduler.step()
        curve.append({"epoch": epoch, "train_mse": train_mse,
                      "seconds": round(time.perf_counter() - epoch_started, 2),
                      "learning_rate": float(optimiser.param_groups[0]["lr"])})

    predicted = b_val.to_sgd(predict_on(model, dev_val))
    overall = metrics(b_val.actual_sgd, predicted)

    out_dir = run_dir / f"final_trial{candidate['trial_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    description = build_for(b_state.numeric_names, b_state.cardinalities(),
                            context_dim=b_fit.query_context.shape[1], config=config)
    description.load_state_dict(cpu_state_dict(model))
    description = description.describe()

    torch.save({"config": config.to_dict(), "state_dict": cpu_state_dict(model),
                "state": b_state.to_dict(), "selected_epoch": epochs,
                "device": str(device), "description": description},
               out_dir / "checkpoint.pt")
    np.savez_compressed(out_dir / "validation_predictions.npz",
                        predicted_sgd=predicted, actual_sgd=b_val.actual_sgd,
                        row_id=b_val.row_id, horizon=b_val.horizon)

    summary = {
        "trial_id": candidate["trial_id"], "rank": candidate["rank"],
        "pooling": config.pooling, "epochs": epochs,
        "schedule_epochs_effective": config.schedule_length(),
        "overall": overall,
        "by_horizon": _by_horizon(b_val, predicted),
        "curve": curve, "parameters": description["parameters"],
        "config": config.to_dict(), "description": description,
        "device": str(device),
        "seconds": round(time.perf_counter() - started, 2),
        "prediction_rows": int(len(predicted)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _by_horizon(b_val, predicted: np.ndarray) -> list[dict]:
    rows = []
    horizon = np.asarray(b_val.horizon)
    actual = np.asarray(b_val.actual_sgd, dtype=np.float64)
    for low, high in ((1, 3), (4, 8), (9, 12), (13, 16)):
        mask = (horizon >= low) & (horizon <= high)
        if not mask.any():
            rows.append({"band": f"{low}-{high}", "n": 0})
            continue
        residual = predicted[mask] - actual[mask]
        rows.append({"band": f"{low}-{high}", "n": int(mask.sum()),
                     "rmse_sgd": float(np.sqrt(np.mean(residual ** 2))),
                     "mae_sgd": float(np.mean(np.abs(residual))),
                     "bias_sgd": float(np.mean(residual))})
    return rows


def verify_reload_on_cpu(out_dir: Path, b_state, b_val, device_used: str) -> dict:
    """Load the saved checkpoint on CPU and re-predict; must reproduce the export."""
    stored = torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = TrainConfig(**{k: v for k, v in stored["config"].items()
                            if k in TrainConfig().__dict__})
    model = build_for(b_state.numeric_names, b_state.cardinalities(),
                      context_dim=b_val.query_context.shape[1], config=config)
    model.load_state_dict(stored["state_dict"])
    model.eval()
    cpu_tensors = DeviceTensors(b_val, torch.device("cpu"))
    rebuilt = b_val.to_sgd(predict_on(model, cpu_tensors))
    exported = np.load(out_dir / "validation_predictions.npz")["predicted_sgd"]
    delta = float(np.abs(rebuilt - exported).max())
    result = {"ok": delta < 1e-4, "max_abs_delta_sgd": delta, "n": int(len(rebuilt)),
              "trained_on": device_used,
              "tensors_saved_on_cpu": all(v.device.type == "cpu"
                                          for v in stored["state_dict"].values())}
    (out_dir / "reload_check.json").write_text(json.dumps(result, indent=2))
    return result
