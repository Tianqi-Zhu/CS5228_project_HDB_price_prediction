"""Training, seeding, metrics and prediction export for the Gate 1 controls.

Stage A chooses the epoch count: fit on ``A/fit``, stop on ``A/holdout``.
Stage B refits on ``B/fit`` for exactly that many epochs and keeps the last weights, so
no holdout or validation label can influence which weights are exported.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .adapter import Tensors
from .models import build_control, parameter_count
from .settings import CAPACITY, TRAIN

torch.set_num_threads(TRAIN["threads"])
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class FitConfig:
    control: str
    seed: int = TRAIN["seed"]
    epochs: int = TRAIN["max_epochs"]
    patience: int = TRAIN["patience"]
    batch_size: int = TRAIN["batch_size"]
    learning_rate: float = TRAIN["learning_rate"]
    weight_decay: float = TRAIN["weight_decay"]
    clip_grad: float = TRAIN["clip_grad"]

    def actual_manifest(self, shapes) -> dict:
        return {
            "control": self.control,
            "capacity": dict(CAPACITY),
            "shapes": {
                "n_categorical": shapes.n_categorical, "static_dim": shapes.static_dim,
                "history_dim": shapes.history_dim, "context_dim": shapes.context_dim,
                "cardinalities": list(shapes.cardinalities),
                "sequence_channels": shapes.sequence_channels,
                "sequence_length": shapes.sequence_length,
            },
            "optimiser": {"name": "AdamW", "lr": self.learning_rate,
                          "weight_decay": self.weight_decay, "clip_grad": self.clip_grad,
                          "scheduler": "CosineAnnealingLR"},
            "batch_size": self.batch_size, "seed": self.seed,
            "max_epochs": self.epochs, "patience": self.patience,
            "threads": torch.get_num_threads(),
            "loss": "MSE on kSGD residual from the contract anchor (anchor kept in SGD)",
            "early_stopping": "A/fit train, A/holdout monitor",
            "refit": "B/fit, fixed epoch count, last weights kept",
        }


@dataclass
class FitResult:
    model: nn.Module
    epochs_run: int
    best_epoch: int
    best_inner_rmse: float
    history: list[dict] = field(default_factory=list)
    seconds: float = 0.0


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "rmse_sgd": float(np.sqrt(np.mean(np.square(error)))),
        "mae_sgd": float(np.mean(np.abs(error))),
        "bias_sgd": float(np.mean(error)),
        "n": int(error.size),
    }


@torch.no_grad()
def predict(model: nn.Module, tensors: Tensors, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for batch in tensors.batches(batch_size, shuffle=False):
        out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def _epoch(model, tensors, config, optimiser, scheduler, epoch: int) -> float:
    loss_fn = nn.MSELoss()
    model.train(True)
    total, seen = 0.0, 0
    for features, target in tensors.batches(config.batch_size, shuffle=True,
                                            seed=config.seed * 1000 + epoch, with_target=True):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss for {config.control}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimiser.step()
        total += float(loss.item()) * len(target)
        seen += len(target)
    scheduler.step()
    return total / max(seen, 1)


def fit_early_stopping(model: nn.Module, train: Tensors, holdout: Tensors,
                       config: FitConfig) -> FitResult:
    import time

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history: list[dict] = []
    started = time.perf_counter()
    actual = holdout.actual_sgd

    for epoch in range(1, config.epochs + 1):
        train_loss = _epoch(model, train, config, optimiser, scheduler, epoch)
        holdout_rmse = metrics(actual, holdout.to_sgd(predict(model, holdout)))["rmse_sgd"]
        history.append({"epoch": epoch, "train_mse_ksgd2": train_loss,
                        "holdout_rmse_sgd": holdout_rmse})
        if holdout_rmse < best - 1e-6:
            best, best_epoch, stale = holdout_rmse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return FitResult(model=model, epochs_run=len(history), best_epoch=best_epoch,
                     best_inner_rmse=best, history=history,
                     seconds=time.perf_counter() - started)


def fit_fixed_epochs(model: nn.Module, train: Tensors, config: FitConfig, epochs: int) -> float:
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    last = float("nan")
    for epoch in range(1, epochs + 1):
        last = _epoch(model, train, config, optimiser, scheduler, epoch)
    return last


def band_report(frame: pd.DataFrame, bands=((1, 3), (4, 8), (9, 12), (13, 16))) -> list[dict]:
    rows = []
    for low, high in bands:
        block = frame.loc[frame["horizon_months"].between(low, high)]
        if block.empty:
            rows.append({"horizon_band": f"{low}-{high}", "n": 0, "rmse_sgd": None,
                         "mae_sgd": None, "bias_sgd": None, "note": "empty band"})
        else:
            rows.append({"horizon_band": f"{low}-{high}",
                         **metrics(block["actual_rent_sgd"].to_numpy(),
                                   block["predicted_rent_sgd"].to_numpy())})
    return rows


def monthly_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, block in frame.groupby("target_month", sort=True):
        rows.append({"target_month": int(month),
                     **metrics(block["actual_rent_sgd"].to_numpy(),
                               block["predicted_rent_sgd"].to_numpy())})
    return pd.DataFrame(rows)


def predictions_frame(population, predicted_sgd: np.ndarray, *, control: str, seed: int) -> pd.DataFrame:
    samples = population.samples
    frame = pd.DataFrame({
        "SOURCE_ROW_ID": samples["SOURCE_ROW_ID"].to_numpy(),
        "split": samples["split_id"].to_numpy(),
        "stage": population.stage,
        "population": population.population,
        "cutoff_month": samples["origin_month_index"].to_numpy(),
        "horizon_months": samples["horizon_months"].to_numpy(),
        "target_month": samples["target_month_index"].to_numpy(),
        "actual_rent_sgd": samples["actual_rent_sgd"].to_numpy(dtype=np.float64),
        "predicted_rent_sgd": np.asarray(predicted_sgd, dtype=np.float64),
        "model": control,
        "seed": int(seed),
    })
    if len(frame) != len(samples):
        raise AssertionError("prediction rows lost during export")
    return frame


def save_json(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def build(control: str, shapes):
    model = build_control(control, shapes)
    return model, parameter_count(model)
