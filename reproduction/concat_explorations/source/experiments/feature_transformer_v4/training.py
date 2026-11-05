"""Training loop for the v4 feature Transformer.

Two rules are enforced structurally rather than by discipline:

* **The A and B learning-rate schedules are identical.** A cosine schedule is built from a
  fixed ``schedule_epochs`` (the trial's max epochs), never from the selected epoch count.
  An earlier design that set ``T_max`` to the chosen epoch would silently train B with a
  different schedule than A, making the two stages incomparable.
* **Epoch selection cannot see B.** `fit` takes a single monitor population and returns the
  curve; nothing here knows that a validation split exists.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from .data import FeatureTensors
from .models import build_model

TRAIN_THREADS = 2


@dataclass
class TrainConfig:
    """All settings that can change a number. Serialised into every run's manifest."""

    pooling: str = "mean"
    d_model: int = 48
    layers: int = 2
    heads: int = 4
    ff_multiplier: int = 2
    dropout: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    clip_grad: float = 1.0
    max_epochs: int = 60
    patience: int = 10
    eligible_after: int = 5
    seed: int = 42
    schedule_epochs: int | None = None      # defaults to max_epochs; never to a chosen count

    def schedule_length(self) -> int:
        return int(self.schedule_epochs or self.max_epochs)

    def to_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["schedule_epochs_effective"] = self.schedule_length()
        return payload


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residual = np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return {
        "rmse_sgd": float(np.sqrt(np.mean(residual ** 2))),
        "mae_sgd": float(np.mean(np.abs(residual))),
        "bias_sgd": float(np.mean(residual)),
        "n": int(residual.size),
    }


def predict(model: nn.Module, tensors: FeatureTensors, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for batch in tensors.batches(batch_size, shuffle=False):
            out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


@dataclass
class EpochRecord:
    epoch: int
    train_mse: float
    monitor_rmse_sgd: float
    seconds: float
    learning_rate: float


@dataclass
class FitResult:
    model: nn.Module
    curve: list[EpochRecord]
    epochs_run: int
    selected_epoch: int | None        # eligible best; None if no eligible epoch improved
    selected_rmse: float | None
    ineligible_best: dict | None      # the best epoch if it was before `eligible_after`
    seconds: float
    config: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(), "epochs_run": self.epochs_run,
            "selected_epoch": self.selected_epoch, "selected_rmse_sgd": self.selected_rmse,
            "ineligible_best": self.ineligible_best,
            "curve": [r.__dict__ for r in self.curve], "seconds": round(self.seconds, 2),
        }


def _epoch(model, tensors, config, optimiser, epoch: int) -> float:
    loss_fn = nn.MSELoss()
    model.train(True)
    total, seen = 0.0, 0
    for features, target in tensors.batches(config.batch_size, shuffle=True,
                                            seed=config.seed * 1000 + epoch,
                                            with_target=True):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model(features), target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optimiser.step()
        total += float(loss.item()) * len(target)
        seen += len(target)
    return total / max(seen, 1)


def fit(train: FeatureTensors, monitor: FeatureTensors, config: TrainConfig,
        model: nn.Module | None = None, log: Callable | None = None) -> FitResult:
    """Train with epoch selection over *eligible* epochs only.

    ``config.eligible_after`` epochs run before any checkpoint may be selected, but every
    epoch is recorded — including a better early one, which is reported as
    ``ineligible_best`` rather than being silently used.
    """
    set_seed(config.seed)
    if model is None:
        raise ValueError("fit needs a constructed model (see build_for)")
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.schedule_length())

    curve: list[EpochRecord] = []
    best_state, best_epoch, best_rmse = None, None, float("inf")
    best_any_state, best_any_epoch, best_any_rmse = None, None, float("inf")
    stale = 0
    started = time.perf_counter()
    actual = monitor.actual_sgd

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_mse = _epoch(model, train, config, optimiser, epoch)
        current_lr = float(optimiser.param_groups[0]["lr"])
        predicted = monitor.to_sgd(predict(model, monitor))
        monitor_rmse = metrics(actual, predicted)["rmse_sgd"]
        scheduler.step()
        curve.append(EpochRecord(epoch=epoch, train_mse=train_mse,
                                 monitor_rmse_sgd=monitor_rmse,
                                 seconds=round(time.perf_counter() - epoch_started, 2),
                                 learning_rate=current_lr))
        if log is not None:
            log(curve[-1])

        if monitor_rmse < best_any_rmse - 1e-9:
            best_any_rmse, best_any_epoch = monitor_rmse, epoch
            best_any_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch >= config.eligible_after:
            if monitor_rmse < best_rmse - 1e-9:
                best_rmse, best_epoch = monitor_rmse, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif best_any_state is not None:
        # never happened to improve after the eligibility floor: keep the last weights and
        # report the situation instead of quietly selecting an ineligible epoch
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    ineligible = None
    if best_any_epoch is not None and (best_epoch is None or best_any_epoch < best_epoch):
        ineligible = {"epoch": best_any_epoch, "rmse_sgd": best_any_rmse,
                      "note": "better than the selected epoch but before the eligibility floor"}
        if best_epoch is None:
            model.load_state_dict(best_any_state)

    return FitResult(model=model, curve=curve, epochs_run=len(curve),
                     selected_epoch=best_epoch, selected_rmse=None if best_epoch is None
                     else best_rmse, ineligible_best=ineligible,
                     seconds=time.perf_counter() - started, config=config)


def build_for(numeric_names, cardinalities, context_dim: int, config: TrainConfig):
    set_seed(config.seed)
    return build_model(numeric_names, cardinalities, context_dim=context_dim,
                       pooling=config.pooling, d_model=config.d_model,
                       layers=config.layers, heads=config.heads,
                       ff_multiplier=config.ff_multiplier, dropout=config.dropout)
