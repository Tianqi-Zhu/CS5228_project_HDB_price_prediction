"""Training loop, inner chronological early stopping, metrics and prediction export."""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from config import HORIZON_BANDS, TARGET_SCALE
from data import Dataset
from models import build_model, count_parameters

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def set_seed(seed: int) -> None:
    """Seed every RNG that affects initialisation, shuffling or dropout."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainingConfig:
    model: str
    seed: int = 42
    epochs: int = 60
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    hidden: int = 128
    d_model: int = 48
    layers: int = 2
    heads: int = 4
    lstm_hidden: int = 32
    clip_grad: float = 1.0

    def model_kwargs(self) -> dict:
        if self.model == "mlp":
            return {"hidden": self.hidden}
        if self.model == "static_transformer":
            return {"d_model": self.d_model, "layers": self.layers, "heads": self.heads}
        if self.model == "lstm_concat":
            return {"hidden": self.hidden, "lstm_hidden": self.lstm_hidden}
        if self.model == "lstm_transformer":
            return {"d_model": self.d_model, "layers": self.layers, "heads": self.heads,
                    "lstm_hidden": self.lstm_hidden}
        raise KeyError(self.model)


@dataclass
class FitResult:
    model: nn.Module
    epochs_run: int
    best_epoch: int
    best_inner_rmse: float
    selected_inner_rmse: float
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


def _plot_curve(history: list[dict], path: Path, title: str) -> None:
    """Small dependency-free SVG curve (matplotlib is not installed)."""
    points = [(h["epoch"], h["inner_rmse_sgd"]) for h in history if np.isfinite(h["inner_rmse_sgd"])]
    if len(points) < 2:
        return
    width, height, pad = 480, 220, 34
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lo, hi = min(ys), max(ys)
    span = max(hi - lo, 1e-9)
    def sx(x): return pad + (x - min(xs)) / max(max(xs) - min(xs), 1) * (width - 2 * pad)
    def sy(y): return height - pad - (y - lo) / span * (height - 2 * pad)
    polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
    best = min(points, key=lambda p: p[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="100%" height="100%" fill="#fff"/>'
        f'<text x="{pad}" y="20" font-family="monospace" font-size="12" fill="#123B5D">{title}</text>'
        f'<polyline fill="none" stroke="#123B5D" stroke-width="1.6" points="{polyline}"/>'
        f'<circle cx="{sx(best[0]):.1f}" cy="{sy(best[1]):.1f}" r="3.2" fill="#EF7C00"/>'
        f'<text x="{pad}" y="{height - 8}" font-family="monospace" font-size="10" fill="#66717E">'
        f'epoch {min(xs)}-{max(xs)} | inner RMSE {lo:.1f}..{hi:.1f} | best {best[1]:.1f} @ ep{best[0]}</text>'
        "</svg>\n", encoding="utf-8")


def fit_epochs(model: nn.Module, dataset: Dataset, config: TrainingConfig, device: torch.device,
               epochs: int) -> float:
    """Train exactly ``epochs`` epochs; no validation, no checkpoint selection."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(epochs, 1))
    loss_fn = nn.MSELoss()
    model.to(device)
    last_loss = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        for features, target in dataset.batches(config.batch_size, shuffle=True,
                                                seed=config.seed * 1000 + epoch, with_target=True):
            features = {k: v.to(device) for k, v in features.items()}
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimiser.step()
            last_loss = float(loss.item())
        scheduler.step()
    return last_loss


def fit_early_stopping(model: nn.Module, train_ds: Dataset, inner_ds: Dataset,
                       config: TrainingConfig, device: torch.device, curve_path: Path | None = None,
                       title: str = "") -> FitResult:
    """Stage A: choose the epoch count on an inner chronological holdout inside fitting."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)
    loss_fn = nn.MSELoss()
    model.to(device)

    best = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history: list[dict] = []
    started = time.perf_counter()
    actual = inner_ds.actual_sgd()

    for epoch in range(1, config.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        for features, target in train_ds.batches(config.batch_size, shuffle=True,
                                                 seed=config.seed * 1000 + epoch, with_target=True):
            features = {k: v.to(device) for k, v in features.items()}
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimiser.step()
            total += float(loss.item()) * len(target)
            seen += len(target)
        scheduler.step()

        predicted = inner_ds.residual_to_sgd(predict(model, inner_ds, device, config.batch_size))
        inner_rmse = metrics(actual, predicted)["rmse_sgd"]
        history.append({"epoch": epoch, "train_mse_ksgd2": total / max(seen, 1), "inner_rmse_sgd": inner_rmse})

        if inner_rmse < best - 1e-6:
            best, best_epoch, stale = inner_rmse, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if curve_path is not None:
        _plot_curve(history, curve_path, title or config.model)
    return FitResult(model=model, epochs_run=len(history), best_epoch=best_epoch,
                     best_inner_rmse=best, selected_inner_rmse=best, history=history,
                     seconds=time.perf_counter() - started)


def final_inner_rmse(model: nn.Module, inner_ds: Dataset, device: torch.device, batch_size: int) -> float:
    if inner_ds is None or len(inner_ds) == 0:
        return float("nan")
    predicted = inner_ds.residual_to_sgd(predict(model, inner_ds, device, batch_size))
    return metrics(inner_ds.actual_sgd(), predicted)["rmse_sgd"]


@torch.no_grad()
def predict(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for batch in dataset.batches(batch_size, shuffle=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        out.append(model(batch).detach().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def band_report(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for low, high in HORIZON_BANDS:
        block = frame.loc[frame["horizon_months"].between(low, high)]
        if block.empty:
            rows.append({"horizon_band": f"{low}-{high}", "n": 0, "rmse_sgd": None,
                         "mae_sgd": None, "bias_sgd": None, "note": "empty band"})
            continue
        rows.append({"horizon_band": f"{low}-{high}", **metrics(block["actual_rent_sgd"].to_numpy(),
                                                               block["predicted_rent_sgd"].to_numpy())})
    return rows


def monthly_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for month, block in frame.groupby("RENT_APPROVAL_DATE", sort=True):
        rows.append({"RENT_APPROVAL_DATE": month, **metrics(block["actual_rent_sgd"].to_numpy(),
                                                           block["predicted_rent_sgd"].to_numpy())})
    return pd.DataFrame(rows)


def predictions_frame(source: pd.DataFrame, predicted_sgd: np.ndarray, *, split_id: str, cutoff_month: str,
                      model_name: str, seed: int) -> pd.DataFrame:
    """One row per source row, in ascending SOURCE_ROW_ID order."""
    frame = pd.DataFrame({
        "SOURCE_ROW_ID": source["SOURCE_ROW_ID"].to_numpy(),
        "split_id": split_id,
        "cutoff_month": cutoff_month,
        "horizon_months": source["HORIZON"].to_numpy(dtype=np.int32),
        "RENT_APPROVAL_DATE": source["RENT_APPROVAL_DATE"].to_numpy(),
        "actual_rent_sgd": source["MONTHLY_RENT"].to_numpy(dtype=np.float64),
        "predicted_rent_sgd": np.asarray(predicted_sgd, dtype=np.float64),
        "model": model_name,
        "seed": int(seed),
    })
    ordered = frame.sort_values("SOURCE_ROW_ID", kind="stable").reset_index(drop=True)
    if len(ordered) != len(source):
        raise AssertionError("Prediction rows lost during export")
    return ordered


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def build_and_count(model_name: str, cardinalities: list[int], group_indices: list[list[int]],
                    context_dim: int, config: TrainingConfig,
                    group_names: dict[str, list[str]] | None = None) -> tuple[nn.Module, int]:
    kwargs = config.model_kwargs()
    if group_names is not None:
        kwargs = {**kwargs, "group_names": group_names}
    model = build_model(model_name, cardinalities, group_indices, context_dim, **kwargs)
    return model, count_parameters(model)
