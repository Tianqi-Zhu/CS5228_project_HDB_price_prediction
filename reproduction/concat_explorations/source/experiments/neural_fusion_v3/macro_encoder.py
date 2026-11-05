"""DBS and COE encoders: two independent small LSTMs producing one token each.

Axes are kept separate and explicit:

* **DBS** — 63 trading days x 7 channels, one LSTM over the day axis.
* **COE** — 12 calendar months x 6 channels, four categories. The category axis is *not* a
  time axis, so the COE encoder runs the month-LSTM per category and then pools the four
  category states with a learned attention-free mean over categories that are observed.
  Concatenating categories into a 48-step "time" axis would silently invent a timeline.

Both encoders consume **fit-population-scaled** inputs: `MacroScaler` is fitted on a
stage's `fit` rows only, so validation cannot influence the scale.

What the embedding means: it is learned through the rental loss. Nothing here supervises it
against GDP, a recession label or any economic state, so it must be described as a
*financial-state embedding*, never as a validated measure of economic health.
"""
from __future__ import annotations

import sys as _sys
from dataclasses import dataclass
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import numpy as np
import torch
from torch import nn

from neural_fusion_v2.settings import CAPACITY

from .macro import COE_CATEGORIES, COE_WINDOW_MONTHS, DBS_WINDOW_DAYS, StageMacro

CLIP_STD = 8.0


@dataclass
class MacroScaler:
    """Per-channel median/mean/std fitted on one stage's ``fit`` rows."""

    dbs_median: np.ndarray
    dbs_mean: np.ndarray
    dbs_std: np.ndarray
    coe_median: np.ndarray
    coe_mean: np.ndarray
    coe_std: np.ndarray
    dbs_scalar_median: np.ndarray
    dbs_scalar_mean: np.ndarray
    dbs_scalar_std: np.ndarray
    coe_scalar_median: np.ndarray
    coe_scalar_mean: np.ndarray
    coe_scalar_std: np.ndarray
    fitted_population: str
    fitted_rows: int
    clip_std: float = CLIP_STD

    @classmethod
    def fit(cls, stage: StageMacro, row_index: np.ndarray, population: str) -> "MacroScaler":
        """Fit on the given rows' windows only (a stage's ``fit`` population)."""
        windows = np.unique(stage.row_window_index[row_index])
        dbs = stage.dbs_seq[windows]
        dbs_mask = stage.dbs_mask[windows]
        coe = stage.coe_seq[windows]
        coe_mask = stage.coe_mask[windows]
        dbs_scalars = stage.dbs_scalars[windows]
        dbs_scalar_mask = stage.dbs_scalar_mask[windows]
        coe_scalars = stage.coe_scalars[windows]
        coe_scalar_mask = stage.coe_scalar_mask[windows]

        d_med, d_mu, d_sd = _fit_channels(dbs, dbs_mask == 1)
        c_med, c_mu, c_sd = _fit_channels(coe, coe_mask == 1)
        ds_med, ds_mu, ds_sd = _fit_channels(dbs_scalars, dbs_scalar_mask == 1)
        cs_med, cs_mu, cs_sd = _fit_channels(coe_scalars, coe_scalar_mask == 1)
        return cls(d_med, d_mu, d_sd, c_med, c_mu, c_sd, ds_med, ds_mu, ds_sd,
                   cs_med, cs_mu, cs_sd, fitted_population=population,
                   fitted_rows=int(len(row_index)))

    def transform(self, stage: StageMacro, row_index: np.ndarray) -> dict[str, np.ndarray]:
        """Gather and standardise; never refits."""
        gathered = stage.gather(row_index)
        dbs = _apply(gathered["dbs_seq"], self.dbs_median, self.dbs_mean, self.dbs_std,
                     self.clip_std, gathered["dbs_mask"])
        coe = _apply(gathered["coe_seq"], self.coe_median, self.coe_mean, self.coe_std,
                     self.clip_std, gathered["coe_mask"])
        dbs_scalars = _apply(gathered["dbs_scalars"], self.dbs_scalar_median,
                             self.dbs_scalar_mean, self.dbs_scalar_std, self.clip_std,
                             gathered["dbs_scalar_mask"])
        coe_scalars = _apply(gathered["coe_scalars"], self.coe_scalar_median,
                             self.coe_scalar_mean, self.coe_scalar_std, self.clip_std,
                             gathered["coe_scalar_mask"])
        return {
            "dbs_seq": dbs.astype(np.float32), "dbs_mask": gathered["dbs_mask"],
            "coe_seq": coe.astype(np.float32), "coe_mask": gathered["coe_mask"],
            "dbs_scalars": dbs_scalars.astype(np.float32),
            "coe_scalars": coe_scalars.astype(np.float32),
            "macro_cutoff_month": gathered["macro_cutoff_month"],
            "window_index": gathered["window_index"],
        }

    def to_dict(self) -> dict:
        return {
            "dbs_median": self.dbs_median.tolist(), "dbs_mean": self.dbs_mean.tolist(),
            "dbs_std": self.dbs_std.tolist(), "coe_median": self.coe_median.tolist(),
            "coe_mean": self.coe_mean.tolist(), "coe_std": self.coe_std.tolist(),
            "dbs_scalar_median": self.dbs_scalar_median.tolist(),
            "dbs_scalar_mean": self.dbs_scalar_mean.tolist(),
            "dbs_scalar_std": self.dbs_scalar_std.tolist(),
            "coe_scalar_median": self.coe_scalar_median.tolist(),
            "coe_scalar_mean": self.coe_scalar_mean.tolist(),
            "coe_scalar_std": self.coe_scalar_std.tolist(),
            "fitted_population": self.fitted_population,
            "fitted_rows": self.fitted_rows, "clip_std": self.clip_std,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MacroScaler":
        return cls(
            dbs_median=np.array(payload["dbs_median"]), dbs_mean=np.array(payload["dbs_mean"]),
            dbs_std=np.array(payload["dbs_std"]), coe_median=np.array(payload["coe_median"]),
            coe_mean=np.array(payload["coe_mean"]), coe_std=np.array(payload["coe_std"]),
            dbs_scalar_median=np.array(payload["dbs_scalar_median"]),
            dbs_scalar_mean=np.array(payload["dbs_scalar_mean"]),
            dbs_scalar_std=np.array(payload["dbs_scalar_std"]),
            coe_scalar_median=np.array(payload["coe_scalar_median"]),
            coe_scalar_mean=np.array(payload["coe_scalar_mean"]),
            coe_scalar_std=np.array(payload["coe_scalar_std"]),
            fitted_population=payload["fitted_population"],
            fitted_rows=payload["fitted_rows"], clip_std=payload.get("clip_std", CLIP_STD))


def _fit_channels(values: np.ndarray, mask: np.ndarray):
    """Median/mean/std per channel from masked positions; ``values`` may be 3-D or 4-D."""
    flat = values.reshape(-1, values.shape[-1])
    mask_flat = mask.reshape(-1)
    channels = values.shape[-1]
    median = np.zeros(channels, dtype=np.float32)
    mean = np.zeros(channels, dtype=np.float32)
    std = np.ones(channels, dtype=np.float32)
    observed = flat[mask_flat == 1]
    if observed.size == 0:
        return median, mean, std
    for j in range(channels):
        column = observed[:, j]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue
        m = float(np.median(finite))
        median[j] = m
        filled = np.where(np.isfinite(column), column, m)
        mean[j] = float(filled.mean())
        std[j] = float(filled.std(ddof=0)) or 1.0
    return median, mean, std


def _apply(values: np.ndarray, median, mean, std, clip: float, mask: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    for j in range(out.shape[-1]):
        column = out[..., j]
        filled = np.where(np.isfinite(column), column, float(median[j]))
        out[..., j] = np.clip((filled - float(mean[j])) / float(std[j]), -clip, clip)
    return out * np.expand_dims(mask, -1)      # padding stays exactly zero


class DBSEncoder(nn.Module):
    """One LSTM over 63 trading days -> one width-48 token."""

    name = "dbs"

    def __init__(self, channels: int, d_model: int = CAPACITY["d_model"],
                 hidden: int = CAPACITY["lstm_hidden"]):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, d_model)
        self.d_model = d_model
        self.n_steps = DBS_WINDOW_DAYS

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_input = seq * mask.unsqueeze(-1)
        hidden, _ = self.lstm(masked_input)
        hidden = hidden * mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = hidden.sum(dim=1) / denom
        return self.proj(pooled)


class COEEncoder(nn.Module):
    """Month-LSTM per category, then pooled across observed categories -> one token.

    A category is not a timestep: the LSTM runs over the 12-month axis for each of the four
    categories, and the four resulting states are averaged with the category mask. Turning
    categories into a 48-step sequence would invent a timeline that does not exist.
    """

    name = "coe"

    def __init__(self, channels: int, d_model: int = CAPACITY["d_model"],
                 hidden: int = CAPACITY["lstm_hidden"]):
        super().__init__()
        self.lstm = nn.LSTM(channels, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, d_model)
        self.d_model = d_model
        self.n_categories = len(COE_CATEGORIES)
        self.n_months = COE_WINDOW_MONTHS

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # seq: (B, categories, months, channels); mask: (B, categories, months)
        batch, categories, months, channels = seq.shape
        flat_seq = seq.reshape(batch * categories, months, channels)
        flat_mask = mask.reshape(batch * categories, months)
        masked_input = flat_seq * flat_mask.unsqueeze(-1)
        hidden, _ = self.lstm(masked_input)
        hidden = hidden * flat_mask.unsqueeze(-1)
        denom = flat_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        per_category = (hidden.sum(dim=1) / denom).reshape(batch, categories, -1)
        present = (flat_mask.sum(dim=1) > 0).reshape(batch, categories).float()
        weights = present / present.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (per_category * weights.unsqueeze(-1)).sum(dim=1)
        return self.proj(pooled)


class MacroBranch(nn.Module):
    """Holds the enabled encoders and emits zero, one or two tokens."""

    def __init__(self, dbs_channels: int, coe_channels: int, mode: str,
                 d_model: int = CAPACITY["d_model"]):
        super().__init__()
        if mode not in ("none", "dbs", "coe", "both"):
            raise ValueError(f"unknown macro mode {mode!r}")
        self.mode = mode
        self.dbs = DBSEncoder(dbs_channels, d_model) if mode in ("dbs", "both") else None
        self.coe = COEEncoder(coe_channels, d_model) if mode in ("coe", "both") else None
        self.d_model = d_model

    @property
    def n_tokens(self) -> int:
        return (1 if self.dbs is not None else 0) + (1 if self.coe is not None else 0)

    def forward(self, batch) -> list[torch.Tensor]:
        tokens: list[torch.Tensor] = []
        if self.dbs is not None:
            tokens.append(self.dbs(batch["macro_dbs"], batch["macro_dbs_mask"]).unsqueeze(1))
        if self.coe is not None:
            tokens.append(self.coe(batch["macro_coe"], batch["macro_coe_mask"]).unsqueeze(1))
        return tokens

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "tokens": self.n_tokens,
            "dbs": None if self.dbs is None else {
                "symbol": "D05.SI", "steps": DBS_WINDOW_DAYS,
                "hidden": self.dbs.lstm.hidden_size, "d_model": self.dbs.d_model,
                "axis": "trading_day"},
            "coe": None if self.coe is None else {
                "categories": list(COE_CATEGORIES), "steps": COE_WINDOW_MONTHS,
                "hidden": self.coe.lstm.hidden_size, "d_model": self.coe.d_model,
                "axis": "calendar_month",
                "category_handling": "per-category LSTM, masked mean over categories"},
            "semantics": "financial-state embedding learned through the rental loss; "
                         "not a validated measure of economic health",
        }
