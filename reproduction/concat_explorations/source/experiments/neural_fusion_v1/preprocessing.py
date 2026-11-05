"""Fit-only preprocessing for static (per-transaction) inputs and the horizon context.

Everything is fitted on the experiment's fitting rows only and exported, so a
reviewer can rebuild checkpoint inference without retraining.  No scikit-learn
import: the repository's ``.venv`` cannot import scipy (broken anaconda binary) and
the transforms are small enough to state explicitly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CAT_FEATURES, CONTEXT_DIM, INDICATOR_FEATURES, NUMERIC_FEATURES, TARGET, TARGET_SCALE


@dataclass
class StaticPreprocessor:
    numeric: list[str]
    median: dict[str, float]
    mean: dict[str, float]
    std: dict[str, float]
    categories: dict[str, list[str]]
    indicators: list[str]
    target_mean: float
    target_std: float
    fitted_rows: int = 0

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "StaticPreprocessor":
        numeric = NUMERIC_FEATURES
        values = frame[numeric].apply(pd.to_numeric, errors="coerce")
        median = {c: float(values[c].median()) for c in numeric}
        filled = values.fillna(median)
        mean = {c: float(filled[c].mean()) for c in numeric}
        std = {c: float(filled[c].std(ddof=0)) or 1.0 for c in numeric}
        categories = {
            c: sorted(frame[c].fillna("unknown").astype(str).unique().tolist()) for c in CAT_FEATURES
        }
        target = pd.to_numeric(frame[TARGET], errors="coerce")
        return cls(
            numeric=numeric, median=median, mean=mean, std=std, categories=categories,
            indicators=INDICATOR_FEATURES, target_mean=float(target.mean()),
            target_std=float(target.std(ddof=0)) or 1.0, fitted_rows=int(len(frame)),
        )

    # -- static numerics ---------------------------------------------------
    def numeric_matrix(self, frame: pd.DataFrame, clip: float = 8.0) -> np.ndarray:
        """Standardised numerics with missing indicators.

        ``clip`` bounds the standardised value.  A calendar index keeps growing past
        the fitting range, and an unbounded linear extrapolation is exactly how the
        naive MLP diverged in the probe run; clipping bounds the input while the
        anchor residual (see ``context_matrix``) carries the level.
        """
        values = frame[self.numeric].apply(pd.to_numeric, errors="coerce")
        blocks = []
        for column in self.numeric:
            missing = values[column].isna().to_numpy(dtype=np.float32)
            scaled = ((values[column].fillna(self.median[column]) - self.mean[column]) / self.std[column])
            scaled = np.clip(scaled.to_numpy(dtype=np.float32), -clip, clip)
            blocks.append(scaled[:, None])
            if column in self.indicators:
                blocks.append(missing[:, None])
        return np.concatenate(blocks, axis=1)

    @property
    def numeric_columns(self) -> list[str]:
        out: list[str] = []
        for column in self.numeric:
            out.append(column)
            if column in self.indicators:
                out.append(f"{column}__missing")
        return out

    def categorical_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for column in CAT_FEATURES:
            mapping = {name: i + 1 for i, name in enumerate(self.categories[column])}  # 0 = unknown
            values = frame[column].fillna("unknown").astype(str)
            columns.append(np.array([mapping.get(v, 0) for v in values], dtype=np.int64))
        return np.stack(columns, axis=1)

    def written(self, frame: pd.DataFrame) -> np.ndarray:
        """Target in kSGD, the units the networks minimise."""
        return pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(dtype=np.float32) / TARGET_SCALE

    def cardinalities(self) -> list[int]:
        return [len(self.categories[c]) + 1 for c in CAT_FEATURES]

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "numeric": self.numeric,
            "numeric_columns_expanded": self.numeric_columns,
            "median": self.median, "mean": self.mean, "std": self.std,
            "categories": self.categories, "indicators": self.indicators,
            "target_mean": self.target_mean, "target_std": self.target_std,
            "target_scale": TARGET_SCALE, "fitted_rows": self.fitted_rows,
            "clip": 8.0,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "StaticPreprocessor":
        return cls(
            numeric=payload["numeric"], median=payload["median"], mean=payload["mean"],
            std=payload["std"], categories=payload["categories"], indicators=payload["indicators"],
            target_mean=payload["target_mean"], target_std=payload["target_std"],
            fitted_rows=payload.get("fitted_rows", 0),
        )

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)


def context_matrix(horizon: np.ndarray, target_month: np.ndarray, summary: dict,
                   with_history: bool, kscale: float = TARGET_SCALE) -> np.ndarray:
    """Horizon/calendar + causal origin anchor. Shape (n, CONTEXT_DIM).

    Positions 9-10 are history-derived summaries and are zeroed for the static
    controls, so the ablation separates "knows the shared monthly history" from
    "knows where the group stood at the origin".
    """
    h = horizon.astype(np.float32)
    month = target_month.astype(np.float32) % 12
    anchor_rent = summary["anchor_rent_sgd"].astype(np.float32) / kscale
    anchor_psm = summary["anchor_psm_sgd"].astype(np.float32) / kscale
    out = np.stack([
        h / 12.0,
        np.log1p(h),
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        month / 12.0,
        anchor_rent,
        anchor_psm,
        np.log1p(summary["anchor_level"].astype(np.float32) + 1.0),
        summary["anchor_missing"].astype(np.float32),
        summary["history_months"].astype(np.float32) / 12.0,
        summary["exact_frac"].astype(np.float32),
    ], axis=1).astype(np.float32)
    if not with_history:
        out[:, 9:11] = 0.0
    if out.shape[1] != CONTEXT_DIM:
        raise AssertionError(f"context dim {out.shape[1]} != {CONTEXT_DIM}")
    return out
