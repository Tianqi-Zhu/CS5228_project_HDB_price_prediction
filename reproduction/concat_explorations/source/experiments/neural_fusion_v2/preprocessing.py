"""Per-stage imputation/scaling state, fitted on a stage's ``fit`` population only.

The real contract ships raw-scale scalars: ``FLOOR_AREA_SQM`` up to 215,
``anchor_rent_sgd`` in the thousands, ``history`` counts up to ~1e5, plus NaNs in
``YEARS_SINCE_LEASE_COMMENCEMENT`` / ``MAX_FLOOR`` / ``BUILDING_AGE_YEARS`` and in a few
history columns. Feeding that straight into an MLP/Transformer is not trainable, so this
module derives a state from one population and applies it — unchanged — to every other
population of the same stage.

Rules enforced here:

1. ``fit`` looks only at the rows it is given. The caller passes ``stageA/fit`` or
   ``stageB/fit`` and nothing else.
2. ``transform`` never mutates the state. Applying it to the holdout or validation
   population cannot move a median, mean or standard deviation.
3. Categorical codes are passed through as integer indices; the vocabulary size comes
   from ``encoding.json`` and is asserted to cover the observed codes.
4. The anchor is handled separately: it stays in **SGD** for bookkeeping and is only
   divided by 1000 where a tensor needs kSGD. The anchor's definition is not changed.
5. ``context`` is scaled with fit-population statistics too (anchor column excluded and
   carried through in kSGD). No block is standardised with the rows being transformed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .contract import Population, StageArrays

TARGET_SCALE = 1000.0
ANCHOR_SGD = "anchor_rent_sgd"
LOG1P_COLUMNS = {
    "hist_n__global", "hist_n__flat_type", "hist_n__town_ft", "hist_n__block_ft",
    "hist_months__global", "hist_months__flat_type", "hist_months__town_ft", "hist_months__block_ft",
}
CLIP_STD = 8.0


@dataclass
class ScalerState:
    """Reproducible per-column state for the numeric part of ``static`` and ``history``."""

    static_columns: list[str]
    static_median: dict[str, float]
    static_mean: dict[str, float]
    static_std: dict[str, float]
    history_columns: list[str]
    history_median: dict[str, float]
    history_mean: dict[str, float]
    history_std: dict[str, float]
    context_columns: list[str]           # all context columns except the anchor
    context_median: dict[str, float]
    context_mean: dict[str, float]
    context_std: dict[str, float]
    log1p_columns: list[str]
    clip_std: float
    fitted_population: str
    fitted_rows: int

    # -- fit ---------------------------------------------------------------
    @classmethod
    def fit(cls, arrays: StageArrays, population: str) -> "ScalerState":
        numeric_cols = arrays.numeric_columns
        static = arrays.static[:, len(arrays.categorical_columns):].astype(np.float64)
        history = arrays.history.astype(np.float64)
        static_median, static_mean, static_std = _fit_block(static, numeric_cols)
        history_cols = list(arrays.history_order)
        log_cols = [c for c in history_cols if c in LOG1P_COLUMNS]
        history_filled = history.copy()
        for column in log_cols:
            j = history_cols.index(column)
            history_filled[:, j] = np.log1p(np.clip(history_filled[:, j], 0, None))
        history_median, history_mean, history_std = _fit_block(history_filled, history_cols)

        # context: everything except the anchor, fitted on the same fit population
        anchor_idx = arrays.anchor_index
        context_cols = [c for i, c in enumerate(arrays.context_order) if i != anchor_idx]
        context_block = np.delete(arrays.context.astype(np.float64), anchor_idx, axis=1)
        context_median, context_mean, context_std = _fit_block(context_block, context_cols)

        return cls(
            static_columns=numeric_cols, static_median=static_median,
            static_mean=static_mean, static_std=static_std,
            history_columns=history_cols, history_median=history_median,
            history_mean=history_mean, history_std=history_std,
            context_columns=context_cols, context_median=context_median,
            context_mean=context_mean, context_std=context_std,
            log1p_columns=log_cols, clip_std=CLIP_STD,
            fitted_population=population, fitted_rows=int(len(arrays.static)),
        )

    # -- transform ---------------------------------------------------------
    def transform(self, arrays: StageArrays) -> dict[str, np.ndarray]:
        """Standardise the numeric blocks; returns float32 arrays and keeps the anchor in SGD."""
        n_cat = len(arrays.categorical_columns)
        static_numeric = arrays.static[:, n_cat:].astype(np.float64)
        history = arrays.history.astype(np.float64)
        for column in self.log1p_columns:
            j = self.history_columns.index(column)
            history[:, j] = np.log1p(np.clip(history[:, j], 0, None))

        static_scaled = _apply_block(static_numeric, self.static_columns,
                                     self.static_median, self.static_mean, self.static_std,
                                     self.clip_std)
        history_scaled = _apply_block(history, self.history_columns,
                                      self.history_median, self.history_mean, self.history_std,
                                      self.clip_std)

        anchor_idx = arrays.anchor_index
        anchor_sgd = arrays.context[:, anchor_idx].astype(np.float64)
        # context is transformed with the fit-population state; never refitted here
        context_raw = np.delete(arrays.context.astype(np.float64), anchor_idx, axis=1)
        context_scaled = _apply_block(context_raw, self.context_columns,
                                      self.context_median, self.context_mean,
                                      self.context_std, self.clip_std)
        anchor_ksgd = (anchor_sgd / TARGET_SCALE).astype(np.float32)

        return {
            "categorical": arrays.static[:, :n_cat].astype(np.int64),
            "static": static_scaled.astype(np.float32),
            "history": history_scaled.astype(np.float32),
            "context": context_scaled.astype(np.float32),
            "anchor_sgd": anchor_sgd.astype(np.float32),
            "anchor_ksgd": anchor_ksgd,
            "context_columns_scaled": list(self.context_columns),
            "sequence": arrays.sequence.astype(np.float32),
            "sequence_mask": arrays.sequence_mask.astype(np.float32),
        }

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "static_columns": self.static_columns, "static_median": self.static_median,
            "static_mean": self.static_mean, "static_std": self.static_std,
            "history_columns": self.history_columns, "history_median": self.history_median,
            "history_mean": self.history_mean, "history_std": self.history_std,
            "context_columns": self.context_columns, "context_median": self.context_median,
            "context_mean": self.context_mean, "context_std": self.context_std,
            "log1p_columns": self.log1p_columns, "clip_std": self.clip_std,
            "fitted_population": self.fitted_population, "fitted_rows": self.fitted_rows,
            "anchor_handling": f"{ANCHOR_SGD} kept in SGD for bookkeeping; /1000 only in tensors",
            "target_scale": TARGET_SCALE,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ScalerState":
        return cls(
            static_columns=payload["static_columns"], static_median=payload["static_median"],
            static_mean=payload["static_mean"], static_std=payload["static_std"],
            history_columns=payload["history_columns"], history_median=payload["history_median"],
            history_mean=payload["history_mean"], history_std=payload["history_std"],
            context_columns=payload["context_columns"], context_median=payload["context_median"],
            context_mean=payload["context_mean"], context_std=payload["context_std"],
            log1p_columns=payload["log1p_columns"], clip_std=payload["clip_std"],
            fitted_population=payload["fitted_population"], fitted_rows=payload["fitted_rows"],
        )

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))


def _fit_block(values: np.ndarray, columns: list[str]) -> tuple[dict, dict, dict]:
    median, mean, std = {}, {}, {}
    for j, column in enumerate(columns):
        column_values = values[:, j]
        finite = column_values[np.isfinite(column_values)]
        if finite.size == 0:
            median[column], mean[column], std[column] = 0.0, 0.0, 1.0
            continue
        m = float(np.median(finite))
        median[column] = m
        filled = np.where(np.isfinite(column_values), column_values, m)
        mu = float(filled.mean())
        sigma = float(filled.std(ddof=0)) or 1.0
        mean[column], std[column] = mu, sigma
    return median, mean, std


def _apply_block(values: np.ndarray, columns: list[str], median: dict, mean: dict,
                 std: dict, clip: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    for j, column in enumerate(columns):
        column_values = values[:, j]
        filled = np.where(np.isfinite(column_values), column_values, median[column])
        scaled = (filled - mean[column]) / std[column]
        out[:, j] = np.clip(scaled, -clip, clip)
    return out


def categorical_cardinalities(encoding: dict, arrays: StageArrays) -> list[int]:
    """Vocabulary sizes from the producer's encoding.json, checked against observed codes."""
    vocab = encoding.get("categorical_vocabularies", {})
    sizes = []
    for i, column in enumerate(arrays.categorical_columns):
        observed = int(arrays.static[:, i].max()) + 1
        if column in vocab:
            declared = len(vocab[column])
            if declared < observed:
                raise ValueError(
                    f"{column}: encoding.json declares {declared} categories but codes reach "
                    f"{observed - 1}; the encoding does not cover the data")
            sizes.append(max(declared, observed))
        else:
            sizes.append(observed)
    return sizes
