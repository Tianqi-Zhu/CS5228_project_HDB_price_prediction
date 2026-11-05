"""Assemble the 56 numeric + 4 categorical tensors for the v4 Transformer.

This module deliberately **does not** re-implement scaling. `neural_fusion_v2.preprocessing`
already encodes three things that are easy to get subtly wrong, and each was wrong in an
earlier draft of this file:

* eight ``hist_n__*`` / ``hist_months__*`` columns are ``log1p``-transformed before
  standardising; skipping that leaves heavy-tailed counts dominated by outliers;
* the anchor stays in **SGD** for bookkeeping and is divided by 1000 only where a tensor
  needs kSGD, so ``anchor_ksgd * 1000 == anchor_sgd``;
* categorical codes are passed through as integer indices, with code 0 reserved for unknown.

So `StageState` wraps a v2 `ScalerState` and adds exactly what the v4 model needs beyond it:
a per-column vocabulary for the four categorical columns, fitted on the stage's `fit`
population only.

Numeric positions, in model order:

    0..20   static spatial/property numeric  (contract `numeric_columns`, 21)
    21..44  history scalar summaries         (contract `history_order`, 24)
    45..55  context excluding the anchor     (contract `context_order` minus anchor, 11)

The anchor is not a feature token: it is the additive baseline, and the model predicts
``actual_sgd/1000 - anchor_ksgd``.
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

from neural_fusion_v2.preprocessing import TARGET_SCALE, ScalerState

N_CATEGORICAL = 4
N_NUMERIC = 56
UNKNOWN_CODE = 0


class FeatureDataError(ValueError):
    pass


@dataclass(frozen=True)
class StageState:
    """A v2 `ScalerState` plus per-column categorical vocabularies."""

    scaler: ScalerState
    vocabularies: dict[str, list[int]]
    categorical_names: list[str]
    fitted_population: str
    n_rows: int

    @property
    def numeric_names(self) -> list[str]:
        return (list(self.scaler.static_columns) + list(self.scaler.history_columns)
                + list(self.scaler.context_columns))

    def cardinalities(self) -> tuple[int, ...]:
        return tuple(len(self.vocabularies[name]) for name in self.categorical_names)

    def to_dict(self) -> dict:
        return {
            "scaler": self.scaler.to_dict(),
            "vocabularies": {k: list(v) for k, v in self.vocabularies.items()},
            "categorical_names": list(self.categorical_names),
            "fitted_population": self.fitted_population, "n_rows": self.n_rows,
            "n_numeric_positions": len(self.numeric_names),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "StageState":
        return cls(
            scaler=ScalerState.from_dict(payload["scaler"]),
            vocabularies={k: [int(c) for c in v]
                          for k, v in payload["vocabularies"].items()},
            categorical_names=list(payload["categorical_names"]),
            fitted_population=payload["fitted_population"], n_rows=int(payload["n_rows"]))


def fit_stage_state(population, population_label: str) -> StageState:
    """Fit scalers and vocabularies on one population. Only fit populations call this."""
    arrays = population.arrays
    scaler = ScalerState.fit(arrays, population=population_label)
    names = (list(scaler.static_columns) + list(scaler.history_columns)
             + list(scaler.context_columns))
    if len(names) != N_NUMERIC:
        raise FeatureDataError(f"expected {N_NUMERIC} numeric columns, got {len(names)}")
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise FeatureDataError(f"numeric names collide across blocks: {duplicates}")

    codes = np.asarray(arrays.static[:, :N_CATEGORICAL], dtype=np.int64)
    vocabularies: dict[str, list[int]] = {}
    for i, name in enumerate(arrays.categorical_columns):
        present = sorted({int(v) for v in codes[:, i].tolist()})
        # Code 0 is a *legal input value* meaning "no category recorded" (e.g. missing), and
        # it maps to the unknown slot. It must never receive a positive index of its own:
        # that would make a missing value indistinguishable from a real category.
        real_codes = [code for code in present if code != UNKNOWN_CODE]
        vocabularies[name] = [UNKNOWN_CODE] + real_codes
        if len(set(vocabularies[name])) != len(vocabularies[name]):
            raise FeatureDataError(f"{name}: duplicate codes in vocabulary")

    return StageState(scaler=scaler, vocabularies=vocabularies,
                      categorical_names=list(arrays.categorical_columns),
                      fitted_population=population_label, n_rows=len(population))


def apply_state(population, state: StageState) -> dict:
    """Standardise and encode with an already-fitted state. Never refits."""
    arrays = population.arrays
    transformed = state.scaler.transform(arrays)

    incoming = list(transformed.get("context_columns_scaled", []))
    expected = list(state.scaler.context_columns)
    if incoming and incoming != expected:
        raise FeatureDataError(f"context column order changed: {incoming} vs {expected}")

    numeric = np.concatenate([transformed["static"], transformed["history"],
                              transformed["context"]], axis=1)
    if numeric.shape[1] != N_NUMERIC:
        raise FeatureDataError(
            f"assembled {numeric.shape[1]} numeric columns, expected {N_NUMERIC}")
    if not np.isfinite(numeric).all():
        raise FeatureDataError(
            f"{population.split}/{population.stage}/{population.population}: "
            "non-finite numeric features after scaling")

    raw_codes = np.asarray(transformed["categorical"], dtype=np.int64)
    codes = np.zeros_like(raw_codes)
    for i, name in enumerate(state.categorical_names):
        lookup = {code: index for index, code in enumerate(state.vocabularies[name])
                  if index > 0}
        codes[:, i] = [lookup.get(int(value), UNKNOWN_CODE)
                       for value in raw_codes[:, i].tolist()]

    anchor_sgd = np.asarray(transformed["anchor_sgd"], dtype=np.float64)
    anchor_ksgd = np.asarray(transformed["anchor_ksgd"], dtype=np.float64)
    if not np.allclose(anchor_ksgd * TARGET_SCALE, anchor_sgd, rtol=0, atol=1e-3):
        raise FeatureDataError("anchor_ksgd does not reconstruct anchor_sgd")

    actual_sgd = population.actual_sgd
    residual = (actual_sgd / TARGET_SCALE - anchor_ksgd).astype(np.float32)
    if not np.isfinite(residual).all():
        raise FeatureDataError(
            f"{population.split}/{population.stage}/{population.population}: "
            "non-finite residual target")

    return {
        "numeric": numeric.astype(np.float32),
        "categorical": codes,
        "query_context": transformed["context"].astype(np.float32),
        "anchor_ksgd": anchor_ksgd,
        "anchor_sgd": anchor_sgd,
        "actual_sgd": actual_sgd.astype(np.float64),
        "target": residual,
        "row_id": np.asarray(arrays.row_id, dtype=np.int64),
        "horizon": np.asarray(arrays.horizon, dtype=np.int32),
    }


@dataclass(frozen=True)
class FeatureTensors:
    """Model-ready tensors for one population."""

    numeric: np.ndarray
    categorical: np.ndarray
    query_context: np.ndarray
    target: np.ndarray
    actual_sgd: np.ndarray
    anchor_ksgd: np.ndarray
    row_id: np.ndarray
    horizon: np.ndarray

    def __len__(self) -> int:
        return len(self.target)

    def subset(self, index) -> "FeatureTensors":
        index = np.asarray(index)
        return FeatureTensors(numeric=self.numeric[index], categorical=self.categorical[index],
                             query_context=self.query_context[index],
                             target=self.target[index], actual_sgd=self.actual_sgd[index],
                             anchor_ksgd=self.anchor_ksgd[index], row_id=self.row_id[index],
                             horizon=self.horizon[index])

    def batches(self, batch_size: int, shuffle: bool = False, seed: int = 0,
                with_target: bool = False):
        order = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, len(self), batch_size):
            index = order[start:start + batch_size]
            features = {
                "numeric": torch.from_numpy(self.numeric[index]),
                "categorical": torch.from_numpy(self.categorical[index]),
                "query_context": torch.from_numpy(self.query_context[index]),
            }
            if with_target:
                yield features, torch.from_numpy(self.target[index])
            else:
                yield features

    def to_sgd(self, raw: np.ndarray) -> np.ndarray:
        """Residual predictions (kSGD) -> rent in SGD."""
        return (np.asarray(raw, dtype=np.float64) + self.anchor_ksgd) * TARGET_SCALE

    def anchor_only_sgd(self) -> np.ndarray:
        """What a zero-residual model scores: the pure anchor, in SGD."""
        return self.anchor_ksgd * TARGET_SCALE

    def parity_signature(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        for name, array in (("numeric", self.numeric), ("categorical", self.categorical),
                            ("query_context", self.query_context), ("target", self.target),
                            ("anchor", self.anchor_ksgd), ("actual", self.actual_sgd),
                            ("row_id", self.row_id), ("horizon", self.horizon)):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
        return digest.hexdigest()


def build_tensors(population, state: StageState) -> FeatureTensors:
    payload = apply_state(population, state)
    return FeatureTensors(numeric=payload["numeric"], categorical=payload["categorical"],
                          query_context=payload["query_context"], target=payload["target"],
                          actual_sgd=payload["actual_sgd"],
                          anchor_ksgd=payload["anchor_ksgd"], row_id=payload["row_id"],
                          horizon=payload["horizon"])
