"""Bridge real macro artifacts into the v2 training loop.

Three jobs, each of which can go silently wrong and is therefore checked here:

1. **Row alignment.** `neural_fusion_v2` and `macro_encoders_v3` read the same supervised
   rows from two separately exported contracts. Positional concatenation would be an
   assumption, so every population is reindexed by `SOURCE_ROW_ID` and the result is
   asserted against the v2 row ids.

2. **Per-stage, fit-only scaling.** `fit_macro_scalers` is called with the stage's own
   *fit* population, so A/holdout and B/validation never influence a mean or a scale.

3. **Batching.** v2's `Tensors.batches` knows nothing about macro arrays. Rather than
   modify v2 (frozen), a `MacroTensors` wrapper yields the same batch dicts with the macro
   keys added, plus a row-offset vector so each batch can select its own macro rows.

The wrapper is a plain iterator, not a subclass, because v2's `Tensors` is a frozen
dataclass used by `predict`/`fit_*`; passing a duck-typed object keeps v2 untouched.

Macro tensors are stored once for the whole stage (`(n_rows, 63, 7)` for DBS,
`(n_rows, 12, 24)` for COE after the producer's category-month flattening) and sliced per
batch. They are not re-derived, resampled or aggregated here.
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

from neural_fusion_v2.adapter import Tensors

from .macro import MacroInvalid, MacroNotReady


class MacroAlignmentError(ValueError):
    """v2 and macro rows do not describe the same supervised population."""


@dataclass(frozen=True)
class StageMacroTensors:
    """Macro inputs for one stage, row-aligned with that stage's v2 populations."""

    split: str
    stage: str
    policy: str
    row_id: np.ndarray
    position: dict[int, int]           # SOURCE_ROW_ID -> row in these arrays
    dbs_seq: np.ndarray                # (n, 63, 7) scaled
    dbs_mask: np.ndarray               # (n, 63)
    coe_seq: np.ndarray                # (n, 12, 24) scaled
    coe_mask: np.ndarray               # (n, 12)
    window_index: np.ndarray           # (n,) which unique window each row uses
    macro_cutoff_month: np.ndarray     # (n,)
    scaler: object                     # macro_encoders_v3.scaling.MacroScalers
    scaler_fitted_on: str

    @property
    def n_rows(self) -> int:
        return len(self.row_id)

    def select(self, row_ids: np.ndarray) -> dict[str, torch.Tensor]:
        """Macro batch tensors for a set of supervised rows, in the given order."""
        ids = np.asarray(row_ids, dtype=np.int64)
        missing = [int(i) for i in ids if int(i) not in self.position]
        if missing:
            raise MacroAlignmentError(
                f"{self.split}/{self.stage}: {len(missing)} row ids absent from the macro "
                f"arrays, first few {missing[:5]}")
        index = np.fromiter((self.position[int(i)] for i in ids), dtype=np.int64, count=len(ids))
        return {
            "macro_row": torch.from_numpy(index),      # stage-level row positions
            "macro_dbs": torch.from_numpy(self.dbs_seq[index]),
            "macro_dbs_mask": torch.from_numpy(self.dbs_mask[index]),
            "macro_coe": torch.from_numpy(self.coe_seq[index]),
            "macro_coe_mask": torch.from_numpy(self.coe_mask[index]),
        }

    def describe(self) -> dict:
        return {
            "split": self.split, "stage": self.stage, "policy": self.policy,
            "n_rows": int(self.n_rows),
            "n_windows": int(len(np.unique(self.window_index))),
            "dbs_shape": list(self.dbs_seq.shape), "coe_shape": list(self.coe_seq.shape),
            "dbs_observed_fraction": round(float(self.dbs_mask.mean()), 6),
            "coe_observed_fraction": round(float(self.coe_mask.mean()), 6),
            "scaler_fitted_on": self.scaler_fitted_on,
        }


def build_stage_macro(manifest: dict, split: str, stage: str, scalers=None) -> StageMacroTensors:
    """Load, validate and scale the macro arrays for one stage.

    ``scalers``, when given, is used **as is**: a checkpoint reload must reproduce the
    scaling its training run used, so this path never refits. Passing ``None`` means
    "fit on this stage's own fit population", which is only correct for a fresh run.
    """
    from experiments.macro_encoders_v3 import api as macro_api
    from experiments.macro_encoders_v3 import data as macro_data
    from experiments.macro_encoders_v3 import scaling as macro_scaling

    try:
        windows = macro_data.load_windows(manifest, split, stage)
        rows = macro_data.load_rows(manifest, split, stage)
        if scalers is None:
            scalers = macro_scaling.fit_macro_scalers(windows, rows)
        tensors = macro_api.build_window_tensors(windows, scalers)
    except (MacroNotReady, MacroInvalid):
        raise
    except (ValueError, KeyError, AssertionError) as error:
        raise MacroInvalid(f"{split}/{stage}: macro artifacts unusable: {error}") from error

    # windows -> rows, using the producer's own per-row mapping (never regenerated here)
    index = np.asarray(rows.row_window_index, dtype=np.int64)
    dbs = tensors.dbs_seq.numpy()[index]
    dbs_mask = tensors.dbs_mask.numpy()[index]
    coe = tensors.coe_seq.numpy()[index]
    coe_mask = tensors.coe_mask.numpy()[index]

    if not np.isfinite(dbs).all() or not np.isfinite(coe).all():
        raise MacroInvalid(f"{split}/{stage}: non-finite macro values after scaling")
    padded = dbs_mask == 0
    if padded.any() and np.abs(dbs[padded]).max(initial=0.0) > 0:
        raise MacroInvalid(f"{split}/{stage}: padded DBS rows are non-zero after scaling")

    row_id = np.asarray(rows.row_id, dtype=np.int64)
    if len(np.unique(row_id)) != len(row_id):
        raise MacroInvalid(f"{split}/{stage}: duplicate SOURCE_ROW_ID in macro rows")
    position = {int(v): i for i, v in enumerate(row_id.tolist())}

    cutoff = np.asarray(rows.macro_cutoff_month, dtype=np.int32)
    return StageMacroTensors(
        split=split, stage=stage, policy=str(rows.policy), row_id=row_id, position=position,
        dbs_seq=dbs.astype(np.float32), dbs_mask=dbs_mask.astype(np.float32),
        coe_seq=coe.astype(np.float32), coe_mask=coe_mask.astype(np.float32),
        window_index=index, macro_cutoff_month=cutoff, scaler=scalers,
        scaler_fitted_on=str(getattr(scalers.dbs, "fitted_on", "unknown")))


def align_population(stage_macro: StageMacroTensors, population) -> np.ndarray:
    """Positions in ``stage_macro`` for a v2 population, in the population's own order."""
    ids = np.asarray(population.arrays.row_id, dtype=np.int64)
    absent = int(sum(1 for i in ids.tolist() if int(i) not in stage_macro.position))
    if absent:
        raise MacroAlignmentError(
            f"{population.split}/{population.stage}/{population.population}: {absent} of "
            f"{len(ids)} row ids are missing from the macro contract")
    return np.fromiter((stage_macro.position[int(i)] for i in ids), dtype=np.int64, count=len(ids))


class MacroTensors:
    """v2's `Tensors`, plus macro keys on every batch.

    Implements the same surface the training loop uses: ``__len__``, ``batches``,
    ``subset``, ``to_sgd``, ``actual_sgd``, ``anchor_ksgd``, ``parity_signature``.
    """

    def __init__(self, base: Tensors, row_ids: np.ndarray, stage_macro: StageMacroTensors):
        self.base = base
        self.row_ids = np.asarray(row_ids, dtype=np.int64)
        self.stage_macro = stage_macro
        if len(self.row_ids) != len(base):
            raise MacroAlignmentError(
                f"row id count {len(self.row_ids)} != tensors {len(base)}")

    def __len__(self) -> int:
        return len(self.base)

    @property
    def actual_sgd(self) -> np.ndarray:
        return self.base.actual_sgd

    @property
    def anchor_ksgd(self) -> np.ndarray:
        return self.base.anchor_ksgd

    def to_sgd(self, raw: np.ndarray) -> np.ndarray:
        return self.base.to_sgd(raw)

    def parity_signature(self) -> str:
        return self.base.parity_signature()

    def subset(self, index) -> "MacroTensors":
        index = np.asarray(index)
        return MacroTensors(self.base.subset(index), self.row_ids[index], self.stage_macro)

    def batches(self, batch_size: int, shuffle: bool = False, seed: int = 0,
                with_target: bool = False):
        """Same batches v2 would yield, with the macro keys added.

        The feature dict is built by v2's own `batches` on this exact index slice, so the
        tabular/sequence construction cannot drift from v2. Only the macro keys are new.
        """
        order = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, len(self), batch_size):
            index = order[start:start + batch_size]
            source = self.base.subset(index).batches(len(index), shuffle=False,
                                                    with_target=False)
            features = next(iter(source))
            features.update(self.stage_macro.select(self.row_ids[index]))
            if with_target:
                yield features, torch.from_numpy(self.base.target[index])
            else:
                yield features

    def describe(self) -> dict:
        return {"n_rows": len(self), "macro": self.stage_macro.describe()}
