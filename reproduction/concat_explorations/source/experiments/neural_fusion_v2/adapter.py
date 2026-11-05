"""Turn a contract population into model tensors.

Parity guarantee: every control receives the same ``static``, ``history``, ``context``,
``categorical``, ``target`` and ``anchor`` tensors. Only ``sequence`` is extra, and only
the sequence models receive it.

Sequence shape from the producer is ``(n, levels, months, channels)`` — four fallback
levels (global / flat_type / town_ft / block_ft) x 12 months x 6 channels. Pooled
sequence models flatten levels and months into one step per (month, level), ordered
month-major so the flattened axis increases chronologically; masked steps are skipped.

Feature columns are selected by name from the producer's declared feature order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .contract import ANCHOR_SGD, Population
from .preprocessing import ScalerState

TARGET_SCALE = 1000.0


class AdapterError(ValueError):
    pass


@dataclass(frozen=True)
class Tensors:
    """Model-ready tensors for one population."""

    static: np.ndarray             # (n, k_static) standardised numeric block
    history: np.ndarray            # (n, k_hist) standardised smoothed history
    context: np.ndarray            # (n, k_ctx) standardised, anchor excluded
    anchor_ksgd: np.ndarray        # (n,) anchor in kSGD for the residual target
    anchor_sgd: np.ndarray         # (n,) anchor in SGD, bookkeeping only
    categorical: np.ndarray        # (n, 4) integer codes
    target: np.ndarray             # (n,) residual kSGD
    actual_sgd: np.ndarray         # (n,) actual rent, SGD
    sequence: np.ndarray | None    # (n, T, C) with T = months * levels
    sequence_mask: np.ndarray | None
    columns: dict[str, list[str]]

    def __len__(self) -> int:
        return len(self.target)

    def subset(self, index) -> "Tensors":
        index = np.asarray(index)
        return Tensors(
            static=self.static[index], history=self.history[index], context=self.context[index],
            anchor_ksgd=self.anchor_ksgd[index], anchor_sgd=self.anchor_sgd[index],
            categorical=self.categorical[index], target=self.target[index],
            actual_sgd=self.actual_sgd[index],
            sequence=None if self.sequence is None else self.sequence[index],
            sequence_mask=None if self.sequence_mask is None else self.sequence_mask[index],
            columns=self.columns)

    def batches(self, batch_size: int, shuffle: bool = False, seed: int = 0,
                with_target: bool = False):
        order = np.arange(len(self))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, len(self), batch_size):
            index = order[start:start + batch_size]
            features = {
                "static": torch.from_numpy(self.static[index]),
                "history": torch.from_numpy(self.history[index]),
                "context": torch.from_numpy(self.context[index]),
                "categorical": torch.from_numpy(self.categorical[index]),
            }
            if self.sequence is not None:
                features["sequence"] = torch.from_numpy(self.sequence[index])
                features["sequence_mask"] = torch.from_numpy(self.sequence_mask[index])
            if with_target:
                yield features, torch.from_numpy(self.target[index])
            else:
                yield features

    def to_sgd(self, raw: np.ndarray) -> np.ndarray:
        return (np.asarray(raw, dtype=np.float64) + self.anchor_ksgd) * TARGET_SCALE

    def parity_signature(self) -> str:
        """Hash of every input all controls must share (sequence deliberately excluded)."""
        import hashlib

        digest = hashlib.sha256()
        for name, array in (("static", self.static), ("history", self.history),
                            ("context", self.context), ("categorical", self.categorical),
                            ("target", self.target), ("anchor_ksgd", self.anchor_ksgd),
                            ("actual", self.actual_sgd)):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(array, dtype=np.float32).tobytes())
        return digest.hexdigest()


def flatten_sequence(sequence: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, levels, months, channels) -> (n, months*levels, channels), month-major."""
    if sequence.ndim != 4:
        raise AdapterError(f"sequence must be 4-D (n, levels, months, channels); got {sequence.shape}")
    n, levels, months, channels = sequence.shape
    ordered = np.transpose(sequence, (0, 2, 1, 3))          # n, months, levels, channels
    flat = ordered.reshape(n, months * levels, channels)
    mask_ordered = np.transpose(mask, (0, 2, 1))            # n, months, levels
    flat_mask = mask_ordered.reshape(n, months * levels)
    if not np.isfinite(flat).all():
        raise AdapterError("non-finite values in sequence")
    if np.abs(flat[flat_mask == 0]).max(initial=0.0) > 0:
        raise AdapterError("padded sequence positions are non-zero")
    return flat.astype(np.float32), flat_mask.astype(np.float32)


def build_tensors(population: Population, state: ScalerState, *, use_sequence: bool) -> Tensors:
    transformed = state.transform(population.arrays)
    anchor_sgd = transformed["anchor_sgd"].astype(np.float64)
    actual = population.actual_sgd
    residual = (actual / TARGET_SCALE - transformed["anchor_ksgd"]).astype(np.float32)
    if not np.isfinite(residual).all():
        raise AdapterError(
            f"{population.split}/{population.stage}/{population.population}: non-finite residual")

    sequence = sequence_mask = None
    if use_sequence:
        sequence, sequence_mask = flatten_sequence(population.arrays.sequence,
                                                   population.arrays.sequence_mask)

    return Tensors(
        static=transformed["static"], history=transformed["history"],
        context=transformed["context"], anchor_ksgd=transformed["anchor_ksgd"],
        anchor_sgd=anchor_sgd, categorical=transformed["categorical"],
        target=residual, actual_sgd=actual, sequence=sequence, sequence_mask=sequence_mask,
        columns={
            "static": list(state.static_columns),
            "history": list(state.history_columns),
            "context": list(state.context_columns),
            "categorical": list(population.arrays.categorical_columns),
            "anchor": [ANCHOR_SGD],
            "sequence_channels": list(population.arrays.sequence_order[:population.arrays.n_channels]),
        },
    )
