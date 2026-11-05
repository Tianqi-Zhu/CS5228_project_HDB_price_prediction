"""The integration layer a consumer actually calls.

Two entry points, deliberately separated so the unique-window encoding cannot be
mistaken for a per-row loop:

``encode_windows(encoders, tensors)``
    Encodes the *unique* macro windows once.  Input tensors are
    ``(W, 63, 7)`` and ``(W, 12, 24)``.  Returns per-window tokens plus missing flags.

``gather_to_rows(window_tokens, row_window_index)``
    Expands per-window tokens to supervised rows by indexing.  This is a live-tensor
    gather: ``tokens[row_window_index]`` keeps the autograd graph, so a batch that
    touches 8,192 rows but only 39 windows produces gradients that accumulate into
    those 39 embeddings exactly once per optimiser step.  Nothing is detached and no
    embedding is cached between steps.

``macro_tokens_for_rows(...)``
    Convenience wrapper that does both and returns exactly the dict a fusion layer
    wants: tokens, masks, and the row/window provenance.

Typical call
------------
::

    manifest = load_manifest()
    windows  = load_windows(manifest, "orig", "B")
    rows     = load_rows(manifest, "orig", "B")
    scalers  = fit_macro_scalers(windows, rows)      # or MacroScalers.load(dir)
    batch    = build_window_tensors(windows, scalers)          # scaled, torch
    encoders = build_encoders()

    per_window = encode_windows(encoders, batch)
    per_row    = gather_to_rows(per_window, rows.row_window_index)

Shapes are stated at every step so a consumer can assert rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from experiments.macro_encoders_v3 import contract, data, encoders as encoder_module


@dataclass
class WindowTensors:
    """Scaled, masked torch tensors for the unique windows of one stage."""

    dbs_seq: torch.Tensor        # (W, 63, 7) float32
    dbs_mask: torch.Tensor       # (W, 63) float32, 1 = observed
    coe_seq: torch.Tensor        # (W, 12, 24) float32
    coe_mask: torch.Tensor       # (W, 12) float32, 1 = month observed
    coe_channel_mask: torch.Tensor  # (W, 12, 24) float32, 1 = that category-month
    window_cutoff_month: np.ndarray  # (W,) int32
    dbs_asof_date: np.ndarray    # (W,) int32 YYYYMMDD
    coe_month_ids: np.ndarray    # (W, 12) int32 YYYYMM
    policy: str
    contract_id: str

    @property
    def n_windows(self) -> int:
        return int(self.dbs_seq.shape[0])


def build_window_tensors(windows: data.MacroWindows,
                         scalers) -> WindowTensors:
    """Apply the fitted scalers and convert to torch without touching masks.

    Padding stays 0.0 in value space (the scaler writes 0 for masked cells) and the
    masks carry the truth.  No NaN is produced for a missing window: that is
    represented by an all-zero mask, and the encoder turns it into an explicit
    ``missing`` flag.
    """
    # Only the sequence blocks are scaled here.  The scalar blocks belong to the
    # tree/flat-feature path, not to these window tensors; scaling them here was
    # dead work and raised on the per-element scalar mask.  Consumers that want the
    # macro scalars use ``scaling.fit_macro_scalers`` directly.
    dbs_values = scalers.dbs.transform(windows.dbs_seq, windows.dbs_mask > 0)
    coe_values = scalers.coe.transform(windows.coe_months_seq,
                                       _coe_month_any(windows.coe_mask) > 0)

    # Per-month channel mask so a category that was not observed that month is
    # distinguishable from a genuine zero premium.
    coe_channel_mask = np.transpose(windows.coe_mask, (0, 2, 1))       # (W, month, cat)
    coe_channel_mask = np.repeat(coe_channel_mask, len(contract.COE_CHANNELS), axis=2)
    coe_channel_mask = coe_channel_mask.astype(np.float32)
    # A value that was masked in the scaler must also be masked here.
    coe_values = np.where(coe_channel_mask > 0, coe_values, 0.0).astype(np.float32)

    return WindowTensors(
        dbs_seq=torch.from_numpy(np.ascontiguousarray(dbs_values)),
        dbs_mask=torch.from_numpy(windows.dbs_mask.astype(np.float32)),
        coe_seq=torch.from_numpy(np.ascontiguousarray(coe_values)),
        coe_mask=torch.from_numpy(_coe_month_any(windows.coe_mask).astype(np.float32)),
        coe_channel_mask=torch.from_numpy(coe_channel_mask),
        window_cutoff_month=windows.window_cutoff_month,
        dbs_asof_date=windows.dbs_asof_date,
        coe_month_ids=windows.coe_month_ids,
        policy=windows.policy,
        contract_id=windows.contract_id,
    )


def _coe_month_any(coe_mask: np.ndarray) -> np.ndarray:
    """(W, cat, month) -> (W, month): 1 when any category was observed."""
    return coe_mask.max(axis=1)


def encode_windows(modules, tensors: WindowTensors) -> dict:
    """Encode each unique window once.

    Returns a dict of ``(W, ...)`` tensors.  Call this **once per optimiser step**
    with the current weights; do not cache the result across steps, or the encoder
    stops learning.
    """
    dbs_token, dbs_missing = modules.dbs(tensors.dbs_seq, tensors.dbs_mask)
    coe_token, coe_missing = modules.coe(tensors.coe_seq, tensors.coe_mask)
    if dbs_token.shape[0] != tensors.n_windows or coe_token.shape[0] != tensors.n_windows:
        raise AssertionError("encoder returned a different number of windows than it was given")
    return {
        "dbs_token": dbs_token,            # (W, 48)
        "dbs_missing": dbs_missing,        # (W, 1)
        "coe_token": coe_token,            # (W, 48)
        "coe_missing": coe_missing,        # (W, 1)
    }


def gather_to_rows(per_window: dict, row_window_index: np.ndarray,
                   row_ids: np.ndarray | None = None) -> dict:
    """Index per-window outputs down to supervised rows, preserving the graph.

    ``row_window_index`` must come from the same stage as the windows.  Row order is
    whatever the caller's row vector is; passing ``row_ids`` records it so a caller
    cannot silently reorder.
    """
    index = torch.as_tensor(np.asarray(row_window_index, dtype=np.int64))
    out = {}
    for key, value in per_window.items():
        if not torch.is_tensor(value):
            raise TypeError(f"{key} is not a tensor")
        out[key] = value.index_select(0, index)
    out["row_window_index"] = index
    if row_ids is not None:
        out["row_id"] = torch.as_tensor(np.asarray(row_ids, dtype=np.int64))
    return out


def macro_tokens_for_rows(modules, tensors: WindowTensors, row_window_index,
                          row_ids=None) -> dict:
    """Encode unique windows and gather to rows in one call."""
    return gather_to_rows(encode_windows(modules, tensors), row_window_index, row_ids)


def batch_indices(row_window_index: np.ndarray, row_index: np.ndarray,
                  n_windows: int) -> tuple[np.ndarray, np.ndarray]:
    """Map one batch of rows to the *unique* windows it needs.

    Returns ``(unique_window_ids, inverse)`` such that
    ``unique_window_ids[inverse] == row_window_index[row_index]``.  Encoding only
    these windows is what makes a deduplicated batch cheap; gradients still flow to
    every window the batch touches.
    """
    needed = np.asarray(row_window_index, dtype=np.int64)[np.asarray(row_index, dtype=np.int64)]
    unique, inverse = np.unique(needed, return_inverse=True)
    if unique.size and (unique.max() >= n_windows or unique.min() < 0):
        raise IndexError(f"window ids out of range for {n_windows} windows")
    return unique, inverse.astype(np.int64)


def select_window_tensors(tensors: WindowTensors, window_ids: np.ndarray) -> WindowTensors:
    """Subset the window tensors to a batch's needed windows."""
    ids = torch.as_tensor(np.asarray(window_ids, dtype=np.int64))
    return WindowTensors(
        dbs_seq=tensors.dbs_seq.index_select(0, ids),
        dbs_mask=tensors.dbs_mask.index_select(0, ids),
        coe_seq=tensors.coe_seq.index_select(0, ids),
        coe_mask=tensors.coe_mask.index_select(0, ids),
        coe_channel_mask=tensors.coe_channel_mask.index_select(0, ids),
        window_cutoff_month=tensors.window_cutoff_month[window_ids],
        dbs_asof_date=tensors.dbs_asof_date[window_ids],
        coe_month_ids=tensors.coe_month_ids[window_ids],
        policy=tensors.policy,
        contract_id=tensors.contract_id,
    )


def encode_batch(modules, tensors: WindowTensors, row_window_index,
                 row_index) -> dict:
    """Encode only the windows a row batch needs, then expand to those rows.

    This is the recommended training path: one forward pass over the batch's unique
    windows, then a live gather.  The encoder sees a variable number of windows per
    step, which is fine — gradient accumulation is over the windows this batch
    references, and a window referenced by several rows receives the summed gradient,
    as it should.
    """
    unique, inverse = batch_indices(row_window_index, row_index, tensors.n_windows)
    subset = select_window_tensors(tensors, unique)
    per_window = encode_windows(modules, subset)
    out = gather_to_rows(per_window, inverse)
    return out


def summarize_tokens(per_row: dict) -> dict:
    """Small JSON-able summary for a report: shapes, missing rates, value ranges."""
    summary = {}
    for key, value in per_row.items():
        if not torch.is_tensor(value):
            continue
        array = value.detach().cpu().numpy()
        entry = {"shape": list(array.shape), "dtype": str(array.dtype)}
        if key.endswith("_missing"):
            entry["rate"] = float(array.mean())
            entry["n_missing"] = int(array.sum())
        elif key.endswith("_token"):
            entry["min"] = float(np.min(array))
            entry["max"] = float(np.max(array))
            entry["mean"] = float(np.mean(array))
            entry["std"] = float(np.std(array))
        summary[key] = entry
    return summary
