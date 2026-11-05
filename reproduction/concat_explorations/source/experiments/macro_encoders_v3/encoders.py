"""DBS and COE sequence encoders producing one 48-d token each.

Architecture
------------
Two independent small recurrent encoders, deliberately *not* one shared trunk so
that a DBS-only and a COE-only arm stay identifiable:

* **DBS**: 63 trading rows x 7 channels -> LSTM(hidden 32) -> last valid state
  -> Linear(32 -> 48) -> token.  Sequence order is oldest -> newest, matching the
  exported tensor.
* **COE**: 12 months x 24 channel values (4 categories x 6 statistics, flattened as
  *channels of the same month*) -> GRU(hidden 32) -> last valid state
  -> Linear(32 -> 48) -> token.  The month is the recurrent axis.  The four
  categories are never laid end to end along time; doing so would turn a 12-month
  history into 48 pseudo-months.

Missingness
-----------
A window with no usable observation at all is reported through an explicit
per-row ``*_missing`` mask (1.0 = missing) alongside a finite token.  Returning NaN
would poison every downstream tensor through the residual path, and a zero token
would be indistinguishable from a real embedding of zero.  The token content in the
missing case is the projection of a learned "empty" vector, and the flag tells the
fusion layer to ignore it.

Training pattern
----------------
``encode_windows`` works on the *unique* window tensors.  ``gather_to_rows`` maps the
per-window tokens onto supervised rows.  The gather is a plain indexing operation on
a live tensor, so gradients accumulate into the shared window embeddings normally.
Nothing is detached and no cached embedding is reused across optimiser steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from experiments.macro_encoders_v3 import contract


def compact_masked(sequence: torch.Tensor, mask: torch.Tensor):
    """Gather the *valid* steps of each window, preserving their real order.

    ``pack_padded_sequence`` assumes the valid steps form a prefix, but the exported
    windows are **leading-padded** (an early window is 43 zeros then 20 real trading
    days) and can contain interior holes.  Passing such a mask straight to ``pack``
    makes the recurrence read padding and drop the real tail.

    This gathers every position where ``mask == 1`` per window, in original order,
    into a dense tensor with a matching length vector:

    * ``sequence`` (B, L, C), ``mask`` (B, L) -> ``(B, L, C)`` compacted values
      (positions beyond a window's length are zero and excluded by ``lengths``)
      and ``lengths`` (B,) as CPU int64.
    * A window with no valid step gets ``lengths == 1`` so the packed call stays
      well-formed; the caller substitutes the learned empty vector for it.
    """
    if sequence.dim() != 3:
        raise ValueError(f"sequence must be (B, L, C), got {tuple(sequence.shape)}")
    if mask.shape != sequence.shape[:2]:
        raise ValueError(f"mask {tuple(mask.shape)} does not match sequence "
                         f"{tuple(sequence.shape[:2])}")
    binary = (mask > 0)
    lengths = binary.sum(dim=1).to(torch.long)

    # Order-preserving compaction: sort by (row, retain-first, position).
    # Positions to keep keep their original sequence order, padding moves to the end.
    positions = torch.arange(sequence.shape[1], device=sequence.device)
    keep = binary
    sort_key = torch.where(keep, positions.unsqueeze(0), positions.unsqueeze(0) + sequence.shape[1])
    order = torch.argsort(sort_key, dim=1, stable=True)
    compacted = torch.gather(
        sequence, 1, order.unsqueeze(-1).expand(-1, -1, sequence.shape[-1]))
    compacted_mask = torch.gather(binary, 1, order).to(dtype=sequence.dtype)
    compacted = compacted * compacted_mask.unsqueeze(-1)
    return compacted, compacted_mask, lengths.clamp(min=1)


def _recurrent_last_state(rnn: nn.RNNBase, sequence: torch.Tensor,
                          mask: torch.Tensor):
    """Run ``rnn`` over the compacted valid steps; return the last real state.

    An ``LSTM`` returns ``(h_n, c_n)`` while a ``GRU`` returns ``h_n``; taking
    ``state[-1]`` on the tuple would silently return the *cell* state and produce a
    token tensor with a leading layer dimension.  The two cases are separated here.
    """
    compacted, compacted_mask, safe_lengths = compact_masked(sequence, mask)
    packed = nn.utils.rnn.pack_padded_sequence(
        compacted, safe_lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, state = rnn(packed)
    if isinstance(state, tuple):          # LSTM: (h_n, c_n)
        hidden = state[0]
    else:                                  # GRU / RNN: h_n
        hidden = state
    return hidden[-1], compacted_mask.sum(dim=1) > 0


class DBSEncoder(nn.Module):
    """63 trading rows -> one 48-d token (plus an explicit missing flag)."""

    def __init__(self, hidden_dim: int = contract.HIDDEN_DIM,
                 token_dim: int = contract.TOKEN_DIM,
                 num_layers: int = 1, dropout: float = 0.0,
                 channels: int = len(contract.DBS_CHANNELS)):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.channels = channels
        self.lstm = nn.LSTM(input_size=channels, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.projection = nn.Linear(hidden_dim, token_dim)
        # Learned representation for "no observations at all".
        self.empty = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor):
        """``sequence`` (B, 63, 7); ``mask`` (B, 63) with 1 = real observation.

        Returns ``(token (B, token_dim), missing (B, 1))`` where ``missing == 1``
        means the window carried no real observation.
        """
        if sequence.dim() != 3 or sequence.shape[-1] != self.channels:
            raise ValueError(f"DBS sequence must be (B, L, {self.channels}), "
                             f"got {tuple(sequence.shape)}")
        if mask.shape != sequence.shape[:2]:
            raise ValueError(f"DBS mask {tuple(mask.shape)} does not match "
                             f"sequence {tuple(sequence.shape)}")
        mask = mask.to(dtype=sequence.dtype)

        # Compacts to the genuinely observed steps in their real order, so leading
        # padding and interior holes cannot displace the tail.  Fully missing
        # windows still run one step and are then replaced by the learned vector.
        hidden, has_any = _recurrent_last_state(self.lstm, sequence, mask)
        hidden = torch.where(has_any.unsqueeze(-1), hidden, self.empty.unsqueeze(0))

        token = self.projection(hidden)
        missing = (~has_any).to(dtype=sequence.dtype).unsqueeze(-1)
        return token, missing


class COEEncoder(nn.Module):
    """12 months x 24 channels -> one 48-d token (plus an explicit missing flag)."""

    def __init__(self, hidden_dim: int = contract.HIDDEN_DIM,
                 token_dim: int = contract.TOKEN_DIM,
                 num_layers: int = 1, dropout: float = 0.0,
                 channels: int = contract.COE_FLAT_CHANNELS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        self.channels = channels
        self.gru = nn.GRU(input_size=channels, hidden_size=hidden_dim,
                          num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.projection = nn.Linear(hidden_dim, token_dim)
        self.empty = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor):
        """``sequence`` (B, 12, 24); ``mask`` (B, 12) with 1 = month observed.

        A month counts as observed when *any* category was observed that month;
        the per-category detail is in the 24 channels and in ``month_channel_mask``.
        """
        if sequence.dim() != 3 or sequence.shape[-1] != self.channels:
            raise ValueError(f"COE sequence must be (B, L, {self.channels}), "
                             f"got {tuple(sequence.shape)}")
        if mask.shape != sequence.shape[:2]:
            raise ValueError(f"COE mask {tuple(mask.shape)} does not match "
                             f"sequence {tuple(sequence.shape)}")
        mask = mask.to(dtype=sequence.dtype)
        hidden, has_any = _recurrent_last_state(self.gru, sequence, mask)
        hidden = torch.where(has_any.unsqueeze(-1), hidden, self.empty.unsqueeze(0))
        token = self.projection(hidden)
        missing = (~has_any).to(dtype=sequence.dtype).unsqueeze(-1)
        return token, missing


class MacroEncoderSet(nn.Module):
    """Both encoders, so ``state_dict``/``train()``/``eval()`` cover the whole branch.

    ``dbs`` and ``coe`` stay separately addressable so a DBS-only or COE-only
    ablation can freeze or skip the other branch.  Deliberately *not* a dataclass:
    a dataclass-generated ``__eq__``/``__hash__`` breaks ``nn.Module`` bookkeeping.
    """

    def __init__(self, dbs: DBSEncoder, coe: COEEncoder):
        super().__init__()
        self.dbs = dbs
        self.coe = coe

    def forward(self, dbs_sequence, dbs_mask, coe_sequence, coe_mask):
        dbs_token, dbs_missing = self.dbs(dbs_sequence, dbs_mask)
        coe_token, coe_missing = self.coe(coe_sequence, coe_mask)
        return {
            "dbs_token": dbs_token,
            "dbs_missing": dbs_missing,
            "coe_token": coe_token,
            "coe_missing": coe_missing,
        }


def build_encoders(*, hidden_dim: int = contract.HIDDEN_DIM,
                   token_dim: int = contract.TOKEN_DIM,
                   dbs_layers: int = 1, coe_layers: int = 1,
                   use_coe: bool = True) -> MacroEncoderSet:
    """Construct both encoders.  ``use_coe=False`` builds a DBS-only set."""
    dbs = DBSEncoder(hidden_dim=hidden_dim, token_dim=token_dim, num_layers=dbs_layers)
    # A DBS-only ablation still needs a COE module object so state_dict keys line up
    # across runs; it is simply never called.
    coe = COEEncoder(hidden_dim=hidden_dim, token_dim=token_dim, num_layers=coe_layers)
    return MacroEncoderSet(dbs=dbs, coe=coe)


def parameter_count(module: nn.Module) -> dict:
    """Explicit parameter counts so a report cannot mis-state the architecture."""
    return {
        "total": sum(p.numel() for p in module.parameters()),
        "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
        "by_submodule": {
            name: sum(p.numel() for p in child.parameters())
            for name, child in module.named_children()
        },
    }
