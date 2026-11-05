"""Block-endpoint macro tokens with explicit calendar age, and the corrected mask path.

## What this package changes and what it must not

The v4 sources stay frozen; this is a parallel wrapper. Two things it exists to get right:

**1. The prefix-mask defect.** `feature_transformer_v4.macro_v4` builds its mask as
`cat([zeros(leading + n_macro), key_padding_mask])` when `leading == 1`. The real sequence is
`[query, 60 features, n_macro macros]`, so that zero block has the right *width* but the wrong
*position*: every padding flag ends up shifted by `n_macro`. Measured on a row whose last five
features are padding, the mask marked positions 58-62 instead of 56-60. Nothing caught it
because the v4 macro path never passes a mask at all, so the existing s42/s43 numbers are not
touched by it — but any arm that needs the mask (empty blocks, which is exactly what v5 has)
would inherit it.

**2. Age is calendar, not ordinal.** The 63-day DBS window becomes three 21-row blocks and the
12-month COE window becomes four 3-month blocks. The naive implementation is
`hidden[:, k]` on a padded batch, which shifts every token to the wrong age as soon as a window
has leading padding or an interior hole. Here each token's position is derived from
`cumsum(mask)` — the count of *observed* steps up to and including each block end — so padding
cannot move it. The relative age (1 = newest block) is passed as its own embedding so the model
can tell "this is 42 days old" from "this is the newest thing I have".

**Causality.** Only observed prefix steps affect a prefix token: the recurrent scan is causal,
and a block-end token is read from the state produced by exactly the steps up to that block end.
Nothing after the block end can influence it.
"""
from __future__ import annotations

import sys as _sys
from dataclasses import dataclass
from pathlib import Path as _Path

import numpy as np
import torch
from torch import nn

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

DBS_BLOCK_ROWS = 21          # 63 trading rows -> 3 blocks
COE_BLOCK_MONTHS = 3         # 12 months -> 4 blocks
TOKEN_DIM = 48


class MacroTimeTokenError(RuntimeError):
    pass


# --------------------------------------------------------------------------- masks
def build_encoder_mask(key_padding_mask: torch.Tensor | None, leading: int,
                       n_macro: int) -> torch.Tensor | None:
    """Position-correct mask for `[leading..., features..., macros...]`.

    This is the fix for the v4 defect. The feature mask goes where the features are; the
    query positions and the macro positions are never padding *unless* the caller says a
    macro token is missing, which is a separate mask and arrives already aligned.
    """
    if key_padding_mask is None:
        return None
    if key_padding_mask.dim() != 2:
        raise MacroTimeTokenError(f"key_padding_mask must be 2-D, got "
                                  f"{tuple(key_padding_mask.shape)}")
    batch, n_features = key_padding_mask.shape
    parts = []
    if leading:
        parts.append(torch.zeros(batch, leading, dtype=torch.bool,
                                 device=key_padding_mask.device))
    parts.append(key_padding_mask)
    if n_macro:
        parts.append(torch.zeros(batch, n_macro, dtype=torch.bool,
                                 device=key_padding_mask.device))
    result = torch.cat(parts, dim=1)
    expected = leading + n_features + n_macro
    if result.shape[1] != expected:
        raise MacroTimeTokenError(f"mask has {result.shape[1]} positions, "
                                  f"sequence has {expected}")
    return result


def combine_masks(*masks: torch.Tensor | None) -> torch.Tensor | None:
    """Union of several 2-D padding masks; `True` means padding."""
    live = [m for m in masks if m is not None]
    if not live:
        return None
    out = live[0].clone()
    for m in live[1:]:
        if m.shape != out.shape:
            raise MacroTimeTokenError(f"mask shape {tuple(m.shape)} != {tuple(out.shape)}")
        out = out | m
    return out


# ---------------------------------------------------------------------- block ends
def block_end_offsets(observed: torch.Tensor, block_rows: int, n_blocks: int,
                      return_observed_index: bool = False):
    """Locate each block's endpoint by *calendar row* first, then resolve the observation.

    `observed` is (B, L) with 1 for a real observation, in calendar order (oldest first, newest
    last). Block ``k`` (0 = oldest) covers calendar rows ``[k*block_rows, (k+1)*block_rows)``,
    so its endpoint is the last **calendar** row of that block — 21, 42, 63 for the DBS window
    and 3, 6, 9, 12 for COE. Those calendar rows are fixed regardless of where the holes are.

    Only after the calendar row is fixed does the observation matter: the token for that block
    is the state after the last observed step at or before that calendar row. That ordering is
    what keeps age correct. Indexing compacted positions directly by 21/42/63 would instead
    land on an arbitrary observation and silently re-age the token (and lose the tail when the
    window has leading padding).

    Returns ``(end_row, valid, age)``:

    * ``end_row`` (B, n_blocks) — 1-based calendar row index of the block's last row, i.e. the
      21/42/63 (DBS) or 3/6/9/12 (COE) endpoints. Constant per block, never shifted by holes.
    * ``valid`` (B, n_blocks) — that block contained at least one observation.
    * ``age`` (B, n_blocks) — ``n_blocks - 1`` is the newest block.

    With ``return_observed_index=True`` a fourth tensor is returned: the 1-based count of
    observed steps at or before ``end_row``, which is the index into the per-step state tensor
    produced by :func:`observed_step_states`. A block with no observation inherits the previous
    count and is flagged invalid rather than duplicating a later state.
    """
    if observed.dim() != 2:
        raise MacroTimeTokenError(f"observed must be (B, L), got {tuple(observed.shape)}")
    batch, length = observed.shape
    if length % block_rows:
        raise MacroTimeTokenError(f"window length {length} is not a multiple of "
                                  f"{block_rows}")
    if length // block_rows != n_blocks:
        raise MacroTimeTokenError(f"window length {length} gives "
                                  f"{length // block_rows} blocks, expected {n_blocks}")

    cumulative = torch.cumsum((observed > 0).to(torch.long), dim=1)      # (B, L)
    # the calendar row of each block end: 21/42/63 for DBS, 3/6/9/12 for COE
    end_row = torch.arange(block_rows, block_rows * n_blocks + 1, block_rows,
                           device=observed.device, dtype=torch.long)
    end_row = end_row.unsqueeze(0).expand(batch, -1).contiguous()

    observed_index = torch.zeros(batch, n_blocks, dtype=torch.long,
                                 device=observed.device)
    valid = torch.zeros(batch, n_blocks, dtype=torch.bool, device=observed.device)
    previous = torch.zeros(batch, dtype=torch.long, device=observed.device)
    for k in range(n_blocks):
        calendar_row = k * block_rows + block_rows        # 1-based, = end_row[:, k]
        here = cumulative[:, calendar_row - 1]
        valid[:, k] = here > previous
        observed_index[:, k] = torch.where(valid[:, k], here, previous)
        previous = observed_index[:, k]

    age = torch.arange(n_blocks - 1, -1, -1, device=observed.device,
                       dtype=torch.long).unsqueeze(0).expand(batch, -1)
    if return_observed_index:
        return end_row, valid, age, observed_index
    return end_row, valid, age


def observed_step_states(rnn: nn.RNNBase, sequence: torch.Tensor,
                         observed: torch.Tensor) -> torch.Tensor:
    """Run `rnn` over the observed steps only and return the per-step states.

    The returned tensor is in **observed-step space**, not calendar space: position ``j`` holds
    the state after the ``j+1``-th observation, in calendar order. That is the space
    ``block_end_offsets`` reports its endpoints in (`cumsum` of the observed flag), so a caller
    indexes this directly as ``states[b, end_index - 1]``.

    An earlier version scattered these states back to calendar positions and then indexed them
    with cumulative counts. On a fully observed window the two orders coincide and the mistake
    is invisible; on a sparse window it silently selected a padding position, so the recurrent
    output never entered the loss and the RNN received exactly zero gradient while the
    projection and embeddings still trained. The gradient test below uses a sparse window
    precisely because a dense one cannot catch it.
    """
    if sequence.dim() != 3:
        raise MacroTimeTokenError(f"sequence must be (B, L, C), got {tuple(sequence.shape)}")
    if observed.shape != sequence.shape[:2]:
        raise MacroTimeTokenError(f"observed {tuple(observed.shape)} does not match "
                                  f"sequence {tuple(sequence.shape[:2])}")
    binary = observed > 0
    lengths = binary.sum(dim=1).to(torch.long).clamp(min=1)

    positions = torch.arange(sequence.shape[1], device=sequence.device)
    sort_key = torch.where(binary, positions.unsqueeze(0),
                           positions.unsqueeze(0) + sequence.shape[1])
    order = torch.argsort(sort_key, dim=1, stable=True)
    compacted = torch.gather(
        sequence, 1, order.unsqueeze(-1).expand(-1, -1, sequence.shape[-1]))
    compacted_mask = torch.gather(binary, 1, order).to(dtype=sequence.dtype)
    compacted = compacted * compacted_mask.unsqueeze(-1)

    packed = nn.utils.rnn.pack_padded_sequence(
        compacted, lengths.cpu(), batch_first=True, enforce_sorted=False)
    output, _ = rnn(packed)
    unpacked, _ = nn.utils.rnn.pad_packed_sequence(
        output, batch_first=True, total_length=sequence.shape[1])
    return unpacked


# ------------------------------------------------------------------------- encoders
class BlockEndpointEncoder(nn.Module):
    """One recurrent pass over a window; block-end hidden states become tokens.

    The recurrent core is shared across all blocks of a window (that is the point: the tokens
    are one causal state progression sampled at four moments, not seven unrelated encoders).
    Each token gets a learned age embedding, produced identically in every arm so the 2-token
    baseline and the 7-token arm differ only in how many block ends are read.
    """

    def __init__(self, channels: int, block_rows: int, n_blocks: int,
                 hidden_dim: int = 32, token_dim: int = TOKEN_DIM,
                 cell: str = "lstm", n_layers: int = 1):
        super().__init__()
        self.channels = channels
        self.block_rows = block_rows
        self.n_blocks = n_blocks
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size=channels, hidden_size=hidden_dim,
                           num_layers=n_layers, batch_first=True)
        self.cell = cell
        self.age_embedding = nn.Embedding(n_blocks, hidden_dim)
        self.projection = nn.Linear(hidden_dim, token_dim)
        # The "whole window was empty" case is carried by `valid`/`window_observed` as a
        # boolean, not by a learned flag: a flag fed through `torch.where(cond, 0, flag)`
        # never receives gradient in either branch, so it would be a permanently dead
        # parameter that still inflated the parameter count.
        self.empty = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, sequence: torch.Tensor, observed: torch.Tensor,
                block_rows: int | None = None) -> dict:
        rows = self.block_rows if block_rows is None else int(block_rows)
        states = observed_step_states(self.rnn, sequence, observed)
        # calendar row first (21/42/63, or 3/6/9/12), then the observation at or before it
        end_row, valid, age, observed_index = block_end_offsets(
            observed, rows, self.n_blocks, return_observed_index=True)
        del end_row

        index = (observed_index - 1).clamp(min=0).unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        picked = torch.gather(states, 1, index)                     # (B, n_blocks, hidden)
        picked = torch.where(valid.unsqueeze(-1), picked,
                             self.empty.view(1, 1, -1).expand_as(picked))
        picked = picked + self.age_embedding(age)
        tokens = self.projection(picked)
        any_observed = valid.any(dim=1, keepdim=True)
        return {"tokens": tokens, "valid": valid, "age": age,
                "window_observed": any_observed}


def build_dbs_block_encoder(hidden_dim: int = 32) -> BlockEndpointEncoder:
    from experiments.macro_encoders_v3 import contract as macro_contract
    return BlockEndpointEncoder(channels=len(macro_contract.DBS_CHANNELS),
                                block_rows=DBS_BLOCK_ROWS, n_blocks=3,
                                hidden_dim=hidden_dim, cell="lstm")


def build_coe_block_encoder(hidden_dim: int = 32) -> BlockEndpointEncoder:
    from experiments.macro_encoders_v3 import contract as macro_contract
    channels = len(macro_contract.COE_CHANNELS) * len(macro_contract.COE_CATEGORIES)
    return BlockEndpointEncoder(channels=channels, block_rows=COE_BLOCK_MONTHS,
                                n_blocks=4, hidden_dim=hidden_dim, cell="gru")
