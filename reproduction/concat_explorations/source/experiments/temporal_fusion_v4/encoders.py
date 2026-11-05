"""Padding-safe recurrent encoder for the rent sequence.

The defect in v2
----------------
v2's `_masked_lstm_pool` zeroes the *input* at masked steps and masks the *outputs*
before averaging, but it still runs the LSTM over every step:

    masked_input = sequence * mask            # padded steps become 0-vectors
    hidden, _ = lstm(masked_input)            # ... but the recurrence still advances
    hidden = hidden * mask                    # masked outputs are dropped from the mean

For trailing padding this is harmless: padded steps come after every real step, so they
cannot corrupt a real step's state. For *leading* padding it is not. At a padded step the
LSTM receives a zero vector and updates `h`, `c` anyway; the first real step then reads a
state built from those zeros, so the pooled result depends on **how many pad steps
preceded it**, not only on the real history.

That is exactly this contract's layout: 31,807 of 52,310 A/fit rows have leading padding,
because the exported mask is `suffix(origin)` — a row whose origin is month `t < 11` can
only fill the last `t + 1` slots of a 12-month window. So for ~61% of training rows v2's
pooling reads a state contaminated by pad count.

`reproduce_legacy_padding_defect` demonstrates this numerically on real windows rather
than asserting it.

The repair
----------
Order-preserving compaction: gather only the real steps into a dense sequence (padded
right), run the LSTM on that, pool with the compacted mask. The recurrence length equals
the real history length and no pad step ever enters it. A row with no real step gets an
explicit zero token plus `present=0`, never NaN.

One encoder class serves both arms
----------------------------------
Both the padding-fixed 48-step arm and the monthly 12-step arm use **this same class with
`kind="lstm"`**. They differ only in the window representation fed to it, so a score
difference is attributable to the time axis rather than to a different recurrent type.
GRU is available but unused; no experiment here decodes one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

RECURRENT_KINDS = ("lstm", "gru")


class TemporalError(ValueError):
    pass


@dataclass(frozen=True)
class TemporalOutput:
    token: torch.Tensor        # (B, token_dim)
    present: torch.Tensor      # (B, 1) 1.0 where the row had at least one real step
    steps_used: torch.Tensor   # (B,) number of real steps fed to the recurrence


# ---------------------------------------------------------------- the v2 behaviour
def legacy_masked_pool(lstm: nn.Module, sequence: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """v2's `_masked_lstm_pool`, reproduced exactly, including the defect."""
    masked_input = sequence * mask.unsqueeze(-1)
    hidden, _ = lstm(masked_input)
    hidden = hidden * mask.unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return hidden.sum(dim=1) / denom


def reproduce_legacy_padding_defect(sequence: torch.Tensor, mask: torch.Tensor,
                                    hidden: int = 8, seed: int = 0) -> dict:
    """Same observations, different pad positions -> different legacy outputs.

    Returns the max absolute difference between the legacy pooling of a left-padded
    window and the *same observations* right-padded. A padding-safe encoder gives 0.
    """
    if sequence.dim() != 3:
        raise TemporalError("sequence must be (B, T, C)")
    generator = torch.Generator().manual_seed(seed)
    lstm = nn.LSTM(sequence.shape[-1], hidden, batch_first=True)
    with torch.no_grad():
        for parameter in lstm.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.5)

    steps = int(mask.sum().item())
    if steps == 0:
        raise TemporalError("need at least one real step")
    total = sequence.shape[1]
    real = sequence[mask > 0].unsqueeze(0)[:, :steps]

    def place(values: torch.Tensor, left_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.zeros(1, total, values.shape[-1])
        m = torch.zeros(1, total)
        out[0, left_pad:left_pad + steps] = values[0]
        m[0, left_pad:left_pad + steps] = 1.0
        return out, m

    legacy = {}
    for pad in (0, total - steps):
        values, m = place(real, pad)
        with torch.no_grad():
            legacy[pad] = legacy_masked_pool(lstm, values, m)

    return {
        "steps": steps,
        "total": total,
        "max_abs_difference": float((legacy[0] - legacy[total - steps]).abs().max()),
    }


# ---------------------------------------------------------------- compaction
def compact_by_mask(sequence: torch.Tensor, mask: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Order-preserving compaction of real steps to the left.

    Returns `(dense, dense_mask, steps_used)`. Real steps keep their relative order and
    the output is right-padded with zeros so it can be batched.
    """
    if sequence.dim() != 3:
        raise TemporalError(f"sequence must be (B, T, C), got {tuple(sequence.shape)}")
    batch, total, channels = sequence.shape
    if mask.shape != (batch, total):
        raise TemporalError(f"mask {tuple(mask.shape)} != {(batch, total)}")

    steps_used = mask.sum(dim=1).long()
    max_steps = int(steps_used.max().item()) if batch else 0
    dense = torch.zeros(batch, max(max_steps, 1), channels,
                        dtype=sequence.dtype, device=sequence.device)
    dense_mask = torch.zeros(batch, max(max_steps, 1),
                             dtype=mask.dtype, device=mask.device)
    if max_steps:
        for row in range(batch):
            count = int(steps_used[row].item())
            if count == 0:
                continue
            dense[row, :count] = sequence[row][mask[row] > 0][:count]
            dense_mask[row, :count] = 1.0
    return dense, dense_mask, steps_used


class PaddingSafeTemporalEncoder(nn.Module):
    """LSTM over compacted real steps; pooling never sees a pad step.

    `kind` defaults to `"lstm"` to match v2's recurrent type, so the comparison isolates
    the padding fix rather than swapping recurrent families.
    """

    def __init__(self, channels: int, hidden: int = 32, token_dim: int = 48,
                 kind: str = "lstm", project: bool = True):
        super().__init__()
        if channels <= 0:
            raise TemporalError("channels must be positive")
        if kind not in RECURRENT_KINDS:
            raise TemporalError(f"unknown recurrent kind {kind!r}; {RECURRENT_KINDS}")
        self.channels = channels
        self.hidden = hidden
        self.token_dim = token_dim
        self.kind = kind
        self.project = bool(project)
        recurrent = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = recurrent(channels, hidden, batch_first=True)
        # With project=False the caller supplies the hidden -> d_model map (v2's
        # `temporal_proj`), so no extra projection is created here.
        self.proj = nn.Linear(hidden, token_dim) if self.project else None
        self.presence = nn.Parameter(torch.zeros(token_dim))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> TemporalOutput:
        dense, dense_mask, steps_used = compact_by_mask(sequence, mask)
        present = (steps_used > 0).to(dense.dtype).unsqueeze(1)
        if int(steps_used.max().item()) == 0:
            zero = torch.zeros(dense.shape[0], self.token_dim, dtype=dense.dtype,
                               device=dense.device)
            return TemporalOutput(token=zero, present=present, steps_used=steps_used)

        with torch.backends.cudnn.flags(enabled=False):
            hidden, _ = self.rnn(dense)
        hidden = hidden * dense_mask.unsqueeze(-1)
        denom = dense_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = hidden.sum(dim=1) / denom
        if self.proj is not None:
            pooled = self.proj(pooled)
        # absent rows carry the learned presence vector instead of a zero state
        token = pooled * present + self.presence * (1.0 - present)
        return TemporalOutput(token=token, present=present, steps_used=steps_used)

    def describe(self) -> dict:
        return {
            "kind": "padding_safe",
            "recurrent": self.kind,
            "channels": self.channels,
            "hidden": self.hidden,
            "token_dim": self.token_dim,
            "padding": ("order-preserving compaction; pad steps never enter the "
                        "recurrence, so the pooled token is invariant to pad position"),
            "zero_history": "explicit: present=0 and a learned presence vector, never NaN",
        }
