"""v4 temporal arms: a pure pooling repair, and a monthly time-axis variant.

`padding_fixed_48` — the controlled repair
------------------------------------------
Constructed exactly like the verified fixed-7 diagnostic:

    set_seed(42)
    build_control("lstm_fusion", shapes, numeric_columns, token_mode="per_feature")

with no extra module built in between, so the token block, v2's `nn.LSTM`,
`temporal_proj`, `gate`, `type_embedding` and the head all start from **the same
weights**. Nothing is re-created and nothing is added:

* `type_embedding` is `base.block.type_embedding`, untouched. The temporal token uses
  the rent-token slot v2 already reserved at index `n_types - 1` — no extra row, so no
  existing type vector changes.
* `temporal_proj` and `gate` are `base.temporal_proj` / `base.gate`.
* No presence parameter is added. A zero-history row emits a raw zero hidden state,
  which `temporal_proj` maps exactly as it maps any other state.

The single behavioural change is the pooling: padded steps are compacted away before the
recurrence instead of being fed in as zeros.

`monthly_12` — the time-axis variant
------------------------------------
Same construction and the **same reused** `temporal_proj`, `gate` and `type_embedding`.
The only necessary difference is the recurrent input width (6 -> 24 channels), which
forces a new `nn.LSTM` because a v2 module of input size 6 cannot accept 24. That
re-initialisation is recorded as a residual difference between the arms, so a score gap
here is not purely a time-axis effect.

Both arms use `kind="lstm"` at the same hidden width, so the recurrent family is fixed.
`assert_init_comparable` proves the claim by rebuilding the reference and comparing every
tensor after mapping `lstm.*` <-> `temporal.rnn.*`.

Nothing global is patched: v2/v3 classes are untouched and `forward` is bound per
instance.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import torch
from torch import nn

from neural_fusion_v2.settings import CAPACITY
from neural_fusion_v3.models import build_control

from .encoders import TemporalOutput, compact_by_mask

ARMS = ("padding_fixed_48", "monthly_12")


class TemporalArmError(ValueError):
    pass


class _PoolingRepair(nn.Module):
    """Wraps an existing `nn.LSTM` and pools only over compacted real steps.

    Owns no parameters, so wrapping v2's module leaves every weight in place and adds
    nothing to the optimiser.
    """

    def __init__(self, lstm: nn.Module):
        super().__init__()
        self.rnn = lstm
        self.channels = int(lstm.input_size)
        self.hidden = int(lstm.hidden_size)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> TemporalOutput:
        dense, dense_mask, steps_used = compact_by_mask(sequence, mask)
        present = (steps_used > 0).to(dense.dtype).unsqueeze(1)
        if int(steps_used.max().item()) == 0:
            zero = torch.zeros(dense.shape[0], self.hidden, dtype=dense.dtype,
                               device=dense.device)
            return TemporalOutput(token=zero, present=present, steps_used=steps_used)
        with torch.backends.cudnn.flags(enabled=False):
            hidden, _ = self.rnn(dense)
        hidden = hidden * dense_mask.unsqueeze(-1)
        denom = dense_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = hidden.sum(dim=1) / denom
        return TemporalOutput(token=pooled * present, present=present,
                              steps_used=steps_used)

    def describe(self) -> dict:
        return {
            "kind": "pooling_repair_on_v2_lstm",
            "recurrent": "lstm (v2's own module, weights inherited)",
            "channels": self.channels,
            "hidden": self.hidden,
            "owned_parameters": 0,
            "padding": ("order-preserving compaction before the recurrence; the pooled "
                        "token is invariant to pad position"),
            "zero_history": "raw zero hidden state -> temporal_proj, no extra parameter",
        }


class TemporalPerFeatureModel(nn.Module):
    """v3 per-feature tabular tokens + one padding-safe temporal token."""

    def __init__(self, shapes, numeric_columns: list[str], arm: str,
                 temporal_channels: int, seed: int = 42,
                 capacity: dict | None = None):
        super().__init__()
        if arm not in ARMS:
            raise TemporalArmError(f"unknown arm {arm!r}; available {ARMS}")
        capacity = dict(capacity or CAPACITY)
        self.arm = arm
        self.capacity = capacity
        self.numeric_columns = list(numeric_columns)
        self.seed = int(seed)
        self.temporal_channels = int(temporal_channels)

        from neural_fusion_v2.training import set_seed
        set_seed(self.seed)
        base = build_control("lstm_fusion", shapes, self.numeric_columns,
                             token_mode="per_feature")

        self.block = base.block                       # type_embedding left untouched
        self.d_model = self.block.d_model
        # The rent token slot v2 already reserved; reusing it adds no embedding row.
        self.temporal_type_index = int(self.block.n_types) - 1
        self.temporal_proj = base.temporal_proj       # reused verbatim in both arms
        self.gate = base.gate                         # reused verbatim in both arms

        if arm == "padding_fixed_48":
            if temporal_channels != int(base.lstm.input_size):
                raise TemporalArmError(
                    f"padding_fixed_48 must keep v2's {base.lstm.input_size}-channel "
                    f"input to reuse its LSTM, got {temporal_channels}")
            self.temporal = _PoolingRepair(base.lstm)
            self.new_recurrent = False
        else:
            # 24 channels cannot enter a 6-input LSTM, so this module is genuinely new.
            self.temporal = _MonthlyLSTMBranch(
                channels=temporal_channels, hidden=capacity["lstm_hidden"])
            self.new_recurrent = True

    # ------------------------------------------------------------------ helpers
    def _tabular(self, batch) -> torch.Tensor:
        encoder = self.block.encoder
        parts = [emb(batch["categorical"][:, i])
                 for i, emb in enumerate(encoder.embeddings)]
        parts.extend(encoder.feature_tokens(batch["static"]).unbind(dim=1))
        parts.append(torch.tanh(encoder.history_proj(batch["history"])))
        parts.append(torch.tanh(encoder.context_proj(batch["context"])))
        width = self.block.proj.in_features
        padded = [part if part.shape[1] >= width else
                  torch.cat([part, torch.zeros(part.shape[0], width - part.shape[1],
                                               device=part.device, dtype=part.dtype)], dim=1)
                  for part in parts]
        return torch.stack(padded, dim=1)

    def forward(self, batch):
        block = self.block
        tokens = block.proj(self._tabular(batch))
        types = torch.arange(1, 1 + tokens.shape[1], device=tokens.device)
        tokens = tokens + block.type_embedding(types)[None]

        temporal = self.temporal(batch["sequence"], batch["sequence_mask"])
        token = self.temporal_proj(temporal.token).unsqueeze(1)
        token = token + block.type_embedding(
            torch.full((1,), self.temporal_type_index, dtype=torch.long,
                       device=token.device))

        sequence = torch.cat([tokens, token], dim=1)
        pooled = block.pooled(sequence)
        return block.head(pooled * self.gate(pooled)).squeeze(-1)

    # ------------------------------------------------------------------ reporting
    def describe(self) -> dict:
        return {
            "arm": self.arm,
            "reused_v2_lstm": not self.new_recurrent,
            "new_recurrent_module": self.new_recurrent,
            "reused_temporal_proj": True,
            "reused_gate": True,
            "reused_type_embedding": True,
            "temporal_type_index": self.temporal_type_index,
            "type_embedding_rows": int(self.block.type_embedding.num_embeddings),
            "temporal_channels": self.temporal_channels,
            "temporal": self.temporal.describe(),
            "d_model": self.d_model,
            "seed": self.seed,
            "numeric_columns": list(self.numeric_columns),
        }


class _MonthlyLSTMBranch(nn.Module):
    """Padding-safe LSTM for the 24-channel monthly reading (new module, 6 != 24)."""

    def __init__(self, channels: int, hidden: int = 32, kind: str = "lstm"):
        super().__init__()
        if channels <= 0:
            raise TemporalArmError("channels must be positive")
        self.channels = channels
        self.hidden = hidden
        self.kind = kind
        self.rnn = (nn.LSTM if kind == "lstm" else nn.GRU)(channels, hidden,
                                                           batch_first=True)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> TemporalOutput:
        dense, dense_mask, steps_used = compact_by_mask(sequence, mask)
        present = (steps_used > 0).to(dense.dtype).unsqueeze(1)
        if int(steps_used.max().item()) == 0:
            zero = torch.zeros(dense.shape[0], self.hidden, dtype=dense.dtype,
                               device=dense.device)
            return TemporalOutput(token=zero, present=present, steps_used=steps_used)
        with torch.backends.cudnn.flags(enabled=False):
            hidden, _ = self.rnn(dense)
        hidden = hidden * dense_mask.unsqueeze(-1)
        denom = dense_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = hidden.sum(dim=1) / denom
        return TemporalOutput(token=pooled * present, present=present,
                              steps_used=steps_used)

    def describe(self) -> dict:
        return {
            "kind": "monthly_12_channels",
            "recurrent": self.kind,
            "channels": self.channels,
            "hidden": self.hidden,
            "owned_parameters": int(sum(p.numel() for p in self.parameters())),
            "axis": "12 calendar months; levels are channels, not timesteps",
            "padding": ("pre-panel padding compacted away; real months without "
                        "transactions are retained"),
        }


def build_arm(arm: str, shapes, numeric_columns: list[str], temporal_channels: int,
              seed: int = 42, capacity: dict | None = None) -> TemporalPerFeatureModel:
    return TemporalPerFeatureModel(shapes, numeric_columns, arm, temporal_channels,
                                   seed, capacity)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def normalise_state(state: dict) -> dict:
    """`temporal.rnn.*` IS v2's `lstm.*`, re-parented; align the names for comparison."""
    out = {}
    for name, tensor in state.items():
        if name.startswith("temporal.rnn."):
            name = "lstm." + name[len("temporal.rnn."):]
        out[name] = tensor
    return out


def assert_init_comparable(shapes, numeric_columns: list[str], temporal_channels: int,
                           seed: int = 42) -> dict:
    """Prove `padding_fixed_48` starts from the reference's weights, tensor by tensor."""
    from neural_fusion_v2.training import set_seed

    arm_model = build_arm("padding_fixed_48", shapes, numeric_columns,
                          temporal_channels, seed)
    set_seed(seed)
    reference = build_control("lstm_fusion", shapes, list(numeric_columns),
                              token_mode="per_feature")

    arm_state = normalise_state(dict(arm_model.named_parameters()))
    ref_state = normalise_state(dict(reference.named_parameters()))
    shared = sorted(set(arm_state) & set(ref_state))
    differing = [name for name in shared
                 if not torch.equal(arm_state[name].detach(), ref_state[name].detach())]
    return {
        "shared_tensors": len(shared),
        "differing": differing,
        "only_in_reference": sorted(set(ref_state) - set(arm_state)),
        "only_in_arm": sorted(set(arm_state) - set(ref_state)),
        "comparable": not differing and not (set(ref_state) - set(arm_state)),
    }
