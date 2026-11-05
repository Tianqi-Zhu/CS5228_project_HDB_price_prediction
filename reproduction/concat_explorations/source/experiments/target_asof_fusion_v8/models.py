"""Target-month as-of fusion: house Transformer + three market representations.

One transaction, one sample. The external market window is `[t-12, ..., t-1]` — nothing from the
target month, no rent history, no anchor. The head predicts rent in kSGD directly.

## Three arms, one head

| arm | market representation |
|---|---|
| `house` | market vector is exactly zeros (the ablation floor) |
| `stats` | six simple statistics per channel from the same 12-month window -> small MLP -> 48 |
| `bilstm` | the same window through a single-layer Bi-LSTM -> 48 |

All three feed `concat[z_house(48), z_market(48), calendar(2)]` into the same MLP head. The house
encoder and the head are constructed from the same seed and copied by name across arms, so the
comparison isolates the market representation rather than the initialisation.

## The Bi-LSTM detail that is easy to get wrong

`output[:, -1]` is **not** a bidirectional summary of the window. For the reverse direction,
position -1 is the *first* time step processed, i.e. the state formed from almost nothing. Taking
it silently throws away the backward pass over the real history.

This module uses `h_n`: shape `(num_layers * num_directions, batch, hidden)`, so layer 0 gives
`h_n[0]` (forward, after the last real step) and `h_n[1]` (backward, after scanning back to the
first real step). Concatenated they are 64 dims, projected to 48. `h_n` is packing-aware, so
prefix padding and interior gaps do not shift it.

Three window shapes must all work:
* **prefix padding** — the window starts before any data exists; those steps are packed out.
* **interior missing months** — the month slot is kept as a time step with its value masked to 0
  and `observed=0`; it is *not* compressed, because dropping it would silently shorten the gap.
* **fully missing** — no observed step at all: a learned `missing` vector is used instead of `h_n`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

ARMS = ("house", "stats", "bilstm")
D_MODEL = 48
HOUSE_LAYERS = 3
HOUSE_HEADS = 4
HOUSE_DROPOUT = 0.05
MARKET_HIDDEN = 32          # per direction; the concatenation is 64
WINDOW_MONTHS = 12


class FusionError(ValueError):
    pass


# --------------------------------------------------------------------------- house
class HouseEncoder(nn.Module):
    """Per-numeric vector token + categorical embedding + CLS, Pre-LN Transformer.

    Mirrors the frozen v4 architecture (one token per column, type ids, CLS query) but exposes
    the pooled 48-d vector instead of a scalar head, because the fusion head consumes it.
    """

    name = "house_encoder"

    def __init__(self, numeric_names, cardinalities, d_model: int = D_MODEL,
                 layers: int = HOUSE_LAYERS, heads: int = HOUSE_HEADS,
                 ff_multiplier: int = 2, dropout: float = HOUSE_DROPOUT):
        super().__init__()
        if d_model % heads:
            raise FusionError(f"d_model {d_model} not divisible by heads {heads}")
        self.numeric_names = list(numeric_names)
        self.n_numeric = len(self.numeric_names)
        self.cardinalities = tuple(int(c) for c in cardinalities)
        self.d_model = d_model

        self.numeric_weight = nn.Parameter(torch.randn(self.n_numeric, d_model) * 0.02)
        self.numeric_bias = nn.Parameter(torch.zeros(self.n_numeric, d_model))
        self.numeric_identity = nn.Parameter(torch.randn(self.n_numeric, d_model) * 0.02)
        self.categorical = nn.ModuleList([nn.Embedding(c, d_model) for c in self.cardinalities])

        n_features = self.n_numeric + len(self.cardinalities)
        self.n_feature_tokens = n_features
        self.query_type_id = n_features + 1
        self.n_types = n_features + 2
        self.type_embedding = nn.Embedding(self.n_types, d_model)
        self.head_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * ff_multiplier, dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)

    def feature_tokens(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        if numeric.shape[1] != self.n_numeric:
            raise FusionError(f"expected {self.n_numeric} numeric columns, "
                              f"got {numeric.shape[1]}")
        # per-column vector token: x_j * w_j + b_j, plus a learned identity direction
        numeric_tokens = numeric.unsqueeze(-1) * self.numeric_weight[None]
        numeric_tokens = numeric_tokens + self.numeric_bias[None] + self.numeric_identity[None]
        cat_tokens = [emb(categorical[:, i]) for i, emb in enumerate(self.categorical)]
        tokens = torch.cat([torch.stack(cat_tokens, dim=1), numeric_tokens], dim=1)
        types = torch.arange(1, 1 + tokens.shape[1], device=tokens.device)
        return tokens + self.type_embedding(types)[None]

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        tokens = self.feature_tokens(numeric, categorical)
        query = self.head_token.expand(tokens.shape[0], 1, self.d_model)
        query_type = torch.full((1,), self.query_type_id, dtype=torch.long,
                                device=tokens.device)
        query = query + self.type_embedding(query_type)[None]
        sequence = torch.cat([query, tokens], dim=1)
        encoded = self.final_norm(self.encoder(sequence))
        return encoded[:, 0]                       # the CLS position


# -------------------------------------------------------------------------- market
class BiLstmMarket(nn.Module):
    """Single-layer Bi-LSTM over the 12-month window; `h_n` is the summary.

    Two masks, with different jobs:

    * ``step_valid`` marks months that exist in the window at all. Before the data starts it is
      0, and those steps are packed away.
    * ``observed`` marks a real value for that channel-month. An interior month with no data
      keeps its time step (``step_valid=1``) and is fed as zeros **plus** its observation
      channel, so the encoder can tell "0.0 because missing" from "0.0 because the value was
      zero". Interior gaps are therefore never compressed — only the leading padding is.

    `h_n` from an LSTM is ``(h_n, c_n)`` with ``h_n`` of shape
    ``(num_layers * num_directions, batch, hidden)``. One layer gives ``h_n[0]`` = forward final
    state (after the last real step) and ``h_n[1]`` = backward final state (after scanning back
    to the first real step). Concatenated they summarise the whole observed window.

    ``output[:, -1]`` is **not** that: for the reverse direction it is the state formed at the
    *first* step, i.e. from almost nothing, so using it would silently discard the backward pass.

    A row with no observed value anywhere gets a learned ``missing`` vector instead.
    """

    name = "bilstm_market"

    def __init__(self, channels: int, hidden: int = MARKET_HIDDEN, out_dim: int = D_MODEL,
                 feed_observation_mask: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.hidden = int(hidden)
        self.feed_observation_mask = bool(feed_observation_mask)
        input_size = self.channels * (2 if self.feed_observation_mask else 1)
        self.bilstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden,
                              num_layers=1, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(2 * self.hidden, out_dim)
        self.missing = nn.Parameter(torch.zeros(out_dim))

    def forward(self, values: torch.Tensor, observed: torch.Tensor,
                step_valid: torch.Tensor) -> torch.Tensor:
        """`values` (B, L, C) masked to 0 where absent; `observed` (B, L, C); `step_valid` (B, L)."""
        if values.dim() != 3 or values.shape[2] != self.channels:
            raise FusionError(f"market values must be (B, L, {self.channels}), "
                              f"got {tuple(values.shape)}")
        if observed.shape != values.shape:
            raise FusionError(f"observed {tuple(observed.shape)} != values "
                              f"{tuple(values.shape)}")
        if step_valid.shape != values.shape[:2]:
            raise FusionError(f"step_valid {tuple(step_valid.shape)} does not match "
                              f"values {tuple(values.shape[:2])}")

        steps = (step_valid > 0)
        any_observed = (observed > 0).any(dim=(1, 2))

        # leading padding is the only thing dropped; an interior gap keeps its slot because
        # `step_valid` stays 1 through it, so `lengths` spans across the hole
        lengths = steps.sum(dim=1).clamp(min=1).to(torch.long)

        inputs = torch.cat([values * (observed > 0).to(values.dtype),
                            (observed > 0).to(values.dtype)], dim=-1) \
            if self.feed_observation_mask else values

        # `pack_padded_sequence` takes the first `lengths[i]` steps; it does NOT skip leading
        # zeros. A window that begins before the data exists therefore has to be left-shifted
        # first, or the padding becomes the model's input and the real months are truncated.
        # Interior gaps are inside the retained span, so they survive this roll unchanged.
        steps_int = steps.to(torch.long)
        first_valid = torch.where(any_observed,
                                 steps_int.argmax(dim=1),
                                 torch.zeros_like(lengths))
        window = inputs.shape[1]
        width = inputs.shape[2]
        positions = torch.arange(window, device=inputs.device)[None]
        shifted = positions + first_valid[:, None]
        gather_index = shifted.clamp(max=window - 1).unsqueeze(-1).expand(-1, -1, width)
        inputs = torch.gather(inputs, 1, gather_index)
        still_in = shifted < window
        inputs = inputs * still_in.unsqueeze(-1).to(inputs.dtype)

        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.bilstm(packed)
        summary = torch.cat([h_n[0], h_n[1]], dim=1)          # forward ++ backward
        projected = self.projection(summary)
        return torch.where(any_observed.unsqueeze(-1), projected,
                           self.missing.unsqueeze(0))


class StatsMarket(nn.Module):
    """Six statistics per channel from the same window, computed on **calendar** months.

    Definitions match the data module exactly (rev2 of the export, `DATA_READY.md` §5):

    | suffix | meaning |
    |---|---|
    | `last_valid` | newest observed value in the window |
    | `mean3` | mean over calendar months `t-3..t-1` |
    | `mean12` | mean over the whole window (`t-12..t-1`) |
    | `recent3_minus_prior3` | `mean(t-3..t-1) - mean(t-6..t-4)` |
    | `n_valid` | number of observed months |
    | `months_since_last_valid` | age of the newest observation, in months |

    These are **not** "last three observations". A definition based on the last N observed
    values silently stretches across missing months, so it disagrees with the data module and
    the two numbers stop meaning the same thing. A calendar window with no observation inside
    it is missing — the window is never widened to find one.

    The window is positioned by `step_valid`: with 12 slots and `L = 12`, `t-3..t-1` is slots
    `9..11` and `t-6..t-4` is slots `6..8`.
    """

    name = "stats_market"
    N_STATS = 6

    def __init__(self, channels: int, out_dim: int = D_MODEL, hidden: int = 64):
        super().__init__()
        self.channels = int(channels)
        self.net = nn.Sequential(
            nn.Linear(self.channels * self.N_STATS, hidden), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(hidden, out_dim))
        self.missing = nn.Parameter(torch.zeros(out_dim))

    @staticmethod
    def statistics(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        """(B, L, C) values + (B, L, C) observed -> (B, C * 6) in calendar-window order."""
        if values.shape != observed.shape:
            raise FusionError(f"observed {tuple(observed.shape)} != values "
                              f"{tuple(values.shape)}")
        batch, length, channels = values.shape
        if length != WINDOW_MONTHS:
            raise FusionError(f"the calendar windows assume {WINDOW_MONTHS} months, "
                              f"got {length}")
        mask = observed.to(values.dtype)

        mean12 = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # calendar t-3..t-1 = the last three slots; t-6..t-4 = the three before those
        recent_vals, recent_mask = values[:, 9:12], mask[:, 9:12]
        prior_vals, prior_mask = values[:, 6:9], mask[:, 6:9]
        recent_n = recent_mask.sum(dim=1)
        prior_n = prior_mask.sum(dim=1)
        mean3 = (recent_vals * recent_mask).sum(dim=1) / recent_n.clamp(min=1)
        mean_prior3 = (prior_vals * prior_mask).sum(dim=1) / prior_n.clamp(min=1)
        # either calendar window being empty makes the difference missing, not zero
        change = torch.where((recent_n > 0) & (prior_n > 0), mean3 - mean_prior3,
                             torch.zeros_like(mean3))

        # newest observed value and its age, in calendar months
        reversed_values = values.flip(1)
        reversed_mask = mask.flip(1)
        has_any = reversed_mask.sum(dim=1) > 0
        first_index = reversed_mask.to(torch.bool).float().argmax(dim=1)   # (B, C)
        last_value = torch.gather(reversed_values, 1,
                                  first_index.unsqueeze(1)).squeeze(1)
        last_value = torch.where(has_any, last_value, torch.zeros_like(last_value))
        age = torch.where(has_any, first_index,
                          torch.full_like(first_index, float(length)))

        # `n_valid` counts every observed month in the window, not just the recent three
        n_valid = mask.sum(dim=1) / length
        stats = torch.stack([last_value, mean3, mean12, change,
                             n_valid, age / length], dim=-1)
        return stats.reshape(batch, channels * StatsMarket.N_STATS)

    def forward(self, values: torch.Tensor, observed: torch.Tensor,
                any_observed: torch.Tensor) -> torch.Tensor:
        features = self.statistics(values, observed)
        projected = self.net(features)
        return torch.where(any_observed.unsqueeze(-1), projected,
                           self.missing.unsqueeze(0))


# -------------------------------------------------------------------------- fusion
class TargetAsOfFusion(nn.Module):
    """The three-arm model: house encoder + market representation + shared head."""

    name = "target_asof_fusion"

    def __init__(self, arm: str, numeric_names, cardinalities, market_channels: int,
                 calendar_dim: int = 2, d_model: int = D_MODEL):
        super().__init__()
        if arm not in ARMS:
            raise FusionError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.house = HouseEncoder(numeric_names, cardinalities, d_model=d_model)
        self.market_channels = int(market_channels)

        if arm == "stats":
            self.market = StatsMarket(self.market_channels, out_dim=d_model)
        elif arm == "bilstm":
            self.market = BiLstmMarket(self.market_channels, out_dim=d_model)
        else:
            self.market = None

        self.head = nn.Sequential(
            nn.Linear(2 * d_model + calendar_dim, d_model), nn.GELU(),
            nn.Dropout(HOUSE_DROPOUT), nn.Linear(d_model, 1))

    def market_vector(self, batch) -> torch.Tensor:
        if self.market is None:
            reference = next(self.house.parameters())
            rows = batch["house_numeric"].shape[0]
            return torch.zeros(rows, self.house.d_model, device=reference.device,
                               dtype=reference.dtype)
        observed = batch["market_observed"]
        any_observed = (observed > 0).any(dim=(1, 2))
        if self.arm == "stats":
            return self.market(batch["market_values"], observed, any_observed)
        return self.market(batch["market_values"], observed, batch["market_step_valid"])

    def forward(self, batch) -> torch.Tensor:
        z_house = self.house(batch["house_numeric"], batch["house_cat"])
        z_market = self.market_vector(batch)
        calendar = batch["calendar"]
        fused = torch.cat([z_house, z_market, calendar], dim=1)
        return self.head(fused).squeeze(-1)

    def describe(self) -> dict:
        def count(module):
            return int(sum(p.numel() for p in module.parameters()))

        market_params = count(self.market) if self.market is not None else 0
        return {
            "name": self.name, "arm": self.arm,
            "house_parameters": count(self.house),
            "market_parameters": market_params,
            "head_parameters": count(self.head),
            "parameters": count(self),
            "market_channels": self.market_channels,
            "d_model": self.house.d_model,
            "market_encoder": (None if self.market is None else self.market.name),
            "market_input": ("values + observation mask channels" if self.arm == "bilstm"
                             else "six masked statistics per channel"),
            "bilstm_hidden_per_direction": (MARKET_HIDDEN if self.arm == "bilstm" else None),
            "bilstm_summary": ("h_n[0] forward + h_n[1] backward, concatenated; not "
                               "output[:,-1]" if self.arm == "bilstm" else None),
            "target": "MONTHLY_RENT/1000 directly; no anchor, no log, no clipping",
        }


def build(arm: str, numeric_names, cardinalities, market_channels: int, **kwargs):
    return TargetAsOfFusion(arm, numeric_names, cardinalities, market_channels, **kwargs)


# ------------------------------------------------------- shared initialisation
def build_three_arms(seed: int, numeric_names, cardinalities, market_channels: int):
    """Build all three arms from one seed, copying the common house+head weights by name.

    The market branch is the only thing that differs in initialisation, which is what the
    ablation is meant to isolate. Copies, not tied tensors.
    """
    models = {}
    reference_state = None
    for arm in ARMS:
        torch.manual_seed(seed)
        model = build(arm, numeric_names, cardinalities, market_channels)
        if reference_state is None:
            reference_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            target = {k: v.detach().clone() for k, v in model.state_dict().items()}
            shared = 0
            for key in target:
                if key.startswith(("house.", "head.")):
                    if key not in reference_state:
                        raise FusionError(f"{arm}: no counterpart for {key}")
                    if reference_state[key].shape != target[key].shape:
                        raise FusionError(f"{arm}: shape mismatch on {key}")
                    target[key] = reference_state[key].clone()
                    shared += 1
            model.load_state_dict(target)
        models[arm] = model
    return models
