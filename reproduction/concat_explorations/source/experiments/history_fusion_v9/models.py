"""History-fusion v9: the v8 Bi-LSTM model plus 50 rent-history feature tokens.

## What changes and what must not

The v8 model is a house Transformer (21 numeric + 3 categorical tokens + CLS) fused with a
Bi-LSTM over external market months. The history arm adds **50 numeric history columns as 50
independent feature tokens** to the house encoder. Everything else — the market branch, the
head, the calendar inputs, the target, the row set — stays exactly as v8.

## The type-id collision this has to avoid

v8 assigns type ids as: features get `1..24`, and CLS gets `n_feature_tokens + 1 = 25`. The
feature ids are `1..24` because `n_numeric = 21` plus 3 categoricals is 24, and id 0 is unused.

Appending 50 history tokens makes 74 feature tokens. If CLS were left at 25, it would collide
with what is now a real feature token — the query would silently read a history feature's
embedding. So CLS moves to `74 + 1 = 75`, and the history tokens occupy `25..74`.

That mapping is carried in `init_parity` so a reviewer can check it rather than trust it.

## Shared-parameter initialisation

The plan is explicit: common parameters come from a **fresh seed-42 initialisation**, never from
the trained v8 checkpoint. The procedure:

1. build the v8-shaped model (21+3 features, CLS at 25) at seed 42, snapshot it;
2. build the history model (21+3+50 features, CLS at 75) at the same seed;
3. copy every shared tensor by name: the 21 numeric weight/bias/identity rows, the 3
   categorical embeddings, all encoder blocks, the head, the market branch, the head_token;
4. move the old CLS type row (25) to its new position (75) and leave the 50 new type rows fresh.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

D_MODEL = 48
HISTORY_COLUMNS = 50
BASE_NUMERIC = 21
BASE_CATEGORICAL = 3


class HistoryFusionError(ValueError):
    pass


class HistoryHouseEncoder(nn.Module):
    """v8's house encoder with extra per-column vector tokens for the history block."""

    name = "history_house_encoder"

    def __init__(self, numeric_names, cardinalities, history_columns: int = HISTORY_COLUMNS,
                 d_model: int = D_MODEL, layers: int = 3, heads: int = 4,
                 ff_multiplier: int = 2, dropout: float = 0.05):
        super().__init__()
        if d_model % heads:
            raise HistoryFusionError(f"d_model {d_model} not divisible by heads {heads}")
        self.numeric_names = list(numeric_names)
        self.n_numeric = len(self.numeric_names)
        self.cardinalities = tuple(int(c) for c in cardinalities)
        self.history_columns = int(history_columns)
        self.d_model = d_model

        # base block: unchanged shape semantics from v8
        self.numeric_weight = nn.Parameter(torch.randn(self.n_numeric, d_model) * 0.02)
        self.numeric_bias = nn.Parameter(torch.zeros(self.n_numeric, d_model))
        self.numeric_identity = nn.Parameter(torch.randn(self.n_numeric, d_model) * 0.02)
        self.categorical = nn.ModuleList([nn.Embedding(c, d_model) for c in self.cardinalities])

        # the history block gets its own per-column parameters so the two blocks are separable
        self.history_weight = nn.Parameter(torch.randn(self.history_columns, d_model) * 0.02)
        self.history_bias = nn.Parameter(torch.zeros(self.history_columns, d_model))
        self.history_identity = nn.Parameter(torch.randn(self.history_columns, d_model) * 0.02)

        base_features = self.n_numeric + len(self.cardinalities)
        self.base_feature_tokens = base_features
        self.n_feature_tokens = base_features + self.history_columns
        # CLS must sit after every feature token; in v8 it was base_features + 1
        self.query_type_id = self.n_feature_tokens + 1
        self.n_types = self.n_feature_tokens + 2
        self.type_embedding = nn.Embedding(self.n_types, d_model)
        self.head_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * ff_multiplier, dropout,
                                           batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------ pieces
    def base_tokens(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """The v8 token block: 3 categoricals then 21 numerics, type ids 1..24."""
        if numeric.shape[1] != self.n_numeric:
            raise HistoryFusionError(f"expected {self.n_numeric} numeric columns, "
                                     f"got {numeric.shape[1]}")
        numeric_tokens = numeric.unsqueeze(-1) * self.numeric_weight[None]
        numeric_tokens = numeric_tokens + self.numeric_bias[None] + self.numeric_identity[None]
        cat_tokens = [emb(categorical[:, i]) for i, emb in enumerate(self.categorical)]
        tokens = torch.cat([torch.stack(cat_tokens, dim=1), numeric_tokens], dim=1)
        types = torch.arange(1, 1 + tokens.shape[1], device=tokens.device)
        return tokens + self.type_embedding(types)[None]

    def history_tokens(self, history: torch.Tensor) -> torch.Tensor:
        """The 50 history columns as 50 tokens, typed immediately after the base block."""
        if history.shape[1] != self.history_columns:
            raise HistoryFusionError(f"expected {self.history_columns} history columns, "
                                     f"got {history.shape[1]}")
        tokens = history.unsqueeze(-1) * self.history_weight[None]
        tokens = tokens + self.history_bias[None] + self.history_identity[None]
        first = self.base_feature_tokens + 1
        types = torch.arange(first, first + self.history_columns, device=tokens.device)
        return tokens + self.type_embedding(types)[None]

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor,
                history: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([self.base_tokens(numeric, categorical),
                            self.history_tokens(history)], dim=1)
        query = self.head_token.expand(tokens.shape[0], 1, self.d_model)
        query_type = torch.full((1,), self.query_type_id, dtype=torch.long,
                                device=tokens.device)
        query = query + self.type_embedding(query_type)[None]
        sequence = torch.cat([query, tokens], dim=1)
        encoded = self.final_norm(self.encoder(sequence))
        return encoded[:, 0]


def build_base_model(seed: int, numeric_names, cardinalities, market_channels: int):
    """A v8-shaped model: no history block, CLS at `base_features + 1`."""
    from experiments.target_asof_fusion_v8 import models as V8

    torch.manual_seed(seed)
    return V8.build("bilstm", numeric_names, cardinalities, market_channels)


class HistoryFusionModel(nn.Module):
    """The v8 fusion model with the wider house encoder and a `history` input.

    Subclassing rather than monkey-patching the frozen v8 class keeps the v8 module untouched:
    the market branch, calendar handling, concat order and head are all reused verbatim, and
    only the house call gains one extra argument.
    """

    name = "history_fusion"

    def __init__(self, v8_model, history_columns: int = HISTORY_COLUMNS):
        super().__init__()
        self.market_channels = v8_model.market_channels
        self.arm = "bilstm_history"
        self.history_columns = int(history_columns)
        # keep the v8 submodules so their state dict keys line up with the frozen model
        self.house = HistoryHouseEncoder(v8_model.house.numeric_names,
                                         v8_model.house.cardinalities,
                                         history_columns=self.history_columns,
                                         d_model=v8_model.house.d_model)
        self.market = v8_model.market
        self.head = v8_model.head

    def market_vector(self, batch):
        observed = batch["market_observed"]
        any_observed = (observed > 0).any(dim=(1, 2))
        return self.market(batch["market_values"], observed, batch["market_step_valid"])

    def forward(self, batch):
        z_house = self.house(batch["house_numeric"], batch["house_cat"], batch["history"])
        z_market = self.market_vector(batch)
        fused = torch.cat([z_house, z_market, batch["calendar"]], dim=1)
        return self.head(fused).squeeze(-1)

    def describe(self) -> dict:
        return {
            "name": self.name, "arm": self.arm,
            "history_columns": self.history_columns,
            "parameters": int(sum(p.numel() for p in self.parameters())),
            "cls_type_id": int(self.house.query_type_id),
            "n_feature_tokens": int(self.house.n_feature_tokens),
            "house_parameters": int(sum(p.numel() for p in self.house.parameters())),
            "market_parameters": int(sum(p.numel() for p in self.market.parameters())),
            "head_parameters": int(sum(p.numel() for p in self.head.parameters())),
        }


def build_history_model(seed: int, numeric_names, cardinalities, market_channels: int,
                        history_columns: int = HISTORY_COLUMNS):
    """The history arm: same market branch and head shapes, 50 extra house tokens."""
    from experiments.target_asof_fusion_v8 import models as V8

    torch.manual_seed(seed)
    base = V8.build("bilstm", numeric_names, cardinalities, market_channels)
    return HistoryFusionModel(base, history_columns=history_columns)


def copy_common_parameters(base_model, history_model) -> dict:
    """Copy every parameter the two models share by name, then move CLS.

    Returns a record of what was copied and what was left fresh, plus the checks that make the
    claim falsifiable.
    """
    base_state = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    target_state = {k: v.detach().clone() for k, v in history_model.state_dict().items()}

    rows = []
    for key, tensor in target_state.items():
        if key not in base_state:
            continue
        source = base_state[key]
        if source.shape != tensor.shape:
            continue
        target_state[key] = source.clone()
        rows.append({"param": key, "action": "copied", "shape": list(source.shape)})

    # the house encoder changed shape; handle its pieces explicitly instead of skipping them
    for key in ("house.numeric_weight", "house.numeric_bias", "house.numeric_identity"):
        if key in base_state and key in target_state:
            target_state[key] = base_state[key].clone()
            rows.append({"param": key, "action": "copied (same shape)", 
                         "shape": list(base_state[key].shape)})
    for key in list(target_state):
        if key.startswith("house.categorical."):
            if key in base_state:
                target_state[key] = base_state[key].clone()
                rows.append({"param": key, "action": "copied (same shape)",
                             "shape": list(base_state[key].shape)})

    # type embedding: base has (base_features+2) rows, history has (base_features+50+2)
    type_key = "house.type_embedding.weight"
    source_types = base_state[type_key]
    base_features = source_types.shape[0] - 2
    old_cls = base_features + 1
    new_cls = base_features + HISTORY_COLUMNS + 1
    destination = target_state[type_key]
    destination[0] = source_types[0]                      # unused row
    destination[1:1 + base_features] = source_types[1:1 + base_features]
    destination[new_cls] = source_types[old_cls]          # CLS moves, does not collide
    target_state[type_key] = destination
    rows.append({"param": type_key, "action": f"features 1..{base_features} copied, "
                                              f"CLS {old_cls} -> {new_cls}, "
                                              f"{HISTORY_COLUMNS} new rows fresh",
                 "old_cls": old_cls, "new_cls": new_cls})

    history_model.load_state_dict(target_state)
    after = {k: v.detach().clone() for k, v in history_model.state_dict().items()}

    checks = []
    for key in ("house.numeric_weight", "house.numeric_bias", "house.numeric_identity"):
        checks.append({"param": key, "scope": "all rows",
                       "bitwise_equal": bool(torch.equal(after[key], base_state[key]))})
    for key in list(base_state):
        if key.startswith(("house.numeric", "house.type_embedding")):
            continue
        if key in after and base_state[key].shape == after[key].shape:
            checks.append({"param": key, "scope": "all",
                           "bitwise_equal": bool(torch.equal(after[key], base_state[key]))})
    checks.append({"param": type_key, "scope": f"features 1..{base_features}",
                   "bitwise_equal": bool(torch.equal(after[type_key][1:1 + base_features],
                                                     base_state[type_key][1:1 + base_features]))})
    checks.append({"param": type_key, "scope": f"CLS {old_cls} -> {new_cls}",
                   "bitwise_equal": bool(torch.equal(after[type_key][new_cls],
                                                     base_state[type_key][old_cls]))})
    checks.append({"param": type_key, "scope": f"old CLS position {old_cls} is now a "
                                               f"history feature (must differ)",
                   "bitwise_equal": bool(torch.equal(after[type_key][old_cls],
                                                     base_state[type_key][old_cls]))})
    checks.append({"param": "house.history_weight", "scope": "fresh, non-zero",
                   "nonzero": bool(after["house.history_weight"].abs().sum() > 0)})

    parity = {
        "source": "fresh seed-42 initialisation, NOT the trained v8 checkpoint",
        "cls_move": {"old": old_cls, "new": new_cls},
        "base_feature_tokens": base_features,
        "history_columns": HISTORY_COLUMNS,
        "rows": rows,
        "checks": checks,
        "all_checks_pass": all(
            (not c["bitwise_equal"]) if "must differ" in c["scope"]
            else (c["bitwise_equal"] if "bitwise_equal" in c else c["nonzero"])
            for c in checks),
        "base_params": int(sum(v.numel() for v in base_state.values())),
        "history_params": int(sum(v.numel() for v in after.values())),
    }
    return parity
