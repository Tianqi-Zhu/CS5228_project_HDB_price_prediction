"""Controls with a selectable numeric-token representation.

Two modes, differing in exactly one place:

``block``
    **Returns an unmodified `neural_fusion_v2` model.** Same `TabularEncoder`, same
    `static_proj` / `history_proj` / `context_proj` (all `tanh(Linear(..., 8))`), same
    context token, same heads, same parameter count. This *is* the v2 reference, so a
    name-for-name state-dict comparison with v2 is possible.

``per_feature``
    v2's structure with **one** substitution: the numeric block's `Linear(n, 8) -> tanh`
    becomes one width-48 token per numeric column
    (`neural_fusion_v3.tokens.PerFeatureTokens`). Categorical embeddings, the history
    token, the context token and its projection, the heads, the LSTM, the optimiser and
    every other capacity remain what v2 has.

That is the whole ablation. An earlier draft also dropped v2's `context_proj` and fed the
raw 11-dimensional context into the heads; that changed several pathways at once and is
kept only as an exploratory run (`tok-*-mismatched`).

Implementation note: v2's encoder produces width-8 tokens because its `embed_dim` is 8.
Per-feature tokens are width 48, so in token mode they are combined with the width-8
tokens by zero-padding the width-8 part to 48 before the shared `Linear`. Padding rather
than a second learned projection keeps the parameter count of the untouched pathways
identical to v2.
"""
from __future__ import annotations

import sys as _sys
import types
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

from dataclasses import dataclass

import torch
from torch import nn

from neural_fusion_v2 import models as v2_models
from neural_fusion_v2.settings import CAPACITY

from .tokens import (PerFeatureTokens, TokenisationError, feature_token_fingerprint,
                     resolve_feature_index)

V2_CONTROLS = {
    "mlp": v2_models.MLPControl,
    "static_transformer": v2_models.StaticTransformerControl,
    "lstm_concat": v2_models.LSTMConcatControl,
    "lstm_fusion": v2_models.LSTMFusionControl,
}


@dataclass(frozen=True)
class Shapes:
    """v2's shape bundle under v3's field name, so block mode is a straight hand-off."""

    n_categorical: int
    numeric_dim: int
    history_dim: int
    context_dim: int
    cardinalities: tuple[int, ...]
    sequence_channels: int = 0
    sequence_length: int = 0

    def to_v2(self):
        return v2_models.Shapes(
            n_categorical=self.n_categorical, static_dim=self.numeric_dim,
            history_dim=self.history_dim, context_dim=self.context_dim,
            cardinalities=self.cardinalities, sequence_channels=self.sequence_channels,
            sequence_length=self.sequence_length)


def _encoder_of(model: nn.Module) -> nn.Module:
    return model.encoder if hasattr(model, "encoder") else model.block.encoder


def _parts(model: nn.Module, batch) -> list[torch.Tensor]:
    """Categorical embeddings, then one width-48 numeric token per column, history, context."""
    encoder = _encoder_of(model)
    parts = [emb(batch["categorical"][:, i]) for i, emb in enumerate(encoder.embeddings)]
    numeric = encoder.feature_tokens(batch["static"])          # (B, n, 48)
    parts.extend(numeric.unbind(dim=1))
    parts.append(torch.tanh(encoder.history_proj(batch["history"])))
    parts.append(torch.tanh(encoder.context_proj(batch["context"])))
    return parts


def _flat(model: nn.Module, batch) -> torch.Tensor:
    return torch.cat(_parts(model, batch), dim=1)


def _context_token(model: nn.Module, batch, block) -> torch.Tensor:
    """v2 emits the context token as the last tabular part; build it explicitly here."""
    encoder = _encoder_of(model)
    token = torch.tanh(encoder.context_proj(batch["context"]))
    token = _pad_to(token, block.d_model).unsqueeze(1)
    return token + block.type_embedding(torch.zeros(1, dtype=torch.long, device=token.device))


def _pad_to(token: torch.Tensor, width: int) -> torch.Tensor:
    """Right-pad a token to ``width``; a wider token is returned unchanged."""
    if token.shape[1] >= width:
        return token
    pad = torch.zeros(token.shape[0], width - token.shape[1],
                      device=token.device, dtype=token.dtype)
    return torch.cat([token, pad], dim=1)


def _projected_tokens(model: nn.Module, batch, out_dim: int) -> torch.Tensor:
    """(B, n_tokens, max_width): width-8 tokens padded up to the numeric token width.

    Numeric tokens are already width 48, so ``out_dim`` is 48 and the categorical, history
    and context tokens are padded from 8 to 48. Padding (rather than a second learned
    projection) keeps every untouched pathway's parameter count identical to v2.
    """
    parts = _parts(model, batch)
    width = max(out_dim, max(part.shape[1] for part in parts))
    return torch.stack([_pad_to(part, width) for part in parts], dim=1)


def install_per_feature(model: nn.Module, numeric_columns: list[str],
                        expected: list[str] | None) -> nn.Module:
    """Swap v2's numeric block for per-feature tokens; leave every other pathway alone."""
    encoder = _encoder_of(model)
    indices = resolve_feature_index(numeric_columns, expected)
    names = [numeric_columns[i] for i in indices]
    encoder.feature_tokens = PerFeatureTokens(names, out_dim=CAPACITY["d_model"])
    # record where each token's column sits in the contract's own numeric matrix, so a
    # report can prove column identity instead of asserting it
    encoder.feature_tokens.feature_indices = list(indices)
    encoder.numeric_columns = names
    encoder.numeric_index = indices
    encoder.uses_feature_tokens = True
    encoder.static_proj = None                    # replaced by `feature_tokens`
    d_model = CAPACITY["d_model"]
    # `proj` maps a token to d_model in the Transformer paths; v2 built it as 8 -> 48, but
    # per-feature tokens are already 48 wide, so it becomes an identity-width map.
    stack = getattr(model, "block", None)
    if stack is not None:
        stack.proj = nn.Linear(d_model, d_model)
    # v2's MLP / lstm_concat head consumed (4+3)*8 = 56 features. It now sees
    # (4+2)*8 + n*48, plus the LSTM state width for lstm_concat. Only the first Linear is
    # re-sized; every later layer keeps its shape and order.
    if hasattr(model, "head"):
        existing = list(model.head.children())
        in_features = (len(encoder.embeddings) + 2) * encoder.embed_dim \
            + len(encoder.feature_tokens.feature_names) * d_model
        if getattr(model, "name", "") == "lstm_concat":
            in_features += CAPACITY["lstm_hidden"]
        model.head = nn.Sequential(nn.Linear(in_features, existing[0].out_features),
                                   *existing[1:])
    # token count changes (1 numeric token -> n), so rebuild the type embedding that
    # v2 sized for its own layout. Same module type, same width, nothing else touched.
    stack = getattr(model, "block", None)
    if stack is not None:
        n_types = 1 + (len(encoder.embeddings) + encoder.feature_tokens.n_tokens + 2) \
            + (1 if model.name == "lstm_fusion" else 0)
        stack.type_embedding = nn.Embedding(n_types, stack.d_model)
        stack.n_tabular_tokens = len(encoder.embeddings) + encoder.feature_tokens.n_tokens + 2
        stack.n_types = n_types
    model.token_mode = "per_feature"
    model.numeric_columns = names
    _bind_forward(model)
    return model


def _bind_forward(model: nn.Module) -> None:
    name = model.name
    d_model = CAPACITY["d_model"]

    if name == "mlp":
        def forward(self, batch):
            return self.head(_flat(self, batch)).squeeze(-1)

    elif name == "static_transformer":
        def forward(self, batch):
            block = self.block
            tokens = block.proj(_projected_tokens(self, batch, block.proj.in_features))
            types = torch.arange(1, 1 + tokens.shape[1], device=tokens.device)
            tokens = tokens + block.type_embedding(types)[None]
            return block.head(block.pooled(tokens)).squeeze(-1)

    elif name == "lstm_concat":
        def forward(self, batch):
            state = v2_models._masked_lstm_pool(self.lstm, batch["sequence"],
                                                batch["sequence_mask"])
            x = torch.cat([_flat(self, batch), state], dim=1)
            return self.head(x).squeeze(-1)

    elif name == "lstm_fusion":
        def forward(self, batch):
            block = self.block
            state = v2_models._masked_lstm_pool(self.lstm, batch["sequence"],
                                                batch["sequence_mask"])
            temporal = self.temporal_proj(state).unsqueeze(1)
            temporal = temporal + block.type_embedding(
                torch.full((1,), block.n_types - 1, dtype=torch.long, device=temporal.device))
            tokens = block.proj(_projected_tokens(self, batch, block.proj.in_features))
            types = torch.arange(1, 1 + tokens.shape[1], device=tokens.device)
            tokens = tokens + block.type_embedding(types)[None]
            context = _context_token(self, batch, block)
            sequence = torch.cat([context, tokens, temporal], dim=1)
            pooled = block.pooled(sequence)
            return block.head(pooled * self.gate(pooled)).squeeze(-1)
    else:
        raise TokenisationError(f"unknown control {name!r}")

    model.forward = types.MethodType(forward, model)


def build_control(name: str, shapes: Shapes, numeric_columns: list[str],
                  token_mode: str = "block",
                  expected_numeric: list[str] | None = None) -> nn.Module:
    if name not in V2_CONTROLS:
        raise KeyError(f"unknown control {name!r}; available {sorted(V2_CONTROLS)}")
    if token_mode == "block":
        model = V2_CONTROLS[name](shapes.to_v2())          # v2's class, unchanged
        model.token_mode = "block"
        return model
    if token_mode != "per_feature":
        raise TokenisationError(f"unknown token mode {token_mode!r}")
    model = V2_CONTROLS[name](shapes.to_v2())
    return install_per_feature(model, numeric_columns, expected_numeric)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def model_description(model: nn.Module) -> dict:
    encoder = _encoder_of(model)
    mode = getattr(model, "token_mode", "block")
    if mode == "per_feature":
        numeric = feature_token_fingerprint(encoder.feature_tokens)
        n_tokens = len(encoder.embeddings) + encoder.feature_tokens.n_tokens + 2
        context_path = "tanh(context_proj(context)) width 8, zero-padded to 48 for attention"
    else:
        numeric = {
            "mode": "block", "n_numeric_in": encoder.static_proj.in_features,
            "n_tokens": 1, "token_width": encoder.embed_dim,
            "bottleneck": f"Linear({encoder.static_proj.in_features}, {encoder.embed_dim})",
        }
        n_tokens = encoder.n_tokens
        context_path = "tanh(context_proj(context)) width 8"
    return {
        "control": getattr(model, "name", type(model).__name__),
        "token_mode": mode,
        "capacity": dict(CAPACITY),
        "encoder": {
            "categorical_tokens": len(encoder.embeddings),
            "numeric": numeric,
            "history_tokens": 1,
            "context_tokens": 1,
            "context_path": context_path,
            "total_tokens": n_tokens,
            "uses_context_projection": encoder.context_proj is not None,
        },
        "parameters": parameter_count(model),
    }
