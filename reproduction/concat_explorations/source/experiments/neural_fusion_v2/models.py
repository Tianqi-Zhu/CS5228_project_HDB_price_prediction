"""Four Gate 1 controls at the frozen capacity (d_model=48, layers=2, heads=4, lstm=32).

Every control consumes the same ``static`` / ``history`` / ``context`` / ``categorical``
tensors. Only ``lstm_concat`` and ``lstm_fusion`` accept ``sequence`` /
``sequence_mask``; the static controls never receive those keys, so a missing sequence
cannot affect them.

Inputs (all produced by ``adapter.build_tensors``):

    static       standardised numeric block
    history      standardised smoothed history summaries (24 cols from the producer)
    context      standardised horizon/calendar/anchor-count block, anchor excluded
    categorical  4 integer category codes
    sequence     (B, T, C) flattened month-major over the 4 fallback levels

Attention weights are never interpreted as causal effects or correlations.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .settings import CAPACITY


@dataclass(frozen=True)
class Shapes:
    """Dimensions taken from the contract at runtime."""

    n_categorical: int
    static_dim: int
    history_dim: int
    context_dim: int
    cardinalities: tuple[int, ...]
    sequence_channels: int = 0
    sequence_length: int = 0

    def tabular_dim(self) -> int:
        return self.static_dim + self.history_dim + self.context_dim


class TabularEncoder(nn.Module):
    """Categorical embeddings + one projection per numeric block."""

    def __init__(self, shapes: Shapes, embed_dim: int = 8):
        super().__init__()
        if len(shapes.cardinalities) != shapes.n_categorical:
            raise ValueError("cardinalities must cover every categorical column")
        self.embeddings = nn.ModuleList([nn.Embedding(s, embed_dim) for s in shapes.cardinalities])
        self.static_proj = nn.Linear(shapes.static_dim, embed_dim)
        self.history_proj = nn.Linear(shapes.history_dim, embed_dim)
        self.context_proj = nn.Linear(shapes.context_dim, embed_dim)
        self.embed_dim = embed_dim
        self.n_tokens = len(shapes.cardinalities) + 3       # categorical + 3 numeric tokens

    def _parts(self, batch) -> list[torch.Tensor]:
        parts = [emb(batch["categorical"][:, i]) for i, emb in enumerate(self.embeddings)]
        parts.append(torch.tanh(self.static_proj(batch["static"])))
        parts.append(torch.tanh(self.history_proj(batch["history"])))
        parts.append(torch.tanh(self.context_proj(batch["context"])))
        return parts

    def flat(self, batch) -> torch.Tensor:
        return torch.cat(self._parts(batch), dim=1)

    def tokens(self, batch) -> torch.Tensor:
        return torch.stack(self._parts(batch), dim=1)

    @property
    def out_dim(self) -> int:
        return self.embed_dim * self.n_tokens


class MLPControl(nn.Module):
    name = "mlp"
    uses_sequence = False

    def __init__(self, shapes: Shapes):
        super().__init__()
        self.encoder = TabularEncoder(shapes)
        hidden = CAPACITY["mlp_hidden"]
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        return self.head(self.encoder.flat(batch)).squeeze(-1)


class _TokenStack(nn.Module):
    """Projection + type embeddings + encoder + masked mean pooling + scalar head."""

    def __init__(self, shapes: Shapes, extra_tokens: int = 0):
        super().__init__()
        self.encoder = TabularEncoder(shapes)
        self.d_model = CAPACITY["d_model"]
        self.proj = nn.Linear(self.encoder.embed_dim, self.d_model)
        self.n_tabular_tokens = self.encoder.n_tokens
        self.n_types = 1 + self.n_tabular_tokens + extra_tokens
        self.type_embedding = nn.Embedding(self.n_types, self.d_model)
        layer = nn.TransformerEncoderLayer(self.d_model, CAPACITY["heads"], self.d_model * 2,
                                          0.05, batch_first=True, norm_first=False)
        self.stack = nn.TransformerEncoder(layer, CAPACITY["layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.GELU(),
                                  nn.Linear(self.d_model, 1))

    def tabular_tokens(self, batch) -> torch.Tensor:
        tokens = self.proj(self.encoder.tokens(batch))
        types = torch.arange(1, 1 + self.n_tabular_tokens, device=tokens.device)
        return tokens + self.type_embedding(types)[None]

    def pooled(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.stack(sequence)
        if mask is None:
            return self.norm(encoded).mean(dim=1)
        weights = mask.unsqueeze(-1).float()
        return (self.norm(encoded) * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class StaticTransformerControl(nn.Module):
    """Same information as the MLP, arranged as tokens for self-attention."""

    name = "static_transformer"
    uses_sequence = False

    def __init__(self, shapes: Shapes):
        super().__init__()
        self.block = _TokenStack(shapes)

    def forward(self, batch):
        tokens = self.block.tabular_tokens(batch)
        return self.block.head(self.block.pooled(tokens)).squeeze(-1)


def _masked_lstm_pool(lstm: nn.LSTM, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool only over real timesteps.

    Zeroing the input at masked steps is not enough: an LSTM carries state forward, so a
    padded step still shifts the hidden state that later real steps read. Here the masked
    steps are zeroed *and* their outputs are excluded from the mean, and the conversion
    from levels/months to the flat axis keeps real steps in chronological order, so the
    result depends only on unmasked history.
    """
    masked_input = sequence * mask.unsqueeze(-1)
    hidden, _ = lstm(masked_input)
    hidden = hidden * mask.unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return hidden.sum(dim=1) / denom


class LSTMConcatControl(nn.Module):
    name = "lstm_concat"
    uses_sequence = True

    def __init__(self, shapes: Shapes):
        if shapes.sequence_channels == 0:
            raise ValueError("lstm_concat needs sequence channels")
        super().__init__()
        self.encoder = TabularEncoder(shapes)
        self.lstm = nn.LSTM(shapes.sequence_channels, CAPACITY["lstm_hidden"], batch_first=True)
        hidden = CAPACITY["mlp_hidden"]
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim + CAPACITY["lstm_hidden"], hidden), nn.ReLU(),
            nn.Dropout(0.05), nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        state = _masked_lstm_pool(self.lstm, batch["sequence"], batch["sequence_mask"])
        return self.head(torch.cat([self.encoder.flat(batch), state], dim=1)).squeeze(-1)


class LSTMFusionControl(nn.Module):
    """One pooled temporal token fused with the tabular tokens by self-attention."""

    name = "lstm_fusion"
    uses_sequence = True

    def __init__(self, shapes: Shapes):
        if shapes.sequence_channels == 0:
            raise ValueError("lstm_fusion needs sequence channels")
        super().__init__()
        self.block = _TokenStack(shapes, extra_tokens=1)
        self.lstm = nn.LSTM(shapes.sequence_channels, CAPACITY["lstm_hidden"], batch_first=True)
        self.temporal_proj = nn.Linear(CAPACITY["lstm_hidden"], self.block.d_model)
        self.gate = nn.Sequential(nn.Linear(self.block.d_model, self.block.d_model), nn.Sigmoid())

    def forward(self, batch):
        state = _masked_lstm_pool(self.lstm, batch["sequence"], batch["sequence_mask"])
        temporal = self.temporal_proj(state).unsqueeze(1)
        temporal = temporal + self.block.type_embedding(
            torch.full((1,), self.block.n_types - 1, dtype=torch.long, device=temporal.device))
        sequence = torch.cat([self.block.tabular_tokens(batch), temporal], dim=1)
        pooled = self.block.pooled(sequence)
        return self.block.head(pooled * self.gate(pooled)).squeeze(-1)


CONTROLS = {
    "mlp": MLPControl,
    "static_transformer": StaticTransformerControl,
    "lstm_concat": LSTMConcatControl,
    "lstm_fusion": LSTMFusionControl,
}


def build_control(name: str, shapes: Shapes) -> nn.Module:
    if name not in CONTROLS:
        raise KeyError(f"unknown control {name!r}; available {sorted(CONTROLS)}")
    return CONTROLS[name](shapes)


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
