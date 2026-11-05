"""Four small regressors isolating recurrence and Transformer fusion.

All four consume the *same* static feature set, the *same* causal origin anchor and
the *same* horizon/calendar context; every model predicts an offset from that anchor.
They differ only in whether they see the shared monthly history and how they combine
it, which is what makes the ablation interpretable:

    mlp                 static features + context                              -> MLP
    static_transformer  categorical + grouped-numeric tokens + context token    -> Transformer
    lstm_concat         LSTM(final state) + static features + context           -> MLP
    lstm_transformer    static tokens + context token + pooled temporal token   -> Transformer

Token layout: one token per categorical feature, one token per *semantic* numeric
group (``config.NUMERIC_GROUPS``), one context token.  That is 10 tokens for the static
Transformer and 11 for the fusion model -- narrower on purpose than one token per
numeric scalar (29 tokens), which measured 39.5 s/epoch on this CPU at 2 threads.

Depth and width are set by ``TrainingConfig`` (``layers=2``, ``d_model=48``,
``heads=4``, ``lstm_hidden=32``), NOT by the constructor defaults below.  The
constructor defaults are only for direct instantiation in tests.  A run's
``model_kwargs_actual`` in its metadata is the authoritative record of what was built.

Column assignment is by *name*: ``group_index_map`` resolves each group's members to
positions in ``StaticPreprocessor.numeric_matrix`` order, and the encoder gathers those
positions.  Never slice the numeric matrix by group size -- the input order is not the
declaration order, and a positional split silently feeds every group the wrong columns
while all shapes stay valid (this bug existed and is covered by
``tests/test_numeric_groups.py``).

Attention weights are NOT causal effects, correlations or feature importances.
"""
from __future__ import annotations

import torch
from torch import nn

from config import NUMERIC_GROUPS, PANEL_CHANNELS


class GroupedStaticEncoder(nn.Module):
    """Categorical embeddings + a learned projection per numeric group.

    Each group selects its columns by *position index* computed from column names
    (``group_index_map``), never by ``torch.split`` over the input matrix: the input
    order follows ``StaticPreprocessor.numeric_matrix``, which differs from the group
    declaration order.  The resolved index map and the column names are stored on the
    module and saved with the checkpoint.
    """

    def __init__(self, cardinalities: list[int], group_indices: list[list[int]],
                 group_names: dict[str, list[str]] | None = None, embed_dim: int = 8):
        super().__init__()
        if not group_indices or sum(len(index) for index in group_indices) == 0:
            raise ValueError("At least one non-empty numeric group is required")
        self.embeddings = nn.ModuleList([nn.Embedding(size, embed_dim) for size in cardinalities])
        self.group_indices = [list(map(int, index)) for index in group_indices]
        self.group_names = group_names or {}
        self.group_linear = nn.ModuleList([nn.Linear(len(index), embed_dim) for index in self.group_indices])
        self._index_cache: dict[tuple[int, str], torch.Tensor] = {}
        self.embed_dim = embed_dim
        self.n_categorical = len(cardinalities)
        self.n_tokens = self.n_categorical + len(self.group_indices)
        self.out_dim = embed_dim * self.n_tokens

    def _index(self, group: int, device: torch.device) -> torch.Tensor:
        key = (group, str(device))
        cached = self._index_cache.get(key)
        if cached is None:
            cached = torch.tensor(self.group_indices[group], dtype=torch.long, device=device)
            self._index_cache[key] = cached
        return cached

    def _parts(self, numeric: torch.Tensor, categorical: torch.Tensor) -> list[torch.Tensor]:
        if numeric.shape[1] != sum(len(index) for index in self.group_indices):
            raise ValueError(
                f"numeric width {numeric.shape[1]} != configured group width "
                f"{sum(len(index) for index in self.group_indices)}")
        parts = [embedding(categorical[:, i]) for i, embedding in enumerate(self.embeddings)]
        for group, linear in enumerate(self.group_linear):
            selected = numeric.index_select(1, self._index(group, numeric.device))
            parts.append(torch.tanh(linear(selected)))
        return parts

    def flat(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """(B, out_dim)."""
        return torch.cat(self._parts(numeric, categorical), dim=1)

    def tokens(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """(B, n_tokens, embed_dim)."""
        return torch.stack(self._parts(numeric, categorical), dim=1)


class MLP(nn.Module):
    name = "mlp"
    uses_history = False
    uses_transformer = False

    def __init__(self, cardinalities, group_indices, context_dim, hidden=128, group_names=None):
        super().__init__()
        self.encoder = GroupedStaticEncoder(cardinalities, group_indices, group_names)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim + context_dim, hidden), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        static = self.encoder.flat(batch["numeric"], batch["categorical"])
        return self.head(torch.cat([static, batch["context"]], dim=1)).squeeze(-1)


class _TransformerStack(nn.Module):
    """Projection, type embeddings, encoder, masked mean-pool and regression head."""

    def __init__(self, encoder: GroupedStaticEncoder, context_dim: int, d_model: int,
                 layers: int, heads: int, n_extra_tokens: int = 0):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.proj = nn.Linear(encoder.embed_dim, d_model)
        self.n_static_tokens = encoder.n_tokens
        self.n_types = 1 + encoder.n_tokens + n_extra_tokens      # context + static + extras
        self.feature_type = nn.Embedding(self.n_types, d_model)
        self.context_proj = nn.Linear(context_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 2, 0.05,
                                          batch_first=True, norm_first=False)
        self.stack = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def static_tokens(self, batch) -> torch.Tensor:
        tokens = self.proj(self.encoder.tokens(batch["numeric"], batch["categorical"]))
        types = torch.arange(1, 1 + self.n_static_tokens, device=tokens.device)
        return tokens + self.feature_type(types)[None]

    def context_token(self, batch) -> torch.Tensor:
        token = self.context_proj(batch["context"]).unsqueeze(1)
        return token + self.feature_type(torch.zeros(1, dtype=torch.long, device=token.device))

    def pool(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.norm(self.stack(sequence)).mean(dim=1)


class StaticTransformer(nn.Module):
    name = "static_transformer"
    uses_history = False
    uses_transformer = True

    def __init__(self, cardinalities, group_indices, context_dim, d_model=48, layers=1, heads=4,
                 group_names=None):
        super().__init__()
        self.block = _TransformerStack(GroupedStaticEncoder(cardinalities, group_indices, group_names),
                                       context_dim, d_model, layers, heads)

    def forward(self, batch):
        context = self.block.context_token(batch)
        static = self.block.static_tokens(batch)
        return self.block.head(self.block.pool(torch.cat([context, static], dim=1))).squeeze(-1)


class LSTMConcat(nn.Module):
    name = "lstm_concat"
    uses_history = True
    uses_transformer = False

    def __init__(self, cardinalities, group_indices, context_dim, hidden=128, lstm_hidden=32,
                 lstm_layers=1, group_names=None):
        super().__init__()
        self.encoder = GroupedStaticEncoder(cardinalities, group_indices, group_names)
        self.lstm = nn.LSTM(len(PANEL_CHANNELS), lstm_hidden, lstm_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim + context_dim + lstm_hidden, hidden), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch):
        state = run_lstm(self.lstm, batch["panel"], batch["panel_mask"])
        static = self.encoder.flat(batch["numeric"], batch["categorical"])
        return self.head(torch.cat([static, batch["context"], state], dim=1)).squeeze(-1)


class LSTMFusion(nn.Module):
    """LSTM history summarised into one temporal token, fused with static tokens.

    The temporal token is a masked mean of the LSTM output sequence, so padded steps
    contribute nothing.  Using every timestep as its own token was also tried and
    costs ~2x the wall time for the same token count budget.
    """

    name = "lstm_transformer"
    uses_history = True
    uses_transformer = True

    def __init__(self, cardinalities, group_indices, context_dim, d_model=48, layers=1, heads=4,
                 lstm_hidden=32, lstm_layers=1, group_names=None):
        super().__init__()
        self.block = _TransformerStack(GroupedStaticEncoder(cardinalities, group_indices, group_names),
                                       context_dim, d_model, layers, heads, n_extra_tokens=1)
        self.lstm = nn.LSTM(len(PANEL_CHANNELS), lstm_hidden, lstm_layers, batch_first=True)
        self.temporal_proj = nn.Linear(lstm_hidden, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, batch):
        panel, mask = batch["panel"], batch["panel_mask"]
        hidden, _ = self.lstm(panel)                                   # (B, T, H)
        weights = mask.unsqueeze(-1)
        pooled_state = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        temporal = self.temporal_proj(pooled_state).unsqueeze(1)
        temporal = temporal + self.block.feature_type(
            torch.full((1,), self.block.n_types - 1, dtype=torch.long, device=hidden.device))

        sequence = torch.cat([self.block.context_token(batch),
                              self.block.static_tokens(batch),
                              temporal], dim=1)
        pooled = self.block.pool(sequence)
        return self.block.head(pooled * self.gate(pooled)).squeeze(-1)


def run_lstm(lstm: nn.LSTM, panel: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Final hidden state of each real (left-aligned, trailing-padded) sequence."""
    lengths = mask.sum(dim=1).long().clamp(min=1, max=panel.shape[1])
    packed = nn.utils.rnn.pack_padded_sequence(panel, lengths.cpu(), batch_first=True, enforce_sorted=False)
    _, (hidden, _) = lstm(packed)
    return hidden[-1]


MODELS = {
    "mlp": MLP,
    "static_transformer": StaticTransformer,
    "lstm_concat": LSTMConcat,
    "lstm_transformer": LSTMFusion,
}


def build_model(name: str, cardinalities, group_indices, context_dim, **kwargs) -> nn.Module:
    return MODELS[name](cardinalities, group_indices, context_dim, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def group_index_map(columns: list[str]) -> tuple[list[list[int]], dict[str, list[str]]]:
    """Map every numeric group to the *positions* of its columns in ``columns``.

    The model's input matrix is emitted in ``StaticPreprocessor.numeric_matrix`` order,
    which is NOT the order of ``config.NUMERIC_GROUPS``.  Slicing that matrix by group
    size (``torch.split``) therefore feeds each group the wrong columns -- a bug that
    leaves every tensor shape valid, so no shape test can catch it.  This function is
    the only place that decides which columns a group receives, and it does so by
    column *name*.
    """
    lookup = {name: i for i, name in enumerate(columns)}
    if len(lookup) != len(columns):
        duplicates = sorted({c for c in columns if columns.count(c) > 1})
        raise KeyError(f"Duplicate numeric columns: {duplicates}")
    indices: list[list[int]] = []
    names: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for group, members in NUMERIC_GROUPS.items():
        unknown = [m for m in members if m not in lookup]
        if unknown:
            raise KeyError(f"Numeric group {group!r} references unknown columns {unknown}")
        overlap = assigned & set(members)
        if overlap:
            raise KeyError(f"Columns assigned to more than one group: {sorted(overlap)}")
        assigned |= set(members)
        indices.append([lookup[m] for m in members])
        names[group] = list(members)
    unused = [c for c in columns if c not in assigned]
    if unused:
        raise KeyError(f"Numeric columns not assigned to any group: {unused}")
    return indices, names


def group_sizes_from(columns: list[str]) -> tuple[list[int], dict[str, int]]:
    """Group sizes for ``config.NUMERIC_GROUPS`` given the expanded numeric columns."""
    indices, names = group_index_map(columns)
    return [len(index) for index in indices], {group: len(members) for group, members in names.items()}
