"""Feature-identity Transformer: one token per numeric feature, three pooling choices.

The point of this architecture is that no numeric field is ever mixed with another before
attention. Every column j gets its own scalar affine map and its own learned 48-d identity:

    token_j = scaled_x_j * weight_j + bias_j + identity_j

`identity_j` is what makes two features with identical scaled values still distinct tokens;
without it the model could not tell "floor area = 3" from "MAX_FLOOR = 3". A shared
`Linear(n, d)` would reintroduce exactly the bottleneck this architecture is meant to remove.

Categorical columns keep one embedding table each, with code 0 reserved for unknown.

Two masks are kept strictly separate:

* ``key_padding_mask`` — which positions attention may read. A macro or prediction token
  that must be attended to but not averaged belongs here, not in the pooling mask.
* ``pool_mask`` — which positions contribute to the pooled vector.

Conflating them is the failure mode this design exists to avoid.
"""
from __future__ import annotations

import math

import torch
from torch import nn

POOLINGS = ("mean", "cls", "horizon_query")
N_HEAD_TOKENS = 1        # the single prediction/query token slot, when a pooling uses it


class FeatureTransformerError(ValueError):
    pass


class FeatureTokens(nn.Module):
    """One width-`d_model` token per named numeric column, no shared projection.

    The projection must be a **vector per feature**:

        token_j = x_j * weight_j + bias_j + identity_j        weight_j, bias_j in R^d

    An earlier version used scalar ``weight_j``/``bias_j``, i.e. ``x_j * w_j * 1_d``. That is
    a *constant* offset across the width, and the model is Pre-LN plus a final LayerNorm,
    which subtracts the per-token mean. A constant offset therefore cancels exactly:
    ``LN(c*1 + v) == LN(v)``. The numeric values had no effect on the representation at all
    (measured prediction change from perturbing all 56 inputs: ~7e-9, pure float noise).

    Two details matter for the same reason:

    * ``weight`` is initialised in the **zero-mean subspace** (each row is centred), so the
      initial gradient direction is the one LayerNorm preserves rather than the one it kills;
    * ``bias`` is also a vector, and ``identity`` gives every column a distinct starting
      point, so two features holding identical values still produce different tokens.
    """

    def __init__(self, feature_names: list[str], d_model: int = 48):
        super().__init__()
        if not feature_names:
            raise FeatureTransformerError("at least one numeric feature is required")
        if len(set(feature_names)) != len(feature_names):
            raise FeatureTransformerError("feature names must be unique")
        n = len(feature_names)
        self.feature_names = list(feature_names)
        self.d_model = d_model

        weight = torch.randn(n, d_model) / math.sqrt(d_model)
        weight = weight - weight.mean(dim=1, keepdim=True)        # zero-mean rows
        self.weight = nn.Parameter(weight)
        bias = torch.randn(n, d_model) * 0.02
        self.bias = nn.Parameter(bias - bias.mean(dim=1, keepdim=True))
        self.identity = nn.Parameter(torch.randn(n, d_model) * 0.02)

    @property
    def n_tokens(self) -> int:
        return len(self.feature_names)

    def forward(self, numeric: torch.Tensor) -> torch.Tensor:
        if numeric.shape[1] != len(self.feature_names):
            raise FeatureTransformerError(
                f"numeric width {numeric.shape[1]} != {len(self.feature_names)} features")
        contribution = numeric.unsqueeze(-1) * self.weight           # (B, n, d)
        return contribution + self.bias + self.identity

    def initial_tokens(self) -> torch.Tensor:
        """Tokens for an all-zero input row."""
        return self.bias.detach() + self.identity.detach()

    def describe(self) -> dict:
        return {
            "mode": "per_feature", "n_tokens": self.n_tokens, "token_width": self.d_model,
            "formula": "x_j * weight_j + bias_j + identity_j  (weight_j, bias_j in R^d)",
            "weight_shape": list(self.weight.shape),
            "shared_projection": None,
            "weight_row_norm_min": float(self.weight.detach().norm(dim=1).min()),
            "feature_names": list(self.feature_names),
        }


class CategoricalTokens(nn.Module):
    """One embedding table per categorical column; code 0 is unknown."""

    def __init__(self, cardinalities: tuple[int, ...], d_model: int = 48):
        super().__init__()
        if not cardinalities:
            raise FeatureTransformerError("at least one categorical column is required")
        if any(int(c) < 2 for c in cardinalities):
            raise FeatureTransformerError(
                f"every vocabulary needs >= 2 entries (code 0 is unknown); got {cardinalities}")
        self.cardinalities = tuple(int(c) for c in cardinalities)
        self.d_model = d_model
        self.embeddings = nn.ModuleList([nn.Embedding(c, d_model) for c in self.cardinalities])

    @property
    def n_tokens(self) -> int:
        return len(self.cardinalities)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[1] != len(self.cardinalities):
            raise FeatureTransformerError(
                f"categorical width {codes.shape[1]} != {len(self.cardinalities)} columns")
        if codes.min() < 0:
            raise FeatureTransformerError("categorical codes must be non-negative")
        for i, table in enumerate(self.embeddings):
            if int(codes[:, i].max()) >= table.num_embeddings:
                raise FeatureTransformerError(
                    f"column {i}: code {int(codes[:, i].max())} outside vocabulary "
                    f"of {table.num_embeddings}")
        return torch.stack([table(codes[:, i]) for i, table in enumerate(self.embeddings)],
                           dim=1)

    def describe(self) -> dict:
        return {"n_tokens": self.n_tokens, "token_width": self.d_model,
                "cardinalities": list(self.cardinalities), "unknown_code": 0}


class FeatureTransformer(nn.Module):
    """Pre-norm Transformer over feature tokens with a scalar residual head."""

    name = "feature_transformer"

    def __init__(self, numeric_features: list[str], cardinalities: tuple[int, ...],
                 d_model: int = 48, layers: int = 2, heads: int = 4, ff_multiplier: int = 2,
                 dropout: float = 0.05, pooling: str = "mean", context_dim_zero: int = 0):
        super().__init__()
        if pooling not in POOLINGS:
            raise FeatureTransformerError(f"unknown pooling {pooling!r}; available {POOLINGS}")
        if d_model % heads != 0:
            raise FeatureTransformerError(f"d_model {d_model} not divisible by heads {heads}")
        if layers < 1:
            raise FeatureTransformerError("at least one Transformer layer is required")

        self.pooling = pooling
        self.d_model = d_model
        self.numeric = FeatureTokens(numeric_features, d_model)
        self.categorical = CategoricalTokens(cardinalities, d_model)

        n_feature_tokens = self.numeric.n_tokens + self.categorical.n_tokens
        # ids 0..n_feature_tokens-1 are unused, feature tokens take 1..n_feature_tokens,
        # and the query/prediction token gets its own id AFTER them. Using n_types-1 for the
        # query collided with the last feature token when N_HEAD_TOKENS was added.
        self.n_feature_tokens = n_feature_tokens
        self.query_type_id = n_feature_tokens + 1
        self.n_types = n_feature_tokens + 1 + N_HEAD_TOKENS
        self.type_embedding = nn.Embedding(self.n_types, d_model)

        if pooling in ("cls", "horizon_query"):
            self.head_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        if pooling == "horizon_query":
            # the conditioning vector is the existing horizon/calendar context, standardised
            # by the same fit-only scaler; it carries no label and no future observation
            self.query_proj = nn.Linear(context_dim_zero, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model, heads, d_model * ff_multiplier, dropout,
            batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, layers,
                                             enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_model, 1))

    # ------------------------------------------------------------------ pieces
    def feature_tokens(self, batch) -> torch.Tensor:
        numeric = self.numeric(batch["numeric"])
        categorical = self.categorical(batch["categorical"])
        tokens = torch.cat([categorical, numeric], dim=1)
        types = torch.arange(1, 1 + tokens.shape[1], device=tokens.device)
        return tokens + self.type_embedding(types)[None]

    def _query_token(self, batch, dtype) -> torch.Tensor:
        token = self.head_token.expand(batch["numeric"].shape[0], 1, self.d_model)
        if self.pooling == "horizon_query":
            token = token + self.query_proj(batch["query_context"]).unsqueeze(1)
        type_id = torch.full((1,), self.query_type_id, dtype=torch.long,
                             device=token.device)
        return token + self.type_embedding(type_id)[None]

    # ------------------------------------------------------------------ forward
    def forward(self, batch, key_padding_mask: torch.Tensor | None = None):
        tokens = self.feature_tokens(batch)

        if self.pooling == "mean":
            sequence = tokens
            pool_mask = torch.ones(sequence.shape[:2], device=sequence.device)
            leading_tokens = 0
        else:
            query = self._query_token(batch, tokens.dtype)
            sequence = torch.cat([query, tokens], dim=1)
            # both token poolings read position 0; the difference between them is only
            # whether that token is conditioned on the horizon, not which positions are pooled
            pool_mask = torch.zeros(sequence.shape[:2], device=sequence.device)
            pool_mask[:, 0] = 1.0
            leading_tokens = 1

        encoder_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.shape[1] != tokens.shape[1]:
                raise FeatureTransformerError(
                    f"key_padding_mask covers {key_padding_mask.shape[1]} feature positions, "
                    f"but the sequence has {tokens.shape[1]}")
            if leading_tokens:
                # the query token is always visible; shift the feature mask by one
                pad = torch.zeros(key_padding_mask.shape[0], leading_tokens,
                                  dtype=torch.bool, device=key_padding_mask.device)
                encoder_mask = torch.cat([pad, key_padding_mask], dim=1)
            else:
                encoder_mask = key_padding_mask

        # `src_key_padding_mask=True` means "this position is padding": attention never
        # reads it. Zeroing the embedding instead would still leave it readable.
        encoded = self.final_norm(self.encoder(sequence, src_key_padding_mask=encoder_mask))

        if encoder_mask is not None:
            # a padding position is not a feature of this row, so it must not be averaged in
            pool_mask = pool_mask * (~encoder_mask).to(pool_mask.dtype)

        if self.pooling != "cls":
            weights = pool_mask.unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoded[:, 0]

        # every feature may be padding (an all-missing history); keep the result finite
        if not torch.isfinite(pooled).all():
            raise FeatureTransformerError("non-finite pooled representation")

        return self.head(pooled).squeeze(-1)

    # ------------------------------------------------------------------ reporting
    def describe(self) -> dict:
        return {
            "name": self.name, "pooling": self.pooling, "d_model": self.d_model,
            "layers": len(self.encoder.layers),
            "heads": self.encoder.layers[0].self_attn.num_heads,
            "ff_width": self.encoder.layers[0].linear1.out_features,
            "dropout": self.encoder.layers[0].dropout.p,
            "numeric": self.numeric.describe(),
            "categorical": self.categorical.describe(),
            "total_feature_tokens": self.n_feature_tokens,
            "query_type_id": self.query_type_id,
            "type_embedding_rows": self.n_types,
            "pooling_tokens": 0 if self.pooling == "mean" else 1,
            "query_conditioned": self.pooling == "horizon_query",
            "parameters": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
        }


def build_model(numeric_features: list[str], cardinalities: tuple[int, ...],
                context_dim: int, **kwargs) -> FeatureTransformer:
    pooling = kwargs.get("pooling", "mean")
    if pooling not in POOLINGS:
        raise FeatureTransformerError(f"unknown pooling {pooling!r}; available {POOLINGS}")
    return FeatureTransformer(numeric_features, cardinalities,
                              context_dim_zero=context_dim, **kwargs)
