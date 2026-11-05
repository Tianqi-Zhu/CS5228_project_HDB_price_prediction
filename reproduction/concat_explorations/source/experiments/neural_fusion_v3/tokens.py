"""Numeric house/spatial tokenisation: one block token versus one token per feature.

Reference (v2, unchanged): all 21 standardised numeric columns -> `Linear(21, 8)` ->
`tanh` -> one token of width 8, exactly like the history and context tokens.

Alternative (`per_feature`): each column gets its own token, projected **directly to 48**:

    token_j = tanh(x_j * w_j + b_j) * scale_j + identity_j

No 21 -> 8 bottleneck, and no stage where a feature is squeezed to width 8 before reaching
`d_model`. `w_j`, `b_j` are scalars; `scale_j` and `identity_j` are 48-wide vectors, so the
token has width 48 from the outset.

Everything else — categorical embeddings, the history token, the context token and its
projection, the heads, the LSTM, the optimiser — is whatever v2 uses. This module only
decides how the numeric block is represented, and `neural_fusion_v3.models` reuses v2's
classes for the block reference so the two cannot drift.

Columns are resolved **by name** against the contract's declared order. A missing,
duplicate or unnamed column raises instead of shifting indices.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

from dataclasses import dataclass

import torch
from torch import nn

FEATURE_TOKEN_MODES = ("block", "per_feature")


class TokenisationError(ValueError):
    pass


def resolve_feature_index(numeric_columns: list[str], expected: list[str] | None = None) -> list[int]:
    """Positions of each numeric column, validated against the contract's own order."""
    if len(set(numeric_columns)) != len(numeric_columns):
        duplicates = sorted({c for c in numeric_columns if numeric_columns.count(c) > 1})
        raise TokenisationError(f"duplicate numeric columns: {duplicates}")
    if expected is None:
        return list(range(len(numeric_columns)))
    lookup = {name: i for i, name in enumerate(numeric_columns)}
    missing = [name for name in expected if name not in lookup]
    if missing:
        raise TokenisationError(f"expected numeric columns absent from the contract: {missing}")
    if len(set(expected)) != len(expected):
        raise TokenisationError("expected feature list contains duplicates")
    extra = [name for name in numeric_columns if name not in set(expected)]
    if extra:
        raise TokenisationError(
            f"contract supplies numeric columns not covered by the feature list: {extra}")
    return [lookup[name] for name in expected]


class PerFeatureTokens(nn.Module):
    """One width-`out_dim` token per numeric column, no shared bottleneck.

    ``token_j = tanh(x_j * w_j + b_j) * scale_j + identity_j``

    A distinct identity per column means two features with identical scaled values still
    produce different tokens, so attention can weight them separately.
    """

    mode = "per_feature"

    def __init__(self, feature_names: list[str], out_dim: int = 48):
        super().__init__()
        if not feature_names:
            raise TokenisationError("per-feature tokenisation needs at least one feature")
        if len(set(feature_names)) != len(feature_names):
            raise TokenisationError("feature names must be unique")
        self.feature_names = list(feature_names)
        n = len(feature_names)
        self.weight = nn.Parameter(torch.ones(n))
        self.bias = nn.Parameter(torch.zeros(n))
        self.scale = nn.Parameter(torch.ones(n, out_dim))
        self.identity = nn.Parameter(torch.randn(n, out_dim) * 0.02)
        self.out_dim = out_dim

    @property
    def n_tokens(self) -> int:
        return len(self.feature_names)

    def forward(self, numeric: torch.Tensor) -> torch.Tensor:
        if numeric.shape[1] != len(self.feature_names):
            raise TokenisationError(
                f"numeric width {numeric.shape[1]} != {len(self.feature_names)} named features")
        projected = torch.tanh(numeric * self.weight + self.bias)      # (B, n)
        return projected.unsqueeze(-1) * self.scale + self.identity     # (B, n, out_dim)

    def describe(self) -> dict:
        return {
            "mode": self.mode,
            "n_numeric_in": len(self.feature_names),
            "n_tokens": self.n_tokens,
            "token_width": self.out_dim,
            "bottleneck": None,
            "formula": "tanh(x_j*w_j+b_j)*scale_j + identity_j, width=out_dim",
            "feature_order": list(self.feature_names),
        }


def feature_token_fingerprint(module: nn.Module) -> dict:
    payload = module.describe()
    payload["feature_indices"] = list(getattr(module, "feature_indices", []))
    payload["feature_names"] = list(getattr(module, "feature_names", []))
    return payload


@dataclass(frozen=True)
class TokenConfig:
    mode: str = "block"

    def key(self) -> str:
        return f"numeric_{self.mode}"
