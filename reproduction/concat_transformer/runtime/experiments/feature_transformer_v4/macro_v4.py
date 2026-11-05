"""Macro memory tokens for the v4 feature Transformer.

`neural_fusion_v3.fusion_macro` cannot be reused here: it is written against v2's control
classes (`ATTENTION_CONTROLS`, `n_tabular_tokens`, a single pooled position). v4 has 60
feature tokens, three poolings and its own type-id space, so this is a parallel wrapper over
the same idea.

What must hold, and what each rule protects:

* **`macro=none` is v4 unchanged.** No wrapper, no extra parameters, same state dict.
* **Macro tokens are attention memory, not pooled.** They are appended after the feature
  tokens, visible to every position through attention, and excluded from the pooled mean by
  the mask. If they were averaged in, the head input would change shape/scale as a function
  of the macro branch and the ablation would no longer isolate information.
* **The head width does not change.** Pooling still yields one `d_model` vector, so a
  `none` checkpoint can be compared to a `both` run without a head-size confound.
* **The 48-d encoder output is projected once** into the model width when they differ, and
  that projection is reported, because it is a learned parameter the counterpart lacks.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import torch
from torch import nn

MACRO_MODES = ("none", "dbs", "coe", "both")


class MacroOnFeatureTransformer(nn.Module):
    """v4 model + up to two macro tokens appended as attention memory."""

    def __init__(self, base, source, macro_mode: str):
        super().__init__()
        if macro_mode not in MACRO_MODES or macro_mode == "none":
            raise ValueError(f"MacroOnFeatureTransformer needs an enabled mode, got {macro_mode!r}")
        self.base = base
        self.source = source
        self.macro_mode = macro_mode
        self.macro_tokens = int(source.n_tokens)
        if self.macro_tokens <= 0:
            raise ValueError(f"macro mode {macro_mode!r} produced no tokens")

        d_model = base.d_model
        # make room for the macro type ids, keeping every existing row in place
        old = base.type_embedding
        grown = nn.Embedding(old.num_embeddings + self.macro_tokens, old.embedding_dim)
        with torch.no_grad():
            grown.weight[: old.num_embeddings] = old.weight
        base.type_embedding = grown
        base.n_types = old.num_embeddings + self.macro_tokens
        self.macro_type_ids = list(range(old.num_embeddings,
                                        old.num_embeddings + self.macro_tokens))

        # a single projection into the model width, only when the encoder width differs
        source_dim = int(getattr(source, "d_model", d_model))
        self.token_proj = nn.Linear(source_dim, d_model) if source_dim != d_model else None

    # ---------------------------------------------------------------- forward
    def forward(self, batch, key_padding_mask=None):
        base = self.base
        tokens = base.feature_tokens(batch)

        if base.pooling == "mean":
            sequence = tokens
            pool_mask = torch.ones(sequence.shape[:2], device=sequence.device)
            leading = 0
        else:
            query = base._query_token(batch, tokens.dtype)
            sequence = torch.cat([query, tokens], dim=1)
            pool_mask = torch.zeros(sequence.shape[:2], device=sequence.device)
            pool_mask[:, 0] = 1.0
            leading = 1

        macro = self.source(batch)                      # (B, n_macro, source_dim)
        if macro.ndim != 3 or macro.shape[1] != self.macro_tokens:
            raise ValueError(f"macro source returned {tuple(macro.shape)}; expected "
                             f"(B, {self.macro_tokens}, ...)")
        if self.token_proj is not None:
            macro = self.token_proj(macro)
        elif macro.shape[2] != base.d_model:
            raise ValueError(f"macro token width {macro.shape[2]} != model width {base.d_model}")

        type_ids = torch.tensor(self.macro_type_ids, device=macro.device, dtype=torch.long)
        macro = macro + base.type_embedding(type_ids)[None]

        n_real = sequence.shape[1]
        sequence = torch.cat([sequence, macro], dim=1)
        # macro positions take part in attention but never in the pooled mean
        pool_mask = torch.cat([pool_mask,
                               torch.zeros(macro.shape[0], self.macro_tokens,
                                           device=pool_mask.device)], dim=1)

        encoder_mask = None
        if key_padding_mask is not None:
            if key_padding_mask.shape[1] != tokens.shape[1]:
                raise ValueError(f"key_padding_mask covers {key_padding_mask.shape[1]} "
                                 f"features, sequence has {tokens.shape[1]}")
            pad = torch.zeros(key_padding_mask.shape[0], leading + self.macro_tokens,
                              dtype=torch.bool, device=key_padding_mask.device)
            encoder_mask = torch.cat([pad, key_padding_mask], dim=1) if leading else \
                torch.cat([key_padding_mask, pad], dim=1)

        encoded = base.final_norm(base.encoder(sequence, src_key_padding_mask=encoder_mask))

        if encoder_mask is not None:
            pool_mask = pool_mask * (~encoder_mask).to(pool_mask.dtype)

        if base.pooling == "cls":
            pooled = encoded[:, 0]
        else:
            weights = pool_mask.unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

        del n_real
        return base.head(pooled).squeeze(-1)

    # ---------------------------------------------------------------- reporting
    def describe(self) -> dict:
        base = self.base
        return {
            "control": "feature_transformer",
            "pooling": base.pooling,
            "macro_mode": self.macro_mode,
            "macro_tokens": self.macro_tokens,
            "macro_role": "appended as attention memory, excluded from the pooled mean; "
                          "pooled width and head shape unchanged from macro=none",
            "macro_projection": (None if self.token_proj is None
                                 else {"in": self.token_proj.in_features,
                                       "out": self.token_proj.out_features}),
            "type_embedding_rows": int(base.n_types),
            "base_parameters": int(sum(p.numel() for p in base.parameters())),
            "source_parameters": int(sum(p.numel() for p in self.source.parameters())),
            "parameters": int(sum(p.numel() for p in self.parameters())),
            "source": (self.source.describe() if hasattr(self.source, "describe")
                       else {"type": type(self.source).__name__,
                             "tokens": self.macro_tokens}),
            "semantics": "financial-state embedding learned through the rental loss; "
                          "not a validated measure of economic health",
        }


def build_with_macro(base, source_factory, macro_mode: str):
    """`none` returns the v4 model itself; an enabled mode wraps it."""
    if macro_mode == "none":
        return base
    if macro_mode not in MACRO_MODES:
        raise ValueError(f"unknown macro mode {macro_mode!r}")
    source = source_factory(macro_mode)
    if source is None:
        raise ValueError(f"macro mode {macro_mode!r} needs a token source")
    return MacroOnFeatureTransformer(base, source, macro_mode)


def describe(model) -> dict:
    if isinstance(model, MacroOnFeatureTransformer):
        return model.describe()
    return {"control": "feature_transformer", "pooling": model.pooling,
            "macro_mode": "none", "macro_tokens": 0,
            "macro_role": "no macro branch; v4 model unchanged",
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "semantics": "financial-state embedding learned through the rental loss; "
                         "not a validated measure of economic health"}
