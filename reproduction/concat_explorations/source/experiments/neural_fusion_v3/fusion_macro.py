"""Attach macro tokens to a v2 Transformer control as extra *memory* tokens.

Design constraint that drives everything here: **`macro=none` must be v2, not a look-alike.**
So `build_fusion(..., macro_mode="none")` returns the v2 model object itself — no wrapper, no
extra module, no change to pooling or token count. Its `state_dict` keys and its forward
output are v2's, bit for bit (see `tests/test_macro_fusion.py`).

When macro tokens are enabled:

* each enabled branch contributes **one** token of width `d_model`
  (DBS: 63 trading days; COE: 12 calendar months x 4 categories) — two tokens for `both`;
* the tokens are appended to the sequence that enters the Transformer, so they are real
  keys/values that every tabular token can attend to;
* they are **excluded from the pooled mean** via the mask argument v2's `pooled()` already
  accepts. Attention reads them; the pooled representation does not simply contain them.
  That keeps the tabular pathway's output shape and head input identical to v2, so any
  change in the score is attributable to information, not to a wider head;
* `type_embedding` is extended by exactly `n_macro_tokens` rows, and the first rows are
  copied from v2 so a `none` checkpoint transfers into an enabled run unchanged.

Only the two attention controls accept macro tokens. `mlp` and `lstm_concat` have no memory
to attend to, so asking for macro there is an error rather than a silent concat into the
flat vector, which would be a different mechanism wearing the same name.

The DBS and COE embeddings are learned through the rental loss. They are financial-state
embeddings, **not** validated measures of economic health.
"""
from __future__ import annotations

import sys as _sys
import types
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import torch
from torch import nn

from neural_fusion_v2.settings import CAPACITY

MACRO_MODES = ("none", "dbs", "coe", "both")
ATTENTION_CONTROLS = ("static_transformer", "lstm_fusion")
SEMANTICS = ("financial-state embedding learned through the rental loss; "
             "not a validated measure of economic health")


class MacroFusionUnsupported(ValueError):
    """Macro tokens were requested on a control that has no attention memory."""


def _token_stack(model: nn.Module) -> nn.Module:
    return model.block if hasattr(model, "block") else model


def _extend_type_embedding(stack: nn.Module, n_macro_tokens: int) -> None:
    """Grow the type embedding by ``n_macro_tokens`` rows, preserving the old rows.

    Copied rather than re-initialised so a checkpoint trained with ``macro=none`` keeps
    every one of its type vectors when the same run is resumed with macro enabled.
    """
    if n_macro_tokens <= 0:
        return
    old = stack.type_embedding
    old_n, width = old.num_embeddings, old.embedding_dim
    grown = nn.Embedding(old_n + n_macro_tokens, width)
    with torch.no_grad():
        grown.weight[:old_n] = old.weight
    stack.type_embedding = grown
    stack.n_types = old_n + n_macro_tokens


class MacroFusion(nn.Module):
    """A v2 attention control plus macro memory tokens.

    ``source`` is any module exposing ``forward(batch) -> (B, n_tokens, d_model)`` and
    ``n_tokens``. It is kept as a submodule so `state_dict`, `train()` and `eval()` cover
    the encoders, and so an optimiser built from `self.parameters()` really does step them.
    """

    def __init__(self, base: nn.Module, source: nn.Module, macro_mode: str):
        super().__init__()
        if macro_mode not in MACRO_MODES or macro_mode == "none":
            raise MacroFusionUnsupported(
                f"MacroFusion is only for enabled modes, got {macro_mode!r}")
        name = getattr(base, "name", type(base).__name__)
        if name not in ATTENTION_CONTROLS:
            raise MacroFusionUnsupported(
                f"{name!r} has no attention memory; macro tokens are only defined for "
                f"{ATTENTION_CONTROLS}. Use a Transformer control or macro=none.")
        self.base = base
        self.source = source
        self.macro_mode = macro_mode
        self.macro_tokens = int(source.n_tokens)
        if self.macro_tokens <= 0:
            raise MacroFusionUnsupported(
                f"macro mode {macro_mode!r} produced no tokens; use 'none' instead")
        _extend_type_embedding(_token_stack(base), self.macro_tokens)

    # ---------------------------------------------------------------- forward
    def _macro_tokens(self, batch, block) -> torch.Tensor:
        tokens = self.source(batch)
        if tokens.ndim != 3 or tokens.shape[2] != block.d_model:
            raise ValueError(
                f"macro source must return (B, n, {block.d_model}); got {tuple(tokens.shape)}")
        if tokens.shape[1] != self.macro_tokens:
            raise ValueError(
                f"macro source declared {self.macro_tokens} tokens but returned "
                f"{tokens.shape[1]}")
        # type ids for macro tokens sit after every v2 type id
        first = block.n_types - self.macro_tokens
        types = torch.arange(first, block.n_types, device=tokens.device)
        return tokens + block.type_embedding(types)[None]

    def forward(self, batch):
        block = _token_stack(self.base)
        name = self.base.name

        if name == "static_transformer":
            tabular = block.tabular_tokens(batch)
            macro = self._macro_tokens(batch, block)
            n_real = tabular.shape[1]
            sequence = torch.cat([tabular, macro], dim=1)
            mask = torch.ones(sequence.shape[:2], device=sequence.device)
            mask[:, n_real:] = 0.0            # attend to macro, never pool it
            return block.head(block.pooled(sequence, mask)).squeeze(-1)

        # lstm_fusion: tabular + one temporal token from v2, then macro memory
        from neural_fusion_v2.models import _masked_lstm_pool
        state = _masked_lstm_pool(self.base.lstm, batch["sequence"], batch["sequence_mask"])
        temporal = self.base.temporal_proj(state).unsqueeze(1)
        temporal = temporal + block.type_embedding(
            torch.full((1,), block.n_types - self.macro_tokens - 1, dtype=torch.long,
                       device=temporal.device))
        tabular = block.tabular_tokens(batch)
        macro = self._macro_tokens(batch, block)
        n_real = tabular.shape[1] + 1
        sequence = torch.cat([tabular, temporal, macro], dim=1)
        mask = torch.ones(sequence.shape[:2], device=sequence.device)
        mask[:, n_real:] = 0.0
        pooled = block.pooled(sequence, mask)
        return block.head(pooled * self.base.gate(pooled)).squeeze(-1)

    # ---------------------------------------------------------------- reporting
    def describe(self) -> dict:
        block = _token_stack(self.base)
        return {
            "control": self.base.name,
            "macro_mode": self.macro_mode,
            "macro_tokens": self.macro_tokens,
            "sequence_length": int(block.n_tabular_tokens
                                   + (1 if self.base.name == "lstm_fusion" else 0)
                                   + self.macro_tokens),
            "pooled_positions": int(block.n_tabular_tokens
                                    + (1 if self.base.name == "lstm_fusion" else 0)),
            "macro_role": "appended to the attention sequence, excluded from pooled mean "
                          "by mask, so head input width is unchanged from macro=none",
            "type_embedding_rows": int(block.n_types),
            "context_projection_kept": True,
            "source": (self.source.describe() if hasattr(self.source, "describe")
                       else {"type": type(self.source).__name__,
                             "tokens": self.macro_tokens}),
            "semantics": SEMANTICS,
        }

    def macro_parameters(self) -> dict[str, int]:
        return {"total": int(sum(p.numel() for p in self.source.parameters())),
                "trainable": int(sum(p.numel() for p in self.source.parameters()
                                     if p.requires_grad))}


class LinearMacroSource(nn.Module):
    """A minimal token source used for wrapper tests; carries no macro data of its own.

    Kept in the package (not in the tests) so the wrapper's contract has one executable
    statement of what a source must provide: `n_tokens` and `(B, n, d_model)` output.
    """

    def __init__(self, n_tokens: int, d_model: int = CAPACITY["d_model"]):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.n_tokens = int(n_tokens)
        self.d_model = d_model
        self.register_buffer("bias_tokens", torch.zeros(1, n_tokens, d_model))

    def forward(self, batch) -> torch.Tensor:
        anchor = batch["context"][:, :1]
        return self.proj(anchor).unsqueeze(1) + self.bias_tokens.expand(anchor.shape[0], -1, -1)

    def describe(self) -> dict:
        return {"type": "LinearMacroSource", "tokens": self.n_tokens,
                "warning": "wrapper-test placeholder, carries no macro data"}


def build_fusion(base_factory, name: str, shapes, macro_mode: str,
                 source_factory=None) -> nn.Module:
    """Build a control; ``macro=none`` returns v2's class untouched.

    ``base_factory(name, shapes)`` must return the v2 model (v3 block mode does exactly
    that), and ``source_factory(macro_mode)`` must return a token source or ``None``.
    """
    base = base_factory(name, shapes)
    if macro_mode == "none":
        return base                                  # literally v2, not a copy of it
    if macro_mode not in MACRO_MODES:
        raise MacroFusionUnsupported(f"unknown macro mode {macro_mode!r}")
    if source_factory is None:
        raise MacroFusionUnsupported(
            f"macro mode {macro_mode!r} needs a token source; none was given")
    source = source_factory(macro_mode)
    if source is None:
        raise MacroFusionUnsupported(
            f"macro mode {macro_mode!r} needs a token source; the factory returned None")
    return MacroFusion(base, source, macro_mode)


def describe(model: nn.Module) -> dict:
    """Manifest-facing description for either a bare v2 control or a fused one.

    Includes the numeric token layout as `encoder`, because the macro fields alone cannot
    distinguish a 6-token block model from a 26-token per-feature one.
    """
    from .models import model_description

    if isinstance(model, MacroFusion):
        payload = model.describe()
        payload["encoder"] = model_description(model.base)["encoder"]
        payload["parameters"] = model_description(model.base)["parameters"]
        return payload
    base = model_description(model)
    return {
        "control": getattr(model, "name", type(model).__name__),
        "macro_mode": "none",
        "macro_tokens": 0,
        "sequence_length": int(_token_stack(model).n_tabular_tokens
                               + (1 if getattr(model, "name", "") == "lstm_fusion" else 0)),
        "pooled_positions": int(_token_stack(model).n_tabular_tokens
                                + (1 if getattr(model, "name", "") == "lstm_fusion" else 0)),
        "macro_role": "no macro branch; v2 model unchanged",
        "type_embedding_rows": int(_token_stack(model).n_types),
        "context_projection_kept": True,
        "source": None,
        "semantics": SEMANTICS,
        "encoder": base["encoder"],
        "parameters": base["parameters"],
    }
