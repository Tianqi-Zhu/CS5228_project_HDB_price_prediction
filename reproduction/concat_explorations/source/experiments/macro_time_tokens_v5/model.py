"""v4 Feature Transformer + v5 block-endpoint macro tokens.

Two arms, compared on orig/A only:

* ``summary2`` — the existing representation: one token per source (DBS, COE) from the
  producer's encoders. This is the reference the plan says to beat.
* ``blocks7`` — three DBS block ends + four COE block ends, each 48-d with a source and a
  calendar-age identity.

**Paired initialisation.** The plan requires identical initial values for every shared base and
RNN/projection parameter, with no live parameter tying between arms. ``paired_initial_state``
builds one seed, one base model and one set of block encoders, then hands out copies: the
``summary2`` arm's producer encoders are *not* the block encoders (they are different modules),
but everything the two arms genuinely share — all 60 feature-token parameters, the encoder, the
head, the type embeddings for the existing rows — starts bitwise identical, and the block
encoders are seeded so ``blocks7`` is reproducible on its own.

**Mask.** The corrected construction from ``tokens.build_encoder_mask`` is used throughout, so
``[query, 60 features, n_macro]`` gets a mask that lines up with the features. Macro positions
are then masked **per token**, from the source's ``(B, n_macro)`` validity: an empty block is
padding for attention even when its sibling blocks are real. A row-level "was anything
observed" flag is not sufficient — it let four empty blocks attend, and the reviewer measured a
7.68 SGD prediction change from perturbing those very tokens.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

import torch
from torch import nn

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

from experiments.macro_time_tokens_v5 import tokens as T

ARMS = ("summary2", "blocks7")


class MacroBlocksOnFeatureTransformer(nn.Module):
    """v4 model + a variable number of macro tokens, all attention memory."""

    def __init__(self, base, source, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise T.MacroTimeTokenError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.base = base
        self.source = source
        self.arm = arm
        self.macro_tokens = int(source.n_tokens)
        if self.macro_tokens <= 0:
            raise T.MacroTimeTokenError(f"arm {arm!r} produced no tokens")

        old = base.type_embedding
        grown = nn.Embedding(old.num_embeddings + self.macro_tokens, old.embedding_dim)
        with torch.no_grad():
            grown.weight[: old.num_embeddings] = old.weight
        base.type_embedding = grown
        base.n_types = old.num_embeddings + self.macro_tokens
        self.macro_type_ids = list(range(old.num_embeddings,
                                         old.num_embeddings + self.macro_tokens))

        source_dim = int(getattr(source, "d_model", base.d_model))
        self.token_proj = nn.Linear(source_dim, base.d_model) if source_dim != base.d_model else None

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

        macro, macro_valid = self.source(batch)          # (B, n_macro, d), (B, n_macro) bool
        if macro.shape[1] != self.macro_tokens:
            raise T.MacroTimeTokenError(f"source returned {tuple(macro.shape)}; "
                                        f"expected (B, {self.macro_tokens}, ...)")
        if macro_valid.shape != (macro.shape[0], self.macro_tokens):
            raise T.MacroTimeTokenError(
                f"source returned validity of shape {tuple(macro_valid.shape)}; expected "
                f"({macro.shape[0]}, {self.macro_tokens}) -- per-token, not row-level")
        if self.token_proj is not None:
            macro = self.token_proj(macro)
        elif macro.shape[2] != base.d_model:
            raise T.MacroTimeTokenError(f"macro width {macro.shape[2]} != model width "
                                        f"{base.d_model}")

        type_ids = torch.tensor(self.macro_type_ids, device=macro.device, dtype=torch.long)
        macro = macro + base.type_embedding(type_ids)[None]

        sequence = torch.cat([sequence, macro], dim=1)
        # macro tokens are attention memory: never pooled
        pool_mask = torch.cat([
            pool_mask,
            torch.zeros(macro.shape[0], self.macro_tokens, device=pool_mask.device)], dim=1)

        # --- the corrected mask path
        feature_padding = T.build_encoder_mask(key_padding_mask, leading, self.macro_tokens)
        # per-token macro validity, placed at the macro positions only
        macro_padding = torch.zeros(sequence.shape[:2], dtype=torch.bool,
                                    device=sequence.device)
        macro_padding[:, sequence.shape[1] - self.macro_tokens:] = ~macro_valid
        encoder_mask = T.combine_masks(feature_padding, macro_padding)

        encoded = base.final_norm(base.encoder(sequence, src_key_padding_mask=encoder_mask))

        if encoder_mask is not None:
            pool_mask = pool_mask * (~encoder_mask).to(pool_mask.dtype)

        if base.pooling == "cls":
            pooled = encoded[:, 0]
        else:
            weights = pool_mask.unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

        if not torch.isfinite(pooled).all():
            raise T.MacroTimeTokenError("non-finite pooled representation")
        return base.head(pooled).squeeze(-1)

    # ---------------------------------------------------------------- reporting
    def describe(self) -> dict:
        base = self.base
        source = self.source.describe() if hasattr(self.source, "describe") else {
            "type": type(self.source).__name__, "tokens": self.macro_tokens}
        return {
            "control": "feature_transformer",
            "arm": self.arm,
            "pooling": base.pooling,
            "macro_mode": self.arm,
            "macro_tokens": self.macro_tokens,
            "macro_role": "attention memory, excluded from the pooled mean; head unchanged",
            "macro_projection": (None if self.token_proj is None else
                                 {"in": self.token_proj.in_features,
                                  "out": self.token_proj.out_features}),
            "type_embedding_rows": int(base.n_types),
            "base_parameters": int(sum(p.numel() for p in base.parameters())),
            "source_parameters": int(sum(p.numel() for p in self.source.parameters())),
            "parameters": int(sum(p.numel() for p in self.parameters())),
            "source": source,
            "semantics": "financial-state embedding learned through the rental loss; not a "
                         "validated measure of economic health",
        }


def paired_initial_state(seed: int, build_base, build_block_dbs, build_block_coe):
    """One seed, one base, both block encoders — the shared starting point for every arm.

    Returns ``(base_state, block_states, fingerprint)``. The base state is deep-copied per arm
    so no live parameter is shared, while both arms still begin from identical values.
    """
    torch.manual_seed(seed)
    import random as _random
    _random.seed(seed)

    base = build_base()
    dbs = build_block_dbs()
    coe = build_block_coe()

    def snapshot(module):
        return {k: v.detach().clone() for k, v in module.state_dict().items()}

    base_state = snapshot(base)
    block_states = {"dbs": snapshot(dbs), "coe": snapshot(coe)}
    fingerprint = {
        "seed": seed,
        "base_sha": _digest(base_state),
        "dbs_sha": _digest(block_states["dbs"]),
        "coe_sha": _digest(block_states["coe"]),
        "base_parameters": int(sum(v.numel() for v in base_state.values())),
        "dbs_parameters": int(sum(v.numel() for v in block_states["dbs"].values())),
        "coe_parameters": int(sum(v.numel() for v in block_states["coe"].values())),
    }
    return base_state, block_states, fingerprint


def _digest(state: dict) -> str:
    """Order-independent-enough digest of a state dict for parity auditing."""
    import hashlib

    hasher = hashlib.sha256()
    for key in sorted(state):
        hasher.update(key.encode())
        hasher.update(state[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]
