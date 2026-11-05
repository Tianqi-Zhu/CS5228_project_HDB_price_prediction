"""Macro encoders v3 — DBS and COE sequence encoders over the v3 macro contract.

Producer-side package.  It reads the *actual* exported v3 artifacts (manifest,
``macro/<split>/stage<X>.npz``, ``features/<split>/stage<X>.npz``) and exposes a
small API for a consumer that wants macro tokens in a Transformer.

The consumer API is documented in ``reports/macro_encoders_v3/API.md``.
"""

from __future__ import annotations

from experiments.macro_encoders_v3.contract import (  # noqa: F401
    COE_CATEGORIES,
    COE_CHANNELS,
    COE_FLAT_CHANNELS,
    COE_LOOKBACK_MONTHS,
    COE_SCALARS,
    CONTRACT_VERSION,
    DBS_CHANNELS,
    DBS_LOOKBACK_ROWS,
    DBS_SCALARS,
    DBS_SYMBOL,
    DEFAULT_TOKEN_DIM,
    HIDDEN_DIM,
    TOKEN_DIM,
    latest_v3_contract_dir,
    load_manifest,
    stage_entry,
)

__all__ = [
    "COE_CATEGORIES",
    "COE_CHANNELS",
    "COE_FLAT_CHANNELS",
    "COE_LOOKBACK_MONTHS",
    "COE_SCALARS",
    "CONTRACT_VERSION",
    "DBS_CHANNELS",
    "DBS_LOOKBACK_ROWS",
    "DBS_SCALARS",
    "DBS_SYMBOL",
    "DEFAULT_TOKEN_DIM",
    "HIDDEN_DIM",
    "TOKEN_DIM",
    "latest_v3_contract_dir",
    "load_manifest",
    "stage_entry",
]
