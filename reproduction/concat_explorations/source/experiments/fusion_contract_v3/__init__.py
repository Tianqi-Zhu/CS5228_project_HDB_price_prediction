"""Fusion Contract v3 — macro (DBS + COE) inputs for the rent comparison.

Producer package.  The consumer API is documented in
``reports/fusion_contract_v3/CONTRACT.md``; this package only implements it.
"""

from __future__ import annotations

from experiments.fusion_contract_v3.contract import (  # noqa: F401
    ARM_INPUTS,
    COE_CATEGORIES,
    COE_SCALARS,
    CONTRACT_VERSION,
    DBS_CHANNELS,
    DBS_LOOKBACK_ROWS,
    DBS_SCALARS,
    DBS_SYMBOL,
    DEFAULT_POLICY,
    POLICIES,
    POLICY_ORIGIN_ASOF,
    POLICY_TARGET_PREVIOUS_MONTH,
    SPLITS,
    arm_scalar_roles,
    arm_uses_coe_sequence,
    arm_uses_dbs_sequence,
    contract_identity,
    month_index,
    month_label,
    split_spec,
)

__all__ = [
    "ARM_INPUTS",
    "COE_CATEGORIES",
    "COE_SCALARS",
    "CONTRACT_VERSION",
    "DBS_CHANNELS",
    "DBS_LOOKBACK_ROWS",
    "DBS_SCALARS",
    "DBS_SYMBOL",
    "DEFAULT_POLICY",
    "POLICIES",
    "POLICY_ORIGIN_ASOF",
    "POLICY_TARGET_PREVIOUS_MONTH",
    "SPLITS",
    "arm_scalar_roles",
    "arm_uses_coe_sequence",
    "arm_uses_dbs_sequence",
    "contract_identity",
    "month_index",
    "month_label",
    "split_spec",
]
