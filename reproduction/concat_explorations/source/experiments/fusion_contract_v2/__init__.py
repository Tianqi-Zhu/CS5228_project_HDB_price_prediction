"""Gate-1 comparison contract: aligned samples, history, sequences.

Consumer documentation lives in ``reports/fusion_contract_v2/CONTRACT.md``; this
package only exposes the implementation it describes.
"""

from __future__ import annotations

from experiments.fusion_contract_v2.contract import (  # noqa: F401
    ARM_INPUTS,
    CONTRACT_VERSION,
    HISTORY_LEN,
    HISTORY_LEVELS,
    MAX_HORIZON,
    SAMPLE_SEED,
    SPLITS,
    arm_scalar_columns,
    arm_uses_sequence,
    contract_identity,
    history_columns,
    month_index,
    month_label,
    split_spec,
)

__all__ = [
    "ARM_INPUTS",
    "CONTRACT_VERSION",
    "HISTORY_LEN",
    "HISTORY_LEVELS",
    "MAX_HORIZON",
    "SAMPLE_SEED",
    "SPLITS",
    "arm_scalar_columns",
    "arm_uses_sequence",
    "contract_identity",
    "history_columns",
    "month_index",
    "month_label",
    "split_spec",
]
