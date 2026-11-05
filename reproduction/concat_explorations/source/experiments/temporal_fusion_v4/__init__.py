"""v4 temporal branch: padding-safe recurrent encoders for the rent sequence.

One encoder class serves both arms (`kind="lstm"`), so the only difference between the
48-step and 12-step arms is the window representation.

Owned by Hermes (Task 2). Imports v2/v3 packages read-only.
"""

from .adapter import (AdapterError, WindowRep, channel_names, legacy_48_step, level_names,
                      month_presence_report, monthly_12_step)
from .encoders import (PaddingSafeTemporalEncoder, TemporalError, TemporalOutput,
                       compact_by_mask, legacy_masked_pool,
                       reproduce_legacy_padding_defect)

__all__ = [
    "AdapterError", "WindowRep", "channel_names", "legacy_48_step", "level_names",
    "month_presence_report", "monthly_12_step",
    "PaddingSafeTemporalEncoder", "TemporalError", "TemporalOutput", "compact_by_mask",
    "legacy_masked_pool", "reproduce_legacy_padding_defect",
]
