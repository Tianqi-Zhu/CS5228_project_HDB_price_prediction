"""Label-blind recent-half selection using complete calendar months."""
from __future__ import annotations

import numpy as np
import pandas as pd


def recent_half_positions(months):
    months = pd.PeriodIndex(months, freq="M")
    if len(months) == 0 or months.isna().any():
        raise ValueError("Nonempty valid training months are required")
    horizon = pd.period_range(months.min(), months.max(), freq="M")
    cutoff = horizon[len(horizon) // 2]
    positions = np.flatnonzero(months >= cutoff)
    return positions, {"first_training_month": str(months.min()), "last_training_month": str(months.max()),
                       "total_calendar_months": len(horizon), "cutoff_month": str(cutoff),
                       "retained_calendar_months": len(horizon) - len(horizon) // 2,
                       "original_context_rows": len(months), "retained_context_rows": len(positions),
                       "selection_rule": "Latest ceil(n_calendar_months/2) complete months; retain the middle month when odd; never split a month or select using labels"}
