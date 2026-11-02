"""COE and stock views frozen at the original per-row forecast origin."""
from __future__ import annotations

import numpy as np
import pandas as pd


VARIANTS = ("coe_origin", "stocks_origin")


def macro_history(market):
    """Trailing-only transforms on a complete monthly grid; no filling."""
    result = market.reindex(pd.period_range(market.index.min(), market.index.max(), freq="M")).copy()
    for category in ("A", "C"):
        premium = result[f"COE_{category}_PREMIUM"]
        logged = np.log(premium.where(premium > 0))
        result[f"COE_{category}_LOG_LEVEL"] = logged
        result[f"COE_{category}_LOG_CHANGE3"] = logged - logged.shift(3)
    return result


def macro_view(frame, months, origins, market, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown macro variant: {variant}")
    months = pd.PeriodIndex(months, freq="M")
    origins = pd.PeriodIndex(origins, freq="M")
    if len(months) != len(frame) or len(origins) != len(frame):
        raise ValueError("Aligned dates required")
    known = ~origins.isna()
    if np.any(months.asi8[known] <= origins.asi8[known]):
        raise ValueError("Origin must strictly precede the prediction month")
    horizon = frame.TEMP_FORECAST_HORIZON.to_numpy(float)
    np.testing.assert_array_equal(horizon[known], months.asi8[known] - origins.asi8[known])
    # Extra one-month buffer for unknown historical release dates. A June
    # forecast has origin May and reads April market statistics at the latest.
    cutoff = origins - 1
    trailing = macro_history(market)
    selected = trailing.reindex(cutoff).set_axis(frame.index).astype(float)
    result = frame.copy()
    damped_h = (1 - .8 ** horizon) / .2
    if variant == "coe_origin":
        added = []
        for category in ("A", "C"):
            for stat in ("LOG_LEVEL", "LOG_CHANGE3", "BID_PRESSURE"):
                source = f"COE_{category}_{stat}"
                name = f"MACRO_{source}"
                result[name] = selected[source]
                added.append(name)
            result[f"MACRO_COE_{category}_MOMENTUM_X_H"] = selected[f"COE_{category}_LOG_CHANGE3"] * damped_h
        result["MACRO_COE_HISTORY_MISSING"] = result[added].isna().any(axis=1).astype(float)
    else:
        added = []
        for stat in ("SG_STOCK_RETURN", "SG_STOCK_RETURN_3M", "SG_STOCK_VOL_3M", "SG_STOCK_COVERAGE"):
            name = f"MACRO_{stat}"
            result[name] = selected[stat]
            added.append(name)
        result["MACRO_SG_STOCK_MOMENTUM_X_H"] = selected.SG_STOCK_RETURN_3M * damped_h
        result["MACRO_STOCK_HISTORY_MISSING"] = result[added].isna().any(axis=1).astype(float)
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    if np.isinf(result.select_dtypes("number").to_numpy(float)).any():
        raise ValueError("Infinite macro features")
    return result
