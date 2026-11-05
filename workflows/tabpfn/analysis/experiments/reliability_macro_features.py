"""Small appended reliability and causally generated national forecast inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

RELIABILITY = ['SIGNATURE_HISTORY_MISSING', 'BLOCK_HISTORY_MISSING',
               'BLOCK_3M_STD_UNSUPPORTED', 'BLOCK_12M_LOW_SUPPORT', 'BLOCK_ID_UNKNOWN']
RENT_FEATURES = ['NATIONAL_RENT_FORECAST_GROWTH', 'NATIONAL_FORECAST_UNAVAILABLE']
COE_FEATURE = 'NATIONAL_COE_INCREMENTAL_GROWTH'
VARIANTS = ('reliability', 'rent_forecast', 'coe_increment')


def reliability(x):
    z = x.copy()
    z[RELIABILITY[0]] = x.HIST_SIGNATURE_MEAN.isna().astype(float)
    z[RELIABILITY[1]] = x.HIST_BLOCK_FLAT_MEAN.isna().astype(float)
    z[RELIABILITY[2]] = x.RECENT_BLOCK_FLAT_3M_COUNT.fillna(0).lt(2).astype(float)
    z[RELIABILITY[3]] = x.RECENT_BLOCK_FLAT_12M_COUNT.fillna(0).lt(5).astype(float)
    z[RELIABILITY[4]] = x.BLOCK_ID.isna().astype(float)
    return z


def dates_and_origins(x):
    indices = x.APPROVAL_MONTH_INDEX.to_numpy(int)
    months = pd.PeriodIndex.from_ordinals(pd.Period('2021-01', 'M').ordinal + indices, freq='M')
    horizon = x.TEMP_FORECAST_HORIZON.to_numpy(float)
    origin = pd.PeriodIndex.from_ordinals(months.asi8 - np.nan_to_num(horizon, nan=0).astype(int), freq='M')
    origin = pd.Series(origin, index=x.index).where(np.isfinite(horizon))
    return months, origin


def add_forecast(x, table, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    z = reliability(x)
    if variant == 'reliability':
        return z
    months, origins = dates_and_origins(x)
    # Known macro observations stop a full month before the existing feature origin.
    cutoff = pd.PeriodIndex(origins, freq='M') - 1
    horizon = months.asi8 - cutoff.asi8
    known = origins.notna().to_numpy()
    h = np.clip(horizon, 1, 10)
    keys = pd.MultiIndex.from_arrays([cutoff.astype(str), h], names=['cutoff', 'horizon'])
    matched = table.set_index(['cutoff', 'horizon']).reindex(keys)
    rent = matched.rent_growth.to_numpy(float)
    coe = matched.coe_growth.to_numpy(float)
    valid = known & np.isfinite(rent) & np.isfinite(coe)
    z[RENT_FEATURES[0]] = np.where(valid, rent, np.nan)
    z[RENT_FEATURES[1]] = (~valid).astype(float)
    if variant == 'coe_increment':
        z[COE_FEATURE] = np.where(valid, coe-rent, np.nan)
    return z


def forecast_table(panel, *, epochs=100, cutoffs=None):
    from analysis.experiments.temporal_sequence_ablation import Spec, forecast
    rows = []
    positions = list(range(15, len(panel))) if cutoffs is None else [panel.index.get_loc(c) for c in cutoffs]
    for position in positions:
        if position < 15:
            continue
        rent = forecast(panel, 'adjusted_mean_rent', Spec('lstm', 'rent', 12), position, epochs=epochs)[1]
        coe = forecast(panel, 'adjusted_mean_rent', Spec('lstm', 'rent_coe', 12), position, epochs=epochs)[1]
        for h in range(1, 11):
            n = max(0, position-h-12+1)
            valid = n >= 3
            rows.append(dict(cutoff=str(panel.index[position]), horizon=h, mature_monthly_labels=n,
                             rent_growth=float(rent[h-1]) if valid else np.nan,
                             coe_growth=float(coe[h-1]) if valid else np.nan))
    return pd.DataFrame(rows, columns=['cutoff', 'horizon', 'mature_monthly_labels', 'rent_growth', 'coe_growth'])
