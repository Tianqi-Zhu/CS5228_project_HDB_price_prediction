"""Origin-safe identifiers, recent rent moments and forecast-head inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.data import normalized_address
from analysis.experiments.recent_history import (
    GRAINS, WINDOWS, RECENT_FEATURES, _prefix_lookup, recent_history_features,
)
from analysis.experiments.spatiotemporal_hybrid import (
    BASE_CATEGORICAL, BASE_FEATURES, HISTORY_FEATURES, MONTH, TARGET,
    historical_features,
)


ID_FEATURES = ['BLOCK_ID', 'STREET_ID']
STD_FEATURES = [f'RECENT_{grain}_{window}M_STD' for grain in GRAINS for window in WINDOWS]
TREND_FEATURES = ['AREA_RECENT_MINUS_LONG', 'BLOCK_RECENT_MINUS_LONG',
                  'BLOCK_MINUS_AREA_RECENT', 'AREA_LAST_MINUS_RECENT']
ORIGINAL = BASE_FEATURES + HISTORY_FEATURES
IDENTIFIERS = ORIGINAL + ID_FEATURES
ENHANCED = IDENTIFIERS + RECENT_FEATURES + STD_FEATURES + TREND_FEATURES
SEASONAL = [c for c in ENHANCED if c not in ('APPROVAL_MONTH_INDEX', 'APPROVAL_MONTH_OF_YEAR')] + ['MONTH_SIN', 'MONTH_COS']
FEATURE_VARIANTS = {'original': ORIGINAL, 'identifiers': IDENTIFIERS,
                    'recent': ENHANCED, 'seasonal_only': SEASONAL}
CATEGORICAL = BASE_CATEGORICAL | set(ID_FEATURES)
HEAD_NUMERIC = [
    'FLOOR_AREA_SQM', 'FLAT_AGE_YEARS', 'LATITUDE', 'LONGITUDE',
    'NEAREST_MRT_KM', 'DISTANCE_TO_CBD_KM',
    *HISTORY_FEATURES,
    'RECENT_AREA_FLAT_3M_MEAN', 'RECENT_AREA_FLAT_12M_MEAN',
    'RECENT_AREA_FLAT_3M_COUNT', 'RECENT_AREA_FLAT_3M_STD',
    'RECENT_BLOCK_FLAT_3M_SHRUNK_MEAN', 'RECENT_BLOCK_FLAT_12M_SHRUNK_MEAN',
    'RECENT_BLOCK_FLAT_3M_STD', *TREND_FEATURES,
]
HEAD_COLUMNS = HEAD_NUMERIC + ['PFN_MEAN', 'FORECAST_HORIZON', 'ORIGIN_MONTH_INDEX']


def identifiers(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    address = normalized_address(frame)
    result['STREET_ID'] = address['TOWN'] + '|' + address['STREET']
    result['BLOCK_ID'] = result['STREET_ID'] + '|' + address['BLOCK']
    return result


def prior_standard_deviations(fitting: pd.DataFrame, queries: pd.DataFrame,
                             *, frozen: bool) -> pd.DataFrame:
    """Sample standard deviations; exclude the entire current month in training."""
    output = pd.DataFrame(index=queries.index)
    for grain, keys in GRAINS.items():
        values = fitting[keys + [MONTH, TARGET]].copy()
        values['_square'] = values[TARGET].astype(float) ** 2
        monthly = values.groupby(keys + [MONTH], dropna=False, observed=True).agg(
            _sum=(TARGET, 'sum'), _n=(TARGET, 'size'), _sq=('_square', 'sum'),
        ).reset_index()
        monthly['group'] = pd.util.hash_pandas_object(monthly[keys], index=False).to_numpy()
        monthly['month_ordinal'] = monthly[MONTH].astype('int64')
        monthly = monthly.sort_values(['group', 'month_ordinal'])
        monthly['cum_count'] = monthly.groupby('group')['_n'].cumsum()
        monthly['cum_sum'] = monthly.groupby('group')['_sum'].cumsum()
        groups = pd.util.hash_pandas_object(queries[keys], index=False).to_numpy()
        upper = (np.full(len(queries), fitting[MONTH].max().ordinal, dtype=np.int64)
                 if frozen else queries[MONTH].astype('int64').to_numpy() - 1)
        upper_sum, upper_n = _prefix_lookup(monthly, groups, upper)
        squares = monthly.copy()
        squares['cum_sum'] = squares.groupby('group')['_sq'].cumsum()
        upper_sq, _ = _prefix_lookup(squares, groups, upper)
        for window in WINDOWS:
            lower_sum, lower_n = _prefix_lookup(monthly, groups, upper - window)
            lower_sq, _ = _prefix_lookup(squares, groups, upper - window)
            n, total, total_sq = upper_n - lower_n, upper_sum - lower_sum, upper_sq - lower_sq
            correction = np.divide(total ** 2, n, out=np.zeros(len(n)), where=n > 0)
            variance = np.divide(total_sq - correction, n - 1,
                                 out=np.full(len(n), np.nan), where=n > 1)
            output[f'RECENT_{grain}_{window}M_STD'] = np.sqrt(np.maximum(variance, 0))
    return output


def enhanced_feature_pair(fitting: pd.DataFrame, future: pd.DataFrame):
    if fitting.empty or future.empty or fitting[MONTH].max() >= future[MONTH].min():
        raise ValueError('Queries must follow all context labels')
    # Future target values are used only by the caller for loss/evaluation.
    blind = future.drop(columns=TARGET, errors='ignore')
    fit_hist, val_hist = historical_features(fitting, blind)
    fit_recent, val_recent = recent_history_features(fitting, blind)
    outputs = []
    for rows, history, recent, frozen in (
        (fitting, fit_hist, fit_recent, False), (blind, val_hist, val_recent, True),
    ):
        frame = identifiers(rows).join(history).join(recent)
        frame = frame.join(prior_standard_deviations(fitting, rows, frozen=frozen))
        frame['AREA_RECENT_MINUS_LONG'] = frame.RECENT_AREA_FLAT_3M_MEAN - frame.RECENT_AREA_FLAT_12M_MEAN
        frame['BLOCK_RECENT_MINUS_LONG'] = frame.RECENT_BLOCK_FLAT_3M_SHRUNK_MEAN - frame.RECENT_BLOCK_FLAT_12M_SHRUNK_MEAN
        frame['BLOCK_MINUS_AREA_RECENT'] = frame.RECENT_BLOCK_FLAT_3M_SHRUNK_MEAN - frame.RECENT_AREA_FLAT_3M_MEAN
        frame['AREA_LAST_MINUS_RECENT'] = frame.HIST_AREA_FLAT_LAST_MEAN - frame.RECENT_AREA_FLAT_3M_MEAN
        angle = 2 * np.pi * frame.APPROVAL_MONTH_OF_YEAR / 12
        frame['MONTH_SIN'], frame['MONTH_COS'] = np.sin(angle), np.cos(angle)
        outputs.append(frame)
    return outputs[0], outputs[1]


def backbone_inputs(fitting, query, columns):
    if TARGET in columns or {'LOG_MONTHLY_RENT', 'RENT_PER_SQM'} & set(columns):
        raise ValueError('Target-derived current-row inputs are forbidden')
    a, b = fitting[columns].copy(), query[columns].copy()
    for name in CATEGORICAL & set(columns):
        known = pd.Index(a[name].dropna().astype(str).unique())
        for frame in (a, b):
            values = frame[name].astype('string')
            frame[name] = pd.Categorical(values.where(values.isin(known)), categories=known)
    return a, b


def head_numeric(query: pd.DataFrame, predicted: np.ndarray, origin: pd.Period):
    result = query[HEAD_NUMERIC].copy()
    result['PFN_MEAN'] = np.asarray(predicted, float)
    result['FORECAST_HORIZON'] = query[MONTH].astype('int64') - origin.ordinal + 1
    result['ORIGIN_MONTH_INDEX'] = origin.ordinal - pd.Period('2021-01', 'M').ordinal
    if (result.FORECAST_HORIZON < 1).any():
        raise ValueError('Head queries must be at or after the forecast origin')
    return result[HEAD_COLUMNS].to_numpy(float)
