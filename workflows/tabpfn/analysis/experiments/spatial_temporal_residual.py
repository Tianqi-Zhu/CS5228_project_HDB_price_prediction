"""Small residual features with one explicit, fixed information origin per batch."""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.experiments.aligned_temporal_features import (
    FEATURES, HISTORY_COLUMNS, LocalTemporalModel, origin_history,
)
from analysis.experiments.calendar_encoding import MONTH_COLUMN
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET

BASIC = ['TABPFN_RENT', 'FLAT_TYPE_CLEAN', 'FLOOR_AREA_SQM', 'FLAT_AGE_YEARS',
         'LATITUDE', 'LONGITUDE', 'TOWN_CLEAN', 'APPROVAL_MONTH_INDEX',
         'APPROVAL_MONTH_OF_YEAR', 'TEMP_FORECAST_HORIZON']
MARKET = ['LOCAL_HIST_MEDIAN', 'LOCAL_GROWTH_3M', 'LOCAL_COUNT_6M']
CATEGORICAL = ['FLAT_TYPE_CLEAN', 'TOWN_CLEAN']


def check_partition(context_months, query_months, context_ids, query_ids, origin):
    origin = pd.Period(origin, 'M')
    cm, qm = pd.PeriodIndex(context_months, freq='M'), pd.PeriodIndex(query_months, freq='M')
    if not len(cm) or not len(qm) or cm.max() >= origin or qm.min() < origin:
        raise ValueError('Context must strictly precede the complete forecast batch')
    if not pd.Index(context_ids).is_unique or not pd.Index(query_ids).is_unique:
        raise ValueError('Duplicate observation IDs')
    if not set(context_ids).isdisjoint(query_ids):
        raise ValueError('A query target belongs to its model context')


def historical_bundle(base, frame, start, stop):
    """Reuse original causal context; rebuild all query target-derived columns."""
    origin, end = pd.Period(start, 'M'), pd.Period(stop, 'M')
    months = pd.PeriodIndex(base['train_month'], freq='M')
    a, b = np.flatnonzero(months < origin), np.flatnonzero((months >= origin) & (months < end))
    x, q = base['X_train'].iloc[a].copy(), base['X_train'].iloc[b].copy()
    check_partition(months[a], months[b], x.index, q.index, origin)
    past, blind = frame.loc[x.index], frame.loc[q.index].drop(columns=TARGET)
    q.loc[:, HISTORY_COLUMNS] = origin_history(past, blind).reindex(q.index).to_numpy()
    q.loc[:, FEATURES] = LocalTemporalModel(past).features(blind, origin-1).reindex(q.index).to_numpy()
    for column in x.select_dtypes(include='category'):
        if column == MONTH_COLUMN:
            continue
        known = pd.Index(x[column].dropna().astype(str).unique())
        for matrix in (x, q):
            values = matrix[column].astype('string')
            matrix[column] = pd.Categorical(values.where(values.isin(known)), categories=known)
    baseline = q.TEMP_LOCAL_FORECAST.to_numpy(float)
    if not np.isfinite(baseline).all() or list(x) != list(q):
        raise ValueError('Invalid baseline or query schema')
    return dict(X_train=x, X_query=q, y_train=np.asarray(base['y_train'])[a],
                y_query=np.asarray(base['y_train'])[b], train_month=np.asarray(base['train_month'])[a],
                query_month=np.asarray(base['train_month'])[b],
                train_baseline=np.asarray(base['train_baseline'])[a], query_baseline=baseline)


def market_features(past, query, origin, min_history=20, min_window=10):
    """Exact historical medians, fallback and growth; no labels from query read.

    All future rows share the batch cutoff, even when their transaction month
    is much later. The window-growth ratio compares two adjacent three-month
    medians, using the same fallback level for both numerator and denominator.
    """
    origin = pd.Period(origin, 'M')
    p = past.loc[past[MONTH] < origin].copy()
    if len(p) == 0 or (query[MONTH] < origin).any():
        raise ValueError('No permitted history, or query precedes origin')
    keys = ['TOWN_CLEAN', 'FLAT_TYPE_CLEAN']
    def maps(part, group):
        return part.groupby(group, observed=True)[TARGET].agg(['median', 'size'])
    local, parent = maps(p, keys), maps(p, ['FLAT_TYPE_CLEAN'])
    recent = p.loc[p[MONTH] >= origin-3]
    earlier = p.loc[(p[MONTH] >= origin-6) & (p[MONTH] < origin-3)]
    count = p.loc[p[MONTH] >= origin-6].groupby(keys, observed=True).size()
    windows = [(maps(recent, keys), maps(earlier, keys)),
               (maps(recent, ['FLAT_TYPE_CLEAN']), maps(earlier, ['FLAT_TYPE_CLEAN']))]
    rows, audit = [], []
    def sufficient(table, key, minimum):
        return key in table.index and table.loc[key, 'size'] >= minimum
    for town, kind in query[keys].itertuples(index=False, name=None):
        key = (town, kind)
        level = 0
        if sufficient(local, key, min_history): median = local.loc[key, 'median']
        elif sufficient(parent, kind, min_history): median, level = parent.loc[kind, 'median'], 1
        else: median, level = p[TARGET].median(), 2
        growth, growth_level = np.nan, -1
        for j, ((now, old), k) in enumerate(zip(windows, [key, kind])):
            if sufficient(now, k, min_window) and sufficient(old, k, min_window):
                growth = (now.loc[k, 'median']-old.loc[k, 'median']) / max(old.loc[k, 'median'], 1.)
                growth_level = j
                break
        if not np.isfinite(growth) and len(recent) >= min_history and len(earlier) >= min_history:
            growth = (recent[TARGET].median()-earlier[TARGET].median()) / max(earlier[TARGET].median(), 1.)
            growth_level = 2
        rows.append([median, growth, count.get(key, 0)])
        audit.append([level, growth_level])
    return (pd.DataFrame(rows, index=query.index, columns=MARKET),
            pd.DataFrame(audit, index=query.index, columns=['median_fallback_level', 'growth_fallback_level']))


def residual_features(matrix, prediction, market=None):
    if len(matrix) != len(prediction) or not matrix.index.is_unique or not np.isfinite(prediction).all():
        raise ValueError('Predictions and unique feature rows must align')
    result = matrix[[c for c in BASIC if c != 'TABPFN_RENT']].copy()
    result.insert(0, 'TABPFN_RENT', np.asarray(prediction, float))
    for c in result:
        if c in CATEGORICAL:
            result[c] = result[c].astype('string').fillna('__missing__').astype(str)
        else:
            result[c] = pd.to_numeric(result[c].astype('string') if isinstance(result[c].dtype, pd.CategoricalDtype) else result[c], errors='raise').astype(float)
    if market is not None:
        if not market.index.equals(matrix.index):
            raise ValueError('Market feature IDs/order differ')
        result = result.join(market[MARKET])
    if list(result) != BASIC + (MARKET if market is not None else []):
        raise ValueError('Residual feature schema changed')
    if np.isinf(result.select_dtypes(include=np.number).to_numpy()).any():
        raise ValueError('Infinite residual feature')
    return result


def corrected(base, residual, strength):
    base, residual = np.asarray(base, float), np.asarray(residual, float)
    if base.shape != residual.shape or not np.isfinite(residual).all() or not 0 <= strength <= 1:
        raise ValueError('Invalid correction')
    return base.copy() if strength == 0 else base + strength * residual


def choose_strength(actual, base, residual, strengths=(0, .25, .5, .75, 1)):
    scores = [(float(s), float(np.sqrt(np.mean((corrected(base, residual, s)-actual)**2)))) for s in strengths]
    return min(scores, key=lambda pair:(pair[1], pair[0]))[0], scores
