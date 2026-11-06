"""Forecast-origin augmentation and distributional comparable-rent experiments.

Only supplied training rents dated strictly before each row's information
cutoff are used. The April 2024 outer window is development evidence, already
examined by earlier project experiments. All selection happens before it.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from analysis.pipeline.config import ROOT, DATA, DATE, TARGET
from analysis.pipeline.data import load_main, sha256
from analysis.pipeline.features import COMPACT, build_feature_frame

OUT = ROOT / 'output/experiment_runs/round_01_history_target_search'
CACHE = ROOT / '.cache/round_01_history_target_search'
ART = ROOT / 'output/models/round_01_history_target_search'
VERSION = 'origin-distribution-v1'
GROUPS = {
    'BLOCK': ('ADDRESS_KEY', 'FLAT_TYPE_CLEAN'),
    'STREET': ('STREET_CLEAN', 'FLAT_TYPE_CLEAN'),
    'AREA': ('PLANNING_AREA_CLEAN', 'FLAT_TYPE_CLEAN'),
    'TOWN': ('TOWN_CLEAN', 'FLAT_TYPE_CLEAN'),
    'MODEL': ('FLAT_MODEL_CLEAN', 'FLAT_TYPE_CLEAN'),
}
STATICS = tuple(dict.fromkeys(COMPACT + ('STREET_CLEAN', 'ADDRESS_KEY',
    'SUBZONE_CLEAN', 'REGION_CLEAN', 'BUILDING_AGE_YEARS', 'ROOM_COUNT',
    'AREA_PER_ROOM', 'COMPLETION_LEASE_GAP')))


def month_number(values):
    dates = pd.to_datetime(values, format='%Y-%m')
    return (dates.dt.year * 12 + dates.dt.month).to_numpy(np.int64)


def score(actual, predicted):
    difference = np.asarray(predicted) - np.asarray(actual)
    return {'rmse_sgd': float(np.sqrt(np.mean(difference ** 2))),
            'mae_sgd': float(np.mean(np.abs(difference))), 'bias_sgd': float(difference.mean())}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def keys(frame, columns):
    values = frame[list(columns)].astype('string').fillna('__MISSING__')
    result = values.iloc[:, 0]
    for column in values.columns[1:]:
        result = result.str.cat(values[column], sep='\x1f')
    return result.to_numpy(str)


class DistributionHistory:
    """Prefix sufficient statistics and fixed-cutoff geographic comparables."""
    def fit(self, source):
        self.source = source.copy()
        self.months = month_number(source[DATE])
        self.target = source[TARGET].to_numpy(float)
        self.type = source['FLAT_TYPE_CLEAN'].to_numpy(str)
        self.type_month = pd.DataFrame({'month': self.months, 'kind': self.type,
            'rent': self.target}).groupby(['kind', 'month']).rent.agg(['mean', 'median', 'count'])
        reference = np.array([self.type_month.loc[(kind, month), 'mean']
            for kind, month in zip(self.type, self.months)])
        ratio = self.target / reference
        self.ratio = ratio
        self.table = {}
        for grain, columns in GROUPS.items():
            frame = pd.DataFrame({'key': keys(source, columns), 'month': self.months,
                'n': 1., 'rent': self.target, 'rent2': self.target ** 2,
                'ratio': ratio, 'ratio2': ratio ** 2,
                'low': (ratio < .65).astype(float), 'high': (ratio > 1.25).astype(float)})
            monthly = frame.groupby(['key', 'month'], as_index=False).sum(numeric_only=True)
            for column in ('n', 'rent', 'rent2', 'ratio', 'ratio2', 'low', 'high'):
                monthly['c_' + column] = monthly.groupby('key', sort=False)[column].cumsum()
            self.table[grain] = monthly.sort_values('month', kind='stable')
        return self

    def lookup(self, grain, query, cutoff):
        left = pd.DataFrame({'key': keys(query, GROUPS[grain]), 'cutoff': cutoff,
            'position': np.arange(len(query))}).sort_values('cutoff', kind='stable')
        return pd.merge_asof(left, self.table[grain], left_on='cutoff', right_on='month',
            by='key', direction='backward', allow_exact_matches=False).sort_values('position')

    def market(self, query, cutoff, damping):
        target_month = month_number(query[DATE])
        types = query['FLAT_TYPE_CLEAN'].to_numpy(str)
        result = np.full((len(query), 8), np.nan)
        for kind, cut in set(zip(types, cutoff)):
            chosen = (types == kind) & (cutoff == cut)
            history = self.type_month.loc[kind]
            history = history.loc[history.index < cut]
            if history.empty:
                continue
            last = float(history['mean'].iloc[-1])
            last_month = int(history.index[-1])
            recent = history.iloc[-3:]
            level = float(np.average(recent['mean'], weights=recent['count']))
            older = history.iloc[-12:]
            mean12 = float(np.average(older['mean'], weights=older['count']))
            trendpart = history.iloc[-6:]
            slope = 0. if len(trendpart) < 3 else float(np.polyfit(
                trendpart.index.to_numpy() - trendpart.index.min(),
                np.log(trendpart['mean'].to_numpy()), 1)[0])
            slope = np.clip(slope, -.025, .025)
            horizon = target_month[chosen] - last_month
            effective = np.zeros_like(horizon, dtype=float) if damping == 0 else (
                damping * (1 - damping ** horizon) / (1 - damping))
            forecast = level * np.exp(slope * effective)
            result[chosen] = np.column_stack([np.full(chosen.sum(), last),
                np.full(chosen.sum(), level), np.full(chosen.sum(), mean12),
                np.full(chosen.sum(), slope), horizon, forecast,
                np.full(chosen.sum(), float(history['median'].iloc[-1])),
                np.full(chosen.sum(), float(recent['count'].sum()))])
        columns = ['MARKET_LAST', 'MARKET_3M', 'MARKET_12M', 'MARKET_LOG_SLOPE',
            'FORECAST_HORIZON', 'MARKET_FORECAST', 'MARKET_MEDIAN', 'MARKET_COUNT_3M']
        return pd.DataFrame(result, columns=columns, index=query.index)

    def transform(self, query, cutoff, *, damping=.85, neighbors=False):
        cutoff = np.asarray(cutoff, dtype=np.int64)
        months = month_number(query[DATE])
        if np.any(cutoff > months):
            raise ValueError('A history cutoff may not follow its target month')
        features = query[list(STATICS)].copy()
        market = self.market(query, cutoff, damping)
        for column in market:
            features[column] = market[column].to_numpy()
        features['INFORMATION_AGE'] = months - cutoff
        features['AREA_X_AGE'] = query['FLOOR_AREA_SQM'] * query['FLAT_AGE_YEARS']
        for grain in GROUPS:
            current = self.lookup(grain, query, cutoff)
            count = current['c_n'].fillna(0).to_numpy(float)
            safe = np.where(count > 0, count, np.nan)
            mean = current['c_rent'].to_numpy(float) / safe
            ratio = current['c_ratio'].to_numpy(float) / safe
            features[f'{grain}_N'] = count
            features[f'{grain}_RENT_MEAN'] = mean
            features[f'{grain}_RENT_SD'] = np.sqrt(np.maximum(0,
                current['c_rent2'].to_numpy(float) / safe - mean ** 2))
            features[f'{grain}_RATIO_MEAN'] = ratio
            features[f'{grain}_RATIO_SD'] = np.sqrt(np.maximum(0,
                current['c_ratio2'].to_numpy(float) / safe - ratio ** 2))
            features[f'{grain}_LOW_SHARE'] = current['c_low'].to_numpy(float) / safe
            features[f'{grain}_HIGH_SHARE'] = current['c_high'].to_numpy(float) / safe
            features[f'{grain}_LAST_MEAN'] = current['rent'].to_numpy(float) / current['n'].to_numpy(float)
            features[f'{grain}_LAST_RATIO'] = current['ratio'].to_numpy(float) / current['n'].to_numpy(float)
            features[f'{grain}_RECENCY'] = months - current['month'].to_numpy(float)
            smooth_ratio = (np.nan_to_num(ratio) * count + 8.) / (count + 8.)
            features[f'{grain}_ADJUSTED_RENT'] = smooth_ratio * market.MARKET_FORECAST.to_numpy()
            for window in (6, 12):
                earlier = self.lookup(grain, query, cutoff - window)
                n = np.maximum(0, count - earlier['c_n'].fillna(0).to_numpy(float))
                denom = np.where(n > 0, n, np.nan)
                features[f'{grain}_N_{window}M'] = n
                features[f'{grain}_MEAN_{window}M'] = (current['c_rent'].fillna(0).to_numpy(float)
                    - earlier['c_rent'].fillna(0).to_numpy(float)) / denom
                features[f'{grain}_RATIO_{window}M'] = (current['c_ratio'].fillna(0).to_numpy(float)
                    - earlier['c_ratio'].fillna(0).to_numpy(float)) / denom
        if neighbors:
            nearby = self.comparables(query, cutoff)
            features = pd.concat([features, nearby], axis=1)
        return features.replace([np.inf, -np.inf], np.nan)

    def comparables(self, query, cutoff):
        # Euclidean distance in meaningful scales: 1 km, 20 sqm, 20 years.
        def coords(frame):
            return np.column_stack([frame.LATITUDE.to_numpy(float) * 111.,
                frame.LONGITUDE.to_numpy(float) * 111.,
                frame.FLOOR_AREA_SQM.to_numpy(float) / 20.,
                frame.LEASE_COMMENCE_DATE.to_numpy(float) / 20.])
        source_coordinates = coords(self.source)
        query_coordinates = coords(query)
        result = np.full((len(query), 12), np.nan)
        query_types = query.FLAT_TYPE_CLEAN.to_numpy(str)
        for kind, cut in set(zip(query_types, cutoff)):
            query_mask = (query_types == kind) & (cutoff == cut) & np.isfinite(query_coordinates).all(axis=1)
            source_mask = (self.type == kind) & (self.months < cut) & (self.months >= cut - 24)
            source_mask &= np.isfinite(source_coordinates).all(axis=1)
            source_rows = np.flatnonzero(source_mask)
            if not len(source_rows) or not query_mask.any():
                continue
            tree = NearestNeighbors(n_neighbors=min(128, len(source_rows)), algorithm='kd_tree', n_jobs=1)
            tree.fit(source_coordinates[source_rows])
            distance, matches = tree.kneighbors(query_coordinates[query_mask])
            matches = source_rows[matches]
            for j, k in enumerate((16, 48, 128)):
                rows = matches[:, :k]
                weights = np.exp2(-(cut - self.months[rows]) / 12.) / (.2 + distance[:, :k])
                weights /= weights.sum(axis=1, keepdims=True)
                values = self.ratio[rows]
                mean = (weights * values).sum(axis=1)
                result[np.flatnonzero(query_mask), j*4:j*4+4] = np.column_stack([
                    mean, np.sqrt((weights * (values - mean[:, None]) ** 2).sum(axis=1)),
                    (weights * (values < .65)).sum(axis=1), distance[:, :k].mean(axis=1)])
        return pd.DataFrame(result, index=query.index, columns=[f'KNN{k}_{stat}'
            for k in (16, 48, 128) for stat in ('RATIO', 'SD', 'LOW_SHARE', 'DISTANCE')])


def frame_data():
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / 'source.joblib'
    fingerprint = {str(p.relative_to(ROOT)): sha256(p) for p in [DATA / 'train.csv',
        *sorted((DATA / 'auxiliary').glob('*.csv'))]}
    if path.exists():
        saved = joblib.load(path)
        if saved['hashes'] == fingerprint:
            return saved['frame'], fingerprint
    frame = build_feature_frame(load_main(DATA / 'train.csv', require_target=True))
    joblib.dump({'frame': frame, 'hashes': fingerprint}, path, compress=1)
    return frame, fingerprint


def prepare(start, end, snapshot, neighbors, damping):
    frame, hashes = frame_data()
    stamp = hashlib.sha256(json.dumps([VERSION, start, end, snapshot, neighbors, damping, hashes], sort_keys=True).encode()).hexdigest()[:16]
    destination = CACHE / f'{stamp}.joblib'
    if destination.exists():
        return joblib.load(destination)
    fitting = frame.loc[frame[DATE] < start].copy()
    validation = frame.loc[(frame[DATE] >= start) & (frame[DATE] < end)].copy()
    encoder = DistributionHistory().fit(fitting)
    months = month_number(fitting[DATE])
    first = int(months.min())
    views = []
    for delay in ((0, 5, 11) if snapshot else (0,)):
        eligible = months - delay >= first + 6
        part = fitting.loc[eligible].copy()
        cutoff = month_number(part[DATE]) - delay
        print(f'Features {start}: delay={delay}, rows={len(part)}, neighbors={neighbors}', flush=True)
        x = encoder.transform(part, cutoff, damping=damping, neighbors=neighbors)
        views.append((x, part[TARGET].to_numpy(float), part.index.to_numpy(), part[DATE].to_numpy()))
    x_fit = pd.concat([v[0] for v in views], ignore_index=True)
    y_fit = np.concatenate([v[1] for v in views])
    cutoff = np.full(len(validation), pd.Period(start, freq='M').year*12 + pd.Period(start, freq='M').month)
    x_val = encoder.transform(validation, cutoff, damping=damping, neighbors=neighbors)
    categories = list(x_fit.select_dtypes(include=['object', 'string', 'category']).columns)
    for column in categories:
        levels = sorted(x_fit[column].astype('string').fillna('__MISSING__').unique())
        x_fit[column] = pd.Categorical(x_fit[column].astype('string').fillna('__MISSING__'), categories=levels)
        x_val[column] = pd.Categorical(x_val[column].astype('string').fillna('__MISSING__'), categories=levels)
    for column in x_fit.columns.difference(categories):
        x_fit[column] = x_fit[column].astype(np.float32)
        x_val[column] = x_val[column].astype(np.float32)
    result = {'x_fit': x_fit, 'x_val': x_val, 'y_fit': y_fit,
        'y_val': validation[TARGET].to_numpy(float), 'categories': categories,
        'row_fit': np.concatenate([v[2] for v in views]), 'date_fit': np.concatenate([v[3] for v in views]),
        'row_val': validation.index.to_numpy(), 'date_val': validation[DATE].to_numpy(),
        'hashes': hashes, 'start': start, 'end': end}
    joblib.dump(result, destination, compress=1)
    return result


# Locked before outer evaluation. No target-period public records or future macro observations.
TRIALS = [
    {'name': 'distribution_lgb31', 'family': 'lgb', 'snapshot': False, 'neighbors': False, 'target': 'raw', 'leaves': 31},
    {'name': 'distribution_lgb63', 'family': 'lgb', 'snapshot': False, 'neighbors': False, 'target': 'raw', 'leaves': 63},
    {'name': 'distribution_residual', 'family': 'lgb', 'snapshot': False, 'neighbors': False, 'target': 'residual', 'leaves': 31},
    {'name': 'distribution_ratio', 'family': 'lgb', 'snapshot': False, 'neighbors': False, 'target': 'ratio', 'leaves': 31},
    {'name': 'snapshot_lgb31', 'family': 'lgb', 'snapshot': True, 'neighbors': False, 'target': 'raw', 'leaves': 31},
    {'name': 'snapshot_lgb63', 'family': 'lgb', 'snapshot': True, 'neighbors': False, 'target': 'raw', 'leaves': 63},
    {'name': 'snapshot_residual', 'family': 'lgb', 'snapshot': True, 'neighbors': False, 'target': 'residual', 'leaves': 31},
    {'name': 'snapshot_ratio', 'family': 'lgb', 'snapshot': True, 'neighbors': False, 'target': 'ratio', 'leaves': 31},
    {'name': 'comparable_lgb', 'family': 'lgb', 'snapshot': False, 'neighbors': True, 'target': 'raw', 'leaves': 31},
    {'name': 'comparable_snapshot', 'family': 'lgb', 'snapshot': True, 'neighbors': True, 'target': 'residual', 'leaves': 31},
    {'name': 'snapshot_flat_market', 'family': 'lgb', 'snapshot': True, 'neighbors': False, 'target': 'residual', 'leaves': 31, 'damping': 0.},
    {'name': 'distribution_cat6', 'family': 'cat', 'snapshot': False, 'neighbors': False, 'target': 'raw', 'depth': 6},
    {'name': 'snapshot_cat6', 'family': 'cat', 'snapshot': True, 'neighbors': False, 'target': 'residual', 'depth': 6},
    {'name': 'snapshot_cat8', 'family': 'cat', 'snapshot': True, 'neighbors': False, 'target': 'residual', 'depth': 8},
]


def fit_model(spec, bundle, iterations=None):
    from lightgbm import LGBMRegressor, early_stopping
    x, q = bundle['x_fit'].copy(), bundle['x_val'].copy()
    y = bundle['y_fit'].copy()
    actual = bundle['y_val']
    base = x.MARKET_FORECAST.to_numpy(float)
    offset = q.MARKET_FORECAST.to_numpy(float)
    mode = spec['target']
    if mode == 'residual':
        y -= base
        target_val = actual - offset
    elif mode == 'ratio':
        y /= base
        target_val = actual / offset
    else:
        target_val = actual
    # Ratio loss is weighted to match squared error in SGD.
    weights = np.ones(len(y)) if mode != 'ratio' else (base / 3000.) ** 2
    if spec.get('half_life'):
        origin = pd.Period(bundle['start'], freq='M').ordinal
        dates = pd.PeriodIndex(bundle['date_fit'], freq='M').asi8
        weights *= np.exp2(-(origin - dates) / spec['half_life'])
    if spec.get('features') == 'relative':
        remove = [c for c in x if (c.startswith('MARKET_') and c not in ('MARKET_LOG_SLOPE', 'MARKET_COUNT_3M'))
            or c == 'APPROVAL_MONTH_INDEX'
            or any(c.startswith(g + '_') and ('RENT' in c or 'MEAN' in c) for g in GROUPS)]
        x, q = x.drop(columns=remove), q.drop(columns=remove)
    elif spec.get('features') == 'no_entities':
        remove = ['STREET_CLEAN', 'ADDRESS_KEY', 'SUBZONE_CLEAN', 'REGION_CLEAN']
        x, q = x.drop(columns=remove), q.drop(columns=remove)
    categories = [c for c in bundle['categories'] if c in x]
    if spec['family'] == 'lgb':
        model = LGBMRegressor(n_estimators=iterations or 1800, learning_rate=.035,
            num_leaves=spec['leaves'], min_child_samples=spec.get('min_child', 100 if spec['snapshot'] else 50),
            reg_lambda=spec.get('l2', 15.), colsample_bytree=.9, verbosity=-1, random_state=42, n_jobs=4)
        options = {} if iterations else {'eval_set': [(q, target_val)],
            'callbacks': [early_stopping(100, verbose=False)]}
        if mode == 'ratio' and not iterations:
            options['eval_sample_weight'] = [(offset / 3000.) ** 2]
        model.fit(x, y, sample_weight=weights, categorical_feature=categories, **options)
        count = iterations or int(model.best_iteration_)
    else:
        from catboost import CatBoostRegressor
        for column in categories:
            x[column] = x[column].astype('string').fillna('__MISSING__').astype(str)
            q[column] = q[column].astype('string').fillna('__MISSING__').astype(str)
        x = x.fillna(-999.)
        q = q.fillna(-999.)
        model = CatBoostRegressor(iterations=iterations or 1800, learning_rate=.045,
            depth=spec['depth'], l2_leaf_reg=15., random_seed=42, thread_count=4,
            loss_function='RMSE', allow_writing_files=False, verbose=False)
        options = {} if iterations else {'eval_set': (q, target_val), 'early_stopping_rounds': 100}
        model.fit(x, y, sample_weight=weights, cat_features=categories, **options)
        count = iterations or int(model.get_best_iteration()) + 1
    prediction = np.asarray(model.predict(q), dtype=float)
    if mode == 'residual':
        prediction += offset
    elif mode == 'ratio':
        prediction *= offset
    return model, prediction, count


def predict_saved(saved, bundle):
    """Predict from a reloaded artifact without reading query labels."""
    spec = saved['spec']
    query = bundle['x_val']
    if 'models' in saved:
        probability = saved['models']['classifier'].predict_proba(query)
        offset = query.MARKET_FORECAST.to_numpy(float)
        return sum(probability[:, j] * (model.predict(query) + offset)
            for j, model in enumerate(saved['models']['experts']))
    model = saved['model']
    if spec.get('family') == 'controlled_lgb':
        from analysis.experiments.frontier_compact import ADDITIONS
        query = query.copy()
        for column in ADDITIONS[spec['group']]:
            query[column] = bundle['extra_val'][column].to_numpy(np.float32)
        return np.asarray(model.predict(query), dtype=float)
    columns = model.feature_name_ if spec['family'] == 'lgb' else model.feature_names_
    x = query[list(columns)].copy()
    if spec['family'] == 'cat':
        for column in x.select_dtypes(include=['category', 'string', 'object']).columns:
            x[column] = x[column].astype('string').fillna('__MISSING__').astype(str)
        x = x.fillna(-999.)
    prediction = np.asarray(model.predict(x), dtype=float)
    if spec['target'] == 'residual':
        prediction += query.MARKET_FORECAST.to_numpy(float)
    elif spec['target'] == 'ratio':
        prediction *= query.MARKET_FORECAST.to_numpy(float)
    return prediction


def run(stage, names=None):
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'plan.json', {'version': VERSION, 'trials': TRIALS,
        'selection_window': ['2023-10', '2024-04'], 'outer_window': ['2024-04', '2025-04'],
        'target_rmse_sgd': 430., 'notes': 'All outer rows retained; snapshots use delays 0/5/11 months; earlier labels only. Outer is repeatedly viewed development data.'})
    trials = [s for s in TRIALS if not names or s['name'] in names]
    for spec in trials:
        destination = OUT / stage / spec['name']
        if (destination / 'results.json').exists():
            print('Already completed', stage, spec['name'], flush=True)
            continue
        if stage == 'screen':
            start, end, iterations = '2023-10', '2024-04', None
        else:
            start, end = '2024-04', '2025-04'
            selected = json.loads((OUT / 'screen' / spec['name'] / 'results.json').read_text())
            iterations = selected['iterations']
        started = time.perf_counter()
        bundle = prepare(start, end, spec['snapshot'], spec['neighbors'], spec.get('damping', .85))
        print('Training', stage, spec['name'], bundle['x_fit'].shape, flush=True)
        model, prediction, count = fit_model(spec, bundle, iterations)
        columns = list(model.feature_name_ if spec['family'] == 'lgb' else model.feature_names_)
        record = {'spec': spec, 'stage': stage, 'start': start, 'end': end, 'iterations': count,
            'n_fit': len(bundle['y_fit']), 'n_validation': len(prediction), 'n_features': len(columns),
            **score(bundle['y_val'], prediction), 'seconds': time.perf_counter()-started, 'source_sha256': bundle['hashes']}
        destination.mkdir(parents=True, exist_ok=True)
        artifact = ART / stage / spec['name']
        artifact.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'source_row': bundle['row_val'], 'month': bundle['date_val'],
            'actual': bundle['y_val'], 'prediction': prediction}).to_csv(artifact / 'predictions.csv', index=False)
        joblib.dump({'model': model, 'spec': spec, 'categories': bundle['categories'],
            'feature_columns': columns, 'iterations': count}, artifact / 'model.joblib', compress=1)
        write_json(destination / 'results.json', record)
        print(f"RESULT {stage} {spec['name']}: RMSE={record['rmse_sgd']:.4f}, iterations={count}, seconds={record['seconds']:.1f}", flush=True)
        del bundle, model
        gc.collect()
        summarize()


def summarize():
    rows = []
    paths = list((OUT / 'screen').glob('*/results.json')) + list((OUT / 'outer').glob('*/results.json'))
    for path in paths:
        record = json.loads(path.read_text(encoding='utf-8'))
        rows.append({k: v for k, v in record.items() if k not in ('source_sha256', 'spec')} | {'name': record['spec']['name']})
    if rows:
        table = pd.DataFrame(rows).sort_values(['stage', 'rmse_sgd'])
        table.to_csv(OUT / 'metrics.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['screen', 'outer', 'summary'])
    parser.add_argument('--names', nargs='+')
    args = parser.parse_args()
    summarize() if args.stage == 'summary' else run(args.stage, args.names)


if __name__ == '__main__':
    main()
