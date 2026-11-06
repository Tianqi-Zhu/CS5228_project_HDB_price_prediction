"""Controlled additions of rent-distribution features to the existing 27 inputs."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from analysis.experiments.frontier_search import (OUT, CACHE, ART, GROUPS,
    DistributionHistory, frame_data, month_number, score, write_json, summarize)
from analysis.pipeline.config import DATE, TARGET
from analysis.pipeline.features import model_inputs, FEATURE_CANDIDATES

ADDITIONS = {
    'control': [],
    'block_shape': [f'BLOCK_{s}' for s in ('LOW_SHARE', 'HIGH_SHARE', 'RENT_SD', 'RATIO_SD', 'N_6M', 'N_12M')],
    'all_shape': [f'{g}_{s}' for g in GROUPS for s in ('LOW_SHARE', 'HIGH_SHARE', 'RENT_SD', 'RATIO_SD')],
    'relative_history': [f'{g}_{s}' for g in GROUPS for s in ('RATIO_MEAN', 'LAST_RATIO', 'RATIO_6M', 'RATIO_12M')],
    'comparables': [f'KNN{k}_{s}' for k in (16, 48, 128) for s in ('RATIO', 'SD', 'LOW_SHARE', 'DISTANCE')],
}
SPECS = [{'name': f'compact_plus_{group}_{leaves}', 'group': group,
    'leaves': leaves, 'family': 'controlled_lgb'} for group in ADDITIONS for leaves in (15, 31, 63)]


def prepare_compact(start, end):
    frame, hashes = frame_data()
    signature = hashlib.sha256(json.dumps([start, end, hashes, 'compact-distribution-v1'], sort_keys=True).encode()).hexdigest()[:16]
    cache = CACHE / f'compact-{signature}.joblib'
    if cache.exists():
        return joblib.load(cache)
    fit = frame.loc[frame[DATE] < start].copy()
    val = frame.loc[(frame[DATE] >= start) & (frame[DATE] < end)].copy()
    x, q = model_inputs(fit, val, FEATURE_CANDIDATES['compact_history_27'])
    encoder = DistributionHistory().fit(fit)
    print('Preparing controlled distribution/comparable inputs', start, flush=True)
    extra = encoder.transform(fit, month_number(fit[DATE]), neighbors=True)
    cutoff = np.full(len(val), int(month_number(pd.Series([start]))[0]))
    query = encoder.transform(val, cutoff, neighbors=True)
    bundle = {'x_fit': x, 'x_val': q, 'extra_fit': extra, 'extra_val': query,
        'y_fit': fit[TARGET].to_numpy(float), 'y_val': val[TARGET].to_numpy(float),
        'row_val': val.index.to_numpy(), 'date_val': val[DATE].to_numpy(), 'hashes': hashes}
    joblib.dump(bundle, cache, compress=1)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['screen', 'outer'])
    parser.add_argument('--names', nargs='+')
    args = parser.parse_args()
    write_json(OUT / 'compact_plan.json', {'trials': SPECS,
        'selection': ['2023-10', '2024-04'], 'outer': ['2024-04', '2025-04'],
        'note': 'Retain all fitting rows; existing 27 columns plus only the named added group. Selection precedes outer evaluation.'})
    start, end = ('2023-10', '2024-04') if args.stage == 'screen' else ('2024-04', '2025-04')
    bundle = prepare_compact(start, end)
    for spec in SPECS:
        if args.names and spec['name'] not in args.names:
            continue
        destination = OUT / args.stage / spec['name']
        if (destination / 'results.json').exists():
            continue
        started = time.perf_counter()
        x, q = bundle['x_fit'].copy(), bundle['x_val'].copy()
        for column in ADDITIONS[spec['group']]:
            x[column] = bundle['extra_fit'][column].to_numpy(np.float32)
            q[column] = bundle['extra_val'][column].to_numpy(np.float32)
        iterations = None if args.stage == 'screen' else json.loads(
            (OUT / 'screen' / spec['name'] / 'results.json').read_text())['iterations']
        model = LGBMRegressor(n_estimators=iterations or 1800, learning_rate=.03,
            num_leaves=spec['leaves'], min_child_samples=40, reg_lambda=10.,
            n_jobs=4, verbosity=-1, random_state=42)
        options = {} if iterations else {'eval_set': [(q, bundle['y_val'])],
            'callbacks': [early_stopping(120, verbose=False)]}
        model.fit(x, bundle['y_fit'], **options)
        prediction = model.predict(q)
        count = iterations or int(model.best_iteration_)
        artifact = ART / args.stage / spec['name']
        artifact.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'source_row': bundle['row_val'], 'month': bundle['date_val'],
            'actual': bundle['y_val'], 'prediction': prediction}).to_csv(artifact / 'predictions.csv', index=False)
        joblib.dump({'model': model, 'spec': spec}, artifact / 'model.joblib', compress=1)
        result = {'spec': spec, 'stage': args.stage, 'start': start, 'end': end,
            'n_fit': len(x), 'n_validation': len(q), 'n_features': x.shape[1],
            'iterations': count, **score(bundle['y_val'], prediction),
            'seconds': time.perf_counter()-started, 'source_sha256': bundle['hashes']}
        write_json(destination / 'results.json', result)
        print(f"RESULT {args.stage} {spec['name']}: {result['rmse_sgd']:.4f}, iterations={count}", flush=True)
        summarize()
        del x, q, model
        gc.collect()


if __name__ == '__main__':
    main()
