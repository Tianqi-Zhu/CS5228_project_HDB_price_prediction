"""Evaluate the open TabICLv2 regressor with causal rental-history features.

All screening precedes April 2024. Full later windows retain every query row.
The pretrained model runs locally, without sending the project table anywhere.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd

from analysis.experiments.frontier_search import ROOT, frame_data, score, write_json, month_number, prepare as prepare_snapshots
from analysis.experiments.frontier_compact import prepare_compact, ADDITIONS
from analysis.experiments.download_tabicl import CHECKPOINT
from analysis.pipeline.data import sha256
from analysis.pipeline.features import COMPACT

OUT = ROOT / 'output/experiment_runs/round_02_anniversary_history/tabicl'
ART = ROOT / 'output/models/round_02_anniversary_history/tabicl'
DEPS = ROOT / '.cache/foundation-deps'
SPECS = [
    {'name': 'compact_raw24k', 'features': 'compact', 'target': 'raw', 'context': 24000, 'recent': False},
    {'name': 'compact_residual24k', 'features': 'compact', 'target': 'residual', 'context': 24000, 'recent': False},
    {'name': 'relative_residual24k', 'features': 'relative', 'target': 'residual', 'context': 24000, 'recent': False},
    {'name': 'compact_recent24k', 'features': 'compact', 'target': 'raw', 'context': 24000, 'recent': True},
    {'name': 'relative_residual48k', 'features': 'relative', 'target': 'residual', 'context': 48000, 'recent': False, 'views': 4},
    {'name': 'relative_residual_full', 'features': 'relative', 'target': 'residual', 'context': 999999, 'recent': False, 'views': 2},
    {'name': 'relative_raw48k', 'features': 'relative', 'target': 'raw', 'context': 48000, 'recent': False, 'views': 4},
    {'name': 'relative_ratio48k', 'features': 'relative', 'target': 'ratio', 'context': 48000, 'recent': False, 'views': 4},
    {'name': 'aligned_residual48k', 'features': 'relative', 'target': 'residual', 'context': 48000, 'recent': False, 'views': 4, 'aligned': True},
    {'name': 'observed_ratio24k', 'features': 'relative', 'target': 'observed_ratio', 'context': 24000, 'recent': False, 'views': 2},
    {'name': 'observed_ratio48k', 'features': 'relative', 'target': 'observed_ratio', 'context': 48000, 'recent': False, 'views': 4},
    {'name': 'cohort_residual24k', 'features': 'relative', 'target': 'residual', 'context': 24000, 'recent': False, 'views': 2, 'cohort': True},
    {'name': 'cohort_observed24k', 'features': 'relative', 'target': 'observed_ratio', 'context': 24000, 'recent': False, 'views': 2, 'cohort': True},
    {'name': 'cohort_observed48k', 'features': 'relative', 'target': 'observed_ratio', 'context': 48000, 'recent': False, 'views': 2, 'cohort': True},
]


def enable():
    sys.path.insert(0, str(DEPS))
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.72)
    from tabicl import TabICLRegressor
    return torch, TabICLRegressor


def context_indices(bundle, spec):
    n = len(bundle['y_fit'])
    eligible = np.ones(n, dtype=bool)
    if spec['target'] != 'raw':
        eligible &= np.isfinite(bundle['extra_fit'].MARKET_FORECAST.to_numpy())
        # Same six-month history warmup as the previous distribution experiments.
        dates = bundle['extra_fit'].APPROVAL_MONTH_INDEX.to_numpy(float)
        if not bundle.get('warmup_complete', False):
            eligible &= dates >= np.min(dates)+6
    if spec['recent']:
        dates = bundle['extra_fit'].APPROVAL_MONTH_INDEX.to_numpy(float)
        eligible &= dates >= np.max(dates)-23
    rows = np.flatnonzero(eligible)
    count = min(spec['context'], len(rows))
    # Sampling never looks at rent values. Fixed uniform selection preserves all regimes.
    return np.sort(np.random.default_rng(42).choice(rows, size=count, replace=False))


def inputs(bundle, spec):
    x, q = bundle['x_fit'].copy(), bundle['x_val'].copy()
    if spec['features'] == 'relative':
        for column in ADDITIONS['relative_history']:
            x[column] = bundle['extra_fit'][column].to_numpy(np.float32)
            q[column] = bundle['extra_val'][column].to_numpy(np.float32)
    if spec.get('cohort'):
        for column in bundle['ann_fit']:
            if column.startswith('ANN_BLOCK') and '_R1_' in column:
                x[column] = bundle['ann_fit'][column].to_numpy(np.float32)
                q[column] = bundle['ann_val'][column].to_numpy(np.float32)
    for column in x.select_dtypes(include=['category','string','object']).columns:
        x[column] = x[column].astype('string').fillna('__MISSING__').astype(str)
        q[column] = q[column].astype('string').fillna('__MISSING__').astype(str)
    target = bundle['y_fit'].copy()
    offset = np.zeros(len(q))
    if spec['target'] == 'residual':
        target -= bundle['extra_fit'].MARKET_FORECAST.to_numpy(float)
        offset = bundle['extra_val'].MARKET_FORECAST.to_numpy(float)
    elif spec['target'] == 'ratio':
        target /= bundle['extra_fit'].MARKET_FORECAST.to_numpy(float)
        offset = bundle['extra_val'].MARKET_FORECAST.to_numpy(float)
    elif spec['target'] == 'observed_ratio':
        # Contemporaneous levels are used only for fitting labels, never query inputs.
        labels = pd.DataFrame({'month':bundle['extra_fit'].APPROVAL_MONTH_INDEX.to_numpy(),
            'kind':x.FLAT_TYPE_CLEAN.to_numpy(), 'rent':target})
        reference = labels.groupby(['month','kind']).rent.transform('mean').to_numpy(float)
        target /= reference
        offset = bundle['extra_val'].MARKET_3M.to_numpy(float)
        remove = [c for c in x if c=='APPROVAL_MONTH_INDEX' or (c.startswith('HIST_') and c.endswith('_MEAN'))
            or (c.startswith('ANN_') and c.endswith('_RENT'))]
        x,q = x.drop(columns=remove),q.drop(columns=remove)
    return x, q, target, offset


def fit_predict(bundle, spec, query_rows=None, *, views=2, smoke=False):
    torch, Regressor = enable()
    indices = context_indices(bundle, spec)
    x, q, target, offset = inputs(bundle, spec)
    # sklearn's mean imputer removes all-NaN columns; TabICL 2.1.1's feature
    # mask expects the original width. Drop unsupported columns explicitly,
    # using only the chosen fitting context, and preserve the query schema.
    supported = x.iloc[indices].notna().any(axis=0)
    x,q = x.loc[:,supported],q.loc[:,supported]
    if query_rows is None:
        query_rows = np.arange(len(q))
    query = q.iloc[query_rows]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # Representation caching keeps the same model/context while reducing the
    # stored ICL cache on this 8-GB device. Larger query batches amortize prefill.
    cache_mode = 'repr' if len(indices) >= 48000 else True
    model = Regressor(n_estimators=views, batch_size=1, kv_cache=cache_mode,
        model_path=str(CHECKPOINT), allow_auto_download=False, device='cuda',
        norm_methods=['none'] if views == 1 else ['none', 'power'],
        use_amp=True, use_fa3=False, offload_mode='auto',
        disk_offload_dir=str(ROOT/'.cache/second_wave_offload'), n_jobs=4,
        random_state=42, verbose=False)
    started = time.perf_counter()
    print(f"TABICL fit: {spec['name']}, context={len(indices)}, features={x.shape[1]}, views={views}", flush=True)
    model.fit(x.iloc[indices], target[indices])
    fitted = time.perf_counter()
    print(f'TABICL cached context in {fitted-started:.1f}s; query={len(query)}', flush=True)
    batches = []
    batch_size = 128 if smoke else (2048 if cache_mode == 'repr' else 256)
    for start in range(0, len(query), batch_size):
        batches.append(np.asarray(model.predict(query.iloc[start:start+batch_size], output_type='mean'), float))
        if start % (batch_size*10) == 0 or start+batch_size >= len(query):
            print(f'TABICL predicted {min(start+batch_size,len(query))}/{len(query)}', flush=True)
    prediction = np.concatenate(batches)
    prediction = prediction*offset[query_rows] if spec['target'] in ('ratio','observed_ratio') else prediction+offset[query_rows]
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite TabICL predictions')
    timing = {'fit_seconds': fitted-started, 'predict_seconds': time.perf_counter()-fitted,
        'peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3,
        'context_rows': len(indices), 'features': x.shape[1], 'views': views,
        'cache_mode': cache_mode, 'query_batch_size': batch_size,
        'input_columns':list(x.columns),
        'checkpoint_sha256': sha256(CHECKPOINT)}
    return model, prediction, query_rows, indices, timing


def screen_rows(bundle):
    # 200 rows per month, chosen without using target values.
    generator = np.random.default_rng(729)
    rows = []
    for month in sorted(set(bundle['date_val'])):
        eligible = np.flatnonzero(bundle['date_val'] == month)
        rows.extend(generator.choice(eligible, min(200,len(eligible)), replace=False))
    return np.sort(rows)


def aligned_bundle(start, end):
    source = prepare_snapshots(start, end, True, False, .85)
    parts = []
    for frame in (source['x_fit'], source['x_val']):
        x = frame[list(COMPACT)].copy()
        for new, old in [('BLOCK_FLAT','BLOCK'),('AREA_FLAT','AREA')]:
            for target, suffix in [('MEAN','RENT_MEAN'),('COUNT','N'),('LAST_MEAN','LAST_MEAN'),('RECENCY_MONTHS','RECENCY')]:
                x[f'HIST_{new}_{target}'] = frame[f'{old}_{suffix}'].to_numpy()
        # Explicit forecast information is identical for context and query views.
        for column in ['INFORMATION_AGE','MARKET_LAST','MARKET_3M','MARKET_12M',
            'MARKET_LOG_SLOPE','FORECAST_HORIZON','MARKET_FORECAST','MARKET_COUNT_3M']:
            x[column] = frame[column].to_numpy()
        parts.append(x)
    return {'x_fit':parts[0],'x_val':parts[1], 'extra_fit':source['x_fit'],
        'extra_val':source['x_val'], 'y_fit':source['y_fit'],'y_val':source['y_val'],
        'row_val':source['row_val'],'date_val':source['date_val'],
        'hashes':source['hashes'],'warmup_complete':True}


def run(stage, names=None):
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {'specs': SPECS, 'views': 2, 'earlier_screen_months': ['2023-10','2024-03'],
        'screen_rows_per_month': 200, 'outer_months': ['2024-04','2025-03'],
        'june_months': ['2024-06','2025-03'],
        'sources': ['https://github.com/soda-inria/tabicl', 'https://huggingface.co/jingang/TabICL'],
        'notes': 'Only past training labels; author checkpoint unchanged; no target-period public rent records. Sampled earlier screening is not a full-window score.'}
    write_json(OUT/'plan.json', plan)
    start, end = ('2023-10','2024-04') if stage in ('smoke','screen','confirm') else (
        ('2024-04','2025-04') if stage == 'outer' else ('2024-06','2025-04'))
    default_bundle = None
    for spec in SPECS:
        if names and spec['name'] not in names:
            continue
        destination = OUT/stage/spec['name']
        if (destination/'results.json').exists():
            print('Already complete:',stage,spec['name'],flush=True)
            continue
        current = spec.copy()
        if spec.get('cohort'):
            from analysis.experiments.second_wave_anniversary import data as cohort_data
            bundle = cohort_data(start,end)
        else:
            bundle = aligned_bundle(start,end) if spec.get('aligned') else default_bundle
        if bundle is None:
            bundle = prepare_compact(start,end)
            default_bundle = bundle
        rows, views = None, spec.get('views',2)
        if stage == 'smoke':
            current['context'] = 1024
            rows, views = np.arange(32), 1
        elif stage == 'screen':
            rows = screen_rows(bundle)
        model, prediction, rows, context, timing = fit_predict(bundle,current,rows,views=views,smoke=stage=='smoke')
        artifact = ART/stage/spec['name']
        artifact.mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'source_row':bundle['row_val'][rows],'month':bundle['date_val'][rows],
            'actual':bundle['y_val'][rows],'prediction':prediction}).to_csv(artifact/'predictions.csv',index=False)
        np.save(artifact/'context_indices.npy',context)
        result = {'stage':stage,'spec':current,'rows':len(rows),
            **score(bundle['y_val'][rows],prediction),'timing':timing,'source_sha256':bundle['hashes']}
        write_json(destination/'results.json',result)
        print('TABICL RESULT',stage,spec['name'],result['rmse_sgd'],flush=True)
        if stage != 'screen':
            model.save(str(artifact/'model.pkl'),save_model_weights=False,save_training_data=True,save_kv_cache=False)
        del model
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        if stage == 'smoke':
            break


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['smoke','screen','confirm','outer','june'])
    parser.add_argument('--names',nargs='+')
    args=parser.parse_args()
    run(args.stage,args.names)
