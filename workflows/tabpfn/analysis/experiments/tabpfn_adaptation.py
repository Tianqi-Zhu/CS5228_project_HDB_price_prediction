"""Temporal feature ablations and frozen TabPFN representations for rent heads."""

from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from analysis.config import DATA, REPORTS, ROOT
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET, FOLDS, SOURCES, spatial_frame
from analysis.experiments.tabpfn_comparison import fixed_origin_split, write_json
from analysis.experiments.tabpfn_adaptation_features import (
    FEATURE_VARIANTS, ENHANCED, HEAD_COLUMNS, enhanced_feature_pair, backbone_inputs, head_numeric,
)
from analysis.experiments.tabpfn_residual_heads import (
    chronological_head_split, fit_ridge_head, fit_mlp_head,
)


OUT = REPORTS / 'tabpfn_adaptation'
HORIZONS = (1, 3, 6, 12)
ORIGINS = [str(p) for p in pd.period_range('2021-07', '2024-01', freq='3M')]
MODELS = ['catboost', *[f'pfn_{name}' for name in FEATURE_VARIANTS],
          'ridge_numeric', 'ridge_embedding', 'mlp_numeric', 'mlp_embedding']


def sampled_episode_queries(frame, origin, *, cap=600, seed=42):
    """Choose query rows without using rents, bounded before the last outer origin."""
    origin = pd.Period(origin, 'M')
    chunks = []
    for horizon in HORIZONS:
        month = origin + horizon - 1
        if month >= pd.Period(FOLDS[-1][0], 'M'):
            continue
        rows = frame[frame[MONTH].eq(month)]
        if len(rows) > cap:
            rng = np.random.default_rng(seed + origin.ordinal * 17 + horizon)
            rows = rows.loc[np.sort(rng.choice(rows.index, cap, replace=False))]
        if not rows.empty:
            chunks.append(rows)
    if not chunks:
        raise ValueError('No eligible historical episode query months')
    return pd.concat(chunks)


def assert_context_query(fitting, query, origin):
    origin = pd.Period(origin, 'M')
    if fitting.empty or query.empty:
        raise ValueError('Context and query must both be nonempty')
    if not (fitting[MONTH] < origin).all() or not (query[MONTH] >= origin).all():
        raise ValueError('Context labels must strictly precede the forecast origin')
    if len(fitting.index.intersection(query.index)):
        raise ValueError('A query row occurs in its labelled context')


def backbone_arrays(args, fitting, query, columns, *, embeddings):
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    x_fit, x_query = backbone_inputs(fitting, query, columns)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V3_5, device='cuda', n_estimators=1, random_state=args.seed,
        fit_mode='fit_with_cache', inference_config={'SUBSAMPLE_SAMPLES': args.context_rows},
    )
    try:
        model.fit(x_fit, fitting[TARGET].to_numpy(float))
        for backbone in model.models_:
            for parameter in backbone.parameters():
                parameter.requires_grad_(False)
        torch.cuda.synchronize()
        fitted = time.perf_counter()
        predictions, representations = [], []
        for offset in range(0, len(query), args.batch_size):
            x = x_query.iloc[offset:offset + args.batch_size]
            predictions.append(np.asarray(model.predict(x, output_type='mean'), float))
            if embeddings:
                embedding = model.get_embeddings(x, data_source='test')
                if embedding.ndim != 3 or embedding.shape[0] != 1:
                    raise ValueError('Expected one estimator and row-wise query embeddings')
                representations.append(np.asarray(embedding[0], np.float32))
            done = min(offset + args.batch_size, len(query))
            if done % 5000 == 0 or done == len(query):
                print(f'    query rows {done}/{len(query)}', flush=True)
        torch.cuda.synchronize()
        p = np.concatenate(predictions)
        e = np.concatenate(representations) if embeddings else np.empty((len(query), 0), np.float32)
        if not np.isfinite(p).all() or not np.isfinite(e).all():
            raise ValueError('Nonfinite backbone output')
        timing = {'fit_seconds': fitted - start,
                  'predict_embedding_seconds': time.perf_counter() - fitted,
                  'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
                  'context_population': len(fitting), 'query_rows': len(query),
                  'embedding_width': e.shape[1]}
        return p, e, timing
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def cached_arrays(path, build, signature):
    manifest = path.with_suffix('.json')
    if path.exists() and manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if saved['signature'] != signature or saved['sha256'] != sha256(path):
            raise ValueError(f'Cached arrays do not match: {path}')
        with np.load(path, allow_pickle=False) as arrays:
            return {name: arrays[name] for name in arrays.files}, saved['audit']
    arrays, audit = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    write_json(manifest, {'signature': signature, 'sha256': sha256(path), 'audit': audit})
    return arrays, audit


def episode_arrays(args, frame, origin, cache, signature):
    def build():
        fitting = frame[frame[MONTH] < pd.Period(origin, 'M')].copy()
        query = sampled_episode_queries(frame, origin, cap=args.episode_cap, seed=args.seed)
        assert_context_query(fitting, query, origin)
        fit, val = enhanced_feature_pair(fitting, query)
        p, e, timing = backbone_arrays(args, fit, val, ENHANCED, embeddings=True)
        horizon = query[MONTH].astype('int64').to_numpy() - pd.Period(origin, 'M').ordinal + 1
        arrays = {'embedding': e, 'numeric': head_numeric(val, p, pd.Period(origin, 'M')),
                  'base': p, 'actual': query[TARGET].to_numpy(float),
                  'row_id': query.index.to_numpy(), 'month': query[MONTH].astype(str).to_numpy(dtype='U7'),
                  'origin': np.full(len(query), origin, dtype='U7'), 'horizon': horizon}
        audit = {'origin': origin, 'context_last_month': str(fitting[MONTH].max()),
                 'query_months': sorted(query[MONTH].astype(str).unique().tolist()),
                 'query_rows': len(query), 'context_query_overlap': 0, **timing}
        return arrays, audit
    return cached_arrays(cache / f'episode_{origin}.npz', build, signature)


def settings_for(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this experiment')
    code = [Path(__file__), Path(__file__).with_name('tabpfn_adaptation_features.py'),
            Path(__file__).with_name('tabpfn_residual_heads.py'),
            ROOT / 'analysis/data.py', ROOT / 'analysis/experiments/recent_history.py',
            ROOT / 'analysis/experiments/spatiotemporal_hybrid.py']
    checkpoint = ROOT / '.cache/tabpfn/tabpfn-v3.5-20260909.safetensors'
    if not checkpoint.is_file():
        raise RuntimeError('Run the existing TabPFN access check to cache v3.5 weights first')
    return {'context_rows': args.context_rows, 'n_estimators': 1, 'seed': args.seed,
            'batch_size': args.batch_size, 'episode_cap': args.episode_cap, 'mlp_epochs': args.epochs,
            'folds': FOLDS, 'episode_origins': ORIGINS, 'horizons': HORIZONS,
            'feature_variants': FEATURE_VARIANTS, 'head_numeric_columns': HEAD_COLUMNS,
            'checkpoint_sha256': sha256(checkpoint),
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in SOURCES},
            'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in code},
            'versions': {name: version(name) for name in ['tabpfn', 'torch', 'numpy', 'pandas', 'scikit-learn']},
            'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
            'ridge_alphas': [10, 100, 1000], 'head_inner_months': 3,
            'mlp': {'widths': [64, 32], 'dropout': .1, 'learning_rate': .001,
                    'weight_decay': .001, 'patience': 6, 'batch_size': 256},
            'gradient_pilot_gate': 'Embedding head beats pfn_recent pooled and on at least two folds',
            'promotion_rule': 'At least 1% pooled RMSE gain and lower RMSE on all three folds',
            'plan_sha256': sha256(ROOT / 'EXPERIMENT_PLAN_TABPFN_ADAPTATION.md')}


def save_scores(rows, out):
    frame = pd.concat(rows, ignore_index=True)
    pooled, folds, months = [], [], []
    for model in MODELS:
        pooled.append({'model': model, 'n': len(frame), **metrics(frame[TARGET], frame[model])})
        for origin, part in frame.groupby('validation_start'):
            folds.append({'model': model, 'validation_start': origin, 'n': len(part),
                          **metrics(part[TARGET], part[model])})
            for month, group in part.groupby('month'):
                months.append({'model': model, 'validation_start': origin, 'month': month,
                               'n': len(group), **metrics(group[TARGET], group[model])})
    pooled = pd.DataFrame(pooled).sort_values('rmse_sgd')
    cat_rmse = float(pooled.loc[pooled.model.eq('catboost'), 'rmse_sgd'].iloc[0])
    pooled['delta_vs_catboost_sgd'] = pooled.rmse_sgd - cat_rmse
    for name, table in [('validation_predictions', frame), ('pooled_scores', pooled),
                        ('fold_scores', pd.DataFrame(folds)), ('monthly_scores', pd.DataFrame(months))]:
        table.to_csv(out / f'{name}.csv', index=False, float_format='%.8f')
    return pooled, pd.DataFrame(folds)


def report(out, settings, pooled, folds, audits, heads):
    cat = folds[folds.model.eq('catboost')].set_index('validation_start').rmse_sgd
    cat_pool = float(pooled.loc[pooled.model.eq('catboost'), 'rmse_sgd'].iloc[0])
    qualified, pilot = [], []
    matched = folds[folds.model.eq('pfn_recent')].set_index('validation_start').rmse_sgd
    matched_pool = float(pooled.loc[pooled.model.eq('pfn_recent'), 'rmse_sgd'].iloc[0])
    for model in MODELS[1:]:
        values = folds[folds.model.eq(model)].set_index('validation_start').rmse_sgd.reindex(cat.index)
        value = float(pooled.loc[pooled.model.eq(model), 'rmse_sgd'].iloc[0])
        if value <= .99 * cat_pool and (values < cat).all():
            qualified.append(model)
        if model in ['ridge_embedding', 'mlp_embedding'] and value < matched_pool and (values < matched).sum() >= 2:
            pilot.append(model)
    selected = next((name for name in pooled.model if name in qualified), 'catboost')
    def table(data):
        lines = ['| ' + ' | '.join(map(str, data.columns)) + ' |', '| ' + ' | '.join(['---'] * len(data.columns)) + ' |']
        for row in data.itertuples(index=False, name=None):
            lines.append('| ' + ' | '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in row) + ' |')
        return '\n'.join(lines)
    text = f'''# TabPFN feature engineering and frozen residual heads

Selected candidate: **{selected}**. Backbone-gradient pilot gate: **{'passed' if pilot else 'not passed'}** ({', '.join(pilot) or 'no qualifying embedding head'}).

## Pooled scores

{table(pooled)}

## Fold RMSE (SGD)

{table(folds.pivot(index='model', columns='validation_start', values='rmse_sgd').reset_index())}

## Design

All new feature and head models use the same frozen TabPFN-3.5 checkpoint, one estimator, {settings['context_rows']:,} context rows and seed {settings['seed']}. `pfn_original` is the matched smaller-backbone control; it is separate from the prior four-estimator/50,000-context benchmark. The reference CatBoost predictions are reused only after source hashes and row labels match.

`pfn_identifiers` adds normalized block/street categoricals. `pfn_recent` additionally uses strictly earlier 3/6/12-month counts, means, shrunken means, sample standard deviations and local contrasts. `pfn_seasonal_only` replaces approval-month trend/ordinal month with sine/cosine seasonality. Heads use the prespecified recent-feature backbone, irrespective of outer scores.

Quarterly historical origins from July 2021 to January 2024 generate horizons 1, 3, 6 and 12. Each query month is sampled without looking at rents, capped at {settings['episode_cap']} rows. Context labels end before the origin, queries begin at the origin, and row IDs are disjoint. Histories freeze at each origin. A source row can represent several forecasting tasks under different earlier origins; every copy goes to the same inner partition by its query month.

Each outer head uses only episode query labels before its own origin. The final three fitting months select Ridge alpha or MLP epoch; outer validation labels never select head hyperparameters. After selection, heads refit on all eligible earlier episodes. All imputation/scaling is fitted within the corresponding head-training partition. The backbone weights are frozen and embeddings are obtained as query tokens, not labelled training tokens.

Numeric-only heads control for calibration without learned representations. Embedding heads add the 1024-dimensional query representation to the same numeric inputs. Both predict an additive residual in SGD. The MLP has hidden widths 64/32, GELU, dropout 0.1, AdamW and patience 6. Model details and inner learning curves are in `head_selection.json`; episode audits are in `episode_audits.json`.

## Limits and reproduction

These outer windows have been examined previously, so scores are development evidence. Episode sampling, context capping and fixed head architectures are deliberate computational limits, not an exhaustive search. No future COE/stock observations are used. The plan predates scoring and defines both promotion and gradient-pilot rules.

Run `.venv/Scripts/python.exe analysis/compare_tabpfn_adaptation.py --resume`. Embeddings and intermediate arrays are under ignored `.cache/tabpfn_adaptation/`, signed by source/code/settings fingerprints. Row predictions retain source IDs. Raw CSVs and earlier model reports are preserved.
'''
    (out / 'report.md').write_text(text, encoding='utf-8')
    write_json(out / 'results.json', {'settings': settings, 'selected_candidate': selected,
                                    'gradient_pilot_models': pilot, 'pooled_scores': pooled.to_dict('records')})
    write_json(out / 'episode_audits.json', audits)
    write_json(out / 'head_selection.json', heads)
    return selected, pilot


def evaluate(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    settings = settings_for(args)
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
    settings_path = out / 'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text()) != json.loads(json.dumps(settings, default=str)):
        raise ValueError('Different experiment settings/code; choose a new output directory')
    write_json(settings_path, settings)
    cache = ROOT / '.cache/tabpfn_adaptation' / signature[:16]
    cache.mkdir(parents=True, exist_ok=True)
    frame = spatial_frame(pd.read_csv(DATA / 'train.csv'))
    if args.smoke:
        raw_fit = frame[frame[MONTH] < pd.Period('2021-07', 'M')]
        query = sampled_episode_queries(frame, '2021-07', cap=2, seed=args.seed)
        fit, val = enhanced_feature_pair(raw_fit, query)
        p, e, timing = backbone_arrays(args, fit.tail(256), val, ENHANCED, embeddings=True)
        print(json.dumps({'predictions': len(p), 'embedding_shape': list(e.shape), 'timing': timing}), flush=True)
        return
    prior = json.loads((REPORTS / 'tabpfn/results.json').read_text())
    if prior['settings']['sources_sha256'] != settings['source_sha256']:
        raise ValueError('Existing CatBoost control uses different source files')
    control = pd.read_csv(REPORTS / 'tabpfn/validation_predictions.csv')
    episodes, audits = [], []
    for origin in ORIGINS:
        print(f'Historical embedding origin {origin}', flush=True)
        arrays, audit = episode_arrays(args, frame, origin, cache, signature)
        episodes.append(arrays)
        audits.append(audit)
        write_json(out / 'episode_audits.json', audits)
    all_episodes = {name: np.concatenate([episode[name] for episode in episodes]) for name in episodes[0]}
    completed, selections = [], []
    for start, end in FOLDS:
        result_path = out / f'predictions_{start}.csv'
        manifest_path = out / f'completed_{start}.json'
        if args.resume and result_path.exists() and manifest_path.exists():
            saved = json.loads(manifest_path.read_text())
            if saved['signature'] != signature or saved['sha256'] != sha256(result_path):
                raise ValueError('Completed fold signature/hash mismatch')
            completed.append(pd.read_csv(result_path))
            selections.extend(saved['heads'])
            continue
        print(f'Outer fold {start}', flush=True)
        fitting, query = fixed_origin_split(frame, start, end)
        assert_context_query(fitting, query, start)
        fit, val = enhanced_feature_pair(fitting, query)
        existing = control[control.validation_start.eq(start)].set_index('row_id').loc[query.index]
        np.testing.assert_array_equal(existing[TARGET], query[TARGET])
        result = pd.DataFrame({'row_id': query.index, 'validation_start': start,
                               'month': query[MONTH].astype(str).to_numpy(),
                               TARGET: query[TARGET].to_numpy(float), 'catboost': existing.catboost.to_numpy()})
        for name, columns in FEATURE_VARIANTS.items():
            def build_variant(columns=columns, name=name):
                p, e, timing = backbone_arrays(args, fit, val, columns, embeddings=name == 'recent')
                return {'base': p, 'embedding': e, 'row_id': query.index.to_numpy()}, timing
            arrays, timing = cached_arrays(cache / f'outer_{start}_{name}.npz', build_variant, signature)
            np.testing.assert_array_equal(arrays['row_id'], query.index)
            result[f'pfn_{name}'] = arrays['base']
            if name == 'recent':
                outer_embedding = arrays['embedding']
            print(f"  pfn_{name}: RMSE {metrics(result[TARGET], arrays['base'])['rmse_sgd']:.2f}", flush=True)
            write_json(out / f'timing_{start}_{name}.json', timing)
        eligible = all_episodes['month'] < start
        head_data = {name: values[eligible] for name, values in all_episodes.items()}
        if not (head_data['origin'] <= head_data['month']).all():
            raise ValueError('Historical episode has a query before its origin')
        training, calibration = chronological_head_split(head_data['month'], start)
        query_numeric = head_numeric(val, result.pfn_recent.to_numpy(), pd.Period(start, 'M'))
        fold_heads = []
        for kind in ('ridge', 'mlp'):
            for representation in ('numeric', 'embedding'):
                name = f'{kind}_{representation}'
                x = head_data['numeric']
                qx = query_numeric
                if representation == 'embedding':
                    x = np.concatenate([head_data['embedding'], x], axis=1)
                    qx = np.concatenate([outer_embedding, qx], axis=1)
                if kind == 'ridge':
                    predicted, selection, fitted = fit_ridge_head(
                        x, head_data['actual'], head_data['base'], qx,
                        result.pfn_recent.to_numpy(), training, calibration,
                    )
                else:
                    predicted, selection, fitted = fit_mlp_head(
                        x, head_data['actual'], head_data['base'], qx,
                        result.pfn_recent.to_numpy(), training, calibration,
                        epochs=args.epochs, seed=args.seed,
                    )
                if not np.isfinite(predicted).all():
                    raise ValueError('Nonfinite head predictions')
                result[name] = predicted
                selection.update({'model': name, 'validation_start': start,
                                  'training_episodes': int(training.sum()), 'inner_rows': int(calibration.sum()),
                                  'latest_head_query_month': max(head_data['month'].tolist()),
                                  'head_input_width': x.shape[1]})
                fold_heads.append(selection)
                joblib.dump(fitted, out / f'head_{start}_{name}.joblib')
                print(f"  {name}: RMSE {metrics(result[TARGET], predicted)['rmse_sgd']:.2f}", flush=True)
        result.to_csv(result_path, index=False, float_format='%.8f')
        write_json(manifest_path, {'signature': signature, 'sha256': sha256(result_path), 'heads': fold_heads})
        selections.extend(fold_heads)
        completed.append(result)
        pooled, folds = save_scores(completed, out)
        write_json(out / 'execution_status.json', {'status': 'running', 'completed_folds': list(pd.concat(completed).validation_start.unique())})
    pooled, folds = save_scores(completed, out)
    selected, pilot = report(out, settings, pooled, folds, audits, selections)
    write_json(out / 'execution_status.json', {'status': 'complete', 'selected_candidate': selected,
                                               'gradient_pilot_models': pilot})
    (out / 'failure.json').unlink(missing_ok=True)
    print(pooled.to_string(index=False), flush=True)
    print(f'Gradient pilot gate: {pilot}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--context-rows', type=int, default=10_000)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--episode-cap', type=int, default=600)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if min(args.context_rows, args.batch_size, args.episode_cap, args.epochs) < 1:
        parser.error('Sizes and epochs must be positive')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('TABPFN_MODEL_CACHE_DIR', str(ROOT / '.cache/tabpfn'))
    os.environ.setdefault('TABPFN_NO_BROWSER', '1')
    os.environ.setdefault('TABPFN_MAX_BATCHED_ESTIMATOR_ROWS', '0')
    os.environ.setdefault('TABPFN_DISABLE_TELEMETRY', '1')
    try:
        evaluate(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise
