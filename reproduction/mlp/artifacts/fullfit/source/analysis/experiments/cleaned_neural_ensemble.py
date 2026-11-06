"""User-requested low-rent removal experiment, with an unchanged outer holdout.

Two predeclared cleaning policies operate ONLY on the model fitting partition.
Architecture, seeds, epoch counts and averaging match the uncleaned ensemble.
January-March 2024 remains a separate, unfiltered interval-calibration set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from analysis.experiments.neural_ensemble_uncertainty import (
    RentIntervalCalibrator, ensemble_summary, export_predictions, interval_metrics,
)


COMPARABLE_KEYS = ['RENT_APPROVAL_DATE', 'TOWN', 'BLOCK', 'STREET',
                   'FLAT_TYPE_CLEAN', 'FLAT_MODEL_CLEAN', 'FLOOR_AREA_SQM', 'LEASE_COMMENCE_DATE']


def training_cleaning_audit(frame, policy):
    """Return row-aligned decisions; use only the supplied fitting rows' labels.

    This encodes an experimental hypothesis, not verified room-rental labels.
    Duplicate maximum-price observations remain; no separate deduplication.
    """
    if policy not in ('obvious_low', 'highest_only'):
        raise ValueError('Unknown cleaning policy')
    if frame.empty or not frame.index.is_unique:
        raise ValueError('Nonempty, uniquely indexed fitting data required')
    f = frame.copy(deep=True)
    required = set(COMPARABLE_KEYS) - {'FLAT_TYPE_CLEAN', 'FLAT_MODEL_CLEAN'}
    required.add('MONTHLY_RENT')
    if not required.issubset(f):
        raise ValueError(f'Missing cleaning fields: {sorted(required.difference(f))}')
    for c in ('TOWN', 'BLOCK', 'STREET'):
        f[c] = f[c].astype('string').str.casefold().str.strip().str.replace(r'\s+', ' ', regex=True)
    if 'FLAT_TYPE_CLEAN' not in f:
        f['FLAT_TYPE_CLEAN'] = f.FLAT_TYPE.astype('string')
    f['FLAT_TYPE_CLEAN'] = f.FLAT_TYPE_CLEAN.str.casefold().str.strip().str.replace(
        r'^(\d+)\s*[- ]\s*room$', r'\1-room', regex=True)
    if 'FLAT_MODEL_CLEAN' not in f:
        f['FLAT_MODEL_CLEAN'] = f.FLAT_MODEL.astype('string')
    f['FLAT_MODEL_CLEAN'] = f.FLAT_MODEL_CLEAN.str.casefold().str.strip()
    y = pd.to_numeric(f.MONTHLY_RENT, errors='raise')
    if not np.isfinite(y.to_numpy()).all() or (y <= 0).any():
        raise ValueError('Cleaning expects positive finite rental labels')
    f['MONTHLY_RENT'] = y
    dates = pd.PeriodIndex(f.RENT_APPROVAL_DATE, freq='M')
    if dates.isna().any():
        raise ValueError('Invalid fitting dates')
    same = f.groupby(COMPARABLE_KEYS, observed=True, dropna=False).MONTHLY_RENT
    maximum, count = same.transform('max'), same.transform('size')
    local = f.groupby(['RENT_APPROVAL_DATE', 'TOWN', 'FLAT_TYPE_CLEAN'], observed=True, dropna=False).MONTHLY_RENT
    median, local_count = local.transform('median'), local.transform('size')
    large_low = (f.FLOOR_AREA_SQM >= 90) & (y <= 1300)
    local_low = (local_count >= 20) & (y < .5 * median)
    comparable_low = ((count >= 2) & (y < .6 * maximum)) if policy == 'obvious_low' else ((count >= 2) & (y < maximum))
    reasons = []
    for a, b, c in zip(large_low, local_low, comparable_low):
        flags = []
        if a: flags.append('large_area_and_rent_le_1300')
        if b: flags.append('rent_lt_half_same_month_town_type_median')
        if c: flags.append('rent_lt_60pct_identical_input_max' if policy == 'obvious_low' else 'below_identical_input_max')
        reasons.append(';'.join(flags))
    return pd.DataFrame({'remove': large_low | local_low | comparable_low, 'reason': reasons,
                         'same_input_n': count, 'same_input_max_rent': maximum,
                         'same_month_town_type_n': local_count, 'same_month_town_type_median': median,
                         'large_area_low_flag': large_low, 'local_low_flag': local_low,
                         'comparable_low_flag': comparable_low}, index=frame.index)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def _normalise(frame):
    result = frame.copy()
    for c in ['TOWN', 'STREET', 'BLOCK']:
        result[c] = result[c].astype(str).str.casefold().str.strip()
    result['STREET_ID'] = result.TOWN+'|'+result.STREET
    result['BUILDING_ID'] = result.STREET_ID+'|'+result.BLOCK
    return result


def main():
    import joblib
    import torch
    from analysis.pipeline.config import DATA, ROOT
    from analysis.pipeline.data import check_output_path, sha256
    from analysis.pipeline.auxiliary import build_auxiliary_features
    from analysis.experiments.embedding_rent_search import fit_model
    from analysis.experiments.tabm_rent_search import fit as fit_tabm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path, default=Path('reports/experiments/neural_ensemble_uncertainty/run-2026-11-02'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = check_output_path(args.output_dir)
    if out.exists() and any(out.iterdir()) and not args.resume:
        parser.error('New output directory required, or --resume')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    original = json.loads((args.baseline_dir/'protocol.json').read_text())
    epochs = {family: int(json.loads((args.baseline_dir/f'{family}_selected.json').read_text())['epochs']) for family in ['mlp', 'tabm']}
    source_paths = [DATA/'train.csv', DATA/'test.csv', DATA/'example-submission.csv', *sorted((DATA/'auxiliary').glob('*.csv'))]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    for name, expected in original['source_sha256'].items():
        if hashes[name] != expected:
            raise ValueError(f'Baseline source differs: {name}')
    code_paths = [Path(__file__), ROOT/'analysis/experiments/embedding_rent_search.py',
                  ROOT/'analysis/experiments/tabm_rent_search.py', ROOT/'analysis/experiments/adjusted_rent_history.py',
                  ROOT/'analysis/experiments/deep_learning.py', ROOT/'analysis/experiments/spatiotemporal_hybrid.py',
                  ROOT/'analysis/experiments/neural_ensemble_uncertainty.py', ROOT/'analysis/pipeline/auxiliary.py', ROOT/'analysis/pipeline/data.py']
    protocol = {'policies': ['obvious_low', 'highest_only'], 'baseline_dir': str(args.baseline_dir),
                'fit': '2021-01..2023-12', 'calibration': '2024-01..2024-03 unfiltered',
                'outer': '2024-04..2025-03 all 35158 original records',
                'specs': original['specs'], 'seeds': original['seeds'], 'fixed_epochs': epochs,
                'aggregation': 'equal independent seed means; hybrid equal family means; no outer selection',
                'obvious_low_rules': ['area >=90 sqm AND rent <=1300 SGD', 'rent < .5 times same-month town/type median with >=20 records',
                                      'rent < .6 times maximum of identical-input group with >=2 records'],
                'highest_only_rules': ['first two obvious-low rules', 'within identical-input group retain only maximum-rent observations, preserving ties'],
                'comparable_keys': COMPARABLE_KEYS, 'history': 'histories, vocabularies, imputation/scaling, target scaling all rebuilt from retained fitting rows only',
                'source_sha256': hashes, 'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in code_paths},
                'versions': original['versions'],
                'limits': ['Suspected partial-rental classification is a user-requested hypothesis, not verified rental scope.',
                           'Target filtering intentionally changes the training population; full holdout is primary.',
                           'Same address is compared only within same month, type, model, area and lease year; not across years or different flat types.',
                           'Historical feature definitions inherit existing static-facility and reused-holdout limitations.',
                           'No calibration/outer/test rows are removed; test labels are unavailable.',
                           'Fixed epochs isolate filtering; no new epoch/hyperparameter tuning.',
                           'Historical experiment design used this development holdout previously.']}
    if args.resume and (not (out/'protocol.json').exists() or json.loads((out/'protocol.json').read_text()) != protocol):
        parser.error('Resume settings or source/code hashes differ')
    _json(out/'protocol.json', protocol)
    for p in code_paths:
        dest = out/'source'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())

    raw = pd.read_csv(DATA/'train.csv')
    frame = _normalise(build_auxiliary_features(raw))
    fitting = frame[frame.RENT_APPROVAL_DATE<'2024-01']
    known_query = frame[frame.RENT_APPROVAL_DATE>='2024-01']
    test = _normalise(build_auxiliary_features(pd.read_csv(DATA/'test.csv')))
    test.index = np.arange(len(frame), len(frame)+len(test))
    queries = pd.concat([known_query, test])
    meta = pd.DataFrame({'source_row': queries.index, 'month': queries.RENT_APPROVAL_DATE,
                         'partition': np.where(queries.index>=len(frame), 'test', np.where(queries.RENT_APPROVAL_DATE<'2024-04', 'calibration', 'outer')),
                         'flat_type': queries.FLAT_TYPE_CLEAN}).reset_index(drop=True)
    sample = pd.read_csv(DATA/'example-submission.csv')
    meta['Id'] = pd.array([None]*len(known_query)+sample.Id.tolist(), dtype='Int64')
    meta['months_after_fit'] = pd.PeriodIndex(meta.month, freq='M').asi8-pd.Period('2023-12', freq='M').ordinal
    masks = {name: meta.partition.eq(name).to_numpy() for name in ['calibration', 'outer', 'test']}
    if masks['outer'].sum()!=35158 or masks['calibration'].sum()!=9005 or masks['test'].sum()!=50000:
        raise ValueError('Original query partitions changed')
    blind = queries.drop(columns='MONTHLY_RENT', errors='ignore')
    comparisons, band_scores, summaries = [], [], []
    # Freeze decisions before opening any baseline outer labels or scoring predictions.
    audits = {policy: training_cleaning_audit(fitting, policy) for policy in protocol['policies']}
    # Read baseline predictions only as fixed controls; no rule is tuned from them.
    for family in ['mlp', 'tabm', 'hybrid']:
        b = pd.read_csv(args.baseline_dir/f'{family}_outer_predictions.csv')
        c = pd.read_csv(args.baseline_dir/f'{family}_calibration_predictions.csv')
        np.testing.assert_array_equal(b.source_row, meta.loc[masks['outer'], 'source_row'])
        calibration = RentIntervalCalibrator().fit(c.actual_sgd, c.prediction_sgd, c.ensemble_std_sgd)
        comparisons.append({'policy': 'uncleaned_baseline', 'model': family, 'n_fit': len(fitting),
                            **interval_metrics(b.actual_sgd.to_numpy(), b.prediction_sgd.to_numpy(), b.ensemble_std_sgd.to_numpy(), calibration)})
        for band, mask in [('rent_le_1900', b.actual_sgd.to_numpy()<=1900),
                           ('rent_1900_to_3800', (b.actual_sgd.to_numpy()>1900)&(b.actual_sgd.to_numpy()<=3800)),
                           ('rent_gt_3800', b.actual_sgd.to_numpy()>3800)]:
            band_scores.append({'policy':'uncleaned_baseline','model':family,'band':band,
                                **interval_metrics(b.actual_sgd.to_numpy()[mask],b.prediction_sgd.to_numpy()[mask],b.ensemble_std_sgd.to_numpy()[mask],calibration)})
    # All cleaning decisions are frozen before any retraining or outer scoring.
    for policy, audit in audits.items():
        folder = out/policy; folder.mkdir(exist_ok=True)
        kept = fitting.loc[~audit['remove']]
        deletion = raw.loc[audit.index[audit['remove']]].copy()
        deletion.insert(0, 'SOURCE_ROW', deletion.index); deletion.insert(1, 'ORIGINAL_CSV_LINE', deletion.index+2)
        deletion = deletion.join(audit.loc[deletion.index])
        deletion.to_csv(folder/'removed_training_rows.csv', index=False)
        saved_clean = raw.loc[kept.index].copy(); saved_clean.insert(0, 'SOURCE_ROW', saved_clean.index);saved_clean.insert(1, 'ORIGINAL_CSV_LINE', saved_clean.index+2)
        saved_clean.to_csv(folder/'retained_training_rows.csv', index=False)
        count = {'policy': policy, 'n_before': len(fitting), 'n_removed': int(audit['remove'].sum()),
                 'n_retained': len(kept), 'removed_fraction': float(audit['remove'].mean()),
                 'large_low_flag': int(audit.large_area_low_flag.sum()), 'local_low_flag': int(audit.local_low_flag.sum()),
                 'comparable_low_flag': int(audit.comparable_low_flag.sum()), 'hypothesis_not_verified_rental_scope': True}
        _json(folder/'cleaning_summary.json', count); summaries.append(count)
        print('Cleaning frozen', json.dumps(count), flush=True)
    _json(out/'cleaning_summary.json', summaries)

    for policy, audit in audits.items():
        folder = out/policy; kept = fitting.loc[~audit['remove']]
        ensembles = {}
        for family in ['mlp', 'tabm']:
            parts = []
            for seed in original['seeds']:
                run = folder/f'{family}_seed{seed}'; run.mkdir(exist_ok=True)
                completion = run/'complete.json'; path = run/'predictions.csv'
                if args.resume and completion.exists():
                    complete = json.loads(completion.read_text())
                    if sha256(path)!=complete['prediction_sha256']:
                        raise ValueError('Saved predictions differ')
                    saved = pd.read_csv(path);np.testing.assert_array_equal(saved.source_row, meta.source_row)
                    pred = saved.prediction_sgd.to_numpy()
                else:
                    print('Training', policy, family, seed, 'rows', len(kept), 'epochs', epochs[family], flush=True)
                    spec = dict(original['specs'][family], seed=seed)
                    if family=='mlp':
                        model, prep, _, _, _, pred, _ = fit_model(kept, blind, spec, epochs=epochs[family])
                    else:
                        model, prep, _, _, _, pred = fit_tabm(kept, blind, spec, epochs=epochs[family])
                    if len(pred)!=len(meta) or not np.isfinite(pred).all():
                        raise ValueError('Bad or incomplete blind prediction')
                    pd.DataFrame({'source_row': meta.source_row, 'prediction_sgd': pred}).to_csv(path, index=False)
                    torch.save(model.state_dict(), run/'model.pt');joblib.dump(prep, run/'preprocessor.joblib')
                    _json(completion, {'seed':seed, 'epochs':epochs[family], 'n_fit':len(kept), 'prediction_sha256':sha256(path),
                                      'model_sha256':sha256(run/'model.pt'),'preprocessor_sha256':sha256(run/'preprocessor.joblib')})
                    print('Completed', policy, family, seed, flush=True)
                parts.append(pred)
            ensembles[family] = np.column_stack(parts)
        ensembles['hybrid'] = np.column_stack([ensembles['mlp'], ensembles['tabm']])
        cal_ids = meta.loc[masks['calibration'], 'source_row']
        # Calibration labels remain untouched and are used only after all models are fitted.
        cal_y = raw.loc[cal_ids, 'MONTHLY_RENT'].to_numpy()
        calibrators = {}
        for family, values in ensembles.items():
            mean, std = ensemble_summary(values)
            calibrators[family] = RentIntervalCalibrator().fit(cal_y, mean[masks['calibration']], std[masks['calibration']])
        _json(folder/'frozen_calibration.json', {name:c.metadata() for name,c in calibrators.items()})
        outer_ids = meta.loc[masks['outer'], 'source_row']; outer_y = raw.loc[outer_ids, 'MONTHLY_RENT'].to_numpy()
        for family, values in ensembles.items():
            exported = export_predictions(meta, values, calibrators[family])
            exported['actual_sgd'] = np.nan
            exported.loc[masks['calibration'], 'actual_sgd'] = cal_y
            exported.loc[masks['outer'], 'actual_sgd'] = outer_y
            for partition, mask in masks.items():
                piece = exported.loc[mask].copy()
                if partition=='test': piece = piece.drop(columns='actual_sgd')
                piece.to_csv(folder/f'{family}_{partition}_predictions.csv', index=False)
            mean, std = ensemble_summary(values)
            comparisons.append({'policy':policy,'model':family,'n_fit':len(kept),
                                **interval_metrics(outer_y, mean[masks['outer']], std[masks['outer']], calibrators[family])})
            for band, mask in [('rent_le_1900',outer_y<=1900),('rent_1900_to_3800',(outer_y>1900)&(outer_y<=3800)),('rent_gt_3800',outer_y>3800)]:
                score = interval_metrics(outer_y[mask],mean[masks['outer']][mask],std[masks['outer']][mask],calibrators[family])
                band_scores.append({'policy':policy,'model':family,'band':band,**score})
            if family=='hybrid':
                case_ids = [123066,123120,121146,147330,123448,124930]
                cases = exported.set_index('source_row').loc[case_ids].copy();cases['ORIGINAL_CSV_LINE']=cases.index+2
                cases.to_csv(folder/'outer_case_comparison.csv',index_label='source_row')
        pd.DataFrame(comparisons).to_csv(out/'comparison_metrics.csv', index=False)
        pd.DataFrame(band_scores).to_csv(out/'rent_band_metrics.csv', index=False)
    for name, expected in hashes.items():
        if sha256(ROOT/name)!=expected:
            raise AssertionError(f'Source changed: {name}')
    _json(out/'verification.json', {'raw_sources_unchanged':True,'n_outer':int(masks['outer'].sum()),
                                   'n_calibration':int(masks['calibration'].sum()),'n_test':int(masks['test'].sum()),
                                   'query_targets_removed_before_inference':True,'epochs_hyperparameters_seeds_unchanged':True,
                                   'cleaning_applied_only_to_model_fit':True})
    print(pd.DataFrame(comparisons).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
