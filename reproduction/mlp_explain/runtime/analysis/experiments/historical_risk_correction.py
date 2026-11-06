"""Test past-rent-distribution residual corrections without fitting to outer cases.

MLP/TabM forecasts at September 2023 provide out-of-time residuals. October-
November fit a small correction, December selects it, and October-December
refit the selected correction. January-March 2024 calibrates prediction
intervals; the unchanged April 2024-March 2025 outer set evaluates accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.experiments.neural_ensemble_uncertainty import RentIntervalCalibrator, interval_metrics


GRAINS = {
    'SIGNATURE': ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE_CLEAN', 'FLAT_MODEL_CLEAN', 'FLOOR_AREA_SQM', 'LEASE_COMMENCE_DATE'],
    'BLOCK_TYPE': ['TOWN', 'BLOCK', 'STREET', 'FLAT_TYPE_CLEAN'],
    'STREET_TYPE': ['TOWN', 'STREET', 'FLAT_TYPE_CLEAN'],
    'TOWN_TYPE_SIZE': ['TOWN', 'FLAT_TYPE_CLEAN', '_SIZE'],
    'TYPE_SIZE': ['FLAT_TYPE_CLEAN', '_SIZE'],
}


def historical_risk_features(fitting, queries, prediction, std, family_difference):
    """All label-derived features use frozen pre-origin fitting labels only."""
    if fitting.empty or queries.empty or fitting.RENT_APPROVAL_DATE.max() >= queries.RENT_APPROVAL_DATE.min():
        raise ValueError('Queries must follow all fitting months')
    f = fitting.copy()
    q = queries.drop(columns='MONTHLY_RENT', errors='ignore').copy()
    for frame in (f, q):
        frame['_SIZE'] = (frame.FLOOR_AREA_SQM / 10).round().astype('Int64')
    market = f.groupby('RENT_APPROVAL_DATE').MONTHLY_RENT.median().sort_index()
    last = float(market.iloc[-1])
    f['_ratio'] = f.MONTHLY_RENT / f.RENT_APPROVAL_DATE.map(market)
    f['_low'] = (f._ratio < .65).astype(float)
    f['_high'] = (f._ratio > 1.35).astype(float)
    result = pd.DataFrame({'prediction': np.asarray(prediction), 'std': np.asarray(std),
                           'family_difference': np.asarray(family_difference)}, index=q.index)
    for column in ['FLOOR_AREA_SQM', 'FLAT_AGE_YEARS', 'LATITUDE', 'LONGITUDE']:
        result[column] = q[column].to_numpy()
    result['prediction_market_ratio'] = np.asarray(prediction) / last
    end = pd.Period(f.RENT_APPROVAL_DATE.max(), freq='M')
    for window in (None, 12):
        part = f if window is None else f[pd.PeriodIndex(f.RENT_APPROVAL_DATE, freq='M') > end - window]
        suffix = 'ALL' if window is None else '12M'
        for grain, keys in GRAINS.items():
            grouped = part.groupby(keys, dropna=False, observed=True)
            stats = grouped.agg(mean=('_ratio', 'mean'), std=('_ratio', 'std'), median=('_ratio', 'median'),
                                count=('_ratio', 'size'), low=('_low', 'sum'), high=('_high', 'sum'))
            quantiles = grouped._ratio.quantile([.1, .9]).unstack(-1)
            stats['q10'], stats['q90'] = quantiles[.1], quantiles[.9]
            looked = stats.reindex(pd.MultiIndex.from_frame(q[keys])).reset_index(drop=True)
            count = looked['count'].to_numpy(float)
            prefix = f'{grain}_{suffix}_'
            result[prefix + 'count'] = np.log1p(count)
            for measure in ['low', 'high']:
                prior = float(part['_'+measure].mean())
                result[prefix + measure + '_share'] = (looked[measure].to_numpy() + 10 * prior) / (count + 10)
            for measure in ['mean', 'median', 'q10', 'q90']:
                result[prefix + measure + '_minus_prediction'] = looked[measure].to_numpy() * last - prediction
            result[prefix + 'std_sgd'] = looked['std'].to_numpy() * last
    return result


def make_corrector(spec):
    if spec['kind'] == 'ridge':
        model = Ridge(alpha=spec['alpha'])
    elif spec['kind'] == 'mlp':
        model = MLPRegressor(hidden_layer_sizes=(32,), alpha=spec['alpha'], random_state=42,
                             max_iter=150, batch_size=256, early_stopping=False, learning_rate_init=.001)
    else:
        raise ValueError('Not a learned corrector')
    return TransformedTargetRegressor(regressor=make_pipeline(
        SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True),
        StandardScaler(), model), transformer=StandardScaler())


def predict_correction(spec, model, features):
    if spec['kind'] == 'identity':
        return np.zeros(len(features))
    if spec['kind'] == 'history_blend':
        delta = features.SIGNATURE_ALL_mean_minus_prediction.fillna(
            features.BLOCK_TYPE_ALL_mean_minus_prediction).fillna(0).to_numpy()
        return np.clip(spec['weight'] * delta, -spec['cap'], spec['cap'])
    return np.clip(model.predict(features), -spec['cap'], spec['cap'])


def select_candidate(scores, minimum_relative_gain=.0025):
    baseline = next(row['rmse_sgd'] for row in scores if row['name'] == 'identity')
    winner = min(scores, key=lambda row: row['rmse_sgd'])
    return winner['name'] if winner['rmse_sgd'] < baseline * (1 - minimum_relative_gain) else 'identity'


def _json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    import joblib
    import torch
    from analysis.pipeline.config import DATA, ROOT
    from analysis.pipeline.auxiliary import build_auxiliary_features
    from analysis.pipeline.data import check_output_path, sha256
    from analysis.experiments.embedding_rent_search import fit_model
    from analysis.experiments.tabm_rent_search import fit as fit_tabm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=Path('reports/experiments/neural_ensemble_uncertainty/run-2026-11-02'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    a = parser.parse_args()
    out = check_output_path(a.output_dir)
    if out.exists() and any(out.iterdir()) and not a.resume:
        parser.error('Use a new output directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    before = sha256(DATA/'train.csv')
    original = json.loads((a.source_dir/'protocol.json').read_text())
    candidates = [{'name': 'identity', 'kind': 'identity'}]
    candidates += [dict(name=f'history_{w}', kind='history_blend', weight=w, cap=600) for w in [.1, .25, .5]]
    candidates += [dict(name=f'ridge_{alpha}', kind='ridge', alpha=alpha, cap=600) for alpha in [100, 1000, 10000]]
    candidates += [dict(name=f'mlp_{alpha}', kind='mlp', alpha=alpha, cap=600) for alpha in [10, 100]]
    protocol = {'source_dir': str(a.source_dir), 'source_train_sha256': before, 'candidates': candidates,
                'code_sha256': sha256(Path(__file__)), 'source_protocol_sha256': sha256(a.source_dir/'protocol.json'),
                'selection': 'October-November 2023 correction fit; December 2023 RMSE selection; minimum .25% gain else identity',
                'correction_refit': 'October-December 2023 out-of-time residuals only',
                'base_oof_origin': '2023-09; inherited fixed 3/4-epoch schedules, no new epoch selection',
                'calibration': '2024-01..2024-03; prediction intervals only',
                'outer': '2024-04..2025-03, unchanged 35158 rows',
                'limits': 'Exploratory; design motivated by previously inspected outer cases. No outer label fits correction or sets thresholds. OOF-to-refit distribution shift possible.',
                'promotion_rule': 'Outer RMSE must improve by >=1 SGD and MAE must not worsen; otherwise retain old predictions.'}
    if a.resume and json.loads((out/'protocol.json').read_text()) != protocol:
        parser.error('Resume inputs/code differ')
    _json(out/'protocol.json', protocol)
    (out/'source.py').write_text(Path(__file__).read_text())
    frame = build_auxiliary_features(pd.read_csv(DATA/'train.csv'))
    for c in ['TOWN', 'STREET', 'BLOCK']:
        frame[c] = frame[c].astype(str).str.casefold().str.strip()
    frame['STREET_ID'] = frame.TOWN+'|'+frame.STREET
    frame['BUILDING_ID'] = frame.STREET_ID+'|'+frame.BLOCK
    f = frame[frame.RENT_APPROVAL_DATE<'2023-10']
    v = frame[(frame.RENT_APPROVAL_DATE>='2023-10') & (frame.RENT_APPROVAL_DATE<'2024-01')]
    pieces = {}
    for family in ['mlp', 'tabm']:
        epoch = json.loads((a.source_dir/f'{family}_selected.json').read_text())['epochs']
        members = []
        for seed in original['seeds']:
            path = out/f'oof_{family}_{seed}.csv'
            if a.resume and path.exists():
                saved = pd.read_csv(path); np.testing.assert_array_equal(saved.source_row, v.index)
                prediction = saved.prediction.to_numpy()
            else:
                print('Earlier-origin refit', family, seed, epoch, flush=True)
                spec = dict(original['specs'][family], seed=seed)
                blind = v.drop(columns='MONTHLY_RENT')
                if family == 'mlp':
                    _, _, _, _, _, prediction, _ = fit_model(f, blind, spec, epochs=epoch)
                else:
                    _, _, _, _, _, prediction = fit_tabm(f, blind, spec, epochs=epoch)
                pd.DataFrame({'source_row': v.index, 'prediction': prediction}).to_csv(path, index=False)
            members.append(prediction)
        pieces[family] = np.column_stack(members)
    members = np.column_stack([pieces['mlp'], pieces['tabm']])
    mean, std = members.mean(1), members.std(1)
    features = historical_risk_features(f, v, mean, std, pieces['mlp'].mean(1)-pieces['tabm'].mean(1))
    features.to_csv(out/'correction_oof_features.csv', index_label='source_row')
    residual = v.MONTHLY_RENT.to_numpy()-mean
    train_mask = (v.RENT_APPROVAL_DATE<'2023-12').to_numpy()
    scores = []
    for spec in candidates:
        model = None
        if spec['kind'] in ['ridge', 'mlp']:
            model = make_corrector(spec).fit(features.loc[train_mask], residual[train_mask])
        adjustment = predict_correction(spec, model, features.loc[~train_mask])
        error = mean[~train_mask]+adjustment-v.MONTHLY_RENT.to_numpy()[~train_mask]
        scores.append({'name': spec['name'], 'rmse_sgd': float(np.sqrt(np.mean(error**2))),
                       'mae_sgd': float(np.mean(abs(error))), 'mean_adjustment': float(adjustment.mean())})
    pd.DataFrame(scores).to_csv(out/'december_selection_scores.csv', index=False)
    name = select_candidate(scores)
    spec = next(spec for spec in candidates if spec['name']==name)
    model = make_corrector(spec).fit(features, residual) if spec['kind'] in ['ridge', 'mlp'] else None
    _json(out/'selected.json', {'name': name, 'spec': spec, 'selection_scores': scores})
    if model is not None:
        joblib.dump(model, out/'corrector.joblib')
    print('Frozen correction selection', name, flush=True)
    # Source predictions were produced by frozen December-2023 models.
    frames = [pd.read_csv(a.source_dir/f'hybrid_{split}_predictions.csv') for split in ['calibration', 'outer', 'test']]
    saved = pd.concat(frames, ignore_index=True)
    if not saved.source_row.is_unique:
        raise ValueError('Duplicate query identifiers')
    test = build_auxiliary_features(pd.read_csv(DATA/'test.csv'))
    for c in ['TOWN', 'STREET', 'BLOCK']:
        test[c] = test[c].astype(str).str.casefold().str.strip()
    test.index = np.arange(len(frame), len(frame)+len(test))
    all_inputs = pd.concat([frame, test]).loc[saved.source_row]
    frozen_fit = frame[frame.RENT_APPROVAL_DATE<'2024-01']
    mean, std = saved.prediction_sgd.to_numpy(), saved.ensemble_std_sgd.to_numpy()
    members = saved[[c for c in saved if c.startswith('member_')]].to_numpy()
    features = historical_risk_features(frozen_fit, all_inputs.drop(columns='MONTHLY_RENT'), mean, std,
                                         members[:, :3].mean(1)-members[:, 3:].mean(1))
    correction = predict_correction(spec, model, features)
    predicted = mean+correction
    cal_mask = saved.partition.eq('calibration').to_numpy()
    outer_mask = saved.partition.eq('outer').to_numpy()
    cal_y = saved.actual_sgd.to_numpy()[cal_mask]
    calibration = RentIntervalCalibrator().fit(cal_y, predicted[cal_mask], std[cal_mask])
    _json(out/'frozen_calibration.json', calibration.metadata())
    output = saved[['source_row', 'month', 'partition', 'flat_type', 'Id', 'months_after_fit']].copy()
    output['base_prediction_sgd'], output['correction_sgd'], output['prediction_sgd'] = mean, correction, predicted
    output['ensemble_std_sgd'] = std
    output['actual_sgd'] = saved.actual_sgd
    for level, (lower, upper) in calibration.intervals(predicted, std).items():
        output[f'pi{level}_lower_sgd'], output[f'pi{level}_upper_sgd'] = lower, upper
    for split in ['calibration', 'outer', 'test']:
        part = output[output.partition.eq(split)].copy()
        if split == 'test':
            part = part.drop(columns='actual_sgd'); part['Id'] = part.Id.astype('int64')
        part.to_csv(out/f'{split}_predictions.csv', index=False)
    outer_y = saved.actual_sgd.to_numpy()[outer_mask]
    baseline_calibration = RentIntervalCalibrator().fit(cal_y, mean[cal_mask], std[cal_mask])
    scores = [{'model': 'baseline', **interval_metrics(outer_y, mean[outer_mask], std[outer_mask], baseline_calibration)},
              {'model': 'selected_correction', **interval_metrics(outer_y, predicted[outer_mask], std[outer_mask], calibration)}]
    pd.DataFrame(scores).to_csv(out/'outer_metrics.csv', index=False)
    old, new = scores
    promoted = new['rmse_sgd']<=old['rmse_sgd']-1 and new['mae_sgd']<=old['mae_sgd']
    ids = [123066, 123120, 121146]
    case = output.set_index('source_row').loc[ids].copy()
    case['csv_line'] = case.index+2
    case.to_csv(out/'jurong_west_case.csv', index_label='source_row')
    results = {'selected': name, 'promoted': promoted, 'baseline_rmse': old['rmse_sgd'], 'corrected_rmse': new['rmse_sgd'],
               'case_base_prediction': float(case.base_prediction_sgd.iloc[0]), 'case_corrected_prediction': float(case.prediction_sgd.iloc[0]),
               'case_base_rmse': float(np.sqrt(np.mean((case.base_prediction_sgd-case.actual_sgd)**2))),
               'case_corrected_rmse': float(np.sqrt(np.mean((case.prediction_sgd-case.actual_sgd)**2))),
               'outer_rows': int(outer_mask.sum()), 'test_rows': int(saved.partition.eq('test').sum()),
               'raw_source_unchanged': sha256(DATA/'train.csv')==before}
    _json(out/'result.json', results)
    features.loc[ids].to_csv(out/'jurong_west_risk_features.csv', index_label='source_row')
    if not results['raw_source_unchanged'] or results['outer_rows']!=35158:
        raise AssertionError('Source or validation rows changed')
    print(pd.DataFrame(scores).to_string(index=False), flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
