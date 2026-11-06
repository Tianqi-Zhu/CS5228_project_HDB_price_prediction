"""Chronological second-stage residual calibration of stored model forecasts.

Calibration rows are out-of-time base forecasts. Component hyperparameters
were selected by earlier project development, so this is development evidence.
The second-stage family/strength is selected before the April outer origin.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor

from analysis.experiments.frontier_search import ROOT, OUT as FIRST, ART as FIRST_ART, frame_data, score, write_json
from analysis.pipeline.features import COMPACT

OUT = ROOT / 'output/experiment_runs/round_02_anniversary_history'
ART = ROOT / 'output/models/round_02_anniversary_history'
JUNE_MODELS = ROOT / 'output/models/round_01_history_target_search/matched_june'


def calibration_data(stage):
    selection = json.loads((FIRST / 'ensemble_selection.json').read_text())['members']
    reference, predictions, baseline = None, {}, None
    for name, member in selection.items():
        if stage == 'screen':
            path = ROOT / member['screen_path']
        elif stage == 'outer':
            path = ROOT / member['outer_path']
        elif stage == 'june':
            path = JUNE_MODELS / name / 'predictions.csv'
        else:
            raise ValueError(stage)
        part = pd.read_csv(path).sort_values('source_row').reset_index(drop=True)
        part = part.rename(columns={'approval_month': 'month'})
        if reference is None:
            reference = part
            baseline = np.zeros(len(part))
        np.testing.assert_array_equal(part.source_row, reference.source_row)
        np.testing.assert_array_equal(part.actual, reference.actual)
        values = part.prediction.to_numpy(float)
        baseline += member['weight']*values
        predictions['PRED_' + name] = values
    source, hashes = frame_data()
    rows = reference.source_row.to_numpy(int)
    source = source.iloc[rows].copy().reset_index(drop=True)
    np.testing.assert_array_equal(source.MONTHLY_RENT, reference.actual)
    x = pd.DataFrame(predictions)
    values = x.to_numpy()
    # Forecast disagreements are available without the query's target.
    x = (x.sub(baseline, axis=0) / 1000.)
    x['PRED_BASE'] = baseline/1000.
    x['PRED_SD'] = np.std(values, axis=1)/1000.
    x['PRED_RANGE'] = (values.max(axis=1)-values.min(axis=1))/1000.
    for column in COMPACT:
        if column != 'APPROVAL_MONTH_INDEX':
            x[column] = source[column].to_numpy()
    return {'x': x, 'baseline': baseline, 'actual': reference.actual.to_numpy(float),
        'row': rows, 'month': reference.month.to_numpy(str), 'hashes': hashes}


def select_columns(x, features):
    if features == 'predictions':
        return x[[c for c in x if c.startswith('PRED_')]]
    if features == 'type':
        return x[[c for c in x if c.startswith('PRED_') or c in ('FLAT_TYPE_CLEAN', 'TOWN_CLEAN')]]
    return x.copy()


def fit_corrector(spec, x, y):
    x = select_columns(x, spec['features'])
    categories = list(x.select_dtypes(include=['object', 'string', 'category']).columns)
    numeric = list(x.columns.difference(categories))
    prep = ColumnTransformer([
        ('numbers', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True,
            add_indicator=True)), ('scale', StandardScaler())]), numeric),
        ('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categories)])
    if spec['family'] == 'ridge':
        model = Ridge(alpha=spec['alpha'])
    elif spec['family'] == 'lgb':
        model = LGBMRegressor(n_estimators=spec['iterations'], num_leaves=spec['leaves'],
            learning_rate=.03, min_child_samples=200, reg_lambda=30.,
            verbosity=-1, random_state=42, n_jobs=4)
    elif spec['family'] == 'extra':
        model = ExtraTreesRegressor(n_estimators=180, max_depth=spec['depth'],
            min_samples_leaf=100, max_features=.8, n_jobs=4, random_state=42)
    else:
        raise ValueError(spec)
    fitted = Pipeline([('preprocess', prep), ('model', model)]).fit(x, y)
    return fitted


def predict_corrector(fitted, spec, x):
    return fitted.predict(select_columns(x, spec['features']))


def trials():
    configurations = []
    for features in ('predictions', 'type', 'property'):
        for alpha in (.1, 1., 10., 100., 1000.):
            configurations.append({'family': 'ridge', 'features': features, 'alpha': alpha})
    for features in ('type', 'property'):
        for leaves in (3, 7, 15):
            for iterations in (100, 300):
                configurations.append({'family': 'lgb', 'features': features,
                    'leaves': leaves, 'iterations': iterations})
        for depth in (3, 6):
            configurations.append({'family': 'extra', 'features': features, 'depth': depth})
    return configurations


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / 'stacking_selection.json'
    screen = calibration_data('screen')
    train = screen['month'] < '2024-01'
    valid = ~train
    if not destination.exists():
        write_json(OUT / 'stacking_plan.json', {'trials': trials(),
            'correction_strengths': [0., .1, .25, .5, 1.],
            'meta_fit': ['2023-10', '2023-12'], 'meta_selection': ['2024-01', '2024-03'],
            'outer': ['2024-04', '2025-03'],
            'note': 'Prior base-model selection reused this six-month development window. No post-March label selects a correction.'})
        target = (screen['actual']-screen['baseline'])/1000.
        evaluations, best = [], {}
        for index, spec in enumerate(trials()):
            name = f"{spec['family']}_{spec['features']}"
            fitted = fit_corrector(spec, screen['x'].loc[train], target[train])
            correction = predict_corrector(fitted, spec, screen['x'].loc[valid])*1000.
            for strength in (0., .1, .25, .5, 1.):
                metrics = score(screen['actual'][valid], screen['baseline'][valid]+strength*correction)
                record = {'spec': spec, 'strength': strength, **metrics}
                evaluations.append({'trial': index, 'group': name, 'strength': strength, **metrics})
                if name not in best or metrics['rmse_sgd'] < best[name]['rmse_sgd']:
                    best[name] = record
            print(f"META SCREEN {index+1}/{len(trials())} {name}: best={best[name]['rmse_sgd']:.4f}, strength={best[name]['strength']}", flush=True)
            del fitted
        pd.DataFrame(evaluations).to_csv(OUT / 'stacking_screen.csv', index=False)
        write_json(destination, {'selected': best, 'selection_before_outer_scoring': True})
    selected = json.loads(destination.read_text())['selected']
    outer = calibration_data('outer')
    # Refit selected strengths/models on the complete pre-April calibration set.
    records = []
    for name, item in selected.items():
        spec, strength = item['spec'], item['strength']
        start = time.perf_counter()
        fitted = fit_corrector(spec, screen['x'], (screen['actual']-screen['baseline'])/1000.)
        prediction = outer['baseline'] + strength*predict_corrector(fitted, spec, outer['x'])*1000.
        directory = ART / 'stacking' / name
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': fitted, 'spec': spec, 'strength': strength}, directory/'model.joblib', compress=1)
        pd.DataFrame({'source_row': outer['row'], 'month': outer['month'],
            'actual': outer['actual'], 'prediction': prediction}).to_csv(directory/'predictions.csv', index=False)
        record = {'name': name, 'spec': spec, 'strength': strength,
            'rows': len(prediction), **score(outer['actual'], prediction), 'seconds': time.perf_counter()-start}
        write_json(OUT / 'stacking' / name / 'results.json', record)
        print('META OUTER', name, record['rmse_sgd'], flush=True)
        records.append({k: v for k,v in record.items() if k != 'spec'})
        del fitted
        gc.collect()
    pd.DataFrame(records).sort_values('rmse_sgd').to_csv(OUT/'stacking_metrics.csv', index=False)


def june():
    # Configurations/strengths stay locked from the earlier Jan-Mar selection.
    selected = json.loads((OUT/'stacking_selection.json').read_text())['selected']
    earlier, april, query = calibration_data('screen'), calibration_data('outer'), calibration_data('june')
    eligible = april['month'] < '2024-06'
    fitting_x = pd.concat([earlier['x'], april['x'].loc[eligible]], ignore_index=True)
    target = np.concatenate([earlier['actual']-earlier['baseline'],
        april['actual'][eligible]-april['baseline'][eligible]])/1000.
    records = []
    for name, item in selected.items():
        fitted = fit_corrector(item['spec'], fitting_x, target)
        predicted = query['baseline']+item['strength']*predict_corrector(fitted, item['spec'], query['x'])*1000.
        directory = ART/'stacking_june'/name
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': fitted, 'spec': item['spec'], 'strength': item['strength']}, directory/'model.joblib', compress=1)
        pd.DataFrame({'source_row': query['row'], 'month': query['month'], 'actual': query['actual'],
            'prediction': predicted}).to_csv(directory/'predictions.csv', index=False)
        result = {'name': name, 'rows': len(predicted), 'strength': item['strength'],
            **score(query['actual'], predicted), 'spec': item['spec']}
        write_json(OUT/'stacking_june'/name/'results.json', result)
        records.append({k: v for k,v in result.items() if k != 'spec'})
        print('META JUNE', name, result['rmse_sgd'], flush=True)
    pd.DataFrame(records).sort_values('rmse_sgd').to_csv(OUT/'stacking_june_metrics.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['run', 'june'])
    args = parser.parse_args()
    run() if args.stage == 'run' else june()
