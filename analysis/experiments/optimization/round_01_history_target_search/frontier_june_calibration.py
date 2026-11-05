"""Adapt June blend weights using only forecasts whose labels predate June.

Oct-Mar predictions were issued from September; Apr-May predictions were
issued from March. Both are out of time. No June-March prediction or label
is read by select(). Two weighting rules are declared before evaluation.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.experiments.frontier_search import OUT, ROOT, score, write_json
from analysis.experiments.frontier_matched import MATCHED, MODELS


def select():
    destination = MATCHED / 'recent_calibration_selection.json'
    if destination.exists():
        return
    members = json.loads((OUT / 'ensemble_selection.json').read_text())['members']
    matrix, reference = [], None
    for name, member in members.items():
        earlier = pd.read_csv(ROOT / member['screen_path'])
        later = pd.read_csv(ROOT / member['outer_path'])
        date = 'month' if 'month' in later else 'approval_month'
        later = later.loc[later[date] < '2024-06'].copy()
        earlier = earlier.rename(columns={'approval_month': 'month'})
        later = later.rename(columns={'approval_month': 'month'})
        part = pd.concat([earlier, later], ignore_index=True).sort_values('source_row')
        if not part.source_row.is_unique or part.month.max() >= '2024-06':
            raise AssertionError('Calibration is not strictly pre-June')
        if reference is None:
            reference = part
        np.testing.assert_array_equal(part.source_row, reference.source_row)
        np.testing.assert_array_equal(part.actual, reference.actual)
        matrix.append(part.prediction.to_numpy(float)/1000.)
    matrix = np.column_stack(matrix)
    actual = reference.actual.to_numpy(float)/1000.
    ages = pd.Period('2024-05', freq='M').ordinal - pd.PeriodIndex(reference.month, freq='M').asi8
    choices = {}
    for name, half_life in [('uniform_eight_month', None), ('recent_half_life3', 3.)]:
        sample_weight = np.ones(len(actual)) if half_life is None else np.exp2(-ages/half_life)
        sample_weight /= sample_weight.sum()
        def objective(weights):
            residual = matrix@weights - actual
            return np.dot(sample_weight, residual**2), 2*matrix.T@(sample_weight*residual)
        fitted = minimize(objective, np.full(matrix.shape[1], 1/matrix.shape[1]), jac=True,
            method='SLSQP', bounds=[(0., .6)]*matrix.shape[1],
            constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1., 'jac': lambda w: np.ones_like(w)}],
            options={'maxiter': 1000, 'ftol': 1e-12})
        if not fitted.success:
            raise RuntimeError(fitted.message)
        choices[name] = {'weights': {key: float(value) for key, value in zip(members, fitted.x)},
            'half_life_months': half_life, 'calibration_rows': len(actual),
            'calibration_months': [str(reference.month.min()), str(reference.month.max())],
            'weighted_calibration_rmse_sgd': float(np.sqrt(fitted.fun)*1000.)}
    write_json(destination, {'variants': choices, 'query_labels_used_for_selection': False,
        'note': 'Adds two development blend variants using only pre-June labels and out-of-time forecasts. Candidate pool is the earlier locked 12 components.'})
    print('Locked pre-June recalibration:', json.dumps(choices, indent=2), flush=True)


def evaluate():
    selection = json.loads((MATCHED / 'recent_calibration_selection.json').read_text())
    results = {}
    for name, variant in selection['variants'].items():
        reference, prediction = None, None
        for member, weight in variant['weights'].items():
            part = pd.read_csv(MODELS / member / 'predictions.csv').sort_values('source_row').reset_index(drop=True)
            if reference is None:
                reference, prediction = part, np.zeros(len(part))
            np.testing.assert_array_equal(part.source_row, reference.source_row)
            np.testing.assert_array_equal(part.actual, reference.actual)
            prediction += weight*part.prediction.to_numpy(float)
        reference['prediction'] = prediction
        reference.to_csv(MODELS / f'{name}_predictions.csv', index=False)
        results[name] = {'scores': score(reference.actual, prediction), 'selection': variant}
    write_json(MATCHED / 'recent_calibration_results.json', results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['select', 'evaluate'])
    args = parser.parse_args()
    select() if args.stage == 'select' else evaluate()
