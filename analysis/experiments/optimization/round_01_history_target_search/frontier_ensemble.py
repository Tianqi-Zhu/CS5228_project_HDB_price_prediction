"""Lock convex blend weights on earlier predictions, then evaluate once."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from analysis.experiments.frontier_search import OUT, ART, ROOT, score, write_json
from analysis.experiments.frontier_refinement import TRIALS as REFINEMENT
from analysis.experiments.frontier_compact import SPECS as CONTROLLED
from analysis.experiments.frontier_mixture import SPECS as MIXTURE

OLD = ROOT / 'output/experiment_runs/comparison_model_search/2024-04'


def module_for(name):
    if name in {s['name'] for s in REFINEMENT}:
        return 'analysis.experiments.frontier_refinement'
    if name in {s['name'] for s in CONTROLLED}:
        return 'analysis.experiments.frontier_compact'
    if name in {s['name'] for s in MIXTURE}:
        return 'analysis.experiments.frontier_mixture'
    return 'analysis.experiments.frontier_search'


def align(path, reference=None):
    frame = pd.read_csv(path).sort_values('source_row').reset_index(drop=True)
    if reference is not None:
        np.testing.assert_array_equal(frame.source_row, reference.source_row)
        np.testing.assert_array_equal(frame.actual, reference.actual)
    return frame


def select():
    if (OUT / 'ensemble_selection.json').exists():
        print('Selection is already locked; preserving it.', flush=True)
        return
    candidates = {}
    for path in OLD.glob('*/results.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['family'] == 'inner_blend':
            continue
        selected = record['selection']
        inner = path.parent / f"trial_{selected['trial_index']}" / 'predictions.csv'
        if inner.exists():
            candidates['old_' + record['family']] = {'screen_path': str(inner.relative_to(ROOT)),
                'outer_path': str((path.parent / 'predictions.csv').relative_to(ROOT)), 'module': None}
    for path in (OUT / 'screen').glob('*/results.json'):
        name = path.parent.name
        candidates[name] = {'screen_path': str((ART / 'screen' / name / 'predictions.csv').relative_to(ROOT)),
            'outer_path': str((ART / 'outer' / name / 'predictions.csv').relative_to(ROOT)),
            'module': module_for(name)}
    reference, predictions, retained = None, [], {}
    diagnostics = []
    for name, candidate in candidates.items():
        frame = align(ROOT / candidate['screen_path'], reference)
        if reference is None:
            reference = frame
        rmse = score(frame.actual, frame.prediction)['rmse_sgd']
        diagnostics.append({'name': name, 'screen_rmse': rmse})
        # A broad fixed near-baseline eligibility threshold; excludes clear failures.
        if rmse <= 556.:
            retained[name] = candidate
            predictions.append(frame.prediction.to_numpy(float))
    matrix = np.column_stack(predictions) / 1000.
    actual = reference.actual.to_numpy(float) / 1000.
    def objective(w):
        residual = matrix @ w - actual
        return np.mean(residual**2), 2 * matrix.T @ residual / len(actual)
    fitted = minimize(objective, np.full(matrix.shape[1], 1/matrix.shape[1]),
        jac=True, method='SLSQP', bounds=[(0., .6)] * matrix.shape[1],
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1., 'jac': lambda w: np.ones_like(w)}],
        options={'maxiter': 1000, 'ftol': 1e-12})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    weights = np.where(fitted.x > 1e-5, fitted.x, 0.)
    weights /= weights.sum()
    selected = {name: value | {'weight': float(weight)} for (name, value), weight
        in zip(retained.items(), weights) if weight > 0}
    result = {'members': selected, 'screen_scores': score(actual*1000, matrix@weights*1000),
        'selection_window': ['2023-10', '2024-04'], 'outer_window': ['2024-04', '2025-04'],
        'n_screen': len(actual), 'candidate_diagnostics': diagnostics,
        'objective': 'Nonnegative least-squares convex blend; each weight <=0.6; no intercept.',
        'outer_scores_used_in_weight_selection': False}
    write_json(OUT / 'ensemble_selection.json', result)
    print(json.dumps(result['members'], indent=2), flush=True)
    print('SELECTED SCREEN', result['screen_scores'], flush=True)


def evaluate():
    result = json.loads((OUT / 'ensemble_selection.json').read_text(encoding='utf-8'))
    for name, member in result['members'].items():
        if not (ROOT / member['outer_path']).exists():
            subprocess.run([sys.executable, '-u', '-m', member['module'], 'outer', '--names', name], check=True)
    reference, prediction = None, None
    component_scores = []
    for name, member in result['members'].items():
        frame = align(ROOT / member['outer_path'], reference)
        if reference is None:
            reference = frame
            prediction = np.zeros(len(frame))
        prediction += member['weight'] * frame.prediction.to_numpy(float)
        component_scores.append({'name': name, 'weight': member['weight'], **score(frame.actual, frame.prediction)})
    reference['prediction'] = prediction
    destination = ART / 'outer' / 'locked_ensemble'
    destination.mkdir(parents=True, exist_ok=True)
    reference.to_csv(destination / 'predictions.csv', index=False)
    final = {'scores': score(reference.actual, prediction), 'components': component_scores,
        'n_validation': len(reference), 'selection': result,
        'target_430_achieved': score(reference.actual, prediction)['rmse_sgd'] <= 430.}
    write_json(OUT / 'ensemble_results.json', final)
    print('OUTER LOCKED ENSEMBLE', final['scores'], flush=True)


def old_control():
    """Same convex optimizer and eligibility rule, using only old model families."""
    if (OUT / 'old_only_control.json').exists():
        return
    reference, candidates, matrix = None, [], []
    for path in OLD.glob('*/results.json'):
        record = json.loads(path.read_text())
        if record['family'] == 'inner_blend':
            continue
        inner = path.parent / f"trial_{record['selection']['trial_index']}" / 'predictions.csv'
        if not inner.exists():
            continue
        frame = align(inner, reference)
        if reference is None:
            reference = frame
        if score(frame.actual, frame.prediction)['rmse_sgd'] <= 556.:
            candidates.append((record['family'], path.parent / 'predictions.csv'))
            matrix.append(frame.prediction.to_numpy()/1000.)
    matrix = np.column_stack(matrix)
    actual = reference.actual.to_numpy()/1000.
    def objective(w):
        difference = matrix@w-actual
        return np.mean(difference**2), 2*matrix.T@difference/len(actual)
    fitted = minimize(objective, np.full(len(candidates), 1/len(candidates)), jac=True,
        method='SLSQP', bounds=[(0., .6)]*len(candidates),
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum()-1., 'jac': lambda w: np.ones_like(w)}],
        options={'maxiter': 1000, 'ftol': 1e-12})
    if not fitted.success:
        raise RuntimeError(fitted.message)
    outer, prediction = None, None
    for (_, path), weight in zip(candidates, fitted.x):
        frame = align(path, outer)
        if outer is None:
            outer, prediction = frame, np.zeros(len(frame))
        prediction += weight*frame.prediction.to_numpy()
    write_json(OUT / 'old_only_control.json', {'weights': {name: float(w) for (name, _), w in zip(candidates, fitted.x)},
        'screen_scores': score(actual*1000, matrix@fitted.x*1000),
        'outer_scores': score(outer.actual, prediction),
        'note': 'Control for expanded model pool: same earlier-window convex selection and 0.6 cap, only old families. Added after primary outer evaluation; no outer labels enter weight optimization.'})
    print('OLD-ONLY CONTROL', score(outer.actual, prediction), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['select', 'evaluate', 'old-control'])
    args = parser.parse_args()
    if args.stage == 'select':
        select()
    elif args.stage == 'evaluate':
        evaluate()
    else:
        old_control()
