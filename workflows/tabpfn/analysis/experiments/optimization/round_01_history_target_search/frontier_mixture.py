"""Conditional-mean mixtures for low/ordinary/high rental regimes.

Membership uses the rent relative to a strictly past market forecast. Gates
predict membership from available features; no unknown query rent is used.
"""
from __future__ import annotations
import argparse
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from analysis.experiments.frontier_search import OUT, ART, prepare, score, write_json, summarize

SPECS = [
    {'name': 'mixture_low65', 'snapshot': False, 'thresholds': [.65]},
    {'name': 'mixture_three', 'snapshot': False, 'thresholds': [.7, 1.15]},
    {'name': 'mixture_snapshot', 'snapshot': True, 'thresholds': [.7, 1.15]},
]


def fit_mixture(spec, bundle, counts=None):
    x, q = bundle['x_fit'], bundle['x_val']
    base = x.MARKET_FORECAST.to_numpy(float)
    offset = q.MARKET_FORECAST.to_numpy(float)
    y, actual = bundle['y_fit'], bundle['y_val']
    group = np.digitize(y / base, spec['thresholds'])
    query_group = np.digitize(actual / offset, spec['thresholds'])
    classifier = LGBMClassifier(n_estimators=counts[0] if counts else 1200,
        learning_rate=.035, num_leaves=15, min_child_samples=150,
        reg_lambda=25., verbosity=-1, random_state=42, n_jobs=4)
    options = {} if counts else {'eval_set': [(q, query_group)],
        'callbacks': [early_stopping(80, verbose=False)]}
    classifier.fit(x, group, categorical_feature=bundle['categories'], **options)
    iterations = [counts[0] if counts else int(classifier.best_iteration_)]
    probability = classifier.predict_proba(q)
    prediction = np.zeros(len(q))
    experts = []
    for part in range(len(spec['thresholds']) + 1):
        mask = group == part
        val_mask = query_group == part
        model = LGBMRegressor(n_estimators=counts[part+1] if counts else 1400,
            learning_rate=.035, num_leaves=15, min_child_samples=70,
            reg_lambda=20., verbosity=-1, random_state=42, n_jobs=4)
        options = {} if counts else {'eval_set': [(q.loc[val_mask], (actual-offset)[val_mask])],
            'callbacks': [early_stopping(80, verbose=False)]}
        model.fit(x.loc[mask], (y-base)[mask], categorical_feature=bundle['categories'], **options)
        iterations.append(counts[part+1] if counts else int(model.best_iteration_))
        prediction += probability[:, part] * (model.predict(q) + offset)
        experts.append(model)
    return {'classifier': classifier, 'experts': experts}, prediction, iterations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['screen', 'outer'])
    parser.add_argument('--names', nargs='+')
    args = parser.parse_args()
    write_json(OUT / 'mixture_plan.json', {'specs': SPECS,
        'selection': ['2023-10', '2024-04'], 'outer': ['2024-04', '2025-04']})
    for spec in SPECS:
        if args.names and spec['name'] not in args.names:
            continue
        destination = OUT / args.stage / spec['name']
        if (destination / 'results.json').exists():
            continue
        start, end = ('2023-10', '2024-04') if args.stage == 'screen' else ('2024-04', '2025-04')
        counts = None if args.stage == 'screen' else json.loads(
            (OUT / 'screen' / spec['name'] / 'results.json').read_text())['iterations']
        begun = time.perf_counter()
        bundle = prepare(start, end, spec['snapshot'], False, .85)
        fitted, prediction, iterations = fit_mixture(spec, bundle, counts)
        artifact = ART / args.stage / spec['name']
        artifact.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'source_row': bundle['row_val'], 'month': bundle['date_val'],
            'actual': bundle['y_val'], 'prediction': prediction}).to_csv(artifact / 'predictions.csv', index=False)
        joblib.dump({'models': fitted, 'spec': spec}, artifact / 'model.joblib', compress=1)
        result = {'spec': spec, 'stage': args.stage, 'start': start, 'end': end,
            'n_fit': len(bundle['y_fit']), 'n_validation': len(prediction),
            'n_features': bundle['x_fit'].shape[1], 'iterations': iterations,
            **score(bundle['y_val'], prediction), 'source_sha256': bundle['hashes'],
            'seconds': time.perf_counter()-begun}
        write_json(destination / 'results.json', result)
        print(f"RESULT {args.stage} {spec['name']}: {result['rmse_sgd']:.4f}, iterations={iterations}", flush=True)
        summarize()
        del bundle, fitted
        gc.collect()


if __name__ == '__main__':
    main()
