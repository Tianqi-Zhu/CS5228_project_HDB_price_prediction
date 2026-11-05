"""Refit the already locked blend at the TabPFN June-2024 information origin.

No new features, iterations, configurations or weights are selected here.
The existing June-March development window is reused for a matched comparison.
"""
from __future__ import annotations
import gc
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from analysis.experiments.frontier_search import (OUT, ART, ROOT, prepare, frame_data,
    fit_model, score, write_json, predict_saved)
from analysis.experiments.frontier_compact import prepare_compact, ADDITIONS
from analysis.experiments.frontier_mixture import fit_mixture
from analysis.experiments.model_search import PROFILES, fit_tree
from analysis.pipeline.features import model_inputs
from analysis.pipeline.neural_models import fit_neural, predict_neural
from analysis.pipeline.config import DATE, TARGET

START, END = '2024-06', '2025-04'
MATCHED = OUT / 'matched_june'
MODELS = ART / 'matched_june'


def old_bundle(profile):
    frame, hashes = frame_data()
    fitting = frame.loc[frame[DATE] < START].copy()
    validation = frame.loc[(frame[DATE] >= START) & (frame[DATE] < END)].copy()
    candidate = PROFILES[profile]
    x, q = model_inputs(fitting, validation, candidate)
    fit_offset, query_offset = np.zeros(len(x)), np.zeros(len(q))
    eligible = np.ones(len(x), dtype=bool)
    if profile.endswith('residual'):
        fit_offset = x.MARKET_TYPE_LAST.to_numpy(float)
        query_offset = q.MARKET_TYPE_LAST.to_numpy(float)
        eligible = np.isfinite(fit_offset) & (fitting[DATE].to_numpy() >= '2021-07')
    return {'x_fit': x.loc[eligible].copy(), 'x_val': q,
        'y_fit': fitting[TARGET].to_numpy(float)[eligible]-fit_offset[eligible],
        'y_val': validation[TARGET].to_numpy(float)-query_offset,
        'offset': query_offset, 'actual': validation[TARGET].to_numpy(float),
        'row_val': validation.index.to_numpy(), 'date_val': validation[DATE].to_numpy(),
        'categorical': candidate.categorical, 'hashes': hashes}


def main():
    selection = json.loads((OUT / 'ensemble_selection.json').read_text())
    write_json(MATCHED / 'plan.json', {'origin': START, 'end_exclusive': END,
        'locked_selection_sha256': __import__('hashlib').sha256((OUT/'ensemble_selection.json').read_bytes()).hexdigest(),
        'notes': 'All 12 component weights and all iteration counts frozen on pre-April selection. Fit labels through May; no query labels used for fitting or selection.'})
    for name, member in selection['members'].items():
        if (MATCHED / name / 'results.json').exists():
            print('Completed matched refit:', name, flush=True)
            continue
        begun = time.perf_counter()
        destination = MODELS / name
        destination.mkdir(parents=True, exist_ok=True)
        print('Matched June refit:', name, flush=True)
        if name.startswith('old_'):
            family = name[4:]
            selected = json.loads((ROOT / 'output/experiment_runs/comparison_model_search/2024-04' / family / 'results.json').read_text())['selection']
            bundle = old_bundle(selected['profile'])
            if family in ('ft_transformer_default', 'tabnet'):
                fitted = fit_neural(family, bundle['x_fit'], bundle['y_fit'], bundle['categorical'],
                    selected['settings'], fixed_epochs=selected['selected_iterations'], device_name='cuda')
                predicted = predict_neural(fitted, bundle['x_val']) + bundle['offset']
                if family == 'tabnet':
                    fitted['model'].save_model(str(destination / 'model'))
                    joblib.dump({k: v for k, v in fitted.items() if k != 'model'}, destination / 'preprocessor.joblib')
                else:
                    import torch
                    torch.save({'state_dict': fitted['model'].state_dict(),
                        'encoder': fitted['encoder'], 'mean': fitted['mean'], 'std': fitted['std'],
                        'selected': selected}, destination / 'checkpoint.pt')
                del fitted
                import torch
                torch.cuda.empty_cache()
            else:
                fitted, prediction, _ = fit_tree(family, bundle, bundle['categorical'],
                    selected['settings'], selected['selected_iterations'])
                predicted = prediction + bundle['offset']
                joblib.dump({'model': fitted, 'selected': selected}, destination / 'model.joblib', compress=1)
                del fitted
            actual = bundle['actual']
            spec = {'name': name, 'family': family, 'selection': selected}
        else:
            earlier = json.loads((OUT / 'screen' / name / 'results.json').read_text())
            spec, iterations = earlier['spec'], earlier['iterations']
            if spec.get('family') == 'controlled_lgb':
                bundle = prepare_compact(START, END)
                x, q = bundle['x_fit'].copy(), bundle['x_val'].copy()
                for column in ADDITIONS[spec['group']]:
                    x[column] = bundle['extra_fit'][column].to_numpy(np.float32)
                    q[column] = bundle['extra_val'][column].to_numpy(np.float32)
                fitted = LGBMRegressor(n_estimators=iterations, learning_rate=.03,
                    num_leaves=spec['leaves'], min_child_samples=40, reg_lambda=10.,
                    n_jobs=4, verbosity=-1, random_state=42).fit(x, bundle['y_fit'])
                predicted = fitted.predict(q)
                saved = {'model': fitted, 'spec': spec}
                del x, q
            else:
                bundle = prepare(START, END, spec['snapshot'], spec.get('neighbors', False), spec.get('damping', .85))
                if spec.get('thresholds'):
                    fitted, predicted, _ = fit_mixture(spec, bundle, iterations)
                    saved = {'models': fitted, 'spec': spec}
                else:
                    fitted, predicted, _ = fit_model(spec, bundle, iterations)
                    saved = {'model': fitted, 'spec': spec}
            # Verify that a serialized component reconstructs the same predictions.
            joblib.dump(saved, destination / 'model.joblib', compress=1)
            restored = predict_saved(joblib.load(destination / 'model.joblib'), bundle)
            np.testing.assert_allclose(restored, predicted, rtol=0, atol=1e-7)
            actual = bundle['y_val']
            del saved, fitted
        result = {'name': name, 'origin': START, 'end_exclusive': END,
            'n_fit': len(bundle['y_fit']), 'n_validation': len(actual),
            'weight': member['weight'], **score(actual, predicted),
            'seconds': time.perf_counter()-begun, 'spec': spec,
            'source_sha256': bundle['hashes']}
        pd.DataFrame({'source_row': bundle['row_val'], 'month': bundle['date_val'],
            'actual': actual, 'prediction': predicted}).to_csv(destination / 'predictions.csv', index=False)
        write_json(MATCHED / name / 'results.json', result)
        print(f"MATCHED {name}: RMSE={result['rmse_sgd']:.4f}, seconds={result['seconds']:.1f}", flush=True)
        del bundle
        gc.collect()
    reference, prediction, components = None, None, []
    for name, member in selection['members'].items():
        part = pd.read_csv(MODELS / name / 'predictions.csv').sort_values('source_row').reset_index(drop=True)
        if reference is None:
            reference = part
            prediction = np.zeros(len(part))
        np.testing.assert_array_equal(part.source_row, reference.source_row)
        np.testing.assert_array_equal(part.actual, reference.actual)
        prediction += member['weight'] * part.prediction.to_numpy(float)
        components.append(json.loads((MATCHED / name / 'results.json').read_text()))
    reference['prediction'] = prediction
    reference.to_csv(MODELS / 'ensemble_predictions.csv', index=False)
    scores = score(reference.actual, prediction)
    result = {'origin': START, 'validation_months': ['2024-06', '2025-03'],
        'context_rows_available': 121005, 'validation_rows': len(reference),
        'scores': scores, 'components': [{k: v for k, v in r.items() if k not in ('spec', 'source_sha256')} for r in components],
        'prior_tabpfn_frozen_rmse_sgd': 455.81357012,
        'prior_tabpfn_best_development_rmse_sgd': 455.07102836,
        'delta_vs_prior_tabpfn_best': scores['rmse_sgd'] - 455.07102836,
        'target_430_achieved': scores['rmse_sgd'] <= 430.,
        'note': 'Same origin/rows/raw SGD target as prior TabPFN reports. Prior TabPFN is a saved branch reference, not rerun; all periods were previously examined.'}
    write_json(MATCHED / 'summary.json', result)
    pd.DataFrame(result['components']).to_csv(MATCHED / 'component_metrics.csv', index=False)
    print('MATCHED JUNE ENSEMBLE', scores, flush=True)


if __name__ == '__main__':
    main()
