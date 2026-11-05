"""Fresh Y/J refits with frozen recipes and pre-origin blend weights.

Validation rents are held separately and replaced by zeros in every fitting
bundle. No early stopping, new weight fitting or model selection is performed.
Outputs belong to a new run; original development evidence is never overwritten.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_out(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def blend(matrix, names, support24, support36, model):
    import numpy as np
    if names != model['names'] or matrix.shape[1] != len(names):
        raise ValueError('Frozen blend member order differs')
    for weights in [model['global_weights'], *[g['weights'] for g in model['group_weights'].values()]]:
        if len(weights) != len(names) or not np.isfinite(weights).all() or min(weights) < -1e-10 or not np.isclose(sum(weights), 1., atol=1e-8):
            raise ValueError('Invalid frozen convex weights')
    values = matrix @ np.asarray(model['global_weights'])
    groups = np.array([str(bool(a)) + '|' + str(bool(b)) for a, b in zip(support24, support36)])
    for group, record in model['group_weights'].items():
        mask = groups == group
        values[mask] = matrix[mask] @ np.asarray(record['weights'])
    return values


def score(actual, predicted):
    import numpy as np
    error = predicted - actual
    return {'n': len(error), 'rmse_sgd': float(np.sqrt(np.mean(error ** 2))),
            'mae_sgd': float(np.mean(abs(error))), 'bias_sgd': float(np.mean(error))}


def configure_runtime():
    # Match the independently verified full-data F1 runtime, not root aliases.
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE / 'runtime'))


def make_bundle(window, destination):
    import numpy as np
    from analysis.experiments import frontier_search, frontier_compact
    # New per-run feature caches; never borrow original development outputs.
    frontier_search.CACHE = frontier_compact.CACHE = destination / 'feature_cache'
    frontier_search.CACHE.mkdir(parents=True, exist_ok=True)
    from analysis.experiments.second_wave_anniversary import data
    from analysis.experiments.frontier_search import frame_data, month_number
    origin = {'Y': '2024-04', 'J': '2024-06'}[window]
    bundle = data(origin, '2025-04')
    frame, hashes = frame_data()
    fitting = frame.loc[frame.RENT_APPROVAL_DATE < origin]
    actual = bundle['y_val'].copy()
    bundle.update(reference={method: fitting.groupby(['RENT_APPROVAL_DATE', 'FLAT_TYPE_CLEAN']).MONTHLY_RENT.transform(method).to_numpy(float)
                             for method in ('mean', 'median')}, raw_fit=fitting,
                  start=origin, end='2025-04', fit_month=month_number(fitting.RENT_APPROVAL_DATE),
                  origin=int(month_number(__import__('pandas').Series([origin]))[0]))
    bundle['y_val'] = np.zeros(len(actual), dtype=float)
    if fitting.RENT_APPROVAL_DATE.max() >= origin:
        raise ValueError('Fitting window reaches query origin')
    return bundle, actual, hashes


def run(window, output):
    configure_runtime()
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from analysis.experiments.sixth_wave_deep import dependencies, seed
    from analysis.experiments.best_ensemble_submission import predict_saved
    if not torch.cuda.is_available():
        raise RuntimeError('Fixed F1 validation recipe requires CUDA; frozen analysis remains CPU-only.')
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT / 'output') or output == ROOT / 'output':
        raise ValueError('New validation run must remain under repository output/.')
    if output.exists():
        raise FileExistsError('Choose a new output directory; no checkpoint/cache reuse across runs.')
    output.mkdir(parents=True)
    dependencies()
    torch.set_num_threads(4)
    recipe = json.loads((HERE / 'training_recipes.json').read_text())
    validation = json.loads((HERE / 'validation_recipe.json').read_text())
    if digest(HERE / 'training_recipes.json') != validation['base_recipe_sha256']:
        raise ValueError('Base recipe differs from the frozen validation specification')
    model = validation['windows'][window]['blend']
    origin = {'Y': '2024-04', 'J': '2024-06'}[window]
    if model['fit_last_month'] >= origin:
        raise ValueError('Frozen meta-model cutoff reaches validation origin')
    start = time.perf_counter()
    paths = [Path(__file__), HERE / 'validation_recipe.json', HERE / 'training_recipes.json', *sorted((HERE / 'runtime').rglob('*.py'))]
    code = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() if (ROOT / '.git').exists() else None
    json_out(output / 'run_identity.json', {'window': window, 'generating_commit': commit,
             'source_sha256': code, 'weights_last_fit_month': model['fit_last_month'],
             'validation_labels_used_for_fitting_or_stopping': False,
             'recipe_selection': 'Fixed historical retrospective recipe; no new independent validation claim',
             'original_generator_commit_for_old_results_recovered': False})
    bundle, actual, hashes = make_bundle(window, output)
    n_expected = 35158 if window == 'Y' else 28995
    assert len(actual) == n_expected and not np.any(bundle['y_val'])
    raw_ids = bundle['row_val']
    names, predictions, records, schemas = [], [], [], {}

    def save(name, payload, pred, count, kind, columns):
        directory = output / 'models' / name
        directory.mkdir(parents=True)
        joblib.dump(payload, directory / 'model.joblib', compress=1)
        np.save(directory / 'predictions.npy', pred)
        restored = joblib.load(directory / 'model.joblib')
        if kind == 'plr':
            from analysis.experiments.tenth_wave_structures import predict_saved as predict_plr
            replay = predict_plr(restored, bundle)
        else:
            replay = predict_saved(restored, bundle, kind)
        difference = float(np.max(abs(replay - pred)))
        np.testing.assert_allclose(replay, pred, rtol=0, atol=1e-5)
        records.append({'name': name, 'count': int(count), 'fit_rows': len(bundle['y_fit']),
                        'reload_max_difference': difference, 'schema_columns': columns,
                        'checkpoint_sha256': digest(directory / 'model.joblib')})
        schemas[name] = columns
        del restored
        gc.collect()
        torch.cuda.empty_cache()
        print('VALIDATION TRAINED', window, name, count, flush=True)

    for item in recipe['history_members']:
        kind, spec = item['kind'], item['spec']
        if kind == 'normalized_market':
            from analysis.experiments.second_wave_normalized import fit, select_inputs
            x, _q, *_ = select_inputs(bundle, spec)
            fitted, pred, count = fit(spec, bundle, item['iterations'])
            payload = {'model': fitted, 'spec': spec}
        elif kind == 'anniversary':
            from analysis.experiments.second_wave_cohort_extensions import fit, features
            x, _q = features(bundle, spec)
            fitted, pred, count, _ = fit(spec, bundle, item['iterations'])
            payload = {'model': fitted, 'spec': spec}
        else:
            from analysis.experiments.third_wave_models import fit
            payload, pred, count, _, _ = fit(spec, bundle, item['iterations'])
            x = None
        columns = list(x) if x is not None else payload['columns']
        names.append(item['name']); predictions.append(np.asarray(pred, float))
        save(item['name'], payload, pred, count, kind, columns)
        del payload
        if 'fitted' in locals(): del fitted
        gc.collect(); torch.cuda.empty_cache()
    from analysis.experiments.third_wave_models import fit as fit_ft
    seeds = []
    for value in recipe['no36']['seeds']:
        seed(value)
        spec = recipe['no36']['spec'] | {'seed': value}
        payload, pred, count, _, _ = fit_ft(spec, bundle, recipe['no36']['epochs'])
        save(f'ft_no36_seed{value}', payload, pred, count, 'models', payload['columns'])
        seeds.append(pred); del payload; gc.collect(); torch.cuda.empty_cache()
    names.append('ft_no36_seedmean'); predictions.append(np.mean(seeds, axis=0))
    from analysis.experiments.tenth_wave_structures import fit as fit_plr
    payload, pred, count = fit_plr(recipe['plr']['spec'], bundle, recipe['plr']['epochs'])
    save('tenth/ft_plr_128_3', payload, pred, count, 'plr', payload['columns'])
    names.append('tenth/ft_plr_128_3'); predictions.append(pred)
    matrix = np.column_stack(predictions)
    support24 = bundle['ann_val'].ANN_BLOCK_24_R1_N.to_numpy() > 0
    support36 = bundle['ann_val'].ANN_BLOCK_36_R1_N.to_numpy() > 0
    prediction = blend(matrix, names, support24, support36, model)
    pd.DataFrame({'source_row': raw_ids, 'month': bundle['date_val'], 'actual': actual,
                  'prediction': prediction}).to_csv(output / 'predictions.csv', index=False, lineterminator='\n')
    np.savez_compressed(output / 'components.npz', predictions=matrix, names=np.asarray(names),
                        row_ids=raw_ids, actual=actual, support24=support24, support36=support36)
    original = np.load(ROOT / 'experiment/main/M00/inputs/validation.npz', allow_pickle=False)
    np.testing.assert_array_equal(raw_ids, original[f'{window}__row'])
    np.testing.assert_array_equal(actual, original[f'{window}__actual'])
    historical = original[f'{window}__F1']
    current_score, old_score = score(actual, prediction), score(actual, historical)
    report = {'window': window, 'origin': origin, 'fit_rows': len(bundle['y_fit']), 'query_rows': n_expected,
              'models_trained': len(records), 'model_records': records, 'component_schemas': schemas,
              'metrics': current_score, 'historical_metrics': old_score,
              'difference_from_historical_predictions_max': float(np.max(abs(prediction - historical))),
              'difference_from_historical_rmse': current_score['rmse_sgd'] - old_score['rmse_sgd'],
              'all_checkpoint_reloads_passed': True, 'validation_targets_hidden_from_all_fit_calls': True,
              'new_weight_fitting_performed': False, 'new_hyperparameter_search_performed': False,
              'pre_origin_meta_weights': model, 'data_sha256': hashes, 'source_sha256': code,
              'generating_commit': commit, 'prediction_sha256': digest(output / 'predictions.csv'),
              'seconds': time.perf_counter() - start, 'gpu': torch.cuda.get_device_name(),
              'python': sys.version.split()[0], 'torch': torch.__version__,
              'scope': 'New fixed-recipe validation refit, distinct from recovery of the old exact generating commit. All windows are reused development evidence.'}
    for path, h in code.items():
        if digest(ROOT / path) != h:
            raise RuntimeError('Training source changed during the run')
    json_out(output / 'validation_run.json', report)
    print('VALIDATION COMPLETE', window, current_score, 'seconds', report['seconds'], flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', required=True, choices=('Y', 'J'))
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.window, args.output_dir)


if __name__ == '__main__': main()
