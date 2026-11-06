"""Replay fresh F1 validation blends and verify their source/data identity."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess

import numpy as np
import pandas as pd

from analysis.pipeline.config import ROOT
from analysis.validation.main import dump, metrics, table
from analysis.validation.prepare_inputs import sha

BASE = ROOT / 'reproduction/f1/validation_evidence'


def audit():
    spec = importlib.util.spec_from_file_location('frozen_validation_recipe', ROOT / 'reproduction/f1/validate.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    raw = pd.read_csv(ROOT / 'data/train.csv')
    historical = np.load(ROOT / 'experiment/main/M00/inputs/validation.npz', allow_pickle=False)
    rows = []
    recipe = json.loads((ROOT / 'reproduction/f1/validation_recipe.json').read_text('utf-8'))
    for window in ('Y', 'J'):
        directory = BASE / window
        report = json.loads((directory / 'validation_run.json').read_text('utf-8'))
        manifest = json.loads((directory / 'artifact-manifest.json').read_text('utf-8'))
        for name, expected in manifest.items():
            if sha(directory / name) != expected:
                raise ValueError('Refit artifact changed: ' + name)
        git_checked = 0
        for name, expected in report['source_sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Generating training source changed: ' + name)
            if (ROOT / '.git').exists():
                blob = subprocess.check_output(['git', 'show', report['generating_commit'] + ':' + name], cwd=ROOT)
                if hashlib.sha256(blob).hexdigest() != expected:
                    raise ValueError('Generating commit does not contain the recorded source: ' + name)
                git_checked += 1
        for name, expected in report['data_sha256'].items():
            if sha(ROOT / name.replace('\\', '/')) != expected:
                raise ValueError('Refit input bytes changed: ' + name)
        bank = np.load(directory / 'components.npz', allow_pickle=False)
        ids, actual = bank['row_ids'], bank['actual']
        np.testing.assert_array_equal(ids, historical[window + '__row'])
        np.testing.assert_array_equal(actual, raw.loc[ids, 'MONTHLY_RENT'])
        np.testing.assert_array_equal(actual, historical[window + '__actual'])
        assert int((raw.RENT_APPROVAL_DATE < report['origin']).sum()) == report['fit_rows']
        assert len(report['model_records']) == report['models_trained'] == 15
        assert report['all_checkpoint_reloads_passed'] and report['validation_targets_hidden_from_all_fit_calls']
        assert not report['new_weight_fitting_performed'] and not report['new_hyperparameter_search_performed']
        assert report['pre_origin_meta_weights'] == recipe['windows'][window]['blend']
        assert report['pre_origin_meta_weights']['fit_last_month'] < report['origin']
        assert all(r['reload_max_difference'] <= 1e-5 for r in report['model_records'])
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            prediction = entry.blend(bank['predictions'], bank['names'].tolist(), bank['support24'],
                                     bank['support36'], report['pre_origin_meta_weights'])
        frame = pd.DataFrame({'source_row': ids, 'month': raw.loc[ids, 'RENT_APPROVAL_DATE'].to_numpy(),
                              'actual': actual, 'prediction': prediction})
        csv_sha = hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest()
        score = metrics(actual, prediction)
        for metric in ('rmse_sgd', 'mae_sgd', 'bias_sgd'):
            np.testing.assert_allclose(score[metric], report['metrics'][metric], rtol=0, atol=1e-10)
        difference = float(np.max(abs(prediction - historical[window + '__F1'])))
        np.testing.assert_allclose(difference, report['difference_from_historical_predictions_max'], rtol=0, atol=1e-10)
        rows.append({'window': window, **score, 'fit_rows': report['fit_rows'],
                     'models_trained': 15, 'logical_components': bank['predictions'].shape[1],
                     'generating_commit': report['generating_commit'], 'source_files_verified': len(report['source_sha256']),
                     'source_git_blobs_verified': git_checked, 'generated_csv_bytes_reproduced': csv_sha == report['prediction_sha256'],
                     'historical_prediction_max_abs_difference': difference,
                     'delta_rmse_vs_historical': score['rmse_sgd'] - report['historical_metrics']['rmse_sgd'],
                     'all_checkpoint_reloads_recorded_passed': True, 'seconds': report['seconds']})
    return {'all_passed': True, 'windows': rows, 'new_models_trained_in_fresh_runs': 30,
            'model_training_performed_by_this_audit': False,
            'checkpoint_scope': 'Fresh run saved/reloaded all 30 predictors. Large checkpoints remain local output; shipped component arrays permit CPU blend replay. CUDA entry point permits refitting from raw data.',
            'scope': 'Fresh fixed-recipe retrospective validation. It does not recover the original unknown generator commit or create a new held-out test.',
            'policy': 'docs/methods/prediction-scenario.json'}


def run(output_dir=None):
    result = audit()
    signature = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]
    destination = Path(output_dir or ROOT / 'output/refit_audit' / signature).resolve()
    if not destination.is_relative_to(ROOT / 'output'):
        raise ValueError('Audit output must stay below output/')
    destination.mkdir(parents=True, exist_ok=False)
    dump(destination / 'audit.json', result)
    table(destination / 'validation_results.csv', result['windows'])
    print('Fresh F1 Y/J source and blend replay verified:', destination)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    return run(args.output_dir)


if __name__ == '__main__': main()
