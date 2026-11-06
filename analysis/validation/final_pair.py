"""Audit the integrated F2 candidate and compare frozen recent F1/F2 predictions.

This is component reconstruction and retrospective evaluation, not fresh inference.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from analysis.pipeline.config import ROOT
from analysis.validation.prepare_inputs import sha

FOLDER = 'reports/zch_anniversary_tabpfn_blend'
PUBLICATION = 'reports/submissions/tabpfn_ensemble_20261106'
SOURCES = [
    'data/train.csv', 'data/example-submission.csv',
    'experiment/main/M00/inputs/validation.npz', 'experiment/main/M00/inputs/metadata.json',
    f'{FOLDER}/predictions_june.csv', f'{FOLDER}/predictions_zch_full.csv',
    f'{FOLDER}/predictions_tabpfn87_full.csv', f'{FOLDER}/submission_zch_tabpfn87_equal_150k.csv',
    f'{FOLDER}/submission_provenance.json', f'{FOLDER}/zch_full_provenance.json',
    f'{FOLDER}/tabpfn_full_provenance.json',
    'reports/tabpfn_anniversary_history_80_20/features_anniversary87.json',
    'review/checks/tabpfn-replication/summary.json',
    'workflows/tabpfn/analysis/compare_zch_anniversary_tabpfn_ensemble.py',
    'analysis/validation/main.py',
    f'{PUBLICATION}/publication.json', f'{PUBLICATION}/submission_ensemble_original_150k.csv',
    f'{PUBLICATION}/submission_ensemble_grouped_150k.csv',
    'review/checks/tabpfn-ensemble/audit.json', 'review/checks/tabpfn-ensemble/training_runner.py',
    'run_tabpfn_ensemble.py',
]


def provenance_hash(path, expected):
    """Allow only proven CSV line-ending conversion, never numerical tolerance."""
    data = Path(path).read_bytes()
    variants = {'original_bytes': data, 'normalized_LF': data.replace(b'\r\n', b'\n'),
                'restored_CRLF': data.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')}
    for mode, content in variants.items():
        if hashlib.sha256(content).hexdigest() == expected:
            return {'current_sha256': hashlib.sha256(data).hexdigest(),
                    'recorded_sha256': expected, 'matching_representation': mode}
    raise ValueError(f'Provenance hash mismatch: {path}')


def rebuild_components(trees, tabpfn, sample):
    if list(trees) != ['row_id', 'predicted'] or list(tabpfn) != ['row_id', 'predicted']:
        raise ValueError('Expected row_id,predicted component schemas')
    if list(sample) != ['Id', 'Predicted'] or sample.Id.duplicated().any():
        raise ValueError('Invalid submission IDs/schema')
    for frame in [trees, tabpfn]:
        if frame.row_id.duplicated().any() or not np.array_equal(frame.row_id, sample.Id):
            raise ValueError('Missing, duplicate or reordered component IDs')
        values = frame.predicted.to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Invalid component rents')
    return (trees.predicted.to_numpy(float) + tabpfn.predicted.to_numpy(float)) / 2


def published_candidates(sample, archived_original):
    folder = ROOT / PUBLICATION
    publication = json.loads((folder / 'publication.json').read_text())
    record = json.loads((ROOT / publication['evidence']).read_text())
    if sha(ROOT / publication['evidence']) != publication['evidence_sha256']:
        raise ValueError('Published ensemble evidence changed')
    if sha(ROOT / 'run_tabpfn_ensemble.py') != record['current_runner_sha256']:
        raise ValueError('Published current runner changed')
    if sha(ROOT / 'review/checks/tabpfn-ensemble/training_runner.py') != record['training_runner_sha256']:
        raise ValueError('Published training source changed')
    rows = []
    for item in publication['submissions']:
        path = (folder / item['file']).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != item['sha256']:
            raise ValueError('Published submission path/hash mismatch')
        frame = pd.read_csv(path, float_precision='round_trip')
        if list(frame) != ['Id', 'Predicted'] or len(frame) != 50000 or frame.Id.duplicated().any() or not np.array_equal(frame.Id, sample.Id):
            raise ValueError('Published submission IDs/schema mismatch')
        values = frame.Predicted.to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Invalid published submission rents')
        row = {'variant': item['variant'], 'path': path.relative_to(ROOT).as_posix(),
               'sha256': sha(path), 'rows': len(frame), 'actual_kaggle_score': None}
        if item['variant'] == 'original':
            difference = float(np.max(abs(values - archived_original)))
            if difference > 0.5e-6 + 1e-10:
                raise ValueError('Fresh original candidate differs from archived F2 beyond CSV rounding')
            row['max_difference_from_six_decimal_archived_F2_sgd'] = difference
        rows.append(row)
    if {r['variant'] for r in rows} != {'original', 'grouped'}:
        raise ValueError('Expected separate original/grouped published candidates')
    return {'submissions': rows, 'source_reported_recommendation': publication['recommended_submission'],
            'fresh_full_original_tabpfn_reported': record['original_full_tabpfn_freshly_fitted_this_audit'],
            'fresh_grouped_validation_and_full_reported': record['grouped_validation_and_full_tabpfn_freshly_fitted_this_audit'],
            'fresh_tree_fits_reported': record['fresh_tree_fits'],
            'validation_scores': publication['validation_scores'],
            'verification_scope': 'CSV/evidence/runner identities independently checked; validation scores and GPU training read from source audit, not rerun locally',
            'F2_in_paired_comparison': 'original categories; grouped retained as a separate sensitivity candidate',
            'kaggle_score_link_verified': False}


def audit():
    from analysis.validation.main import metrics, grouped, validate_prediction

    before = {p: sha(ROOT / p) for p in SOURCES}
    folder = ROOT / FOLDER
    load = lambda name: json.loads((folder / name).read_text(encoding='utf-8'))
    provenance = load('submission_provenance.json')
    tree_record, tab_record = load('zch_full_provenance.json'), load('tabpfn_full_provenance.json')
    if provenance['weights'] != {'zch_original': 0.5, 'tabpfn87': 0.5} or len(tree_record['members']) != 7:
        raise ValueError('F2 archived member composition/weights changed')
    identities = {}
    for filename, expected in [
        ('submission_zch_tabpfn87_equal_150k.csv', provenance['sha256']),
        ('predictions_zch_full.csv', provenance['zch_prediction_sha256']),
        ('predictions_tabpfn87_full.csv', provenance['tabpfn_prediction_sha256']),
    ]:
        identities[filename] = provenance_hash(folder / filename, expected)
    assert tree_record['prediction_sha256'] == provenance['zch_prediction_sha256']
    assert tab_record['prediction_sha256'] == provenance['tabpfn_prediction_sha256']
    for recorded_path, expected in tree_record['source_sha256'].items():
        if sha(ROOT / recorded_path.replace('\\', '/')) != expected:
            raise ValueError('F2 tree raw input changed: ' + recorded_path)
    if tab_record['source_train_sha256'] != before['data/train.csv']:
        raise ValueError('F2 TabPFN raw training input changed')
    if sha(ROOT / 'data/test.csv') != tab_record['source_test_sha256']:
        raise ValueError('F2 TabPFN raw test input changed')
    read = lambda name: pd.read_csv(folder / name, float_precision='round_trip')
    sample = pd.read_csv(ROOT / 'data/example-submission.csv')
    submission = read('submission_zch_tabpfn87_equal_150k.csv')
    rebuilt = rebuild_components(read('predictions_zch_full.csv'), read('predictions_tabpfn87_full.csv'), sample)
    if len(sample) != 50000 or list(submission) != list(sample) or not np.array_equal(submission.Id, sample.Id):
        raise ValueError('F2 submission differs from the 50,000-row template')
    if not np.isfinite(submission.Predicted).all() or (submission.Predicted <= 0).any():
        raise ValueError('Invalid F2 candidate predictions')
    difference = float(np.max(abs(rebuilt - submission.Predicted.to_numpy(float))))
    tolerance = 0.5e-6 + 1e-10  # Six-decimal archived CSV; no model tolerance.
    if difference > tolerance:
        raise ValueError('F2 does not reproduce its recorded equal-weight components')
    published = published_candidates(sample, submission.Predicted.to_numpy(float))

    raw = pd.read_csv(ROOT / 'data/train.csv')
    bank = np.load(ROOT / 'experiment/main/M00/inputs/validation.npz', allow_pickle=False)
    paired = read('predictions_june.csv')
    ids, actual = bank['J__row'], bank['J__actual']
    if not np.array_equal(paired.row_id, ids) or not np.array_equal(paired.actual, actual):
        raise ValueError('F2 recent IDs/labels do not match F1')
    if not np.array_equal(paired.month, raw.loc[ids, 'RENT_APPROVAL_DATE']):
        raise ValueError('F2 recent months do not match raw source rows')
    features = json.loads((ROOT / 'reports/tabpfn_anniversary_history_80_20/features_anniversary87.json').read_text())
    assert features['features'] == 87 and len(features['added_columns']) == 10
    assert features['fitting_rows'] == 121005 and features['query_rows'] == 28995
    assert features['feature_pool_last_month'] == '2024-05'
    predictions = {'F1': bank['J__F1'], 'F2': (paired.zch_original.to_numpy() + paired.tabpfn87.to_numpy()) / 2}
    scores, groups = [], []
    months = raw.loc[ids, 'RENT_APPROVAL_DATE'].to_numpy()
    support = np.array([f'24={bool(a)}|36={bool(b)}' for a, b in zip(bank['J__support24'], bank['J__support36'])])
    for name, pred in predictions.items():
        score = validate_prediction(raw, ids, actual, pred, '2024-06', '2025-04', '2024-05')
        scores.append({'window': 'J', 'model': name, **score})
        for dimension, labels in [('month', months), ('support24_36', support),
                                  ('flat_type', raw.loc[ids, 'FLAT_TYPE'].str.lower().str.strip().str.replace(r'\s*-\s*', '-', regex=True))]:
            groups.extend(grouped(actual, pred, labels, 'J', name, dimension))
    monthly = []
    for month in sorted(set(months)):
        mask = months == month
        a, b = [metrics(actual[mask], predictions[name][mask]) for name in ['F1', 'F2']]
        monthly.append({'month': month, 'n': int(mask.sum()), 'F1_rmse_sgd': a['rmse_sgd'],
                        'F2_rmse_sgd': b['rmse_sgd'], 'delta_F2_minus_F1_rmse': b['rmse_sgd'] - a['rmse_sgd']})
    replication = json.loads((ROOT / 'review/checks/tabpfn-replication/summary.json').read_text())
    manifest = {'status': 'partial', 'candidate_reconstruction_verified': True,
                'filename': 'submission_zch_tabpfn87_equal_150k.csv', 'csv_sha256': before[f'{FOLDER}/submission_zch_tabpfn87_equal_150k.csv'],
                'members': {'seven_tree_ensemble': list(tree_record['members']), 'frozen_tabpfn87': tab_record},
                'weights': provenance['weights'], 'F1_is_a_component': False,
                'generator': 'review_tabpfn.py run anniversary-blend -- --stage submission',
                'schema': {'count': 87, 'anniversary_added_columns': features['added_columns'], 'complete_ordered_dictionary_verified': False},
                'target': 'Components are SGD rents; equal blend adds no residual baseline again',
                'displayed_score_user_screenshot': 413.19635, 'screenshot_score_identity_verified': False,
                'standalone_validation_replication_record': 'review/checks/tabpfn-replication/summary.json',
                'standalone_replay_reported_by_source': replication['new_cli_validation']['fresh_fit_and_inference'],
                'subsequent_full_data_replication_record': 'review/checks/tabpfn-ensemble/audit.json',
                'published_candidates': published,
                'missing': ['Complete exported ordered 87-input dictionary', 'Candidate association with the historical screenshot score']}
    protocol = {'status': 'partial', 'paired_comparison_performed': True, 'eligible': True,
                'eligible_windows': ['J'], 'unavailable_windows': ['Y'], 'n': len(ids), 'fit_rows': 121005,
                'fit_end_exclusive': '2024-06', 'query_end_exclusive': '2025-04',
                'row_ids_labels_months_match': True, 'information_policies_identical_verified': False,
                'comparison_scope': 'Complete-pipeline retrospective scores on common records; no isolated architecture or identical information-policy claim',
                'development_period_reused': True, 'Y_J_independent': False,
                'delta_F2_minus_F1_rmse_sgd': scores[1]['rmse_sgd'] - scores[0]['rmse_sgd'],
                'delta_F2_minus_F1_mae_sgd': scores[1]['mae_sgd'] - scores[0]['mae_sgd'],
                'F2_better_months': sum(r['delta_F2_minus_F1_rmse'] < 0 for r in monthly),
                'missing': ['F2 annual-origin predictions', 'Complete input/information-policy equivalence audit']}
    reconstruction = {'status': 'passed_component_reconstruction', 'rows': len(sample),
                      'max_abs_difference_sgd': difference, 'tolerance_sgd': tolerance,
                      'provenance_identity_checks': identities, 'fresh_checkpoint_inference_this_run': False,
                      'full_retraining_this_run': False, 'screenshot_score_link_verified': False}
    after = {p: sha(ROOT / p) for p in SOURCES}
    if before != after:
        raise AssertionError('Source inputs changed during final-pair audit')
    return {'manifest': manifest, 'protocol': protocol, 'reconstruction': reconstruction,
            'metrics': scores, 'groups': groups, 'monthly': monthly,
            'published_candidates': published,
            'source_sha256': before, 'raw_inputs_unchanged': True, 'new_models_fitted': 0}


def write_evidence(out, result):
    from analysis.validation.main import dump, table
    dump(out / 'M07/f2_manifest.json', result['manifest'])
    dump(out / 'M07/source_manifest.json', result['source_sha256'])
    dump(out / 'M08/protocol_checks.json', result['protocol'])
    table(out / 'M08/final_pair_metrics.csv', result['metrics'])
    table(out / 'M08/monthly_pair.csv', result['monthly'])
    table(out / 'M08/grouped_pair.csv', result['groups'])
    dump(out / 'M09/f2_reconstruction.json', result['reconstruction'])


def run(output_dir=None):
    result = audit()
    signature = hashlib.sha256(json.dumps([result['source_sha256'], sha(Path(__file__))], sort_keys=True).encode()).hexdigest()[:16]
    out = Path(output_dir or ROOT / 'output/final_pair' / signature).resolve()
    if not out.is_relative_to(ROOT / 'output') or out == ROOT / 'output':
        raise ValueError('Final-pair outputs must stay under output/')
    out.mkdir(parents=True, exist_ok=False)
    write_evidence(out, result)
    print(json.dumps({'output': str(out), 'metrics': result['metrics'], 'protocol': result['protocol'],
                      'reconstruction': result['reconstruction'], 'new_models_fitted': 0}, indent=2))
    return out
