"""Verify imported M01 evidence; recompute matched cleaning scores, not models."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from analysis.pipeline.config import ROOT
from analysis.validation.main import dump, metrics, table
from analysis.validation.prepare_inputs import sha

M01 = ROOT / 'experiment/main/M01'


def normalized_sha(path):
    """Git source blobs use LF; original model inputs retain their CRLF bytes."""
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def checked_scores(raw, rows, actual, prediction):
    expected = raw.index[(raw.RENT_APPROVAL_DATE >= '2024-04') &
                         (raw.RENT_APPROVAL_DATE < '2025-04')].to_numpy()
    np.testing.assert_array_equal(rows, expected)
    np.testing.assert_array_equal(actual, raw.loc[rows, 'MONTHLY_RENT'].to_numpy())
    return metrics(actual, prediction)


def matched_deltas(frame):
    frame = frame.copy()
    if frame.duplicated(['policy', 'model']).any():
        raise ValueError('Duplicate policy/model control')
    for i, row in frame.iterrows():
        control = frame[(frame.model == row.model) & (frame.policy == 'uncleaned_baseline')]
        if len(control) != 1 or row['n'] != control.iloc[0]['n']:
            raise ValueError('A matched model and identical query population are required')
        frame.loc[i, 'delta_rmse_vs_matched_uncleaned'] = row.rmse_sgd - control.iloc[0].rmse_sgd
        frame.loc[i, 'delta_mae_vs_matched_uncleaned'] = row.mae_sgd - control.iloc[0].mae_sgd
    return frame


def audit():
    imports = json.loads((M01 / 'imported-evidence.json').read_text('utf-8'))
    for item in imports:
        if sha(ROOT / item['path']) != item['sha256'] or item['sha256'] != item['source_blob_sha256']:
            raise ValueError('Imported source bytes changed: ' + item['path'])
    spatial = M01 / 'spatial_evidence/source'
    cleaning = M01 / 'cleaning_evidence'
    versions = []
    for source, key in [(spatial / 'build_manifest.json', 'raw_sha256'),
                        (cleaning / 'source/protocol.json', 'source_sha256')]:
        for path, expected in json.loads(source.read_text('utf-8'))[key].items():
            actual = normalized_sha(ROOT / path)
            if actual != expected:
                raise ValueError('Imported EDA/cleaning uses a different input: ' + path)
            versions.append({'source_manifest': str(source.relative_to(ROOT)), 'path': path,
                             'current_raw_sha256': sha(ROOT / path), 'lf_normalized_sha256': actual,
                             'source_blob_sha256': expected, 'same_content_after_LF_normalization': True})
    raw = pd.read_csv(ROOT / 'data/train.csv')
    fit_n = int((raw.RENT_APPROVAL_DATE < '2024-01').sum())
    assert fit_n == 105837
    policies = json.loads((cleaning / 'source/cleaning_summary.json').read_text('utf-8'))
    for policy in policies:
        assert policy['n_before'] == fit_n
        assert policy['n_removed'] + policy['n_retained'] == fit_n
        assert policy['hypothesis_not_verified_rental_scope']
    source_scores = pd.read_csv(cleaning / 'source/comparison_metrics.csv', float_precision='round_trip')
    comparisons = matched_deltas(source_scores)
    provenance = json.loads((cleaning / 'prediction-provenance.json').read_text('utf-8'))
    if sha(cleaning / 'matched_predictions.npz') != provenance['npz_sha256']:
        raise ValueError('Cleaning prediction archive changed')
    bank = np.load(cleaning / 'matched_predictions.npz', allow_pickle=False)
    recomputed = []
    comparisons['verification_scope'] = 'source aggregate and provenance only'
    for source in provenance['sources']:
        key = source['key']
        policy, model = key.split('__')
        score = checked_scores(raw, bank[key + '__row'], bank[key + '__actual'], bank[key + '__prediction'])
        mask = (comparisons.policy == policy) & (comparisons.model == model)
        assert mask.sum() == 1
        for field in ['rmse_sgd', 'mae_sgd', 'bias_sgd']:
            np.testing.assert_allclose(score[field], comparisons.loc[mask, field].iloc[0], rtol=0, atol=1e-7)
        comparisons.loc[mask, 'verification_scope'] = 'row IDs, labels and metrics independently recomputed'
        recomputed.append({'policy': policy, 'model': model, **score})
    fields = ['DISTANCE_TO_CBD_KM', 'NEAREST_OPEN_MRT_KM', 'NEAREST_MALL_KM',
              'NEAREST_PRIMARY_SCHOOL_KM', 'MAX_FLOOR', 'BUILDING_AGE_YEARS']
    correlations = pd.read_csv(spatial / 'all_numeric_correlations.csv')
    selected = correlations.set_index('feature').loc[fields].reset_index()[
        ['feature', 'n_valid', 'n_adjusted', 'pearson', 'adjusted_pearson']]
    assert len(selected) == 6 and selected.n_valid.between(149841, 150000).all()
    manifest = json.loads((spatial / 'analysis_manifest.json').read_text('utf-8'))
    assert manifest['common_complete_rows'] + manifest['excluded_rows'] == len(raw)
    bins = pd.read_csv(spatial / 'amenity_bin_effects.csv')
    bins = bins[bins.specification == 'joint_cbd'].copy()
    assert len(bins) == 18
    for _amenity, group in bins.groupby('amenity'):
        assert group.n_rows.sum() == manifest['common_complete_rows']
    assert (bins.ci95_low_sgd <= bins.adjusted_difference_sgd).all()
    assert (bins.ci95_high_sgd >= bins.adjusted_difference_sgd).all()
    report = {'all_passed': True, 'source_files_byte_verified': len(imports),
              'input_versions': versions, 'cleaning_fit_rows': fit_n,
              'cleaning_row_level_metrics_recomputed': recomputed,
              'cleaning_aggregate_rows': len(comparisons),
              'cleaning_other_rows': 'Source aggregates only; not independently replayed from all row predictions',
              'spatial_scope': 'Frozen source statistics and input identity verified; regressions not independently refitted',
              'spatial_complete_rows': manifest['common_complete_rows'],
              'spatial_uncertainty': manifest['uncertainty'],
              'spatial_controls': manifest['control_columns'],
              'new_training_performed': False,
              'policy': 'docs/methods/prediction-scenario.json',
              'interpretation': 'Associations are not causal effects; cleaning evidence is not an F1 ablation'}
    return report, selected, bins, comparisons


def run(output_dir=None):
    report, spatial, bins, cleaning = audit()
    signature = sha(M01 / 'imported-evidence.json')[:12] + '-' + sha(Path(__file__))[:8]
    destination = Path(output_dir or ROOT / 'output/evidence_audit' / signature).resolve()
    if not destination.is_relative_to(ROOT / 'output'):
        raise ValueError('New evidence audit output must be below output/')
    destination.mkdir(parents=True, exist_ok=False)
    dump(destination / 'audit.json', report)
    table(destination / 'raw_adjusted_comparison.csv', spatial.to_dict('records'))
    table(destination / 'facility_bin_controls.csv', bins.to_dict('records'))
    table(destination / 'matched_cleaning_comparison.csv', cleaning.to_dict('records'))
    print('M01 evidence verified:', destination)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    return run(args.output_dir)


if __name__ == '__main__': main()
