"""Scientific controls for schema corrections and imported evidence."""
import json
import unittest

import numpy as np
import pandas as pd

from analysis.pipeline.config import ROOT
from analysis.validation.feature_catalog import adjusted_schema, bridge, resolved_record_schema
from analysis.validation.evidence_audit import checked_scores, matched_deltas, normalized_sha


class MainEvidenceCompletionTests(unittest.TestCase):
    def test_equal_width_does_not_mean_same_features(self):
        meta = json.loads((ROOT / 'experiment/main/M00/inputs/metadata.json').read_text('utf-8'))
        old = meta['feature_schemas']['Y_ladder_62']
        new = adjusted_schema(old)
        diff = bridge([c['name'] for c in old], [c['name'] for c in new])
        self.assertEqual(len(new), len(old))
        self.assertEqual(len(diff['added']), 5)
        self.assertEqual(len(diff['removed']), 5)
        self.assertFalse(diff['same_ordered_schema'])
        self.assertFalse(diff['strictly_nested'])

    def test_corrected_ple_plr_inputs_share_adjusted_contract(self):
        meta = json.loads((ROOT / 'experiment/main/M00/inputs/metadata.json').read_text('utf-8'))
        for window in ('Y', 'J'):
            records = [r for r in meta['predictions'] if r['window'] == window and r['model'] in ('single_PLE', 'single_PLR')]
            self.assertEqual(resolved_record_schema(records[0], meta), resolved_record_schema(records[1], meta))
            self.assertEqual(len(resolved_record_schema(records[0], meta)), 62)
            self.assertEqual(len(records[0]['schema']), 57)  # Preserve the historical archive and document its correction.

    def test_cleaning_uses_each_models_own_uncleaned_control(self):
        frame = pd.DataFrame([{'policy': 'uncleaned_baseline', 'model': 'mlp', 'n': 2, 'rmse_sgd': 10., 'mae_sgd': 7.},
                              {'policy': 'uncleaned_baseline', 'model': 'hybrid', 'n': 2, 'rmse_sgd': 8., 'mae_sgd': 6.},
                              {'policy': 'obvious_low', 'model': 'mlp', 'n': 2, 'rmse_sgd': 9., 'mae_sgd': 6.}])
        self.assertEqual(matched_deltas(frame).iloc[2].delta_rmse_vs_matched_uncleaned, -1.)
        with self.assertRaises(ValueError): matched_deltas(frame.drop(index=0))
        frame.loc[2, 'n'] = 1
        with self.assertRaises(ValueError): matched_deltas(frame)

    def test_cleaning_ids_and_labels_must_match_raw_validation_population(self):
        raw = pd.DataFrame({'RENT_APPROVAL_DATE': ['2024-03', '2024-04', '2024-05'], 'MONTHLY_RENT': [1., 2., 3.]})
        self.assertEqual(checked_scores(raw, np.array([1, 2]), np.array([2., 3.]), [2., 3.])['rmse_sgd'], 0)
        for ids, actual in [([1, 1], [2., 2.]), ([2, 1], [3., 2.]), ([1, 2], [2., 4.])]:
            with self.assertRaises(AssertionError): checked_scores(raw, np.array(ids), np.array(actual), [2., 3.])

    def test_source_content_identity_preserves_current_recipe_bytes(self):
        build = json.loads((ROOT / 'experiment/main/M01/spatial_evidence/source/build_manifest.json').read_text('utf-8'))
        for name, expected in build['raw_sha256'].items():
            self.assertEqual(normalized_sha(ROOT / name), expected)


if __name__ == '__main__': unittest.main()
