"""Scientific failure cases for frozen-prediction alignment and diagnostics."""
import unittest
import numpy as np
import pandas as pd
from analysis.validation.main import metrics, grouped, validate_prediction
from analysis.validation.prepare_inputs import same_input_codes


class MainValidationTests(unittest.TestCase):
    def test_pooled_metrics_are_row_weighted_and_bias_is_prediction_minus_actual(self):
        score = metrics([0, 0, 0, 0], [1, 1, 1, 3])
        self.assertAlmostEqual(score['rmse_sgd'], np.sqrt(3))
        self.assertEqual(score['bias_sgd'], 1.5)
        rows = grouped(np.zeros(4), np.array([1, 1, 1, 3]), ['a', 'a', 'a', 'b'], 'Y', 'm', 'month')
        self.assertEqual(sum(r['n'] for r in rows), 4)
        self.assertAlmostEqual(sum(r['sse_share'] for r in rows), 1.)
        self.assertNotAlmostEqual(np.mean([r['rmse_sgd'] for r in rows]), score['rmse_sgd'])

    def test_bad_ids_labels_and_training_cutoffs_fail(self):
        raw = pd.DataFrame({'RENT_APPROVAL_DATE': ['2024-03', '2024-04', '2024-05'],
                            'MONTHLY_RENT': [1., 2., 3.]})
        for ids, actual, cutoff in [([1, 1], [2, 3], '2024-03'), ([2, 1], [3, 2], '2024-03'),
                                    ([1, 2], [2, 4], '2024-03'), ([1, 2], [2, 3], '2024-04')]:
            with self.subTest(ids=ids, actual=actual, cutoff=cutoff), self.assertRaises(ValueError):
                validate_prediction(raw, ids, actual, [2, 3], '2024-04', '2024-06', cutoff)

    def test_equal_inputs_treat_missing_values_identically_without_rounding(self):
        frame = pd.DataFrame({'num': [np.nan, np.nan, 1.00000000001, 1.00000000002],
                              'cat': pd.Categorical(['a', 'a', 'b', 'b'])})
        codes = same_input_codes(frame)
        self.assertEqual(codes[0], codes[1])
        self.assertNotEqual(codes[2], codes[3])

    def test_invalid_predictions_and_misaligned_groups_fail(self):
        for pred in ([1.], [1., np.nan], [1., np.inf]):
            with self.assertRaises(ValueError):metrics([1., 2.], pred)
        with self.assertRaises(ValueError):grouped(np.zeros(2), np.ones(2), ['one'], 'Y', 'm', 'month')

    def test_target_based_grouping_remains_explicitly_posthoc(self):
        rows = grouped(np.array([1., 2.]), np.array([2., 3.]), ['low', 'high'], 'Y', 'm', 'rent_band_fit_q10_q90')
        self.assertTrue(all(r['diagnostic_only_uses_query_labels'] for r in rows))


if __name__ == '__main__':unittest.main()
