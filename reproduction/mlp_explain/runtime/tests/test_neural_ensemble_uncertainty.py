import unittest
import numpy as np
import pandas as pd
from analysis.experiments.neural_ensemble_uncertainty import (
    RentIntervalCalibrator, conformal_quantile, ensemble_summary,
    export_predictions, interval_metrics, partition_masks,
)


class EnsembleUncertaintyTests(unittest.TestCase):
    def test_disagreement_is_not_standard_error(self):
        mean, std = ensemble_summary([[1000, 2000, 3000], [3000, 3000, 3000]])
        np.testing.assert_allclose(mean, [2000, 3000])
        np.testing.assert_allclose(std, [np.sqrt(2e6 / 3), 0])
        with self.assertRaises(ValueError):
            ensemble_summary([[1000]])

    def test_finite_sample_rank_is_not_interpolated_quantile(self):
        self.assertEqual(conformal_quantile(np.arange(1, 10), .2), 8)
        self.assertEqual(conformal_quantile(np.arange(1, 10), .1), 9)
        self.assertTrue(np.isinf(conformal_quantile([1, 2], .1)))

    def test_normalized_intervals_and_constant_width_reference(self):
        actual = np.arange(1, 1001)
        normalized = RentIntervalCalibrator(floor_sgd=100).fit(actual, np.zeros(1000), np.ones(1000))
        intervals = normalized.intervals([3000, 3000], [0, 200])
        np.testing.assert_allclose(intervals['90'][1] - 3000, [901, 1802])
        self.assertGreater(intervals['95'][1][0], intervals['90'][1][0])
        constant = RentIntervalCalibrator(normalized=False).fit(actual, np.zeros(1000), np.ones(1000))
        np.testing.assert_allclose(constant.intervals([3000, 3000], [0, 200])['90'][1], [3901, 3901])

    def test_inference_has_no_actual_target_and_does_not_mutate_calibration(self):
        cal = RentIntervalCalibrator().fit(np.arange(100), np.zeros(100), np.zeros(100))
        before = cal.metadata()
        metadata = pd.DataFrame({'source_row': [77], 'month': ['2025-01']})
        result = export_predictions(metadata, np.array([[3000, 3200, 3100]]), cal)
        self.assertNotIn('actual', result)
        self.assertNotIn('actual_sgd', result)
        self.assertEqual(cal.metadata(), before)
        self.assertEqual(result.source_row.iloc[0], 77)
        self.assertEqual(result.prediction_sgd.iloc[0], 3100)

    def test_chronological_tune_calibration_and_outer_are_disjoint(self):
        dates = pd.period_range('2021-01', '2025-03', freq='M').astype(str)
        masks = partition_masks(dates)
        for a, b in [('final_fit', 'calibration'), ('final_fit', 'outer'),
                     ('selection_tune', 'calibration'), ('calibration', 'outer')]:
            self.assertFalse(np.any(masks[a] & masks[b]))
        self.assertEqual(dates[masks['final_fit']][-1], '2023-12')
        self.assertEqual(dates[masks['calibration']][0], '2024-01')

    def test_interval_score_penalizes_missed_targets(self):
        cal = RentIntervalCalibrator(normalized=False).fit(np.ones(100), np.zeros(100), np.zeros(100))
        good = interval_metrics(np.array([0.]), np.array([0.]), np.array([0.]), cal)
        bad = interval_metrics(np.array([5.]), np.array([0.]), np.array([0.]), cal)
        self.assertEqual(good['coverage_90'], 1)
        self.assertEqual(bad['coverage_90'], 0)
        self.assertGreater(bad['interval_score_90_sgd'], good['interval_score_90_sgd'])

    def test_invalid_inputs_do_not_silently_generate_confidence(self):
        for scores in ([], [np.nan], [-1]):
            with self.assertRaises(ValueError):
                conformal_quantile(scores, .1)
        with self.assertRaises(ValueError):
            RentIntervalCalibrator(floor_sgd=0)
        with self.assertRaises(ValueError):
            RentIntervalCalibrator().intervals([1], [0])


if __name__ == '__main__':
    unittest.main()
