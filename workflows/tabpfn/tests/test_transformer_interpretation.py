"""Meaningful invariants for post-hoc model diagnostics."""
import unittest

import numpy as np
import pandas as pd

from analysis.interpret_spatial_temporal_transformer import (
    ale_1d, donor_permutation, feature_groups, moran_i, sample_months, spatial_weights,
)


class InterpretationTests(unittest.TestCase):
    def test_month_sample_weights_recover_population_and_ignore_labels(self):
        months = np.array(["2024-06"]*7+["2024-07"]*13)
        ids, weights = sample_months(months, per_month=4)
        self.assertEqual(len(ids), 8)
        self.assertAlmostEqual(weights.sum(), 20)
        self.assertEqual(len(set(ids)), len(ids))
        np.testing.assert_array_equal(ids, sample_months(months, per_month=4)[0])

    def test_donors_stay_inside_calendar_and_flat_type_cohort(self):
        months = np.array(["2024-06"]*10+["2024-07"]*10)
        kinds = np.array(["3-room", "4-room"]*10)
        ids = donor_permutation(months, kinds, 42)
        np.testing.assert_array_equal(months[ids], months)
        np.testing.assert_array_equal(kinds[ids], kinds)
        self.assertEqual(len(set(ids)), len(ids))

    def test_ale_recovers_known_linear_effect_despite_correlated_nuisance(self):
        x = np.linspace(1, 20, 600)
        frame = pd.DataFrame({"area": x, "nuisance": x*x})
        def model(f):
            return 4*f.area.to_numpy()+100*f.nuisance.to_numpy()
        result = ale_1d(frame, "area", model)
        expected = 4*(result.midpoint-np.average(result.midpoint, weights=result.n))
        np.testing.assert_allclose(result.ale_sgd, expected, atol=1e-10)
        self.assertAlmostEqual(np.average(result.ale_sgd, weights=result.n), 0.)

    def test_ale_constant_response_has_zero_effect(self):
        frame = pd.DataFrame({"area": np.arange(600.)})
        result = ale_1d(frame, "area", lambda f: np.full(len(f), 1000.))
        np.testing.assert_array_equal(result.ale_sgd, 0.)

    def test_sparse_ale_does_not_accumulate_across_unsupported_bins(self):
        result = ale_1d(pd.DataFrame({"area": np.arange(20.)}), "area", lambda f: f.area)
        self.assertTrue(result.empty)

    def test_discrete_area_sparse_bins_are_merged_without_losing_supported_curve(self):
        x = np.repeat([60., 65., 67., 70., 80.], [10, 100, 300, 100, 90])
        frame = pd.DataFrame({"area": x})
        result = ale_1d(frame, "area", lambda f: 3*f.area.to_numpy())
        self.assertFalse(result.empty)
        self.assertTrue((result.n >= 20).all())
        self.assertEqual(result.n.sum(), len(x))
        expected = 3*(result.midpoint-np.average(result.midpoint, weights=result.n))
        np.testing.assert_allclose(result.ale_sgd, expected)

    def test_spatial_graph_has_no_self_edges_and_normalizes_rows(self):
        xy = np.array([[0, 0], [.1, 0], [.2, 0], [10, 10.]])
        weights = spatial_weights(xy, neighbors=2, radius_km=1)
        np.testing.assert_array_equal(weights.diagonal(), 0)
        np.testing.assert_allclose(np.asarray(weights.sum(axis=1)).ravel(), [1, 1, 1, 0])

    def test_moran_is_invariant_to_offset_and_scale(self):
        xy = np.column_stack([np.arange(10.)/10, np.zeros(10)])
        weights = spatial_weights(xy, neighbors=2)
        values = np.arange(10.)**2
        self.assertAlmostEqual(moran_i(values, weights), moran_i(100+3*values, weights))
        self.assertGreater(moran_i(values, weights), 0)

    def test_geographic_group_covers_location_duplicates_and_calendar_is_excluded(self):
        columns = ["LATITUDE", "LONGITUDE", "BLOCK_ID", "TOWN_CLEAN", "MRT_COUNT_1.0KM",
                   "FLOOR_AREA_SQM", "TEMP_LOCAL_FORECAST", "TEMP_FORECAST_HORIZON", "HIST_AREA_FLAT_MEAN"]
        groups = feature_groups(columns)
        self.assertTrue({"LATITUDE", "LONGITUDE", "BLOCK_ID", "MRT_COUNT_1.0KM"}.issubset(groups["Geography + amenities"]))
        self.assertIn("TEMP_LOCAL_FORECAST", groups["Rental-history bundle"])
        self.assertNotIn("TEMP_FORECAST_HORIZON", groups["Rental-history bundle"])


if __name__ == "__main__":
    unittest.main()
