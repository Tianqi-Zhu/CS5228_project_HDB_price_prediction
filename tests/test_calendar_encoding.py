"""Calendar availability, wraparound, and controlled-ablation guarantees."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.calendar_encoding import encoded_features, stratified_queries


class CalendarEncodingTests(unittest.TestCase):
    def frame(self):
        months = pd.period_range("2021-01", periods=36, freq="M")
        return pd.DataFrame({"APPROVAL_MONTH_INDEX": np.arange(36), "APPROVAL_MONTH_OF_YEAR": months.month,
                             "NEAREST_SCHOOL_KM": 0.3, "SCHOOL_COUNT_1.0KM": 4, "HISTORY": np.nan},
                            index=np.arange(100, 136)), months

    def test_all_variants_complete_from_first_month_and_keep_histories_and_time(self):
        x, months = self.frame()
        for encoding in ("integer", "categorical", "cyclical"):
            result = encoded_features(x, months, encoding, calendar=True, interactions=True)
            new = [c for c in result if c.startswith("CAL_")]
            self.assertTrue(np.isfinite(result[new].to_numpy(float)).all())
            pd.testing.assert_series_equal(result.HISTORY, x.HISTORY)
            pd.testing.assert_series_equal(result.APPROVAL_MONTH_INDEX, x.APPROVAL_MONTH_INDEX)

    def test_category_mapping_is_identical_in_context_and_query(self):
        x, months = self.frame()
        a = encoded_features(x.iloc[:12], months[:12], "categorical")
        b = encoded_features(x.iloc[12:], months[12:], "categorical")
        self.assertEqual(a.APPROVAL_MONTH_OF_YEAR.dtype, b.APPROVAL_MONTH_OF_YEAR.dtype)
        self.assertEqual(list(a.APPROVAL_MONTH_OF_YEAR.cat.categories), list(range(1, 13)))

    def test_cyclical_distance_wraps_december_to_january(self):
        x, months = self.frame()
        a = encoded_features(x, months, "cyclical")[["CAL_MONTH_SIN", "CAL_MONTH_COS"]].to_numpy()
        self.assertAlmostEqual(np.linalg.norm(a[11] - a[12]), np.linalg.norm(a[0] - a[1]))
        np.testing.assert_allclose(a[:12], a[12:24], atol=1e-15)

    def test_future_rows_and_labels_cannot_change_calendar_features(self):
        x, months = self.frame()
        full = encoded_features(x, months, "cyclical", calendar=True, interactions=True)
        prefix = encoded_features(x.iloc[:13], months[:13], "cyclical", calendar=True, interactions=True)
        pd.testing.assert_frame_equal(full.iloc[:13], prefix)
        changed = x.copy(); changed["MONTHLY_RENT"] = 1e9
        with_label = encoded_features(changed, months, "cyclical", calendar=True, interactions=True)
        pd.testing.assert_frame_equal(full, with_label.drop(columns="MONTHLY_RENT"))

    def test_unknown_school_location_has_finite_interactions_and_explicit_flag(self):
        x, months = self.frame(); x.loc[100, "NEAREST_SCHOOL_KM"] = np.nan
        result = encoded_features(x, months, "integer", calendar=True, interactions=True)
        self.assertEqual(result.loc[100, "CAL_SCHOOL_LOCATION_MISSING"], 1)
        self.assertEqual(result.loc[100, "CAL_PRIMARY_START_X_SCHOOL_ACCESS"], 0)

    def test_sample_is_reproducible_and_weights_reconstruct_month_counts(self):
        months = np.repeat(pd.period_range("2021-01", periods=2, freq="M"), [7, 13])
        ids = np.arange(20)
        positions, weights = stratified_queries(months, ids, per_month=3)
        again, _ = stratified_queries(months, ids, per_month=3)
        np.testing.assert_array_equal(positions, again)
        self.assertEqual(len(positions), 6)
        self.assertAlmostEqual(weights.sum(), 20)
        self.assertAlmostEqual(weights[np.asarray(months)[positions] == months[0]].sum(), 7)


if __name__ == "__main__":
    unittest.main()
