"""Future-data invariance and warmup checks for the dynamic market model."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.comprehensive_temporal import ComprehensiveTemporalModel, FEATURES
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET
from tests.test_aligned_temporal_features import panel


def example():
    frame = panel()
    frame["TOWN_CLEAN"] = frame.TOWN
    frame["FLAT_AGE_YEARS"] = frame[MONTH].dt.year - frame.LEASE_COMMENCE_DATE
    coe = pd.Series(40000.0 + 500 * np.arange(36), index=pd.period_range("2021-01", periods=36, freq="M"))
    return frame, coe


class ComprehensiveTemporalTests(unittest.TestCase):
    def test_correction_training_excludes_future_targets_even_when_not_selected(self):
        frame, coe = example()
        cutoff = pd.Period("2023-05", "M")
        changed = frame.copy()
        changed.loc[changed[MONTH] > cutoff, TARGET] = 1e9
        changed_coe = coe.copy()
        changed_coe.loc[changed_coe.index > cutoff] = 1e9
        original = ComprehensiveTemporalModel(frame, coe)
        modified = ComprehensiveTemporalModel(changed, changed_coe)
        index = cutoff.ordinal - original.first.ordinal
        for economic in (False, True):
            a = original.fit_correction(index, economic, 10.0)
            b = modified.fit_correction(index, economic, 10.0)
            self.assertIsNotNone(a)
            for name in ("mean", "scale", "coef"):
                np.testing.assert_allclose(a[name], b[name], rtol=0, atol=1e-10)
            self.assertEqual(a["last_target_month"], str(cutoff))

    def test_future_rents_and_coe_cannot_change_forecasts_or_selection(self):
        frame, coe = example()
        cutoff = pd.Period("2023-05", "M")
        query = frame[frame[MONTH] > cutoff].iloc[:16]
        first_model = ComprehensiveTemporalModel(frame, coe)
        first = first_model.features(query, cutoff, np.full(len(query), 1500.0))
        changed = frame.copy()
        changed.loc[changed[MONTH] > cutoff, TARGET] = 1e9
        changed_coe = coe.copy()
        changed_coe.loc[changed_coe.index > cutoff] = 1e9
        second_model = ComprehensiveTemporalModel(changed, changed_coe)
        second = second_model.features(query, cutoff, np.full(len(query), 1500.0))
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_model.choices, second_model.choices)

    def test_initial_reference_is_never_used_before_it_is_available(self):
        frame, coe = example()
        cutoff = pd.Period("2021-06", "M")
        query = frame[frame[MONTH] == cutoff + 1].copy()
        fallback = np.arange(len(query), dtype=float) + 2000
        a = ComprehensiveTemporalModel(frame, coe).features(query, cutoff, fallback)
        frame.loc[frame[MONTH] > cutoff, TARGET] = 1e9
        b = ComprehensiveTemporalModel(frame, coe).features(query, cutoff, fallback)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(a.DYN_PROPERTY_REFERENCE.isna().all())
        np.testing.assert_array_equal(a.DYN_BASELINE_FORECAST, fallback)

    def test_calendar_features_and_seventeen_month_horizon_are_known(self):
        frame, coe = example()
        cutoff = pd.Period("2022-12", "M")
        query = frame[frame[MONTH] == cutoff + 1].iloc[:1].copy()
        query[MONTH] = cutoff + 17
        result = ComprehensiveTemporalModel(frame, coe).features(query, cutoff, np.array([2000.0]))
        self.assertEqual(result.DYN_FORECAST_HORIZON.iloc[0], 17)
        self.assertAlmostEqual(result.DYN_TARGET_MONTH_SIN.iloc[0], np.sin(2 * np.pi * 5 / 12))
        self.assertTrue(np.isfinite(result.DYN_BASELINE_FORECAST).all())
        self.assertEqual(list(result.columns), FEATURES)

    def test_reference_fit_ends_at_the_first_year(self):
        frame, coe = example()
        model = ComprehensiveTemporalModel(frame, coe)
        self.assertEqual(model.reference_end, pd.Period("2021-12", "M"))
        frame.loc[frame[MONTH] > model.reference_end, TARGET] = 1e9
        changed = ComprehensiveTemporalModel(frame, coe)
        columns = ["FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "TOWN_CLEAN", "FLOOR_AREA_SQM", "FLAT_AGE_YEARS"]
        np.testing.assert_allclose(model.reference.predict(frame[columns]), changed.reference.predict(frame[columns]))


if __name__ == "__main__":
    unittest.main()
