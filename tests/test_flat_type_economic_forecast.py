"""Temporal availability and aggregation checks for the COE panel model."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.flat_type_economic_forecast import (
    coe_monthly, forecast, history_features, training_examples,
)


def synthetic():
    months = pd.period_range("2021-01", periods=30, freq="M")
    rows = []
    for i, month in enumerate(months):
        for kind, extra in (("3-room", 0), ("4-room", 500)):
            value = 1000.0 + 20 * i + extra
            rows.append({"month": month, "flat_type": kind, "mean": value,
                         "count": 10, "rent_sum": value * 10})
    coe = pd.Series(40000.0 + np.arange(len(months)) * 500, index=months)
    return pd.DataFrame(rows), coe


class EconomicForecastTests(unittest.TestCase):
    def test_auction_premiums_aggregate_by_month_and_category(self):
        data = pd.DataFrame({"YEAR": [2021] * 4, "MONTH": ["jan", "jan", "feb", "jan"],
                             "CATEGORY": ["a", "A", "a", "b"],
                             "QUOTA_PREMIUM": ["$40,000", "$42,000", "$45,000", "$99,999"]})
        result = coe_monthly(data)
        self.assertEqual(result.loc[pd.Period("2021-01", "M")], 41000.0)
        self.assertEqual(result.loc[pd.Period("2021-02", "M")], 45000.0)

    def test_history_windows_include_origin_but_exclude_future(self):
        panel, coe = synthetic()
        origin = pd.Period("2021-12", "M")
        features, anchor = history_features(panel, coe, "3-room", origin, 3, economic=True)
        self.assertEqual(anchor, np.mean([1180.0, 1200.0, 1220.0]))
        self.assertAlmostEqual(features["short_minus_long"], anchor - np.mean(1000 + 20 * np.arange(12)))
        changed = panel.copy()
        changed.loc[changed.month > origin, ["mean", "rent_sum"]] = 1e9
        changed_coe = coe.copy()
        changed_coe.loc[changed_coe.index > origin] = 1e9
        other, other_anchor = history_features(changed, changed_coe, "3-room", origin, 3, economic=True)
        self.assertEqual(features, other)
        self.assertEqual(anchor, other_anchor)

    def test_training_targets_and_forecasts_obey_the_cutoff(self):
        panel, coe = synthetic()
        cutoff = pd.Period("2022-12", "M")
        _, _, _, _, audits = training_examples(panel, coe, cutoff, economic=True)
        self.assertTrue(all(origin < target <= cutoff for origin, target, _ in audits))
        a, _ = forecast(panel, coe, cutoff, [1, 3, 6], economic=True, alpha=100)
        changed = panel.copy()
        changed.loc[changed.month > cutoff, ["mean", "rent_sum"]] = 1e9
        changed_coe = coe.copy()
        changed_coe.loc[changed_coe.index > cutoff] = 1e9
        b, _ = forecast(changed, changed_coe, cutoff, [1, 3, 6], economic=True, alpha=100)
        pd.testing.assert_frame_equal(a, b)

    def test_strong_regularization_shrinks_the_entire_change(self):
        panel, coe = synthetic()
        cutoff = pd.Period("2022-12", "M")
        prediction, _ = forecast(panel, coe, cutoff, [1, 6], economic=True, alpha=1e12)
        for row in prediction.itertuples(index=False):
            _, anchor = history_features(panel, coe, row.flat_type, cutoff, row.horizon, economic=True)
            self.assertLess(abs(row.predicted - anchor), 0.01)


if __name__ == "__main__":
    unittest.main()
