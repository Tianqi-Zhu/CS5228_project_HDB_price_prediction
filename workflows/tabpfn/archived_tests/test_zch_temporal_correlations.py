"""Check lag alignment in the monthly time-series diagnostic."""

import unittest

import numpy as np
import pandas as pd

from analysis.studies.temporal_correlations import lag_correlations


class TemporalCorrelationTests(unittest.TestCase):
    def test_lag_three_aligns_a_signal_two_months_after_lag_one(self):
        signal = np.random.default_rng(7).normal(size=24)
        frame = pd.DataFrame({
            "rent_adjusted_growth": pd.Series(signal).shift(2).to_numpy(),
            "SOURCE_LAG1": signal,
        }, index=pd.period_range("2021-01", periods=24, freq="M"))
        result = lag_correlations(frame, indicators={"SOURCE_LAG1": "source"},
                                  max_lag=4, bootstrap_repeats=0)
        peak = result.loc[result["pearson_r"].idxmax()]
        self.assertEqual(int(peak["lag_months"]), 3)
        self.assertAlmostEqual(float(peak["pearson_r"]), 1.0)


if __name__ == "__main__":
    unittest.main()
