"""Check causal information boundaries of forecast-origin features."""
import unittest
import numpy as np
import pandas as pd
from analysis.experiments.frontier_search import DistributionHistory, STATICS, month_number
from analysis.pipeline.config import DATE, TARGET


class FrontierHistoryTests(unittest.TestCase):
    def frame(self):
        rows = []
        for month in range(1, 13):
            for block in ('a', 'b'):
                row = {column: 1. for column in STATICS}
                row.update({DATE: f'2022-{month:02}', TARGET: 1500 + month*100,
                    'FLAT_TYPE_CLEAN': '3-room', 'FLAT_MODEL_CLEAN': 'standard',
                    'ADDRESS_KEY': block, 'STREET_CLEAN': 'road', 'PLANNING_AREA_CLEAN': 'area',
                    'TOWN_CLEAN': 'town', 'LATITUDE': 1.3, 'LONGITUDE': 103.8,
                    'LEASE_COMMENCE_DATE': 1980., 'FLOOR_AREA_SQM': 70.})
                rows.append(row)
        return pd.DataFrame(rows)

    def test_cutoff_excludes_same_month_and_all_future_targets(self):
        source = self.frame()
        query = source.iloc[[-1]].copy()
        cutoff = month_number(pd.Series(['2022-08']))
        before = DistributionHistory().fit(source).transform(query, cutoff, neighbors=True)
        source.loc[source[DATE] >= '2022-08', TARGET] += 100000
        after = DistributionHistory().fit(source).transform(query, cutoff, neighbors=True)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(before.BLOCK_N.iloc[0], 7.)
        self.assertEqual(before.INFORMATION_AGE.iloc[0], 4.)

    def test_query_rents_are_not_features(self):
        source = self.frame()
        query = source.iloc[[-1]].copy()
        cutoff = month_number(pd.Series(['2022-10']))
        encoder = DistributionHistory().fit(source)
        before = encoder.transform(query, cutoff)
        query[TARGET] = -999999.
        after = encoder.transform(query, cutoff)
        pd.testing.assert_frame_equal(before, after)
        self.assertTrue(np.isfinite(before.MARKET_FORECAST).all())

    def test_no_history_has_zero_count_and_missing_statistics(self):
        source = self.frame()
        query = source.iloc[[0]].copy()
        cutoff = month_number(query[DATE])
        result = DistributionHistory().fit(source).transform(query, cutoff)
        self.assertEqual(result.BLOCK_N.iloc[0], 0)
        self.assertTrue(pd.isna(result.BLOCK_RENT_MEAN.iloc[0]))


if __name__ == '__main__':
    unittest.main()
