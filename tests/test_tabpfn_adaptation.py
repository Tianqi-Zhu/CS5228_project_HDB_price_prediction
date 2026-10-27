"""Time, label-isolation and alignment checks for feature/head adaptation."""

import unittest
import numpy as np
import pandas as pd

from analysis.experiments.tabpfn_adaptation_features import (
    ENHANCED, STD_FEATURES, TREND_FEATURES, HEAD_COLUMNS, HEAD_NUMERIC,
    backbone_inputs, enhanced_feature_pair, head_numeric, identifiers,
)
from analysis.experiments.tabpfn_adaptation import sampled_episode_queries, assert_context_query
from analysis.experiments.tabpfn_residual_heads import chronological_head_split
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET


def rows():
    return pd.DataFrame({
        MONTH: pd.PeriodIndex(['2021-01', '2021-02', '2021-02', '2021-03', '2021-05'], freq='M'),
        TARGET: [1000., 2000., 3000., 4000., 5000.],
        'TOWN': ['a'] * 5, 'BLOCK': ['1'] * 5, 'STREET': ['road'] * 5,
        'FLAT_TYPE_CLEAN': ['3-room'] * 5, 'FLAT_MODEL_CLEAN': ['improved'] * 5,
        'FLOOR_AREA_SQM': [70.] * 5, 'LEASE_COMMENCE_DATE': [1980] * 5,
        'PLANNING_AREA': ['a'] * 5, 'APPROVAL_MONTH_OF_YEAR': [1, 2, 2, 3, 5],
    })


class AdaptationTests(unittest.TestCase):
    def test_new_statistics_exclude_same_month_and_freeze_at_origin(self):
        data = rows()
        fitting, query = data.iloc[:3], data.iloc[[4, 3]]
        fit, val = enhanced_feature_pair(fitting, query)
        self.assertTrue(np.isnan(fit.loc[1, 'RECENT_BLOCK_FLAT_3M_STD']))
        self.assertTrue(np.isnan(fit.loc[2, 'RECENT_BLOCK_FLAT_3M_STD']))
        np.testing.assert_allclose(val.RECENT_BLOCK_FLAT_3M_STD, [1000., 1000.])
        np.testing.assert_allclose(val.RECENT_BLOCK_FLAT_3M_COUNT, [3., 3.])
        np.testing.assert_allclose(val.RECENT_BLOCK_FLAT_3M_MEAN, [2000., 2000.])
        self.assertEqual(val.index.tolist(), [4, 3])
        self.assertNotIn(TARGET, val)

    def test_query_rents_cannot_change_features(self):
        data = rows()
        fit, query = data.iloc[:3], data.iloc[3:]
        a, b = enhanced_feature_pair(fit, query)
        changed = query.copy()
        changed[TARGET] = 999999.
        c, d = enhanced_feature_pair(fit, changed)
        pd.testing.assert_frame_equal(a, c)
        pd.testing.assert_frame_equal(b, d)

    def test_ids_normalize_case_and_distinguish_same_block_number(self):
        data = pd.DataFrame({'TOWN': [' A ', 'a', 'b'], 'STREET': [' ROAD ', 'road', 'road'], 'BLOCK': ['1', '1', '1']})
        result = identifiers(data)
        self.assertEqual(result.BLOCK_ID.iloc[0], result.BLOCK_ID.iloc[1])
        self.assertNotEqual(result.BLOCK_ID.iloc[1], result.BLOCK_ID.iloc[2])
        fit, query = backbone_inputs(result.iloc[:1], result.iloc[1:], ['BLOCK_ID'])
        self.assertEqual(fit.BLOCK_ID.cat.categories.tolist(), ['a|road|1'])
        self.assertTrue(pd.isna(query.BLOCK_ID.iloc[1]))

    def test_episode_sampling_does_not_depend_on_rent(self):
        months = pd.period_range('2021-01', '2024-03', freq='M')
        data = pd.DataFrame({MONTH: months.repeat(10), TARGET: np.arange(len(months) * 10)})
        first = sampled_episode_queries(data, '2021-07', cap=3)
        data[TARGET] = -data[TARGET]
        second = sampled_episode_queries(data, '2021-07', cap=3)
        self.assertEqual(first.index.tolist(), second.index.tolist())
        self.assertEqual(len(first), 12)
        self.assertEqual(first[MONTH].astype(str).unique().tolist(), ['2021-07', '2021-09', '2021-12', '2022-06'])
        assert_context_query(data[data[MONTH] < pd.Period('2021-07', 'M')], first, '2021-07')
        with self.assertRaises(ValueError):
            assert_context_query(data, first, '2021-07')

    def test_inner_split_keeps_repeated_targets_together_and_excludes_outer(self):
        months = np.array(['2022-06', '2022-09', '2022-10', '2022-10', '2022-12'])
        training, validation = chronological_head_split(months, '2023-01')
        self.assertEqual(training.tolist(), [True, True, False, False, False])
        self.assertEqual(validation.tolist(), [False, False, True, True, True])
        with self.assertRaises(ValueError):
            chronological_head_split(np.array(['2022-01', '2023-01']), '2023-01')

    def test_head_horizon_is_explicit_and_feature_sets_have_no_current_target(self):
        query = pd.DataFrame({name: [0., 0.] for name in HEAD_NUMERIC})
        query[MONTH] = pd.PeriodIndex(['2021-03', '2021-05'], freq='M')
        numeric = head_numeric(query, np.array([1000., 2000.]), pd.Period('2021-03', 'M'))
        np.testing.assert_array_equal(numeric[:, HEAD_COLUMNS.index('FORECAST_HORIZON')], [1, 3])
        np.testing.assert_array_equal(numeric[:, HEAD_COLUMNS.index('ORIGIN_MONTH_INDEX')], [2, 2])
        self.assertEqual(len(ENHANCED), len(set(ENHANCED)))
        self.assertEqual(len(STD_FEATURES), 6)
        self.assertEqual(len(TREND_FEATURES), 4)
        for forbidden in (TARGET, 'LOG_MONTHLY_RENT', 'RENT_PER_SQM'):
            self.assertNotIn(forbidden, ENHANCED)
        with self.assertRaises(ValueError):
            backbone_inputs(rows(), rows(), [TARGET])


if __name__ == '__main__':
    unittest.main()
