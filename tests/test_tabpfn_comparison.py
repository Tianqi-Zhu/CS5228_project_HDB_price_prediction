"""Leakage and alignment checks; no model download or GPU required."""

import unittest

import numpy as np
import pandas as pd

from analysis.experiments.tabpfn_comparison import (
    BLEND, MONTH, TARGET, add_forward_blend, blend_weight,
    feature_frames, fixed_origin_split, tabpfn_inputs,
    select_candidate,
)


class TabPFNComparisonTests(unittest.TestCase):
    def test_split_excludes_entire_origin_month_and_preserves_source_order(self):
        rows = pd.DataFrame({MONTH: pd.PeriodIndex(['2023-02', '2022-12', '2023-01', '2023-03'], freq='M')},
                            index=[9, 4, 2, 8])
        fit, future = fixed_origin_split(rows, '2023-01', '2023-03')
        self.assertEqual(fit.index.tolist(), [4])
        self.assertEqual(future.index.tolist(), [9, 2])
        with self.assertRaises(ValueError):
            fixed_origin_split(rows, '2020-01', '2021-01')

    def test_history_ignores_future_labels_and_same_month_rents(self):
        rows = pd.DataFrame({
            MONTH: pd.PeriodIndex(['2021-01', '2021-02', '2021-02', '2021-03', '2021-05'], freq='M'),
            TARGET: [1000., 2000., 3000., 4000., 5000.],
            'TOWN': ['a'] * 5, 'BLOCK': ['1'] * 5, 'STREET': ['road'] * 5,
            'FLAT_TYPE_CLEAN': ['3-room'] * 5, 'FLAT_MODEL_CLEAN': ['improved'] * 5,
            'FLOOR_AREA_SQM': [70.] * 5, 'LEASE_COMMENCE_DATE': [1980] * 5,
            'PLANNING_AREA': ['a'] * 5,
        })
        fitting, future = rows.iloc[:3], rows.iloc[[4, 3]]
        fit, val = feature_frames(fitting, future)
        col = 'HIST_BLOCK_FLAT_MEAN'
        self.assertTrue(np.isnan(fit.loc[0, col]))
        self.assertEqual(fit.loc[1, col], 1000.)
        self.assertEqual(fit.loc[2, col], 1000.)
        np.testing.assert_allclose(val[col], [2000., 2000.])
        changed = future.copy()
        changed[TARGET] = 999999.
        _, second = feature_frames(fitting, changed)
        history = [c for c in val if c.startswith('HIST_')]
        pd.testing.assert_frame_equal(val[history], second[history])
        self.assertEqual(val.index.tolist(), [4, 3])
        self.assertEqual(val['HIST_BLOCK_FLAT_RECENCY_MONTHS'].tolist(), [3, 1])

    def test_blend_weight_uses_only_earlier_folds(self):
        rows = pd.DataFrame({'validation_start': ['2023-01', '2023-01', '2023-07', '2024-04'],
                             TARGET: [125., 225., 10000., -10000.],
                             'catboost': [100., 200., 100., 100.],
                             'tabpfn_history': [200., 300., 200., 200.]})
        result, weights = add_forward_blend(rows)
        np.testing.assert_allclose(result.loc[:1, BLEND], rows.loc[:1, 'catboost'])
        self.assertEqual(weights.iloc[1].tabpfn_weight, .25)
        changed = rows.copy()
        changed.loc[changed.validation_start.ge('2023-07'), TARGET] = -999999.
        second, _ = add_forward_blend(changed)
        np.testing.assert_allclose(result.loc[:2, BLEND], second.loc[:2, BLEND])

    def test_blend_is_constrained_and_handles_identical_models(self):
        rows = pd.DataFrame({TARGET: [0.], 'catboost': [100.], 'tabpfn_history': [200.]})
        self.assertEqual(blend_weight(rows), 0.)
        rows[TARGET] = 1000.
        self.assertEqual(blend_weight(rows), 1.)
        rows['tabpfn_history'] = rows['catboost']
        self.assertEqual(blend_weight(rows), 0.)

    def test_inputs_forbid_target_and_handle_unseen_categories(self):
        fitting = pd.DataFrame({'TOWN_CLEAN': ['a', 'b'], 'FLOOR_AREA_SQM': [50., 70.]})
        future = pd.DataFrame({'TOWN_CLEAN': ['new', 'a'], 'FLOOR_AREA_SQM': [60., np.nan]}, index=[7, 3])
        fit, val = tabpfn_inputs(fitting, future, list(fitting))
        self.assertEqual(val.index.tolist(), [7, 3])
        self.assertTrue(pd.isna(val.loc[7, 'TOWN_CLEAN']))
        self.assertTrue(pd.isna(val.loc[3, 'FLOOR_AREA_SQM']))
        self.assertEqual(fit['TOWN_CLEAN'].cat.categories.tolist(), ['a', 'b'])
        for forbidden in [TARGET, 'LOG_MONTHLY_RENT', 'RENT_PER_SQM']:
            with self.assertRaises(ValueError):
                tabpfn_inputs(fitting, future, [forbidden])

    def test_blend_can_qualify_with_uncalibrated_first_fold_tie(self):
        rows = []
        models = ['catboost', 'tabpfn_spatial', 'tabpfn_history', BLEND]
        for model in models:
            errors = [100., 90., 90.] if model == BLEND else [100., 100., 100.]
            for start, error in zip(['2023-01', '2023-07', '2024-04'], errors):
                rows.append({'validation_start': start, 'model': model, 'rmse_sgd': error})
        folds = pd.DataFrame(rows)
        pooled = pd.DataFrame({'model': models, 'rmse_sgd': [100., 100., 100., 94.]})
        self.assertEqual(select_candidate(pooled, folds), BLEND)
        # A pooled gain cannot hide a regression on a later calibrated fold.
        folds.loc[folds.model.eq(BLEND) & folds.validation_start.eq('2024-04'), 'rmse_sgd'] = 101.
        self.assertEqual(select_candidate(pooled, folds), 'catboost')


if __name__ == '__main__':
    unittest.main()
