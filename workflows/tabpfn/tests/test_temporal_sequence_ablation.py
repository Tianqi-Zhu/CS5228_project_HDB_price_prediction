"""Availability, mature-label and selection invariants for sequence forecasts."""
import unittest

import numpy as np
import pandas as pd
import torch

from analysis.experiments.temporal_sequence_ablation import (
    Spec, channels, evaluation_cases, examples, forecast, select, standardize,
)


class TemporalSequenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False
        dates = pd.period_range('2021-01', '2025-03', freq='M')
        t = np.arange(len(dates))
        self.panel = pd.DataFrame({
            'mean_rent': 2000 * np.exp(.007*t + .005*np.sin(t)),
            'adjusted_mean_rent': 2100 * np.exp(.007*t + .003*np.cos(t)),
            'cap_seeded_22': 100 * np.exp(.003*t + .02*np.sin(t)),
            'equal_weight_matched': 100 * np.exp(.004*t + .02*np.sin(t)),
            'coe_a': 40000 * np.exp(.01*t + .03*np.cos(t)),
            'coe_c': 50000 * np.exp(.008*t + .02*np.sin(t)), 'n': 1000,
        }, index=dates)

    def test_no_future_label_is_mature_and_query_has_no_labels(self):
        _, _, y, mask, origins = examples(self.panel, 'mean_rent', Spec('lstm', 'rent_stock_coe', 12), 30)
        for i, origin in enumerate(origins):
            for h in range(1, 11):
                self.assertEqual(mask[i, h-1], origin+h <= 30)
                if mask[i, h-1]:
                    expected = 100*np.log(self.panel.mean_rent.iloc[origin+h]/self.panel.mean_rent.iloc[origin])/h
                    self.assertAlmostEqual(y[i, h-1], expected)
        self.assertFalse(mask[-1].any())

    def test_future_rents_stocks_coe_cannot_change_any_model_forecast(self):
        changed = self.panel.copy()
        changed.iloc[31:] *= 123
        for method in ('persistence', 'momentum3', 'ewma', 'ridge_summary', 'ridge_lags', 'mlp', 'lstm'):
            spec = Spec(method, 'rent_stock_coe', 6)
            a = forecast(self.panel, 'mean_rent', spec, 30, epochs=3)
            b = forecast(changed, 'mean_rent', spec, 30, epochs=3)
            np.testing.assert_array_equal(a[0], b[0], err_msg=method)

    def test_current_origin_is_allowed_but_target_month_is_not(self):
        spec = Spec('persistence')
        a = forecast(self.panel, 'mean_rent', spec, 30)[0]
        changed = self.panel.copy()
        changed.loc[changed.index[30], 'mean_rent'] *= 2
        b = forecast(changed, 'mean_rent', spec, 30)[0]
        np.testing.assert_allclose(b, 2*a)

    def test_input_delay_excludes_current_month_and_uses_available_anchor(self):
        changed = self.panel.copy()
        changed.loc[changed.index[30], ['mean_rent', 'coe_a', 'coe_c', 'cap_seeded_22']] *= 10
        for method in ('persistence', 'ridge_lags', 'lstm'):
            spec = Spec(method, 'rent_stock_coe', 6)
            a = forecast(self.panel, 'mean_rent', spec, 30, delay=1, epochs=3)[0]
            b = forecast(changed, 'mean_rent', spec, 30, delay=1, epochs=3)[0]
            np.testing.assert_array_equal(a, b)
            seq, _, _, _, _ = examples(changed, 'mean_rent', spec, 30, delay=1)
            expected = channels(self.panel.iloc[:30], 'mean_rent', spec.group).iloc[-6:].to_numpy()
            np.testing.assert_allclose(seq[-1], expected)

    def test_query_values_do_not_fit_scaler(self):
        sequence = np.array([[[1.], [3.]], [[2.], [4.]], [[1000.], [2000.]]])
        mask = np.array([[True], [True], [False]])
        first = standardize(sequence, mask)
        sequence[-1] *= 100
        second = standardize(sequence, mask)
        np.testing.assert_array_equal(first[:-1], second[:-1])

    def test_selection_does_not_use_outer_errors(self):
        records = []
        for phase in ('development', 'outer'):
            for spec, value in [(Spec('persistence'), 1.), (Spec('lstm', 'rent_stock_coe', 3), 2.)]:
                records.append(dict(target='mean_rent', scenario='rolling', phase=phase,
                                    model=spec.name, method=spec.method, group=spec.group,
                                    window=spec.window, normalized_mse=value))
        rank = pd.DataFrame(records)
        first = select(rank)
        rank.loc[rank.phase.eq('outer'), 'normalized_mse'] *= -100
        pd.testing.assert_frame_equal(first, select(rank))

    def test_multistep_persistence_does_not_use_intervening_observations(self):
        result = forecast(self.panel, 'mean_rent', Spec('persistence'), 30)[0]
        np.testing.assert_array_equal(result, np.repeat(self.panel.mean_rent.iloc[30], 10))

    def test_development_paths_end_before_outer_period(self):
        cases = evaluation_cases(self.panel.index)
        self.assertTrue((cases.loc[cases.phase.eq('development'), 'month'] < pd.Period('2024-06', 'M')).all())
        self.assertTrue((cases.origin < cases.month).all())


if __name__ == '__main__':
    unittest.main()
