"""Small input ablations must preserve matrices and information cutoffs."""
import unittest
import numpy as np
import pandas as pd
import torch
from analysis.experiments.reliability_macro_features import (
    RELIABILITY, COE_FEATURE, add_forecast, dates_and_origins, forecast_table, reliability,
)
from tests.test_temporal_sequence_ablation import TemporalSequenceTests


class ReliabilityMacroTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(dict(APPROVAL_MONTH_INDEX=[41, 42, 42], TEMP_FORECAST_HORIZON=[1., 2., np.nan],
            HIST_SIGNATURE_MEAN=[np.nan, 3000., 3000.], HIST_BLOCK_FLAT_MEAN=[3000., np.nan, 3000.],
            RECENT_BLOCK_FLAT_3M_COUNT=[0., 1., 2.], RECENT_BLOCK_FLAT_12M_COUNT=[4., 5., 6.],
            BLOCK_ID=pd.Categorical(['known', None, 'known'])), index=[10, 20, 30])

    def table(self):
        return pd.DataFrame(dict(cutoff=['2024-04', '2024-04'], horizon=[2, 3],
                                 rent_growth=[1., 2.], coe_growth=[1.5, 2.7]))

    def test_reliability_flags_mean_what_the_names_say(self):
        z = reliability(self.frame())
        np.testing.assert_array_equal(z[RELIABILITY], [[1,0,1,1,0], [0,1,1,0,1], [0,0,0,0,0]])

    def test_original_values_categories_and_indices_are_preserved(self):
        x = self.frame()
        for variant, count in [('reliability',5), ('rent_forecast',7), ('coe_increment',8)]:
            z = add_forecast(x, self.table(), variant)
            pd.testing.assert_frame_equal(x, z[x.columns])
            self.assertEqual(z.shape[1], x.shape[1]+count)

    def test_macro_inputs_use_origin_minus_one_month(self):
        z = add_forecast(self.frame(), self.table(), 'coe_increment')
        np.testing.assert_allclose(z.NATIONAL_RENT_FORECAST_GROWTH.iloc[:2], [1., 2.])
        np.testing.assert_allclose(z[COE_FEATURE].iloc[:2], [.5, .7])
        self.assertTrue(np.isnan(z.NATIONAL_RENT_FORECAST_GROWTH.iloc[2]))
        self.assertEqual(z.NATIONAL_FORECAST_UNAVAILABLE.iloc[2], 1.)

    def test_unavailable_table_rows_cannot_change_inputs(self):
        x = self.frame(); before = add_forecast(x, self.table(), 'coe_increment')
        poisoned = pd.concat([self.table(), pd.DataFrame(dict(cutoff=['2024-05'], horizon=[1],
                                                          rent_growth=[1e9], coe_growth=[-1e9]))])
        pd.testing.assert_frame_equal(before, add_forecast(x, poisoned, 'coe_increment'))

    def test_forecast_table_ignores_future_rents_and_coe(self):
        fixture = TemporalSequenceTests(); fixture.setUp()
        panel = fixture.panel
        cut = panel.index[29]
        a = forecast_table(panel, epochs=3, cutoffs=[cut])
        changed = panel.copy(); changed.iloc[30:] *= 123
        b = forecast_table(changed, epochs=3, cutoffs=[cut])
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue((a.mature_monthly_labels >= 3).all())


if __name__ == '__main__':
    unittest.main()
