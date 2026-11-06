"""Prevent label-driven context selection and query-label preprocessing leakage."""
import unittest
import numpy as np
import pandas as pd
from analysis.experiments.second_wave_foundation import context_indices, inputs
from analysis.experiments.second_wave_anniversary import AnniversaryHistory


class SecondWaveTests(unittest.TestCase):
    def bundle(self):
        return {'x_fit': pd.DataFrame({'value': np.arange(80.),
                'kind': pd.Categorical(['a','b']*40)}),
            'x_val': pd.DataFrame({'value': [81.,82.], 'kind': ['a','new']}),
            'extra_fit': pd.DataFrame({'APPROVAL_MONTH_INDEX': np.arange(80.),
                'MARKET_FORECAST': np.r_[np.full(6,np.nan),np.full(74,2000.)]}),
            'extra_val': pd.DataFrame({'MARKET_FORECAST': [2100.,2200.]}),
            'y_fit': np.arange(80.)*20+1500, 'y_val': np.array([2500.,2600.])}

    def test_context_sampling_ignores_rent_values_and_excludes_warmup(self):
        bundle = self.bundle()
        spec = {'target':'residual','recent':False,'context':16}
        before = context_indices(bundle,spec)
        bundle['y_fit'] = bundle['y_fit'][::-1]*100
        after = context_indices(bundle,spec)
        np.testing.assert_array_equal(before,after)
        self.assertEqual(len(np.unique(before)),16)
        self.assertGreaterEqual(before.min(),6)

    def test_recent_context_has_only_last_twenty_four_months(self):
        bundle = self.bundle()
        chosen = context_indices(bundle,{'target':'raw','recent':True,'context':16})
        self.assertGreaterEqual(chosen.min(),56)
        self.assertLessEqual(chosen.max(),79)

    def test_query_rent_is_not_used_by_feature_or_target_preparation(self):
        bundle = self.bundle()
        spec = {'features':'compact','target':'residual'}
        x,q,y,offset = inputs(bundle,spec)
        bundle['y_val'][:] = -100000
        xx,qq,yy,other_offset = inputs(bundle,spec)
        pd.testing.assert_frame_equal(x,xx)
        pd.testing.assert_frame_equal(q,qq)
        np.testing.assert_array_equal(y,yy)
        np.testing.assert_array_equal(offset,other_offset)

    def rent_history(self):
        return pd.DataFrame([{'RENT_APPROVAL_DATE':str(month),'FLAT_TYPE_CLEAN':'4-room',
            'ADDRESS_KEY':'block-a','STREET_CLEAN':'street-a','PLANNING_AREA_CLEAN':'area-a',
            'MONTHLY_RENT':2000.+i*10} for i,month in enumerate(pd.period_range('2021-01','2023-12',freq='M'))])

    def test_anniversary_lookup_ignores_same_month_and_future_labels(self):
        source=self.rent_history()
        query=source.iloc[[-1]].copy()
        before=AnniversaryHistory().fit(source).transform(query)
        source.loc[source.RENT_APPROVAL_DATE>='2023-12','MONTHLY_RENT']=999999.
        query.MONTHLY_RENT=-99999.
        after=AnniversaryHistory().fit(source).transform(query)
        pd.testing.assert_frame_equal(before,after)
        self.assertEqual(before.ANN_BLOCK_12_R0_N.iloc[0],1.)

    def test_long_horizon_anniversary_does_not_fill_unobserved_months(self):
        source=self.rent_history()
        fitted=source.loc[source.RENT_APPROVAL_DATE<'2023-04']
        query=source.iloc[[0]].copy()
        query.RENT_APPROVAL_DATE='2024-03'
        features=AnniversaryHistory().fit(fitted).transform(query)
        self.assertEqual(features.ANN_BLOCK_12_R1_N.iloc[0],2.)
        query.RENT_APPROVAL_DATE='2024-04'
        features=AnniversaryHistory().fit(fitted).transform(query)
        self.assertEqual(features.ANN_BLOCK_12_R0_N.iloc[0],0.)
        self.assertEqual(features.ANN_BLOCK_12_R1_N.iloc[0],1.)


if __name__ == '__main__':
    unittest.main()
