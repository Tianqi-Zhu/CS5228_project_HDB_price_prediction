"""Guard the forecast information boundary in third-wave experiments."""
import unittest
import numpy as np
import pandas as pd
from analysis.experiments.third_wave_features import KernelCohorts
from analysis.experiments.third_wave_training import training_mask,adjusted_baseline
from analysis.experiments.third_wave_cross_type import normalized_source


class ThirdWaveTests(unittest.TestCase):
    def source(self):
        rows=[]
        for i,month in enumerate(pd.period_range('2021-01','2023-12',freq='M')):
            for address,rent in [('a',1500+10*i),('b',2500+5*i)]:
                rows.append({'RENT_APPROVAL_DATE':str(month),'ADDRESS_KEY':address,
                    'STREET_CLEAN':'street','PLANNING_AREA_CLEAN':'area',
                    'FLAT_TYPE_CLEAN':'4-room','MONTHLY_RENT':rent})
        return pd.DataFrame(rows)

    def test_kernel_ignores_query_and_recent_labels(self):
        source=self.source()
        query=source.iloc[[-2]].copy()
        before=KernelCohorts().fit(source).transform(query)
        # The latest window is 12 months back plus three months. All newer
        # rents, including the query-month rent, must be irrelevant.
        source.loc[source.RENT_APPROVAL_DATE>'2023-03','MONTHLY_RENT']=-999999
        query.MONTHLY_RENT=888888
        after=KernelCohorts().fit(source).transform(query)
        pd.testing.assert_frame_equal(before,after)

    def test_kernel_has_no_unobserved_history(self):
        source=self.source()
        query=source.iloc[[0]].copy()
        query.RENT_APPROVAL_DATE='2025-04'
        values=KernelCohorts().fit(source).transform(query)
        self.assertEqual(values.KERNEL_BLOCK_12_N.iloc[0],0)
        self.assertTrue(np.isnan(values.KERNEL_BLOCK_12_RATIO.iloc[0]))
        self.assertGreater(values.KERNEL_BLOCK_24_N.iloc[0],0)

    def test_weighted_effective_count_and_constant_ratios(self):
        source=self.source()
        source.MONTHLY_RENT=2000.
        query=source.iloc[[0]].copy()
        query.RENT_APPROVAL_DATE='2023-06'
        values=KernelCohorts().fit(source).transform(query)
        self.assertAlmostEqual(values.KERNEL_BLOCK_24_RATIO.iloc[0],1.)
        self.assertAlmostEqual(values.KERNEL_BLOCK_24_SD.iloc[0],0.)
        self.assertGreater(values.KERNEL_BLOCK_24_EFFECTIVE_N.iloc[0],1.)
        self.assertLessEqual(values.KERNEL_BLOCK_24_EFFECTIVE_N.iloc[0],7.)

    def test_training_policy_preserves_historical_pool(self):
        bundle={'fit_month':np.arange(100,136),'origin':136,
            'ann_fit':pd.DataFrame({'ANN_BLOCK_24_R1_N':np.r_[np.zeros(24),np.ones(12)]})}
        original=bundle['fit_month'].copy()
        selected=training_mask(bundle,{'training':'last12'})
        np.testing.assert_array_equal(np.flatnonzero(selected),np.arange(24,36))
        np.testing.assert_array_equal(bundle['fit_month'],original)

    def test_shrinkage_uses_past_support_and_level(self):
        ann=pd.DataFrame({'ANN_BLOCK_24_R1_N':[0.,2.],
            'ANN_BLOCK_24_R1_RATIO':[np.nan,.5]})
        extra=pd.DataFrame({'BLOCK_RATIO_MEAN':[1.,1.],
            'AREA_RATIO_MEAN':[1.,1.],'MARKET_3M':[3000.,3000.], 'BLOCK_N':[8.,8.]})
        raw=pd.DataFrame({'FLOOR_AREA_SQM':[90.,90.]})
        prediction,features=adjusted_baseline(ann,extra,raw)
        np.testing.assert_allclose(prediction,[3000.,2250.])
        np.testing.assert_allclose(features['COHORT24_SHRUNK_RATIO'],[1.,.75])

    def test_cross_type_normalization_uses_each_observed_type_month(self):
        source=self.source()
        source.loc[source.ADDRESS_KEY=='b','FLAT_TYPE_CLEAN']='5-room'
        normalized=normalized_source(source)
        self.assertEqual(normalized.FLAT_TYPE_CLEAN.nunique(),1)
        # Each type contains one observation in this toy data. Different
        # nominal rents must therefore have exactly the same premium.
        np.testing.assert_allclose(normalized.MONTHLY_RENT,1000.)
        self.assertNotEqual(source.MONTHLY_RENT.nunique(),1)

    def test_cross_type_features_cannot_use_newer_source_rents(self):
        source=self.source()
        query=source.iloc[[-2]].copy()
        query.FLAT_TYPE_CLEAN='ALL_TYPES'
        before=KernelCohorts().fit(normalized_source(source)).transform(query)
        source.loc[source.RENT_APPROVAL_DATE>'2023-03','MONTHLY_RENT']=999999
        query.MONTHLY_RENT=-12345
        after=KernelCohorts().fit(normalized_source(source)).transform(query)
        pd.testing.assert_frame_equal(before,after)


if __name__=='__main__':
    unittest.main()
