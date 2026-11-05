"""Cutoff, hierarchical fallback and out-of-time residual availability checks."""
import unittest
import numpy as np
import pandas as pd
from analysis.experiments.targeted_spatial_economic import hierarchy_features,row_cutoffs,residual_smoother
from analysis.run_targeted_housing_supply import supply_features,numeric_rows


class TargetedFeatureTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for i,month in enumerate(pd.period_range('2021-01',periods=18,freq='M')):
            for block in ['1','2']:
                rows.append(dict(month=month,TOWN_CLEAN='town',FLAT_TYPE_CLEAN='3-room',STREET='street',BLOCK=block,
                                 MONTHLY_RENT=2000+i*50+(block=='2')*100))
        source=pd.DataFrame(rows)
        q=source.iloc[[-2,-1]].drop(columns='MONTHLY_RENT').copy()
        q.index=[100,101]
        return source,q

    def test_future_labels_and_query_labels_never_enter_history(self):
        source,q=self.fixture();cut=pd.Series(pd.Period('2022-04','M'),index=q.index)
        before,_=hierarchy_features(source,q,cut)
        source.loc[source.month>pd.Period('2022-04','M'),'MONTHLY_RENT']=999999
        q['MONTHLY_RENT']=-999999
        after,_=hierarchy_features(source,q,cut)
        pd.testing.assert_frame_equal(before,after)

    def test_unknown_street_and_building_shrink_to_town(self):
        source,q=self.fixture();q['STREET']='unknown';q['BLOCK']='unknown'
        f,_=hierarchy_features(source,q,pd.Series(pd.Period('2022-04','M'),index=q.index))
        np.testing.assert_allclose(f.HD_STREET_RECENT_SHRUNK,f.HD_TOWN_RECENT_SHRUNK)
        np.testing.assert_allclose(f.HD_BUILDING_RECENT_SHRUNK,f.HD_STREET_RECENT_SHRUNK)
        self.assertTrue(f.HD_STREET_COUNT_6M.eq(0).all())
        self.assertTrue(f.HD_STREET_STD_6M.ge(0).all())

    def test_no_history_stays_missing_and_has_zero_support(self):
        source,q=self.fixture();f,_=hierarchy_features(source,q,pd.Series(pd.Period('2020-12','M'),index=q.index))
        self.assertTrue(f.HD_TOWN_RECENT_SHRUNK.isna().all())
        self.assertTrue(f.HD_HISTORY_MISSING.eq(1).all())
        self.assertTrue(f.HD_TOWN_COUNT_6M.eq(0).all())

    def test_feature_cutoff_respects_existing_pseudo_origin_and_extra_month(self):
        x=pd.DataFrame(dict(APPROVAL_MONTH_INDEX=[0,41,50],TEMP_FORECAST_HORIZON=[np.nan,1,10]))
        months,cutoffs=row_cutoffs(x)
        self.assertTrue(pd.isna(cutoffs.iloc[0]))
        self.assertEqual(str(months[1]),'2024-06')
        self.assertEqual(str(cutoffs.iloc[1]),'2024-04')
        self.assertEqual(str(cutoffs.iloc[2]),'2024-04')

    def test_only_mature_out_of_time_residuals_are_used(self):
        source,q=self.fixture();h=source.copy();h['residual']=50.;h['weight']=1.;h['origin']='2021-01'
        h['context_last_month']=h.month-1
        cutoff=pd.Period('2022-04','M')
        before,_=residual_smoother(h,q,cutoff)
        h.loc[h.month>cutoff,'residual']=999999
        after,meta=residual_smoother(h,q,cutoff)
        np.testing.assert_array_equal(before,after)
        self.assertEqual(meta['latest_residual_month'],'2022-04')
        self.assertTrue(((before>0)&(before<50)).all())
        h.loc[h.month<=cutoff,'context_last_month']=h.loc[h.month<=cutoff,'month']
        with self.assertRaisesRegex(ValueError,'in-sample'):residual_smoother(h,q,cutoff)

    def test_supply_is_available_by_release_not_reference_period(self):
        q=pd.DataFrame({'TOWN_CLEAN':['town','town']},index=[10,20])
        cut=pd.Series(pd.PeriodIndex(['2023-12','2024-01'],freq='M'),index=q.index)
        v=pd.DataFrame([dict(year=2023,town='town',stock=100,completed=10,available='2024-01')])
        f=supply_features(q,cut,v)
        self.assertTrue(pd.isna(f.loc[10,'SUPPLY_COMPLETIONS_STOCK_RATIO']))
        self.assertEqual(f.loc[10,'SUPPLY_UNAVAILABLE'],1.)
        self.assertAlmostEqual(f.loc[20,'SUPPLY_COMPLETIONS_STOCK_RATIO'],.1)
        self.assertEqual(f.loc[20,'SUPPLY_OBSERVATION_AGE_MONTHS'],10.)

    def test_future_supply_vintage_cannot_change_earlier_features(self):
        q=pd.DataFrame({'TOWN_CLEAN':['town','unknown','zero']})
        cut=pd.Series(pd.Period('2022-04','M'),index=q.index)
        v=pd.DataFrame([dict(year=2019,town='town',stock=100,completed=10,available='2020-01'),
                        dict(year=2023,town='town',stock=1000,completed=999,available='2024-01'),
                        dict(year=2019,town='zero',stock=0,completed=0,available='2020-01')])
        before=supply_features(q,cut,v)
        v.loc[v.year==2023,['stock','completed']]=999999
        after=supply_features(q,cut,v)
        pd.testing.assert_frame_equal(before,after)
        self.assertAlmostEqual(before.loc[0,'SUPPLY_COMPLETIONS_STOCK_RATIO'],.1)
        self.assertTrue(before.loc[[1,2],'SUPPLY_UNAVAILABLE'].eq(1).all())

    def test_archive_numeric_row_order_is_spatial_not_pdf_text_order(self):
        class Page:
            def get_text(self,mode):
                return [(400.,100.,410.,110.,'15748',0,0,0),
                        (200.,100.3,210.,110.,'1062350',0,0,0),
                        (500.,100.,510.,110.,'69457',0,0,0)]
        self.assertEqual(numeric_rows(Page(),3),[[1062350,15748,69457]])


if __name__=='__main__':unittest.main()
