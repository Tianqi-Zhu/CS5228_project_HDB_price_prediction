"""Forecast-cutoff tests for horizon augmentation and comparable cohorts."""
import unittest
import numpy as np
import pandas as pd
from analysis.experiments.fourth_wave_horizons import CutoffCohorts,inputs as horizon_inputs
from analysis.experiments.fourth_wave_spatial import CalendarComparables
from analysis.experiments.frontier_search import month_number
from analysis.experiments.fourth_wave_flows import batch_features
from analysis.experiments.second_wave_anniversary import AnniversaryHistory


class FourthWaveTests(unittest.TestCase):
    def source(self):
        rows=[]
        for i,month in enumerate(pd.period_range('2021-01','2023-12',freq='M')):
            for key,longitude,value in [('a',103.8,1800+i*10),('b',103.81,2200+i*20)]:
                rows.append({'RENT_APPROVAL_DATE':str(month),'FLAT_TYPE_CLEAN':'4-room',
                    'ADDRESS_KEY':key,'STREET_CLEAN':'street','PLANNING_AREA_CLEAN':'area',
                    'LATITUDE':1.3,'LONGITUDE':longitude,'FLOOR_AREA_SQM':90.,
                    'LEASE_COMMENCE_DATE':2000.,'MONTHLY_RENT':value})
        return pd.DataFrame(rows)

    def test_cutoff_cohorts_hide_post_cutoff_records_even_in_past_windows(self):
        source=self.source()
        query=source.iloc[[-2]].copy()
        query.RENT_APPROVAL_DATE='2024-12'
        cutoff=month_number(pd.Series(['2023-12']))
        before=CutoffCohorts().fit(source).transform(query,cutoff)
        # 12-month cohort has November as the only available observed month.
        self.assertEqual(before.ANN_BLOCK_12_R1_N.iloc[0],1.)
        source.loc[source.RENT_APPROVAL_DATE>='2023-12','MONTHLY_RENT']=999999.
        query.MONTHLY_RENT=-999999.
        after=CutoffCohorts().fit(source).transform(query,cutoff)
        pd.testing.assert_frame_equal(before,after)

    def test_augmented_horizon_can_have_zero_anniversary_support(self):
        source=self.source()
        query=source.iloc[[-2]].copy()
        query.RENT_APPROVAL_DATE='2024-12'
        cutoff=month_number(pd.Series(['2023-11']))
        values=CutoffCohorts().fit(source).transform(query,cutoff)
        self.assertEqual(values.ANN_BLOCK_12_R1_N.iloc[0],0.)
        self.assertTrue(np.isnan(values.ANN_BLOCK_12_R1_RENT.iloc[0]))

    def test_spatial_cohorts_exclude_same_block(self):
        source=self.source()
        query=source.iloc[[-2]].copy()
        before=CalendarComparables().fit(source).transform(query,month_number(query.RENT_APPROVAL_DATE))
        # Only the other block's three observed months may contribute.
        self.assertEqual(before.SPATIAL_COHORT_12_K8_N.iloc[0],3.)
        self.assertGreater(before.SPATIAL_COHORT_12_K8_DISTANCE.iloc[0],0.)

    def test_spatial_query_labels_and_post_cutoff_rents_are_irrelevant(self):
        source=self.source()
        query=source.iloc[[-2]].copy()
        query.RENT_APPROVAL_DATE='2024-12'
        cutoff=month_number(pd.Series(['2023-12']))
        before=CalendarComparables().fit(source).transform(query,cutoff)
        source.loc[source.RENT_APPROVAL_DATE>='2023-12','MONTHLY_RENT']=999999.
        query.MONTHLY_RENT=-99999.
        after=CalendarComparables().fit(source).transform(query,cutoff)
        pd.testing.assert_frame_equal(before,after)

    def test_cutoff_after_target_is_rejected(self):
        source=self.source()
        query=source.iloc[[-2]].copy()
        cutoff=month_number(query.RENT_APPROVAL_DATE)+1
        with self.assertRaises(ValueError):
            CutoffCohorts().fit(source).transform(query,cutoff)
        with self.assertRaises(ValueError):
            CalendarComparables().fit(source).transform(query,cutoff)

    def test_batch_features_never_read_query_rents(self):
        source=self.source()
        query=source.iloc[-4:].copy()
        ann=AnniversaryHistory().fit(source.iloc[:-4]).transform(query)
        before=batch_features(query,ann)
        query.MONTHLY_RENT=-999999.
        after=batch_features(query,ann)
        pd.testing.assert_frame_equal(before,after)
        np.testing.assert_array_equal(before.BATCH_BLOCK_COUNT,1.)

    def test_batch_counts_are_per_month_and_property_not_row_order(self):
        source=self.source()
        query=pd.concat([source.iloc[[-2]],source.iloc[[-2]],source.iloc[[-1]]],ignore_index=True)
        ann=AnniversaryHistory().fit(source.iloc[:-2]).transform(query)
        values=batch_features(query,ann)
        np.testing.assert_array_equal(values.BATCH_BLOCK_COUNT,[2.,2.,1.])
        np.testing.assert_array_equal(values.BATCH_BUILDING_COUNT,[2.,2.,1.])
        np.testing.assert_array_equal(values.BATCH_TYPE_COUNT,3.)

    def test_augmentation_preserves_total_weight_of_each_source_record(self):
        first={'x':pd.DataFrame({'value':[1.,2.]}),'y':np.array([2000.,3000.]),
            'level':np.array([1800.,2800.]),'reference':np.array([2100.,3100.]),'source_row':np.array([7,8])}
        later={'x':pd.DataFrame({'value':[2.]}),'y':np.array([3000.]),
            'level':np.array([2500.]),'reference':np.array([3100.]),'source_row':np.array([8])}
        bundle={'views':{0:first,6:later,12:later},'horizon_query':pd.DataFrame({'value':[3.]}),
            'query_level':np.array([3000.]),'y_val':np.array([3200.])}
        _,q,_,_,weights,_,offset=horizon_inputs(bundle,{'view':'long','target':'raw'})
        np.testing.assert_allclose(weights,[1.,1/3,1/3,1/3])
        bundle['y_val'][:]=-999999
        _,other_q,_,_,_,_,other_offset=horizon_inputs(bundle,{'view':'long','target':'raw'})
        pd.testing.assert_frame_equal(q,other_q)
        np.testing.assert_array_equal(offset,other_offset)


if __name__=='__main__':
    unittest.main()
