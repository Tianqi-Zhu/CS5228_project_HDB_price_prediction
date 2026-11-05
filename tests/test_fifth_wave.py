"""Protect historical information and static-only building feature sources."""
import unittest
import numpy as np
import pandas as pd
from analysis.experiments.fifth_wave_features import RentShapeHistory,BuildingContext
from analysis.experiments.frontier_search import month_number


class FifthWaveTests(unittest.TestCase):
    def source(self):
        return pd.DataFrame([{'RENT_APPROVAL_DATE':str(month),'ADDRESS_KEY':key,
            'STREET_CLEAN':'street','FLAT_TYPE_CLEAN':'4-room','MONTHLY_RENT':rent}
            for month in pd.period_range('2021-01','2023-12',freq='M')
            for key,rent in [('a',1000.),('b',3000.)]])

    def test_exact_quantiles_and_low_history(self):
        source=self.source()
        query=source.iloc[[-2]].copy()
        features=RentShapeHistory().fit(source).transform(query,month_number(query.RENT_APPROVAL_DATE))
        self.assertEqual(features.SHAPE_BLOCK_0_N.iloc[0],35.)
        self.assertEqual(features.SHAPE_BLOCK_0_Q25.iloc[0],.5)
        self.assertEqual(features.SHAPE_BLOCK_0_MEDIAN.iloc[0],.5)
        self.assertEqual(features.SHAPE_BLOCK_0_LOW_SHARE.iloc[0],1.)
        self.assertEqual(features.SHAPE_BLOCK_0_LOW_RECENCY.iloc[0],1.)
        self.assertEqual(features.SHAPE_BLOCK_0_ENTROPY.iloc[0],0.)

    def test_query_and_post_cutoff_rents_do_not_change_features(self):
        source=self.source()
        query=source.iloc[[-2]].copy()
        cutoff=month_number(pd.Series(['2023-03']))
        before=RentShapeHistory().fit(source).transform(query,cutoff)
        source.loc[source.RENT_APPROVAL_DATE>='2023-03','MONTHLY_RENT']=999999.
        query.MONTHLY_RENT=-88888.
        after=RentShapeHistory().fit(source).transform(query,cutoff)
        pd.testing.assert_frame_equal(before,after)

    def test_long_horizon_statistics_stay_frozen_but_recency_ages(self):
        source=self.source()
        query=source.iloc[[-2]].copy()
        cutoff=month_number(pd.Series(['2023-01']))
        query.RENT_APPROVAL_DATE='2024-01'
        before=RentShapeHistory().fit(source).transform(query,cutoff)
        query.RENT_APPROVAL_DATE='2024-02'
        after=RentShapeHistory().fit(source).transform(query,cutoff)
        self.assertEqual(before.SHAPE_BLOCK_0_MEDIAN.iloc[0],after.SHAPE_BLOCK_0_MEDIAN.iloc[0])
        self.assertEqual(before.SHAPE_BLOCK_0_LOW_RECENCY.iloc[0]+1,after.SHAPE_BLOCK_0_LOW_RECENCY.iloc[0])

    def test_unknown_and_coldstart_groups(self):
        source=self.source()
        query=source.iloc[[0]].copy()
        feature=RentShapeHistory().fit(source).transform(query,month_number(query.RENT_APPROVAL_DATE))
        self.assertEqual(feature.SHAPE_BLOCK_0_N.iloc[0],0.)
        self.assertTrue(np.isnan(feature.SHAPE_BLOCK_0_MEDIAN.iloc[0]))

    def buildings(self):
        return pd.DataFrame({'TOWN':['town']*3,'BLOCK':['1','2','3'],'STREET':['street']*3,
            'LATITUDE':[1.3]*3,'LONGITUDE':[103.8,103.801,103.802],
            'MAX_FLOOR':[10.,20.,30.],'YEAR_COMPLETED':[1980.,1990.,2000.]})

    def test_building_context_excludes_self_and_ignores_rent(self):
        buildings=self.buildings()
        query=buildings.iloc[[0]].copy()
        query['MONTHLY_RENT']=1000.
        context=BuildingContext().fit(buildings)
        before=context.transform(query)
        query.MONTHLY_RENT=999999.
        pd.testing.assert_frame_equal(before,context.transform(query))
        self.assertEqual(before.NEIGHBOR_HEIGHT_MEAN.iloc[0],25.)
        self.assertEqual(before['BUILDING_COUNT_0.3KM'].iloc[0],2.)

    def test_missing_coordinates_are_not_zero_distance(self):
        context=BuildingContext().fit(self.buildings())
        query=self.buildings().iloc[[0]].copy()
        query.LATITUDE=np.nan
        features=context.transform(query)
        self.assertTrue(features.isna().all().all())


if __name__=='__main__':
    unittest.main()
