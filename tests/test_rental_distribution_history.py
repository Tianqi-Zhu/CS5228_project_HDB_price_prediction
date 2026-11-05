import unittest
import pandas as pd
from analysis.experiments.rental_distribution_history import distribution_history


class DistributionTests(unittest.TestCase):
    def frames(self):
        f=pd.DataFrame({'RENT_APPROVAL_DATE':['2021-01','2021-02','2021-03'],'TOWN':'a','BLOCK':'1','STREET':'road',
            'FLAT_TYPE_CLEAN':'3-room','FLAT_MODEL_CLEAN':'m','FLOOR_AREA_SQM':70.,'PLANNING_AREA':'area',
            'MONTHLY_RENT':[1000.,2000.,3000.]})
        v=f.iloc[:2].copy();v.RENT_APPROVAL_DATE=['2021-04','2021-07']
        return f,v
    def test_same_month_and_prediction_targets_excluded(self):
        f,v=self.frames();a,b=distribution_history(f,v)
        self.assertEqual(a.loc[1,'DIST_STREET_TYPE_MEDIAN'],1.)
        self.assertEqual(b.DIST_STREET_TYPE_MEDIAN.tolist(),[1.,1.])
        f.loc[2,'MONTHLY_RENT']=99999
        aa,_=distribution_history(f,v)
        pd.testing.assert_series_equal(a.loc[2],aa.loc[2])
        f,v=self.frames();v.MONTHLY_RENT=-99999
        aa,bb=distribution_history(f,v)
        pd.testing.assert_frame_equal(a,aa);pd.testing.assert_frame_equal(b,bb)


if __name__=='__main__':unittest.main()
