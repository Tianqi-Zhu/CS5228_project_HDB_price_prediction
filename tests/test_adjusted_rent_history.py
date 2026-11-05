import unittest
import pandas as pd
from analysis.experiments.adjusted_rent_history import adjusted_history


class AdjustedHistoryTests(unittest.TestCase):
    def frames(self):
        f=pd.DataFrame({'RENT_APPROVAL_DATE':['2021-01','2021-02','2021-03'],'TOWN':'a','BLOCK':'1','STREET':'road',
            'FLAT_TYPE_CLEAN':'3-room','FLAT_MODEL_CLEAN':'m','FLOOR_AREA_SQM':70.,'LEASE_COMMENCE_DATE':2000,
            'PLANNING_AREA':'area','MONTHLY_RENT':[1000.,2000.,3000.]})
        v=f.iloc[:2].copy();v.RENT_APPROVAL_DATE=['2021-04','2021-08']
        return f,v
    def test_known_values_and_frozen_origin(self):
        f,v=self.frames();a,b=adjusted_history(f,v)
        self.assertTrue(pd.isna(a.loc[0,'ADJ_HIST_SIGNATURE_MEAN']))
        self.assertEqual(a.loc[1,'ADJ_HIST_SIGNATURE_MEAN'],1000.)
        self.assertEqual(a.loc[2,'ADJ_HIST_SIGNATURE_MEAN'],2000.)
        self.assertEqual(b.ADJ_HIST_SIGNATURE_MEAN.tolist(),[3000.,3000.])
        v.MONTHLY_RENT=-99999
        aa,bb=adjusted_history(f,v)
        pd.testing.assert_frame_equal(a,aa);pd.testing.assert_frame_equal(b,bb)
    def test_same_month_label_excluded(self):
        f,v=self.frames();a,_=adjusted_history(f,v)
        f.loc[2,'MONTHLY_RENT']=99999
        b,_=adjusted_history(f,v)
        pd.testing.assert_series_equal(a.loc[2],b.loc[2])


if __name__=='__main__':unittest.main()
