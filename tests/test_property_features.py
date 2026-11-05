import unittest
import pandas as pd
import numpy as np
from analysis.experiments.property_features import augment


class PropertyFeaturesTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'RENT_APPROVAL_DATE':['2024-01','2024-12'],'LEASE_COMMENCE_DATE':[1980,2026],
            'YEAR_COMPLETED':[1979,2000],'FLOOR_AREA_SQM':[70.,100.],'FLAT_AGE_YEARS':[44.,np.nan],
            'NEAREST_MRT_KM':[.5,1.],'TOWN_CLEAN':['a','a'],'FLAT_TYPE_CLEAN':['3-room','4-room'],
            'STREET_ID':['a|s','a|s'],'BUILDING_ID':['a|s|1','a|s|2'],'MONTHLY_RENT':[2000,4000]})
    def test_features_ignore_target_and_handle_invalid_lease(self):
        f=self.frame();a=augment(f)
        self.assertTrue(pd.isna(a.loc[1,'XNUM_LEASE_YEAR']))
        self.assertEqual(a.loc[0,'XNUM_LEASE_COMPLETION_GAP'],1)
        f.MONTHLY_RENT=-999999
        b=augment(f)
        columns=[c for c in a if c.startswith(('XNUM_','XCAT_'))]
        pd.testing.assert_frame_equal(a[columns],b[columns])
        self.assertNotEqual(a.loc[0,'XCAT_BUILDING_TYPE'],a.loc[1,'XCAT_BUILDING_TYPE'])


if __name__=='__main__':unittest.main()
