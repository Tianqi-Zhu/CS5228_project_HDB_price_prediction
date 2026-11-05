"""Comparable retrieval must exclude current/future labels at each origin."""
import unittest
import numpy as np
import pandas as pd
from analysis.experiments.historical_neighbors import comparable_features


class HistoricalNeighborsTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'RENT_APPROVAL_DATE':pd.period_range('2021-01',periods=8,freq='M').astype(str),
            'LATITUDE':1.3,'LONGITUDE':103.8,'FLOOR_AREA_SQM':70.,'FLAT_AGE_YEARS':20.,
            'FLAT_TYPE_CLEAN':'3-room','MONTHLY_RENT':np.arange(8)*100.+2000})

    def test_current_label_does_not_enter_own_features(self):
        a=self.frame();original=comparable_features(a,'2021-07',k=3)
        a.loc[2,'MONTHLY_RENT']=99999
        changed=comparable_features(a,'2021-07',k=3)
        pd.testing.assert_series_equal(original.loc[2],changed.loc[2])
        self.assertNotEqual(original.loc[3,'COMP_MEAN'],changed.loc[3,'COMP_MEAN'])

    def test_validation_labels_and_more_future_rows_cannot_change_features(self):
        a=self.frame();original=comparable_features(a,'2021-05',k=3)
        a.loc[4:,'MONTHLY_RENT']=-99999
        changed=comparable_features(a,'2021-05',k=3)
        pd.testing.assert_frame_equal(original,changed)
        self.assertTrue(original.iloc[0].isna().all())
        self.assertTrue(np.isfinite(original.iloc[1:].values).all())
        self.assertEqual(original.loc[1,'COMP_COUNT'],1)


if __name__=='__main__':unittest.main()
