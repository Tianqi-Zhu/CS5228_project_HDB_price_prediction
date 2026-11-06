import unittest
import numpy as np
import pandas as pd
from analysis.experiments.cleaned_neural_ensemble import training_cleaning_audit


class RentCleaningTests(unittest.TestCase):
    def rows(self, rents, dates=None):
        return pd.DataFrame({'RENT_APPROVAL_DATE':dates or ['2023-06']*len(rents),
                             'TOWN':'town','BLOCK':'1','STREET':'road','FLAT_TYPE':'5 room',
                             'FLAT_MODEL':'improved','FLOOR_AREA_SQM':110.,'LEASE_COMMENCE_DATE':2002,
                             'MONTHLY_RENT':rents},index=np.arange(40,40+len(rents)))

    def test_obvious_policy_and_strict_policy_differ_as_declared(self):
        frame=self.rows([1500,5000,5200]);before=frame.copy(deep=True)
        ordinary=training_cleaning_audit(frame,'obvious_low')
        strict=training_cleaning_audit(frame,'highest_only')
        self.assertEqual(ordinary['remove'].tolist(),[True,False,False])
        self.assertEqual(strict['remove'].tolist(),[True,True,False])
        pd.testing.assert_frame_equal(frame,before)

    def test_different_months_and_flat_types_are_not_compared(self):
        frame=self.rows([2000,6000],['2021-01','2023-06'])
        self.assertFalse(training_cleaning_audit(frame,'highest_only')['remove'].any())
        frame=self.rows([2000,6000]);frame.loc[40,'FLAT_TYPE']='3-room'
        self.assertFalse(training_cleaning_audit(frame,'highest_only')['remove'].any())

    def test_lower_labels_without_extreme_gap_are_kept_in_obvious_policy(self):
        audit=training_cleaning_audit(self.rows([4000,5000,6000]),'obvious_low')
        self.assertFalse(audit['remove'].any())

    def test_maximum_ties_are_preserved_and_normalised_keys_match(self):
        frame=self.rows([2500,4000,4000]);frame.loc[40,'STREET']=' Road '
        frame.loc[41,'FLAT_TYPE']='5-room'
        self.assertEqual(training_cleaning_audit(frame,'highest_only')['remove'].tolist(),[True,False,False])

    def test_absolute_floor_requires_large_area(self):
        frame=self.rows([1000,1000],['2023-05','2023-06']);frame.loc[40,'FLOOR_AREA_SQM']=40.
        self.assertEqual(training_cleaning_audit(frame,'obvious_low')['remove'].tolist(),[False,True])

    def test_external_future_labels_cannot_change_training_cleaning(self):
        train=self.rows([2000,4000,4000]);future=self.rows([999999],['2025-01'])
        first=training_cleaning_audit(train,'highest_only')
        future.MONTHLY_RENT=-999999
        second=training_cleaning_audit(train,'highest_only')
        pd.testing.assert_frame_equal(first,second)
        self.assertTrue(first.index.equals(train.index))

    def test_same_month_local_reference_requires_twenty_examples(self):
        frame=self.rows([1600]+[4000]*19)
        # Give each property a distinct block, so only the town/month rule triggers.
        frame['BLOCK']=[str(i) for i in range(20)]
        audit=training_cleaning_audit(frame,'obvious_low')
        self.assertTrue(audit.loc[40,'local_low_flag'])
        self.assertFalse(training_cleaning_audit(frame.iloc[:19],'obvious_low').loc[40,'local_low_flag'])


if __name__=='__main__':unittest.main()
