"""Scientific contracts for paired/clustered appendix diagnostics."""
import unittest
import numpy as np
import pandas as pd
from analysis.appendix_statistics import (aggregate_errors,differences,moving_month_indices,
    paired_intervals,validate_identity,blend_moments)

class AppendixStatisticsTests(unittest.TestCase):
    def test_alignment_rejects_shuffled_predictions_and_duplicate_source_ids(self):
        raw=pd.DataFrame({'MONTHLY_RENT':[100.,200.,300.], 'RENT_APPROVAL_DATE':['2024-06','2024-07','2024-08']})
        validate_identity(np.arange(3),np.array([100.,200.,300.]),raw.RENT_APPROVAL_DATE.to_numpy(),raw)
        with self.assertRaises(ValueError):validate_identity(np.arange(3),np.array([200.,100.,300.]),raw.RENT_APPROVAL_DATE.to_numpy(),raw)
        with self.assertRaises(ValueError):validate_identity(np.array([0,0]),np.array([100.,100.]),np.array(['2024-06']*2),raw)

    def test_cluster_multiplicity_matches_explicitly_repeated_rows(self):
        y=np.zeros(4);candidate=np.array([1.,2.,3.,9.]);reference=np.array([2.,4.,6.,1.])
        _,stats=aggregate_errors(y,candidate,reference,['A','A','A','B'])
        # Draw A twice and B once: all three A rows appear twice, not once.
        expected_ids=[0,1,2,0,1,2,3]
        c=candidate[expected_ids];r=reference[expected_ids]
        expected=[np.sqrt(np.mean(c*c))-np.sqrt(np.mean(r*r)),np.mean(abs(c))-np.mean(abs(r))]
        np.testing.assert_allclose(differences(stats[[0,0,1]].sum(axis=0)),expected)

    def test_moving_blocks_do_not_wrap_at_end_of_ten_month_window(self):
        draws=moving_month_indices(10,repeats=300,block=3,seed=42)
        self.assertEqual(draws.shape,(300,10))
        self.assertTrue(((draws>=0)&(draws<10)).all())
        for start in [0,3,6]:np.testing.assert_array_equal(np.diff(draws[:,start:start+3]),np.ones((300,2)))
        self.assertTrue((draws[:,:9].reshape(-1,3)[:,0]<=7).all())
        np.testing.assert_array_equal(draws,moving_month_indices(10,300,3,42))

    def test_identical_pipelines_have_zero_intervals_under_both_cluster_schemes(self):
        actual=np.arange(10.);pred=actual+np.sin(actual)
        for method in ['moving_month_block','building_cluster']:
            rows=paired_intervals(actual,pred,pred,np.arange(10),method,repeats=40)
            for row in rows:
                self.assertEqual(row['candidate_minus_reference'],0)
                self.assertEqual(row['p025'],0)
                self.assertEqual(row['p975'],0)

    def test_cluster_intervals_are_invariant_to_row_order(self):
        y=np.arange(9.);c=y+np.sin(y);r=y+.5;groups=np.array(['A','B','C']*3)
        a=paired_intervals(y,c,r,groups,'building_cluster',repeats=50)
        idx=np.array([8,2,5,1,7,4,3,0,6])
        b=paired_intervals(y[idx],c[idx],r[idx],groups[idx],'building_cluster',repeats=50)
        for x,z in zip(a,b):np.testing.assert_allclose([x['p025'],x['p975']],[z['p025'],z['p975']],atol=1e-12)

    def test_blend_identity_does_not_assume_low_correlation_or_improvement(self):
        result=blend_moments(np.zeros(3),np.ones(3),np.array([5.,6.,7.]))
        self.assertAlmostEqual(result['observed_blend_mse'],result['reconstructed_blend_mse'])
        self.assertGreater(result['observed_blend_mse'],result['tree_mse'])
        self.assertFalse(result['weight_search_performed'])

    def test_invalid_inputs_and_unknown_resampling_are_rejected(self):
        with self.assertRaises(ValueError):aggregate_errors([1],[np.nan],[1],['A'])
        with self.assertRaises(ValueError):moving_month_indices(2,30,3)
        with self.assertRaises(ValueError):paired_intervals([1],[2],[3],['A'],'rows',repeats=3)

if __name__=='__main__':unittest.main()
