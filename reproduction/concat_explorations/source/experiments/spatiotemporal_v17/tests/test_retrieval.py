import unittest
import numpy as np
from experiments.spatiotemporal_v17.retrieval import retrieve
class RetrievalTest(unittest.TestCase):
    def test_legal_deduplicated_and_query_independent_of_target(self):
        fit=dict(row_id=np.arange(8),month=np.arange(1,9),flat_type=np.array(['a']*8),lat=np.zeros(8),lon=np.arange(8)*.01)
        query=dict(row_id=3,origin=5,flat_type='a',lat=0.,lon=.03)
        ids=retrieve(fit,query,4,seed=42)
        self.assertEqual(len(ids),4)
        self.assertEqual(len(set(ids)),4)
        self.assertNotIn(3,ids)
        self.assertTrue(all(fit['month'][i]<=5 for i in ids))
        query['y_sgd']=1e10
        np.testing.assert_array_equal(ids,retrieve(fit,query,4,seed=42))
