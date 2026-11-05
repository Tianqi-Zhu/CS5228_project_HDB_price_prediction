import unittest
import numpy as np
from run import align,to_sgd

class ContractTest(unittest.TestCase):
    def setUp(self):
        self.h={'row_id':np.array([11,12]),'actual_sgd':np.array([1000.,2000.]),'horizon':np.array([1,2])}
        self.f={'row_id':np.array([12,11]),'actual_sgd':np.array([2000.,1000.]),'horizon':np.array([2,1])}
    def test_reorder_and_bad_ids(self):
        np.testing.assert_array_equal(align(self.h,self.f),[1,0])
        self.f['row_id'][0]=99
        with self.assertRaises(ValueError): align(self.h,self.f)
    def test_duplicate(self):
        self.f['row_id'][:]=11
        with self.assertRaises(ValueError): align(self.h,self.f)
    def test_label_and_horizon(self):
        for key in ('actual_sgd','horizon'):
            changed={k:v.copy() for k,v in self.f.items()}; changed[key][0]+=1
            with self.assertRaises(ValueError): align(self.h,changed)
    def test_units(self): np.testing.assert_allclose(to_sgd(np.array([.25,-.5]),np.array([2.,3.])),[2250.,2500.])
if __name__=='__main__': unittest.main()
