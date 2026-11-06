import unittest
from unittest.mock import patch
import numpy as np,pandas as pd,torch
from analysis.experiments.tabm_rent_search import fit


class EpochSelectionTests(unittest.TestCase):
    def test_mixture_auxiliary_loss_does_not_overwrite_epoch(self):
        torch.set_num_threads(1)
        rng=np.random.default_rng(7)
        f=pd.DataFrame({'MONTHLY_RENT':np.linspace(1000,4000,20)})
        v=pd.DataFrame({'MONTHLY_RENT':np.linspace(1500,3500,8)})
        arrays=[torch.zeros(20,1,dtype=torch.long),torch.tensor(rng.normal(size=(20,2)),dtype=torch.float32),
                torch.zeros(8,1,dtype=torch.long),torch.tensor(rng.normal(size=(8,2)),dtype=torch.float32)]
        meta={'vocab':{'c':['a']},'categories':['c']}
        spec=dict(encoding='none',mixture=True,k=4,width=32,blocks=2,lr=.001)
        with patch('analysis.experiments.tabm_rent_search.prepare',return_value=(arrays,meta)):
            _,_,selected,_,_,pred=fit(f,v,spec,epochs=2)
        self.assertIsInstance(selected,int)
        self.assertGreater(selected,0)
        self.assertEqual(pred.shape,(8,))
        self.assertTrue(np.isfinite(pred).all())


if __name__=='__main__':unittest.main()
