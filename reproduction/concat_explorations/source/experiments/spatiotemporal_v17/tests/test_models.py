import unittest
import torch
from experiments.spatiotemporal_v17.models import TemporalFT

class ModelsTest(unittest.TestCase):
    def test_masked_values_and_empty_histories_are_safe(self):
        torch.manual_seed(1)
        for mode in ('concat','cross'):
            m=TemporalFT(['x','z'],(3,),4,mode=mode,dropout=0.)
            b={'numeric':torch.randn(3,2),'categorical':torch.ones(3,1,dtype=torch.long),'rent_sequence':torch.randn(3,12,4),'sequence_mask':torch.rand(3,12,4)>.4}
            b['sequence_mask'][0]=False
            p=m(b); b2=dict(b); b2['rent_sequence']=torch.where(b['sequence_mask'],b['rent_sequence'],torch.full_like(b['rent_sequence'],999999))
            self.assertTrue(torch.isfinite(p).all()); torch.testing.assert_close(p,m(b2)); p.square().mean().backward()
            self.assertTrue(all(torch.isfinite(x.grad).all() for x in m.parameters() if x.grad is not None))
    def test_modes_have_matched_input_contract(self):
        m=TemporalFT(['x'],(2,),4,mode='concat')
        b={'numeric':torch.ones(2,1),'categorical':torch.ones(2,1,dtype=torch.long),'rent_sequence':torch.ones(2,11,4),'sequence_mask':torch.ones(2,11,4,dtype=torch.bool)}
        with self.assertRaises(ValueError): m(b)
