import unittest,importlib.util
class ConcatIntegration(unittest.TestCase):
 def test_entry_exists(self):
  self.assertIsNotNone(importlib.util.find_spec('analysis.experiments.concat_transformer'))
 def test_model_gradient(self):
  from analysis.experiments.concat_transformer import runtime_path
  import sys,torch
  sys.path.insert(0,str(runtime_path()))
  from experiments.concat_capacity_v21.model import make_model
  torch.set_num_threads(2)
  net=make_model(['HIST_A','ANN_BLOCK_24_R1_RATIO'],[3],128,6)
  y=net({'numeric':torch.randn(2,2),'categorical':torch.zeros(2,1,dtype=torch.long)})
  y.square().mean().backward();self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
if __name__=='__main__':unittest.main()
