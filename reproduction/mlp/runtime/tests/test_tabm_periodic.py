import unittest
import torch
from analysis.experiments.tabm_rent_search import Model


class PeriodicTests(unittest.TestCase):
    def test_periodic_embeddings_are_trainable(self):
        torch.manual_seed(42)
        for lite in [False,True]:
            net=Model([5],3,None,dict(encoding='periodic',frequency_scale=.1,lite=lite,k=4,width=32,blocks=2))
            c=torch.randint(0,5,(20,1));n=torch.randn(20,3)
            out=net(c,n)
            self.assertEqual(out.shape,(20,4))
            out.square().mean().backward()
            gradients=[p.grad for p in net.num.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)


if __name__=='__main__':unittest.main()
