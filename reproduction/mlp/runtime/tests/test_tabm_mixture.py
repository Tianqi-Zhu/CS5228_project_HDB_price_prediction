import unittest
import torch
from analysis.experiments.tabm_rent_search import Model


class MixtureTests(unittest.TestCase):
    def test_input_only_gate_and_all_heads_receive_gradients(self):
        torch.manual_seed(42)
        net=Model([5],2,None,dict(k=4,width=32,blocks=2,mixture=True))
        cats=torch.randint(0,5,(16,1));nums=torch.randn(16,2)
        out=net(cats,nums)
        self.assertEqual(out.shape,(16,4))
        torch.testing.assert_close(net.gate_logits.softmax(-1).sum(-1),torch.ones(16,4))
        out.square().mean().backward()
        grad=net.backbone.output.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(),0)
        net.eval()
        with torch.no_grad():
            a=net(cats,nums);b=net(cats,nums)
        torch.testing.assert_close(a,b)


if __name__=='__main__':unittest.main()
