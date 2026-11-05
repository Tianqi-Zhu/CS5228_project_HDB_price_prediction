"""Zero-initialized piecewise corrections must receive a training gradient."""
import unittest
import torch
import rtdl_num_embeddings as ne
from analysis.experiments.tabm_rent_search import Model


class NumericGradientTests(unittest.TestCase):
    def test_piecewise_branch_learns(self):
        torch.manual_seed(42)
        x=torch.randn(20,2);categories=torch.randint(0,4,(20,1))
        bins=ne.compute_bins(x,n_bins=4)
        net=Model([4],2,bins,dict(k=4,width=32,blocks=2))
        loss=(net(categories,x)-torch.randn(20,1)).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(net.num.linear.weight.grad).all())
        self.assertGreater(float(net.num.linear.weight.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
