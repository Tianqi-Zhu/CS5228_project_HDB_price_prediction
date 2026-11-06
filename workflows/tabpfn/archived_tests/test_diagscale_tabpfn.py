"""Identity and gradient checks for the TabPFN-3.5 DiagScale adapter."""

import unittest

import torch
from torch import nn

from tabpfn.architectures.tabpfn_v3_5 import SoftmaxScalingMLP

from analysis.experiments.diagscale_tabpfn import DiagScale, inject_diagscale


class DiagScaleTest(unittest.TestCase):
    def test_zero_adapter_preserves_scaling_and_only_delta_trains(self):
        torch.manual_seed(42)
        module = nn.Module()
        module.layer = SoftmaxScalingMLP(num_heads=2, head_dim=4)
        q = torch.randn(1, 3, 2, 4)
        before = module.layer(q, 20).detach().clone()
        audit = inject_diagscale(module)
        self.assertEqual(audit["scaling_modules"], 1)
        self.assertEqual(audit["trainable_parameters"], 8)
        self.assertIsInstance(module.layer, DiagScale)
        torch.testing.assert_close(module.layer(q, 20), before, rtol=0, atol=0)
        module.layer(q, 20).sum().backward()
        self.assertIsNotNone(module.layer.delta.grad)
        self.assertTrue(torch.isfinite(module.layer.delta.grad).all())
        self.assertTrue(all(p.grad is None for p in module.layer.base.parameters()))
        with self.assertRaises(ValueError):
            inject_diagscale(module)


if __name__ == "__main__":
    unittest.main()
