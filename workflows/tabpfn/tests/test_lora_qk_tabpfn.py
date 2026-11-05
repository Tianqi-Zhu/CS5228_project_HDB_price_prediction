"""Identity, target selection, and gradient checks for Q/K LoRA."""

import unittest

import torch
from torch import nn

from analysis.experiments.lora_qk_tabpfn import LoRALinear, inject_lora_qk


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_projection = nn.Linear(5, 7, bias=False)
        self.k_projection = nn.Linear(5, 3, bias=False)
        self.v_projection = nn.Linear(5, 3, bias=False)
        self.out_projection = nn.Linear(7, 5, bias=False)

    def forward(self, x):
        return self.q_projection(x), self.k_projection(x)


class LoRAQKTest(unittest.TestCase):
    def test_zero_adapter_preserves_qk_and_freezes_other_weights(self):
        torch.manual_seed(42)
        module = ToyAttention()
        x = torch.randn(2, 4, 5)
        before = tuple(t.detach().clone() for t in module(x))
        audit = inject_lora_qk(module, rank=2, alpha=2)
        self.assertEqual(audit["attention_modules"], 1)
        self.assertEqual(audit["trainable_parameters"], 40)
        self.assertIsInstance(module.q_projection, LoRALinear)
        self.assertIsInstance(module.k_projection, LoRALinear)
        for actual, expected in zip(module(x), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        sum(t.square().sum() for t in module(x)).backward()
        for name, parameter in module.named_parameters():
            if ".lora_b." in name:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            elif ".lora_a." in name:
                self.assertIsNotNone(parameter.grad)
            else:
                self.assertIsNone(parameter.grad)
                self.assertFalse(parameter.requires_grad)
        with self.assertRaises(ValueError):
            inject_lora_qk(module)


if __name__ == "__main__":
    unittest.main()
