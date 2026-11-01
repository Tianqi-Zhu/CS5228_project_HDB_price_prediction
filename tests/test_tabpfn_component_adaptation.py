"""Stage selection, identity, frozen weights and saved component verification."""

import copy
import unittest

import torch
from torch import nn

from analysis.experiments.tabpfn_component_adaptation import (
    configure_component, copy_component, frozen_digest,
)
from analysis.experiments.lora_qk_tabpfn import LoRALinear


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_projection = nn.Linear(4, 4)
        self.k_projection = nn.Linear(4, 4)
        self.v_projection = nn.Linear(4, 4)

    def forward(self, x):
        return self.q_projection(x) + self.k_projection(x) + self.v_projection(x)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_distribution_embedder = Attention()
        self.icl_blocks = nn.ModuleList([nn.Module() for _ in range(3)])
        for block in self.icl_blocks:
            block.icl_attention = Attention()
        self.heads = nn.Module()
        self.heads.mlp_regression = nn.Linear(4, 4)
        self.heads.output_projection = nn.Linear(4, 2)
        self.heads.mlp_classification = nn.Linear(4, 3)

    def forward(self, x):
        x = self.feature_distribution_embedder(x)
        for block in self.icl_blocks:
            x = block.icl_attention(x)
        return self.heads.output_projection(self.heads.mlp_regression(x))


class ComponentTests(unittest.TestCase):
    def test_icl_only_zero_identity_and_frozen_backbone(self):
        torch.manual_seed(42)
        model = Toy()
        x = torch.randn(5, 4)
        expected = model(x).detach()
        audit = configure_component(model, {"component": "icl_lora", "rank": 2})
        self.assertEqual(len(audit["targets"]), 6)
        self.assertTrue(all(n.startswith("icl_blocks.") for n in audit["trainable_names"]))
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        before = frozen_digest(model)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        model(x).square().mean().backward()
        opt.step()
        self.assertEqual(before, frozen_digest(model))
        self.assertFalse(torch.equal(model(x), expected))
        with self.assertRaises(ValueError):
            configure_component(model, {"component": "icl_lora", "rank": 2})

    def test_last_blocks_only(self):
        model = Toy()
        audit = configure_component(model, {"component": "icl_lora", "rank": 4, "last_blocks": 1})
        self.assertEqual(len(audit["targets"]), 2)
        self.assertTrue(all(n.startswith("icl_blocks.2.") for n in audit["trainable_names"]))
        self.assertNotIsInstance(model.icl_blocks[0].icl_attention.q_projection, LoRALinear)

    def test_head_identity_schema_reload_and_gradients(self):
        torch.manual_seed(42)
        model = Toy()
        clone = copy.deepcopy(model)
        x = torch.randn(5, 4)
        expected = model(x).detach()
        cfg = {"component": "head"}
        audit = configure_component(model, cfg)
        configure_component(clone, cfg)
        torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
        self.assertTrue(all(n.startswith(("heads.mlp_regression.", "heads.output_projection."))
                            for n in audit["trainable_names"]))
        before = frozen_digest(model)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        model(x).square().mean().backward()
        opt.step()
        self.assertEqual(before, frozen_digest(model))
        state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        copy_component(clone, state)
        torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            copy_component(clone, {"wrong": torch.ones(1)})


if __name__ == "__main__":
    unittest.main()
