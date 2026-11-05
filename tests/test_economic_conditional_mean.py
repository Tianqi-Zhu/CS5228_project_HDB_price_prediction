import unittest

import numpy as np
import torch

from analysis.experiments.economic_conditional_mean import EconomicConditionalMean
from analysis.experiments.spatial_temporal_transformer import SequencePreprocessor


class EconomicConditionalMeanTests(unittest.TestCase):
    def inputs(self):
        return (torch.randn(4, 3), torch.zeros(4, 3), torch.zeros(4, 1, dtype=torch.long),
                torch.randn(4, 30), torch.randn(4, 12, 34), torch.randn(4, 4))

    def test_initial_conditioning_is_zero_and_finite_for_missing_history(self):
        model = EconomicConditionalMean(3, [2]).eval()
        inputs = list(self.inputs())
        inputs[4].zero_()
        with torch.no_grad():
            base, shift, interaction = model.components(*inputs)
            output = model(*inputs)
        self.assertTrue(torch.equal(shift, torch.zeros_like(shift)))
        self.assertTrue(torch.equal(interaction, torch.zeros_like(interaction)))
        self.assertTrue(torch.equal(output, base))
        self.assertTrue(torch.isfinite(output).all())

    def test_loss_backpropagates_into_economic_and_property_encoders(self):
        model = EconomicConditionalMean(3, [2]).eval()
        # One zero-initialized-head update precedes representation gradients.
        torch.nn.init.normal_(model.mean_parameters[-1].weight, std=.1)
        model(*self.inputs()).square().mean().backward()
        for parameter in (model.economic_encoder.weight_ih_l0,
                          model.property_encoder.layers[0].linear1.weight,
                          model.property_loading[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_stock_channels_masked_and_only_coe_changes_information(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(5, 12, 17)).astype(np.float32)
        pre = SequencePreprocessor().fit(raw)
        rent, coe = [pre.transform(raw, name) for name in ("rent", "rent_coe")]
        np.testing.assert_array_equal(rent[:, :, :12], coe[:, :, :12])
        np.testing.assert_array_equal(rent[:, :, 17:29], coe[:, :, 17:29])
        for values in (rent, coe):
            np.testing.assert_array_equal(values[:, :, [16, 33]], 0.)
        np.testing.assert_array_equal(rent[:, :, 12:16], 0.)
        self.assertGreater(float(abs(coe[:, :, 12:16]).sum()), 0.)


if __name__ == "__main__":
    unittest.main()
