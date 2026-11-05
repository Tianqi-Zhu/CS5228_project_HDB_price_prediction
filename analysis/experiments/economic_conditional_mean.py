"""Joint property transformer and temporal conditional-mean representation."""
import math

import torch
from torch import nn

from analysis.experiments.spatial_temporal_transformer import (
    Experiment, RentalTransformer, SEQUENCE_CHANNELS,
)


class EconomicConditionalMean(RentalTransformer):
    """A temporal mean shift plus eight property-dependent temporal loadings.

    Inputs are available histories, not precomputed neural forecasts. The final
    output is a standardized residual; the runner restores the original rent
    baseline exactly once. All parameters train under individual-rental MSE.
    """

    def __init__(self, n_numeric, cardinalities, spec=None):
        super().__init__(n_numeric, cardinalities, Experiment("T0", "none", "none"))
        self.spec = spec or Experiment("conditional", "rent_coe", "lstm")
        self.economic_encoder = nn.LSTM(2 * len(SEQUENCE_CHANNELS), 8, batch_first=True)
        self.property_loading = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 8))
        self.mean_parameters = nn.Sequential(nn.Linear(12, 16), nn.GELU(), nn.Linear(16, 9))
        self.gate = nn.Parameter(torch.tensor(-2.2))
        nn.init.zeros_(self.mean_parameters[-1].weight)
        nn.init.zeros_(self.mean_parameters[-1].bias)

    def components(self, numeric, missing, categorical, spatial, sequence, context):
        tokens = numeric[:, :, None] * self.numeric_weight + self.numeric_bias + missing[:, :, None] * self.missing_weight
        cats = [module(categorical[:, i])[:, None] for i, module in enumerate(self.categories)]
        tokens = torch.cat([self.cls.expand(len(numeric), -1, -1), tokens, *cats,
                            self.spatial(spatial)[:, None]], dim=1)
        property_state = self.property_encoder(tokens)[:, 0]
        property_mean = self.property_head(property_state).squeeze(-1)
        _, (hidden, _) = self.economic_encoder(sequence)
        parameters = self.mean_parameters(torch.cat([hidden[-1], context], dim=1))
        shift = parameters[:, 0]
        interaction = (parameters[:, 1:] * self.property_loading(property_state)).sum(dim=1) / math.sqrt(8)
        return property_mean, self.gate.sigmoid() * shift, self.gate.sigmoid() * interaction

    def forward(self, *inputs):
        property_mean, shift, interaction = self.components(*inputs)
        # Use the same FP32 sum as the supervised training loss under BF16.
        return property_mean.float() + shift.float() + interaction.float()
