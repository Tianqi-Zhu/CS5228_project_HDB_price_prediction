"""Small rent-history corrections on a frozen, previously fitted T0 backbone."""
from __future__ import annotations

import torch
from torch import nn


class RentCorrection(nn.Module):
    def __init__(self, *, conditioned: bool, stabilized: bool, window: int = 12):
        super().__init__()
        self.conditioned = conditioned
        self.stabilized = stabilized
        self.input = nn.Linear(34, 32)
        self.position = nn.Parameter(torch.randn(1, window, 32) * .02)
        self.context = nn.Linear(4, 32)
        if conditioned:
            self.property_query = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 32))
        layer = nn.TransformerEncoderLayer(32, 2, 64, .1, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 1))
        self.gate_logit = nn.Parameter(torch.tensor(-2.2))
        if stabilized:
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
        else:
            nn.init.normal_(self.head[-1].weight, std=.001)

    def forward(self, property_cls, sequence, context):
        query = self.context(context)
        if self.conditioned:
            query = query + self.property_query(property_cls)
        tokens = torch.cat([self.input(sequence) + self.position, query[:, None]], dim=1)
        correction = self.gate_logit.sigmoid() * self.head(self.encoder(tokens)[:, -1]).squeeze(-1)
        return correction

    def regularizer(self, correction):
        if self.stabilized:
            return .01 * correction.square().mean()
        return .01 * self.gate_logit.sigmoid().square()
