"""Historical-rental retrieval and a small property-conditioned correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import torch
from torch import nn


def xy_km(frame):
    longitude = frame.LONGITUDE.to_numpy(float)
    latitude = frame.LATITUDE.to_numpy(float)
    return np.column_stack([(longitude-103.85)*111.2*np.cos(np.deg2rad(1.35)),
                            (latitude-1.35)*111.2])


def retrieve(bank, query, origins, target_baseline, *, slots=32, search=128):
    """Return nearest same-type rent tokens available by each forecast origin."""
    if len(query) != len(origins) or len(query) != len(target_baseline):
        raise ValueError("Query, origins and baselines must align")
    bank_month = pd.PeriodIndex(bank._MONTH, freq="M").asi8
    query_month = pd.PeriodIndex(query._MONTH, freq="M").asi8
    origin = pd.PeriodIndex(origins, freq="M").asi8
    if np.any(query_month <= origin):
        raise ValueError("A query cannot contain an observed label at its origin")
    xy_b, xy_q = xy_km(bank), xy_km(query)
    output = np.zeros((len(query), slots, 7), np.float32)
    present = np.zeros((len(query), slots), bool)
    btype = bank.FLAT_TYPE_CLEAN.astype(str).to_numpy()
    qtype = query.FLAT_TYPE_CLEAN.astype(str).to_numpy()
    by_block = bank[["TOWN", "STREET", "BLOCK"]].astype(str).agg("|".join, axis=1).to_numpy()
    q_block = query[["TOWN", "STREET", "BLOCK"]].astype(str).agg("|".join, axis=1).to_numpy()
    bank_area = bank.FLOOR_AREA_SQM.to_numpy(float)
    query_area = query.FLOOR_AREA_SQM.to_numpy(float)
    bank_rent = bank.MONTHLY_RENT.to_numpy(float)
    for kind in np.unique(qtype):
        bpos = np.flatnonzero(btype == kind)
        qpos = np.flatnonzero(qtype == kind)
        if not len(bpos):
            continue
        tree = cKDTree(xy_b[bpos])
        for start in range(0, len(qpos), 1024):
            ids = qpos[start:start+1024]
            distance, neighbors = tree.query(xy_q[ids], k=min(search, len(bpos)))
            distance = np.asarray(distance).reshape(len(ids), -1)
            neighbors = np.asarray(neighbors).reshape(len(ids), -1)
            for j, q in enumerate(ids):
                candidates = bpos[neighbors[j]]
                mask = (bank_month[candidates] <= origin[q]) & (distance[j] <= 5.)
                candidates = candidates[mask][:slots]
                m = len(candidates)
                if not m:
                    continue
                displacement = xy_b[candidates]-xy_q[q]
                output[q, :m, 0] = (bank_rent[candidates]-target_baseline[q])/1000.
                output[q, :m, 1:3] = displacement/5.
                output[q, :m, 3] = (origin[q]-bank_month[candidates])/12.
                output[q, :m, 4] = (bank_area[candidates]-query_area[q])/20.
                output[q, :m, 5] = (by_block[candidates] == q_block[q]).astype(float)
                output[q, :m, 6] = (bank_month[candidates] == origin[q]).astype(float)
                present[q, :m] = True
    if np.any(present.sum(1) > slots):
        raise AssertionError("Too many comparables")
    if not np.isfinite(output).all():
        raise ValueError("Nonfinite comparable tokens")
    return output, present


class ComparableAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 32))
        self.context = nn.Linear(4, 32)
        self.token = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 32))
        self.null_token = nn.Parameter(torch.zeros(1, 1, 32))
        self.attention = nn.MultiheadAttention(32, 2, dropout=.1, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 1))
        self.gate_logit = nn.Parameter(torch.tensor(-2.2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, property_cls, context, comparables, present):
        q = (self.query(property_cls)+self.context(context))[:, None]
        tokens = torch.cat([self.token(comparables), self.null_token.expand(len(q), -1, -1)], 1)
        mask = torch.cat([~present, torch.zeros((len(q), 1), dtype=torch.bool,
                                               device=present.device)], 1)
        attended, _ = self.attention(q, tokens, tokens, key_padding_mask=mask,
                                     need_weights=False)
        return self.gate_logit.sigmoid()*self.head(attended[:, 0]).squeeze(-1)
