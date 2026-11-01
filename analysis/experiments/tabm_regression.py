"""Training-only preprocessing and official TabM regression helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import tabm
import rtdl_num_embeddings


class TabMPreprocessor:
    """Median imputation, missingness, standardization, and categorical vocabularies.

    All statistics and category levels are learned from fitting rows only.
    The source features are unchanged; neural representations encode missingness.
    """

    def fit(self, frame):
        self.columns = list(frame)
        self.categorical = [c for c in frame if isinstance(frame[c].dtype, pd.CategoricalDtype)]
        self.numerical = [c for c in frame if c not in self.categorical]
        raw = frame[self.numerical].to_numpy(float)
        if np.isinf(raw).any():
            raise ValueError("Infinite numerical input")
        self.median = np.asarray([np.nanmedian(v) if np.isfinite(v).any() else 0.0 for v in raw.T])
        numeric = np.column_stack([np.where(np.isnan(raw), self.median, raw), np.isnan(raw).astype(float)])
        self.mean = numeric.mean(axis=0)
        self.std = numeric.std(axis=0)
        self.keep = self.std > 1e-12
        self.vocabularies = {}
        for c in self.categorical:
            known = frame[c].astype("string").dropna().unique().tolist()
            self.vocabularies[c] = {v: i + 1 for i, v in enumerate(known)}
        self.cardinalities = [len(self.vocabularies[c]) + 1 for c in self.categorical]
        return self

    def transform(self, frame):
        if list(frame) != self.columns:
            raise ValueError("Source feature order changed")
        raw = frame[self.numerical].to_numpy(float)
        if np.isinf(raw).any():
            raise ValueError("Infinite numerical input")
        numeric = np.column_stack([np.where(np.isnan(raw), self.median, raw), np.isnan(raw).astype(float)])
        numeric = ((numeric[:, self.keep]-self.mean[self.keep])/self.std[self.keep]).astype(np.float32)
        categorical = np.column_stack([
            frame[c].astype("string").map(self.vocabularies[c]).fillna(0).to_numpy(np.int64)
            for c in self.categorical])
        if not np.isfinite(numeric).all():
            raise ValueError("Nonfinite transformed input")
        return numeric, categorical


def make_model(config, numeric, cardinalities, bins=None):
    if config["embedding"] == "ple":
        if bins is None:
            bins = rtdl_num_embeddings.compute_bins(torch.as_tensor(numeric), n_bins=48)
        embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins, d_embedding=16, activation=False, version="B")
    else:
        embeddings = None
    model = tabm.TabM.make(n_num_features=numeric.shape[1], cat_cardinalities=cardinalities,
                           num_embeddings=embeddings, d_out=1, k=config["k"],
                           n_blocks=3, d_block=256, dropout=.1, arch_type="tabm")
    return model, bins


def member_mse(prediction, target):
    """Train each internal member, not just the average of their predictions."""
    if prediction.ndim != 3 or prediction.shape[-1] != 1 or len(prediction) != len(target):
        raise ValueError("Unexpected TabM regression shapes")
    return (prediction.squeeze(-1).float()-target[:, None].float()).square().mean()


@torch.inference_mode()
def predict(model, numeric, categorical, target_mean, target_std, batch=256):
    device = next(model.parameters()).device
    model.eval()
    results = []
    for start in range(0, len(numeric), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            values = model(numeric[start:start+batch], categorical[start:start+batch])
        values = values.squeeze(-1).float().mean(dim=1)
        results.append(values.cpu().numpy().astype(float)*target_std+target_mean)
    result = np.concatenate(results)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite predictions")
    return result
