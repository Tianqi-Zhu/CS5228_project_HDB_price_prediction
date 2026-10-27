"""Frozen TabPFN query-embedding heads and chronological evaluation helpers."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC = (
    "FLOOR_AREA_SQM", "FLAT_AGE_YEARS", "LATITUDE", "LONGITUDE",
    "NEAREST_MRT_KM", "NEAREST_MALL_KM", "NEAREST_SCHOOL_KM", "DISTANCE_TO_CBD_KM",
    "HIST_SIGNATURE_MEAN", "HIST_SIGNATURE_COUNT", "HIST_BLOCK_FLAT_COUNT",
    "RECENT_BLOCK_FLAT_3M_SHRUNK_MEAN", "RECENT_AREA_FLAT_3M_MEAN",
    "RECENT_AREA_FLAT_12M_MEAN", "TEMP_LOCAL_FORECAST", "TEMP_LOCAL_WEIGHT",
    "TEMP_FORECAST_RMSE", "TEMP_FORECAST_HORIZON", "APPROVAL_MONTH_INDEX",
)


def numeric_features(frame: pd.DataFrame, months, origins, base_prediction):
    """Only query-known values; no label or observed future rents."""
    periods = pd.PeriodIndex(months, freq="M")
    origin_periods = pd.PeriodIndex(origins, freq="M")
    if len(frame) != len(periods) or len(frame) != len(base_prediction):
        raise ValueError("Numeric head features and rows differ")
    values = frame[list(NUMERIC)].to_numpy(float)
    angle = 2 * np.pi * (periods.month.to_numpy() - 1) / 12
    extra = np.column_stack((np.asarray(base_prediction, float),
                             origin_periods.asi8 - pd.Period("2021-01", "M").ordinal,
                             np.sin(angle), np.cos(angle)))
    return np.column_stack((values, extra))


def population_weights(months, origins, episode_weights):
    """Count each target month once even when multiple origins forecast it."""
    months, origins = np.asarray(months), np.asarray(origins)
    result = np.asarray(episode_weights, float).copy()
    if len(months) != len(origins) or len(months) != len(result):
        raise ValueError("Head sample metadata lengths differ")
    for month in np.unique(months):
        mask = months == month
        # Each episode's 200 sampled rows represent the same month population.
        result[mask] /= len(np.unique(origins[mask]))
    return result


def weighted_rmse(actual, predicted, weight):
    return float(np.sqrt(np.average((np.asarray(predicted) - np.asarray(actual)) ** 2, weights=weight)))


def shrink_choice(actual, base, correction, weights, strengths=(0.0, 0.25, 0.5, 1.0)):
    score = {float(s): weighted_rmse(actual, base + s * correction, weights) for s in strengths}
    chosen = min(strengths, key=lambda s: (score[float(s)], s))
    return float(chosen), score


class HeadFeatures:
    def __init__(self, kind):
        self.kind = kind
        self.numeric = make_pipeline(SimpleImputer(strategy="median", add_indicator=True,
                                                   keep_empty_features=True), StandardScaler())
        self.embedding = None
        self.pca = None

    def fit(self, numeric, embedding):
        self.numeric.fit(numeric)
        if self.kind != "numeric":
            self.embedding = StandardScaler().fit(embedding)
            if self.kind in ("pca", "pca_only"):
                self.pca = PCA(n_components=32, svd_solver="randomized", random_state=42)
                self.pca.fit(self.embedding.transform(embedding))
        return self

    def transform(self, numeric, embedding):
        number = self.numeric.transform(numeric).astype(np.float32)
        if self.kind == "numeric":
            return number
        latent = self.embedding.transform(embedding)
        if self.pca is not None:
            latent = self.pca.transform(latent)
        latent = latent.astype(np.float32)
        return latent if self.kind == "pca_only" else np.column_stack((number, latent))


def make_mlp(width):
    import torch

    model = torch.nn.Sequential(torch.nn.Linear(width, 32), torch.nn.GELU(),
                                torch.nn.Dropout(0.2), torch.nn.Linear(32, 1))
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def train_mlp(x, residual, weight, validation, *, seed=42, max_epochs=80, patience=10):
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_mlp(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    xx = torch.as_tensor(x, dtype=torch.float32, device=device)
    yy = torch.as_tensor(residual / 1000, dtype=torch.float32, device=device)
    ww = torch.as_tensor(weight / np.mean(weight), dtype=torch.float32, device=device)
    vx, va, vb, vw = validation
    vx = torch.as_tensor(vx, dtype=torch.float32, device=device)
    va = torch.as_tensor(va, dtype=torch.float32, device=device)
    vb = torch.as_tensor(vb, dtype=torch.float32, device=device)
    vw = torch.as_tensor(vw, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(max_epochs):
        model.train()
        for ids in torch.randperm(len(x), generator=generator).split(256):
            ids = ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            error = model(xx[ids]).squeeze(-1) - yy[ids]
            loss = torch.sum(ww[ids] * error.square()) / torch.sum(ww[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = vb + 1000 * model(vx).squeeze(-1)
            score = torch.sum(vw * (prediction - va).square()) / torch.sum(vw)
            score = float(score.cpu())
        if score < best - 1e-6:
            best, best_epoch, best_state, stale = score, epoch + 1, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.cpu()
    return model, best_epoch, float(np.sqrt(best))


def mlp_correction(model, x, batch_size=4096):
    import torch

    model.eval()
    with torch.no_grad():
        return np.concatenate([1000 * model(torch.as_tensor(x[i:i + batch_size], dtype=torch.float32)).squeeze(-1).numpy()
                               for i in range(0, len(x), batch_size)])


def ridge_fit(x, residual, weight, alpha):
    return Ridge(alpha=alpha, solver="lsqr", tol=1e-5, max_iter=2000).fit(x, residual, sample_weight=weight)
