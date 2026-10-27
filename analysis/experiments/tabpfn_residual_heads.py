"""Nested chronological selection for numeric and embedding residual heads."""

from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def chronological_head_split(months: np.ndarray, outer_origin: str):
    import pandas as pd

    periods = pd.PeriodIndex(months, freq='M')
    origin = pd.Period(outer_origin, 'M')
    if (periods >= origin).any():
        raise ValueError('A head cannot train or select on outer validation labels')
    cutoff = origin - 3
    training, calibration = np.asarray(periods < cutoff), np.asarray(periods >= cutoff)
    if not training.any() or not calibration.any():
        raise ValueError('Both chronological head partitions must be nonempty')
    return training, calibration


def head_transform():
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                         StandardScaler())


def fit_ridge_head(x, actual, base, future_x, future_base, training, calibration):
    started = time.perf_counter()
    prep = head_transform()
    a = prep.fit_transform(x[training])
    b = prep.transform(x[calibration])
    residual = (actual - base) / 1000.0
    candidates = []
    for alpha in (10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha, solver='lsqr', tol=1e-5, max_iter=2000)
        model.fit(a, residual[training])
        predicted = base[calibration] + 1000 * model.predict(b)
        candidates.append((float(np.sqrt(np.mean((predicted - actual[calibration]) ** 2))), alpha))
    inner_rmse, alpha = min(candidates)
    # The selected hyperparameter is refit using all pre-outer episode labels.
    final_prep = head_transform()
    all_x = final_prep.fit_transform(x)
    final = Ridge(alpha=alpha, solver='lsqr', tol=1e-5, max_iter=2000).fit(all_x, residual)
    predicted = future_base + 1000 * final.predict(final_prep.transform(future_x))
    return predicted, {'alpha': alpha, 'inner_rmse_sgd': inner_rmse,
                       'fit_predict_seconds': time.perf_counter() - started}, (final_prep, final)


def make_mlp(width: int):
    import torch

    return torch.nn.Sequential(torch.nn.Linear(width, 64), torch.nn.GELU(),
                               torch.nn.Dropout(.1), torch.nn.Linear(64, 32),
                               torch.nn.GELU(), torch.nn.Dropout(.1), torch.nn.Linear(32, 1))


def _train_mlp(x, y, *, epochs, seed, validation=None):
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_mlp(x.shape[1]).to('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    inputs = torch.as_tensor(np.asarray(x, np.float32), device='cuda')
    targets = torch.as_tensor(np.asarray(y, np.float32), device='cuda')
    generator = torch.Generator().manual_seed(seed)
    best_loss, best_epoch, best_state, stale, history = float('inf'), 0, None, 0, []
    if validation is not None:
        vx, vy = validation
        vx = torch.as_tensor(np.asarray(vx, np.float32), device='cuda')
        vy = torch.as_tensor(np.asarray(vy, np.float32), device='cuda')
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=generator)
        for offset in range(0, len(x), 256):
            indexes = order[offset:offset + 256].to('cuda')
            optimizer.zero_grad(set_to_none=True)
            error = model(inputs[indexes]).squeeze(-1) - targets[indexes]
            loss = (error ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if validation is None:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = float(((model(vx).squeeze(-1) - vy) ** 2).mean().item())
        history.append({'epoch': epoch + 1, 'inner_rmse_sgd': 1000 * np.sqrt(val_loss)})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 6:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch or epochs, history


def fit_mlp_head(x, actual, base, future_x, future_base, training, calibration,
                 *, epochs=40, seed=42):
    import torch

    started = time.perf_counter()
    prep = head_transform()
    a = prep.fit_transform(x[training]).astype(np.float32)
    b = prep.transform(x[calibration]).astype(np.float32)
    residual = (actual - base) / 1000.0
    selected, best_epoch, history = _train_mlp(
        a, residual[training], epochs=epochs, seed=seed,
        validation=(b, residual[calibration]),
    )
    inner_rmse = min(row['inner_rmse_sgd'] for row in history)
    del selected
    torch.cuda.empty_cache()
    final_prep = head_transform()
    all_x = final_prep.fit_transform(x).astype(np.float32)
    final, _, _ = _train_mlp(all_x, residual, epochs=best_epoch, seed=seed)
    final.eval()
    q = torch.as_tensor(final_prep.transform(future_x).astype(np.float32), device='cuda')
    with torch.no_grad():
        correction = final(q).squeeze(-1).cpu().numpy()
    predicted = future_base + 1000 * correction
    state = {name: value.detach().cpu().clone() for name, value in final.state_dict().items()}
    del final, q
    torch.cuda.empty_cache()
    return predicted, {'epochs': best_epoch, 'inner_rmse_sgd': inner_rmse,
                       'inner_history': history, 'fit_predict_seconds': time.perf_counter() - started}, (final_prep, state)
