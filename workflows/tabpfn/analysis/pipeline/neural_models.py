"""Training-only tabular encoding and official neural model adapters."""

from __future__ import annotations

import gc
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import torch
from torch import nn

from analysis.pipeline.config import ROOT


def enable_search_dependencies():
    location = ROOT / ".cache" / "model-deps"
    if location.is_dir() and str(location) not in sys.path:
        sys.path.insert(0, str(location))


class NeuralFrameEncoder:
    """Reserve category 0 for unseen values; never fit on validation rows."""

    def __init__(self, categorical, quantile=True, min_category_count=1):
        self.categorical = tuple(categorical)
        self.quantile = quantile
        self.min_category_count = min_category_count

    def _numbers(self, frame):
        numbers = frame[self.numeric_].apply(pd.to_numeric, errors="coerce").copy()
        for column in self.numeric_:
            if column.endswith("_COUNT"):
                numbers[column] = np.log1p(numbers[column].clip(lower=0))
        return numbers

    def fit(self, frame):
        self.numeric_ = [column for column in frame if column not in self.categorical]
        self.levels_ = {}
        for column in self.categorical:
            values = frame[column].astype("string").fillna("__MISSING__")
            counts = values.value_counts()
            levels = sorted(counts.index[counts >= self.min_category_count])
            self.levels_[column] = {value: index + 1 for index, value in enumerate(levels)}
        numbers = self._numbers(frame)
        self.missing_ = [column for column in self.numeric_ if numbers[column].isna().any()]
        self.medians_ = numbers.median().fillna(0)
        filled = numbers.fillna(self.medians_)
        self.means_ = filled.mean()
        self.stds_ = filled.std(ddof=0).replace(0, 1).fillna(1)
        self.quantile_columns_ = [column for column in self.numeric_ if column not in (
            "APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_SIN", "APPROVAL_MONTH_COS", "MARKET_HORIZON_MONTHS")]
        self.quantile_ = None
        if self.quantile and self.quantile_columns_:
            self.quantile_ = QuantileTransformer(n_quantiles=min(512, len(frame)),
                output_distribution="normal", subsample=20000, random_state=42)
            self.quantile_.fit(filled[self.quantile_columns_])
        return self

    def transform(self, frame):
        cats = np.column_stack([frame[column].astype("string").fillna("__MISSING__")
            .map(self.levels_[column]).fillna(0).to_numpy(np.int64) for column in self.categorical])
        numbers = self._numbers(frame)
        missing = numbers[self.missing_].isna().to_numpy(np.float32)
        filled = numbers.fillna(self.medians_)
        standardized = (filled - self.means_) / self.stds_
        if self.quantile_ is not None:
            standardized[self.quantile_columns_] = self.quantile_.transform(filled[self.quantile_columns_])
        nums = np.concatenate([standardized.to_numpy(np.float32), missing], axis=1)
        if not np.isfinite(nums).all():
            raise ValueError("Non-finite neural input after training-only preprocessing")
        return cats.copy(), nums.copy()

    @property
    def cardinalities(self):
        return [len(self.levels_[column]) + 1 for column in self.categorical]


class EmbeddingMLP(nn.Module):
    def __init__(self, cardinalities, n_numeric, width=256, layers=3, dropout=.1):
        super().__init__()
        dimensions = [min(32, max(4, round(2 * size ** .25))) for size in cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim, padding_idx=0)
            for size, dim in zip(cardinalities, dimensions)])
        modules = []
        current = n_numeric + sum(dimensions)
        for _ in range(layers):
            modules.extend([nn.Linear(current, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(dropout)])
            current = width
        self.body = nn.Sequential(*modules, nn.Linear(width, 1))

    def forward(self, nums, cats):
        inputs = torch.cat([nums] + [layer(cats[:, index]) for index, layer in enumerate(self.embeddings)], dim=1)
        return self.body(inputs).squeeze(-1)


class EmbeddingResNet(nn.Module):
    def __init__(self, cardinalities, n_numeric, width=256, layers=3, dropout=.1):
        super().__init__()
        dimensions = [min(32, max(4, round(2 * size ** .25))) for size in cardinalities]
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim, padding_idx=0)
            for size, dim in zip(cardinalities, dimensions)])
        self.input = nn.Linear(n_numeric + sum(dimensions), width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout)) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, nums, cats):
        value = self.input(torch.cat([nums] + [layer(cats[:, index])
            for index, layer in enumerate(self.embeddings)], dim=1))
        for block in self.blocks:
            value = value + block(value)
        return self.output(value).squeeze(-1)


def build_network(family, encoder, n_numeric, settings):
    if family == "embedding_mlp":
        return EmbeddingMLP(encoder.cardinalities, n_numeric, settings.get("width", 256),
            settings.get("layers", 3), settings.get("dropout", .1))
    if family == "embedding_resnet":
        return EmbeddingResNet(encoder.cardinalities, n_numeric, settings.get("width", 256),
            settings.get("layers", 3), settings.get("dropout", .1))
    if family.startswith("ft_transformer"):
        enable_search_dependencies()
        from rtdl_revisiting_models import FTTransformer
        return FTTransformer(n_cont_features=n_numeric, cat_cardinalities=encoder.cardinalities,
            d_out=1, n_blocks=settings.get("layers", 3), d_block=settings.get("width", 96),
            attention_n_heads=8, attention_dropout=.2, ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3, ffn_dropout=.1, residual_dropout=0.)
    raise ValueError(f"Unknown neural architecture {family}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)


def predict_torch(model, cats, nums, device, batch_size=512):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(nums), batch_size):
            x_num = torch.from_numpy(nums[start:start + batch_size]).to(device)
            x_cat = torch.from_numpy(cats[start:start + batch_size]).to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                result.append(model(x_num, x_cat).reshape(-1).float().cpu().numpy())
    return np.concatenate(result)


def fit_neural(family, x_fit, y_fit, categorical, settings, x_eval=None, y_eval=None,
               fixed_epochs=None, device_name="cuda"):
    """Select epochs only on x_eval; fixed_epochs refits never inspect outer labels."""
    seed_everything(settings.get("seed", 42))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(.25)
    encoder = NeuralFrameEncoder(categorical, settings.get("quantile", True),
        settings.get("min_category_count", 1)).fit(x_fit)
    fit_cats, fit_nums = encoder.transform(x_fit)
    target_mean, target_std = float(np.mean(y_fit)), float(np.std(y_fit))
    target_std = max(target_std, 1.)
    target = ((np.asarray(y_fit) - target_mean) / target_std).astype(np.float32)
    eval_encoded = encoder.transform(x_eval) if x_eval is not None else None
    max_epochs = int(fixed_epochs or settings.get("max_epochs", 40))
    batch_size = settings.get("batch_size", 256 if family.startswith("ft_transformer") else 1024)
    history = []
    if family == "tabnet":
        enable_search_dependencies()
        from pytorch_tabnet.tab_model import TabNetRegressor
        fit_array = np.concatenate([fit_nums, fit_cats.astype(np.float32)], axis=1)
        model = TabNetRegressor(n_d=settings.get("width", 24), n_a=settings.get("width", 24),
            n_steps=settings.get("layers", 4), gamma=1.5, lambda_sparse=1e-4,
            cat_idxs=list(range(fit_nums.shape[1], fit_array.shape[1])), cat_dims=encoder.cardinalities,
            cat_emb_dim=[min(16, max(4, round(size ** .25 * 2))) for size in encoder.cardinalities],
            optimizer_fn=torch.optim.Adam, optimizer_params={"lr": settings.get("lr", .01)},
            scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params={"step_size": 15, "gamma": .5},
            mask_type="entmax", seed=settings.get("seed", 42), device_name=device.type, verbose=0)
        eval_set = None
        if eval_encoded is not None:
            eval_cats, eval_nums = eval_encoded
            eval_set = [(np.concatenate([eval_nums, eval_cats.astype(np.float32)], axis=1),
                ((np.asarray(y_eval) - target_mean) / target_std).astype(np.float32).reshape(-1, 1))]
        started = time.perf_counter()
        model.fit(fit_array, target.reshape(-1, 1), eval_set=eval_set, eval_metric=["rmse"],
            max_epochs=max_epochs, patience=0 if fixed_epochs else settings.get("patience", 7),
            batch_size=settings.get("batch_size", 1024), virtual_batch_size=128,
            num_workers=0, drop_last=True, compute_importance=False)
        epochs = max_epochs if fixed_epochs else int(model.best_epoch) + 1
        history = [{"epoch": index + 1, "rmse_sgd": float(value * target_std)}
            for index, value in enumerate(model.history.history.get("val_0_rmse", []))]
        result = {"model": model, "encoder": encoder, "mean": target_mean, "std": target_std,
            "family": family, "epochs": epochs, "history": history, "device": device.type,
            "elapsed_seconds": time.perf_counter() - started}
        return result
    model = build_network(family, encoder, fit_nums.shape[1], settings).to(device)
    parameters = model.make_parameter_groups() if family.startswith("ft_transformer") else model.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=settings.get("lr", 1e-3), weight_decay=settings.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=.5)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    t_cats, t_nums, t_y = (torch.from_numpy(array).to(device) for array in (fit_cats, fit_nums, target))
    best, state, best_epoch, stale = float("inf"), None, 1, 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(target), device=device)
        for start in range(0, len(target), batch_size):
            positions = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                output = model(t_nums[positions], t_cats[positions]).reshape(-1)
                loss = (output - t_y[positions]).square().mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        if eval_encoded is not None:
            prediction = predict_torch(model, *eval_encoded, device, batch_size * 2) * target_std + target_mean
            rmse = float(np.sqrt(np.mean((prediction - y_eval) ** 2)))
            history.append({"epoch": epoch, "rmse_sgd": rmse})
            if rmse < best - .05:
                best, best_epoch, stale = rmse, epoch, 0
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if epoch % 5 == 0:
                print(f"  {family} epoch {epoch}: inner RMSE={rmse:.2f}, best={best:.2f}", flush=True)
            if not fixed_epochs and stale >= settings.get("patience", 7):
                break
    if state is not None and not fixed_epochs:
        model.load_state_dict(state)
    del optimizer, t_cats, t_nums, t_y
    gc.collect()
    return {"model": model, "encoder": encoder, "mean": target_mean, "std": target_std,
        "family": family, "epochs": max_epochs if fixed_epochs else best_epoch,
        "history": history, "device": device.type, "elapsed_seconds": time.perf_counter() - started}


def predict_neural(bundle, frame):
    cats, nums = bundle["encoder"].transform(frame)
    if bundle["family"] == "tabnet":
        prediction = bundle["model"].predict(np.concatenate([nums, cats.astype(np.float32)], axis=1)).reshape(-1)
    else:
        prediction = predict_torch(bundle["model"], cats, nums, torch.device(bundle["device"]))
    return np.asarray(prediction, dtype=float) * bundle["std"] + bundle["mean"]
