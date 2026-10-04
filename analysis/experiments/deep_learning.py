"""GPU tabular deep-learning experiments with entity embeddings.

Run from the repository root, for example:
    python -m analysis.experiments.deep_learning --variant auxiliary --epochs 40
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from analysis.pipeline.auxiliary import MACRO_NUMERIC, build_auxiliary_features
from analysis.pipeline.config import DATA, DATE, REPORTS, TARGET, VALIDATION_START
from analysis.pipeline.data import check_output_path, load_main, protect_sources, temporal_split


MAIN_CATEGORICAL = ["FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "TOWN_CLEAN"]
AUX_CATEGORICAL = MAIN_CATEGORICAL + [
    "STREET_CLEAN", "BLOCK_CLEAN", "ADDRESS_KEY", "POSTAL_CODE_CLEAN",
    "SUBZONE_CLEAN", "PLANNING_AREA_CLEAN", "REGION_CLEAN",
]
MAIN_NUMERIC = [
    "FLOOR_AREA_SQM", "APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_OF_YEAR", "FLAT_AGE_YEARS",
]
AUX_NUMERIC = MAIN_NUMERIC + [
    "APPROVAL_YEAR", "APPROVAL_MONTH_SIN", "APPROVAL_MONTH_COS", "LEASE_COMMENCE_DATE",
    "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED", "BUILDING_AGE_YEARS",
    "COMPLETION_LEASE_GAP", "ROOM_COUNT", "AREA_PER_ROOM", "CBD_DIST_KM",
    "MRT_DIST_KM", "MRT_COUNT_0_5KM", "MRT_COUNT_1_0KM", "MRT_COUNT_2_0KM",
    "MALL_DIST_KM", "MALL_COUNT_0_5KM", "MALL_COUNT_1_0KM", "MALL_COUNT_2_0KM",
    "PRIMARY_SCHOOL_DIST_KM", "PRIMARY_SCHOOL_COUNT_0_5KM",
    "PRIMARY_SCHOOL_COUNT_1_0KM", "PRIMARY_SCHOOL_COUNT_2_0KM",
]


@dataclass
class ExperimentConfig:
    variant: str = "auxiliary"
    architecture: str = "resnet"
    validation_start: str = VALIDATION_START
    fit_window_months: int | None = None
    recency_half_life: float | None = 24.0
    target_encoding: bool = False
    target_encoding_window: int = 18
    width: int = 256
    blocks: int = 3
    dropout: float = 0.10
    batch_size: int = 1024
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    seed: int = 42
    macro_auxiliary: bool = False
    residual_baseline: bool = False
    residual_warmup_months: int = 12


TARGET_ENCODING_GROUPS = [
    ("ADDRESS_KEY",),
    ("ADDRESS_KEY", "FLAT_TYPE_CLEAN"),
    ("STREET_CLEAN", "FLAT_TYPE_CLEAN"),
    ("TOWN_CLEAN", "FLAT_TYPE_CLEAN"),
    ("SUBZONE_CLEAN", "FLAT_TYPE_CLEAN"),
]


def add_chronological_baseline(
    fitting: pd.DataFrame, validation: pd.DataFrame, *, warmup_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add past-only hierarchical predictions suitable for residual learning."""
    from analysis.experiments.historical_priors import hierarchical_prediction

    fitting = fitting.copy()
    validation = validation.copy()
    months = sorted(fitting[DATE].unique())
    baseline = pd.Series(np.nan, index=fitting.index, dtype=float)
    for position, month in enumerate(months):
        if position < warmup_months:
            continue
        history = fitting.loc[fitting[DATE] < month]
        current = fitting.loc[fitting[DATE] == month]
        prediction, _ = hierarchical_prediction(
            history, current, base_method="linear", base_months=12,
            premium_window=36, alpha_scale=1.7,
        )
        baseline.loc[current.index] = prediction
    validation_baseline, _ = hierarchical_prediction(
        fitting, validation, base_method="linear", base_months=12,
        premium_window=36, alpha_scale=1.7,
    )
    fitting["HISTORICAL_BASELINE"] = baseline
    validation["HISTORICAL_BASELINE"] = validation_baseline
    fitting = fitting.loc[fitting["HISTORICAL_BASELINE"].notna()].copy()
    if fitting.empty:
        raise ValueError("Residual warmup removed every fitting row")
    return fitting, validation


def _group_key(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    return frame[list(columns)].fillna("unknown").astype(str).agg("\x1f".join, axis=1)


def add_target_encodings(
    fitting: pd.DataFrame, validation: pd.DataFrame, *, window_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Add fitting-only smoothed encodings; training rows use leave-one-out values."""
    fitting = fitting.copy()
    validation = validation.copy()
    cutoff = pd.to_datetime(fitting[DATE].max(), format="%Y-%m") + pd.DateOffset(months=1)
    dates = pd.to_datetime(fitting[DATE], format="%Y-%m")
    recent_mask = dates >= cutoff - pd.DateOffset(months=window_months)
    recent_global = float(fitting.loc[recent_mask, TARGET].mean())
    monthly_mean = fitting.groupby(DATE)[TARGET].transform("mean")
    residual = fitting[TARGET] - monthly_mean
    added = []
    for columns in TARGET_ENCODING_GROUPS:
        label = "_".join(column.replace("_CLEAN", "").replace("_KEY", "") for column in columns)
        fit_key = _group_key(fitting, columns)
        val_key = _group_key(validation, columns)

        recent = pd.DataFrame({"key": fit_key[recent_mask], "target": fitting.loc[recent_mask, TARGET]})
        stats = recent.groupby("key")["target"].agg(["sum", "count"])
        sums = fit_key.map(stats["sum"]).fillna(0.0).to_numpy(float)
        counts = fit_key.map(stats["count"]).fillna(0.0).to_numpy(float)
        included = recent_mask.to_numpy(float)
        alpha = 20.0
        column = f"TE_RECENT_{label}"
        fitting[column] = (
            sums - included * fitting[TARGET].to_numpy(float) + alpha * recent_global
        ) / np.maximum(counts - included + alpha, 1.0)
        validation[column] = (
            val_key.map((stats["sum"] + alpha * recent_global) / (stats["count"] + alpha))
            .fillna(recent_global).to_numpy(float)
        )
        added.append(column)

        premium = pd.DataFrame({"key": fit_key, "residual": residual})
        premium_stats = premium.groupby("key")["residual"].agg(["sum", "count"])
        premium_alpha = 12.0
        premium_column = f"TE_PREMIUM_{label}"
        fitting[premium_column] = (
            fit_key.map(premium_stats["sum"]).to_numpy(float) - residual.to_numpy(float)
        ) / np.maximum(fit_key.map(premium_stats["count"]).to_numpy(float) - 1 + premium_alpha,
                       premium_alpha)
        validation[premium_column] = (
            val_key.map(premium_stats["sum"] / (premium_stats["count"] + premium_alpha))
            .fillna(0.0).to_numpy(float)
        )
        added.append(premium_column)
    return fitting, validation, added


class TabularEncoder:
    def fit(self, frame: pd.DataFrame, categorical: list[str], numeric: list[str]):
        self.categorical = categorical
        self.numeric = numeric
        self.levels = {}
        for column in categorical:
            values = frame[column].fillna("unknown").astype(str)
            self.levels[column] = {value: i + 1 for i, value in enumerate(sorted(values.unique()))}
        numbers = frame[numeric].apply(pd.to_numeric, errors="coerce")
        self.medians = numbers.median().fillna(0.0)
        filled = numbers.fillna(self.medians)
        self.means = filled.mean()
        self.stds = filled.std().replace(0, 1.0).fillna(1.0)
        return self

    def transform(self, frame: pd.DataFrame):
        cats = np.column_stack([
            frame[column].fillna("unknown").astype(str).map(self.levels[column]).fillna(0).to_numpy(np.int64)
            for column in self.categorical
        ])
        numbers = frame[self.numeric].apply(pd.to_numeric, errors="coerce")
        nums = ((numbers.fillna(self.medians) - self.means) / self.stds).to_numpy(np.float32)
        return cats, nums

    @property
    def cardinalities(self):
        return [len(self.levels[column]) + 1 for column in self.categorical]

    def metadata(self):
        return {
            "categorical": self.categorical,
            "numeric": self.numeric,
            "cardinalities": self.cardinalities,
            "medians": self.medians.to_dict(),
            "means": self.means.to_dict(),
            "stds": self.stds.to_dict(),
            "levels": self.levels,
        }


def _imports():
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, DataLoader, TensorDataset


def embedding_dim(cardinality: int) -> int:
    return int(min(48, max(4, round(1.6 * cardinality ** 0.25))))


def build_network(cardinalities, n_numeric, config: ExperimentConfig):
    torch, nn, _, _ = _imports()

    if config.architecture == "transformer":
        class FeatureTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                dim = config.width
                self.cat_embeddings = nn.ModuleList([nn.Embedding(cardinality, dim) for cardinality in cardinalities])
                self.numeric_weight = nn.Parameter(torch.empty(n_numeric, dim))
                self.numeric_bias = nn.Parameter(torch.zeros(n_numeric, dim))
                self.cls = nn.Parameter(torch.zeros(1, 1, dim))
                nn.init.normal_(self.numeric_weight, std=0.02)
                nn.init.normal_(self.cls, std=0.02)
                heads = 8 if dim % 8 == 0 else 4
                layer = nn.TransformerEncoderLayer(
                    d_model=dim, nhead=heads, dim_feedforward=dim * 2,
                    dropout=config.dropout, activation="gelu", batch_first=True,
                    norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(
                    layer, num_layers=config.blocks, enable_nested_tensor=False,
                )
                self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

            def forward(self, cats, nums):
                cat_tokens = [layer(cats[:, i]).unsqueeze(1) for i, layer in enumerate(self.cat_embeddings)]
                numeric_tokens = nums.unsqueeze(2) * self.numeric_weight.unsqueeze(0) + self.numeric_bias.unsqueeze(0)
                cls = self.cls.expand(len(cats), -1, -1)
                tokens = torch.cat([cls, *cat_tokens, numeric_tokens], dim=1)
                return self.output(self.transformer(tokens)[:, 0]).squeeze(1)

        return FeatureTransformer()

    class ResidualBlock(nn.Module):
        def __init__(self, width, dropout):
            super().__init__()
            self.block = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, width * 2), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(width * 2, width), nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + self.block(x)

    class TabularNet(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [embedding_dim(c) for c in cardinalities]
            self.embeddings = nn.ModuleList([nn.Embedding(c, d) for c, d in zip(cardinalities, dims)])
            input_width = sum(dims) + n_numeric
            self.input = nn.Sequential(nn.Linear(input_width, config.width), nn.SiLU())
            if config.architecture == "resnet":
                self.body = nn.Sequential(*[ResidualBlock(config.width, config.dropout) for _ in range(config.blocks)])
            elif config.architecture == "mlp":
                layers = []
                for _ in range(config.blocks):
                    layers.extend([nn.Linear(config.width, config.width), nn.BatchNorm1d(config.width),
                                   nn.SiLU(), nn.Dropout(config.dropout)])
                self.body = nn.Sequential(*layers)
            else:
                raise ValueError("architecture must be 'resnet', 'mlp' or 'transformer'")
            self.output = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, 1))

        def forward(self, cats, nums):
            embedded = [layer(cats[:, i]) for i, layer in enumerate(self.embeddings)]
            x = torch.cat([*embedded, nums], dim=1)
            return self.output(self.body(self.input(x))).squeeze(1)

    return TabularNet()


def predict(model, cats, nums, batch_size, device):
    torch, _, DataLoader, TensorDataset = _imports()
    dataset = TensorDataset(torch.from_numpy(cats), torch.from_numpy(nums))
    loader = DataLoader(dataset, batch_size=batch_size * 2, shuffle=False)
    output = []
    model.eval()
    with torch.inference_mode():
        for cat_batch, num_batch in loader:
            output.append(model(cat_batch.to(device), num_batch.to(device)).cpu().numpy())
    return np.concatenate(output)


def train(config: ExperimentConfig, output_dir: Path):
    torch, _, DataLoader, TensorDataset = _imports()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda")
    source_paths = [DATA / "train.csv", DATA / "auxiliary" / "sg-hdb-block.csv",
                    DATA / "auxiliary" / "sg-mrt-stations.csv",
                    DATA / "auxiliary" / "sg-shopping-malls.csv",
                    DATA / "auxiliary" / "sg-schools.csv"]
    if config.macro_auxiliary:
        source_paths += [DATA / "auxiliary" / "sg-coe-prices.csv",
                         DATA / "auxiliary" / "sg-stock-prices.csv"]
    started = time.time()
    with protect_sources(source_paths) as hashes:
        raw = load_main(DATA / "train.csv", require_target=True)
        frame = build_auxiliary_features(raw, include_macro=config.macro_auxiliary)
        fitting, validation = temporal_split(frame, config.validation_start, config.fit_window_months)
        categorical = AUX_CATEGORICAL if config.variant == "auxiliary" else MAIN_CATEGORICAL
        numeric = list(AUX_NUMERIC if config.variant == "auxiliary" else MAIN_NUMERIC)
        if config.macro_auxiliary:
            if config.variant != "auxiliary":
                raise ValueError("Macro auxiliary features require the auxiliary feature variant")
            numeric += MACRO_NUMERIC
        if config.target_encoding:
            if config.variant != "auxiliary":
                raise ValueError("Target encoding requires the auxiliary feature variant")
            fitting, validation, encoded = add_target_encodings(
                fitting, validation, window_months=config.target_encoding_window,
            )
            numeric += encoded
        if config.residual_baseline:
            fitting, validation = add_chronological_baseline(
                fitting, validation, warmup_months=config.residual_warmup_months,
            )
            numeric += ["HISTORICAL_BASELINE"]
        encoder = TabularEncoder().fit(fitting, categorical, numeric)
        fit_cats, fit_nums = encoder.transform(fitting)
        val_cats, val_nums = encoder.transform(validation)
        if config.residual_baseline:
            fit_target = fitting[TARGET] - fitting["HISTORICAL_BASELINE"]
        else:
            fit_target = fitting[TARGET]
        target_mean = float(fit_target.mean())
        target_std = float(fit_target.std())
        fit_y = ((fit_target.to_numpy(np.float32) - target_mean) / target_std).astype(np.float32)
        val_y = validation[TARGET].to_numpy(np.float32)
        if config.recency_half_life:
            cutoff = pd.to_datetime(config.validation_start)
            fit_dates = pd.to_datetime(fitting[DATE])
            age = ((cutoff.year - fit_dates.dt.year) * 12 + cutoff.month - fit_dates.dt.month).to_numpy(float)
            weights = np.exp(-math.log(2) * age / config.recency_half_life).astype(np.float32)
            weights /= weights.mean()
        else:
            weights = np.ones(len(fitting), dtype=np.float32)

        dataset = TensorDataset(torch.from_numpy(fit_cats), torch.from_numpy(fit_nums),
                                torch.from_numpy(fit_y), torch.from_numpy(weights))
        generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                            generator=generator, num_workers=0, pin_memory=True)
        model = build_network(encoder.cardinalities, len(numeric), config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
        scaler = torch.amp.GradScaler("cuda")
        history, best_state, best_rmse, stale = [], None, float("inf"), 0
        for epoch in range(1, config.epochs + 1):
            model.train(); running = 0.0; seen = 0
            for cat_batch, num_batch, y_batch, w_batch in loader:
                cat_batch = cat_batch.to(device, non_blocking=True)
                num_batch = num_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                w_batch = w_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    error = model(cat_batch, num_batch) - y_batch
                    loss = (w_batch * error.square()).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update()
                running += float(loss.detach()) * len(y_batch); seen += len(y_batch)
            pred_z = predict(model, val_cats, val_nums, config.batch_size, device)
            predictions = pred_z * target_std + target_mean
            if config.residual_baseline:
                predictions = predictions + validation["HISTORICAL_BASELINE"].to_numpy(float)
            rmse = float(mean_squared_error(val_y, predictions) ** 0.5)
            mae = float(mean_absolute_error(val_y, predictions))
            scheduler.step(rmse)
            record = {"epoch": epoch, "train_loss": running / seen, "rmse": rmse, "mae": mae,
                      "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(record)
            print(json.dumps(record))
            if rmse < best_rmse - 0.05:
                best_rmse, stale = rmse, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
                if stale >= config.patience:
                    break
        model.load_state_dict(best_state)
        pred_z = predict(model, val_cats, val_nums, config.batch_size, device)
        predictions = pred_z * target_std + target_mean
        if config.residual_baseline:
            predictions = predictions + validation["HISTORICAL_BASELINE"].to_numpy(float)
        metrics = {
            "rmse": float(mean_squared_error(val_y, predictions) ** 0.5),
            "mae": float(mean_absolute_error(val_y, predictions)),
            "bias": float(np.mean(predictions - val_y)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": validation.index, DATE: validation[DATE].to_numpy(),
                  "actual": val_y, "prediction": predictions}).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    artifact = {
        "config": asdict(config), "metrics": metrics,
        "fitting_rows": len(fitting), "validation_rows": len(validation),
        "fitting_months": [fitting[DATE].min(), fitting[DATE].max()],
        "validation_months": [validation[DATE].min(), validation[DATE].max()],
        "target_mean": target_mean, "target_std": target_std,
        "categorical_cardinalities": dict(zip(categorical, encoder.cardinalities)),
        "source_sha256": hashes, "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0), "elapsed_seconds": time.time() - started,
        "history": history,
    }
    (output_dir / "results.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    torch.save({"state_dict": best_state, "encoder": encoder.metadata(), "artifact": artifact},
               output_dir / "model.pt")
    print(json.dumps(metrics, indent=2))
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GPU entity-embedding models for HDB rent")
    parser.add_argument("--variant", choices=("main", "auxiliary"), default="auxiliary")
    parser.add_argument("--architecture", choices=("resnet", "mlp", "transformer"), default="resnet")
    parser.add_argument("--validation-start", default=VALIDATION_START)
    parser.add_argument("--fit-window-months", type=int)
    parser.add_argument("--recency-half-life", type=float, default=24.0)
    parser.add_argument("--target-encoding", action="store_true")
    parser.add_argument("--target-encoding-window", type=int, default=18)
    parser.add_argument("--macro-auxiliary", action="store_true")
    parser.add_argument("--residual-baseline", action="store_true")
    parser.add_argument("--residual-warmup-months", type=int, default=12)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output_dir or REPORTS / "deep_learning" / (
        f"{args.variant}_{args.architecture}_seed{args.seed}"
    )
    output = check_output_path(output)
    config = ExperimentConfig(**{key: value for key, value in vars(args).items() if key != "output_dir"})
    return train(config, output)


if __name__ == "__main__":
    main()
