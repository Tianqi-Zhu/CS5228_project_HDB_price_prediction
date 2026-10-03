"""TabM-style neural challenger and tree/neural blend for HDB rent.

The model combines categorical embeddings with a shared MLP whose linear
layers have inexpensive per-member rank-one adapters.  It is inspired by
TabM/BatchEnsemble, but implemented locally so the experiment is transparent
and does not depend on a third-party TabM package.

All target-derived inputs reuse the fixed-origin, leakage-safe history builder
from ``spatiotemporal_hybrid``.  Fold validation targets are never used as
features or fed back into later validation months.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # Keep preprocessing/tests importable without PyTorch.
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None

from analysis.config import DATA
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.spatiotemporal_hybrid import (
    BASE_CATEGORICAL, BASE_FEATURES, FOLDS, HISTORY_FEATURES, MONTH, OUT as ST_OUT,
    TARGET, fit_row_model, historical_features, spatial_frame,
)


DEFAULT_OUT = ST_OUT.parent / "neural_tabm"
ADDRESS_CATEGORICAL = ["BLOCK_ID", "PROPERTY_SIGNATURE_ID"]
CATEGORICAL_FEATURES = sorted(BASE_CATEGORICAL) + ADDRESS_CATEGORICAL
NUMERIC_FEATURES = [column for column in BASE_FEATURES if column not in BASE_CATEGORICAL] + HISTORY_FEATURES
VARIANTS = ["historical_tree", "tabm", "blend_50"]


@dataclass(frozen=True)
class TrainSettings:
    epochs: int = 40
    batch_size: int = 2048
    ensemble_size: int = 8
    hidden_dim: int = 256
    depth: int = 3
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    seed: int = 42
    num_workers: int = 0


@dataclass
class NeuralPreprocessor:
    """Training-only category vocabularies and numeric/target scaling."""

    categories: dict[str, list[str]]
    numeric_median: dict[str, float]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    target_mean: float
    target_std: float

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "NeuralPreprocessor":
        categories = {}
        for column in CATEGORICAL_FEATURES:
            values = frame[column].astype("string").fillna("__MISSING__")
            categories[column] = sorted(values.unique().tolist())
        numeric = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        median = numeric.median().fillna(0.0)
        filled = numeric.fillna(median)
        mean = filled.mean()
        std = filled.std(ddof=0).replace(0, 1.0).fillna(1.0)
        target = frame[TARGET].to_numpy(float)
        target_std = float(np.std(target))
        return cls(
            categories=categories,
            numeric_median=median.astype(float).to_dict(),
            numeric_mean=mean.astype(float).to_dict(),
            numeric_std=std.astype(float).to_dict(),
            target_mean=float(np.mean(target)),
            target_std=target_std if target_std > 0 else 1.0,
        )

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        categorical = np.zeros((len(frame), len(CATEGORICAL_FEATURES)), dtype=np.int64)
        for position, column in enumerate(CATEGORICAL_FEATURES):
            vocabulary = pd.Index(self.categories[column])
            values = frame[column].astype("string").fillna("__MISSING__")
            categorical[:, position] = vocabulary.get_indexer(values) + 1
        numeric = frame[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        median = pd.Series(self.numeric_median)
        mean = pd.Series(self.numeric_mean)
        std = pd.Series(self.numeric_std)
        numeric = ((numeric.fillna(median) - mean) / std).to_numpy(np.float32)
        return numeric, categorical

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=float) - self.target_mean) / self.target_std).astype(np.float32)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.target_std + self.target_mean

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.categories[column]) + 1 for column in CATEGORICAL_FEATURES]


def add_entity_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Create stable high-cardinality IDs that embeddings can learn."""
    result = frame.copy()
    address = (
        result["TOWN"].astype("string") + "|" + result["BLOCK"].astype("string")
        + "|" + result["STREET"].astype("string")
    )
    result["BLOCK_ID"] = address
    result["PROPERTY_SIGNATURE_ID"] = (
        address + "|" + result["FLAT_TYPE_CLEAN"].astype("string")
        + "|" + result["FLAT_MODEL_CLEAN"].astype("string")
        + "|" + result["FLOOR_AREA_SQM"].astype("string")
        + "|" + result["LEASE_COMMENCE_DATE"].astype("string")
    )
    return result


def neural_feature_frames(fitting: pd.DataFrame, prediction: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the same leakage-safe histories used by the winning tree model."""
    fit_history, pred_history = historical_features(fitting, prediction)
    return add_entity_ids(fitting.join(fit_history)), add_entity_ids(prediction.join(pred_history))


def select_blend_weight(actual: np.ndarray, tree_prediction: np.ndarray,
                        neural_prediction: np.ndarray, step: float = .02) -> tuple[float, float]:
    """Choose the tree weight minimizing RMSE on out-of-fold predictions."""
    best_weight, best_rmse = 0.0, math.inf
    for weight in np.arange(0.0, 1.0 + step / 2, step):
        prediction = weight * tree_prediction + (1 - weight) * neural_prediction
        score = metrics(actual, prediction)["rmse_sgd"]
        if score < best_rmse:
            best_weight, best_rmse = float(weight), float(score)
    return best_weight, best_rmse


if nn is not None:
    class BatchEnsembleLinear(nn.Module):
        """Shared linear map with per-member rank-one input/output adapters."""

        def __init__(self, in_features: int, out_features: int, ensemble_size: int):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            self.bias = nn.Parameter(torch.zeros(out_features))
            self.input_scale = nn.Parameter(torch.ones(ensemble_size, in_features))
            self.output_scale = nn.Parameter(torch.ones(ensemble_size, out_features))
            self.member_bias = nn.Parameter(torch.zeros(ensemble_size, out_features))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.normal_(self.input_scale, mean=1.0, std=.05)
            nn.init.normal_(self.output_scale, mean=1.0, std=.05)

        def forward(self, values):
            adapted = values * self.input_scale.unsqueeze(0)
            output = F.linear(adapted, self.weight, self.bias)
            return output * self.output_scale.unsqueeze(0) + self.member_bias.unsqueeze(0)


    class TabMStyleRegressor(nn.Module):
        """Embedding MLP with jointly trained parameter-efficient members."""

        def __init__(self, cardinalities: list[int], n_numeric: int, settings: TrainSettings):
            super().__init__()
            self.ensemble_size = settings.ensemble_size
            dimensions = [min(32, max(4, int(round(cardinality ** .25 * 2)))) for cardinality in cardinalities]
            self.embeddings = nn.ModuleList([
                nn.Embedding(cardinality, dimension, padding_idx=0)
                for cardinality, dimension in zip(cardinalities, dimensions)
            ])
            input_dim = n_numeric + sum(dimensions)
            layers = []
            for layer_index in range(settings.depth):
                in_dim = input_dim if layer_index == 0 else settings.hidden_dim
                layers.append(BatchEnsembleLinear(in_dim, settings.hidden_dim, settings.ensemble_size))
            self.layers = nn.ModuleList(layers)
            self.output = BatchEnsembleLinear(settings.hidden_dim, 1, settings.ensemble_size)
            self.dropout = nn.Dropout(settings.dropout)

        def forward(self, numeric, categorical):
            embedded = [embedding(categorical[:, index]) for index, embedding in enumerate(self.embeddings)]
            base = torch.cat([numeric] + embedded, dim=1)
            values = base.unsqueeze(1).expand(-1, self.ensemble_size, -1)
            for layer in self.layers:
                values = self.dropout(F.relu(layer(values)))
            return self.output(values).squeeze(-1)
else:
    BatchEnsembleLinear = None
    TabMStyleRegressor = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for the neural experiment. Install requirements-model.txt "
            "or run `python -m pip install torch`."
        )


def resolve_device(requested: str) -> str:
    require_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(frame: pd.DataFrame, preprocessor: NeuralPreprocessor,
                settings: TrainSettings, device: str):
    """Train all efficient ensemble members jointly using standardized MSE."""
    set_seed(settings.seed)
    numeric, categorical = preprocessor.transform(frame)
    target = preprocessor.transform_target(frame[TARGET].to_numpy(float))
    dataset = TensorDataset(
        torch.from_numpy(numeric), torch.from_numpy(categorical), torch.from_numpy(target)
    )
    generator = torch.Generator().manual_seed(settings.seed)
    loader = DataLoader(
        dataset, batch_size=settings.batch_size, shuffle=True,
        num_workers=settings.num_workers, generator=generator,
    )
    model = TabMStyleRegressor(preprocessor.cardinalities, len(NUMERIC_FEATURES), settings).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(settings.epochs, 1))
    losses = []
    for epoch in range(settings.epochs):
        model.train()
        total_loss, total_rows = 0.0, 0
        for batch_numeric, batch_categorical, batch_target in loader:
            batch_numeric = batch_numeric.to(device)
            batch_categorical = batch_categorical.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            member_prediction = model(batch_numeric, batch_categorical)
            loss = F.mse_loss(member_prediction, batch_target.unsqueeze(1).expand_as(member_prediction))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_target)
            total_rows += len(batch_target)
        scheduler.step()
        epoch_loss = total_loss / max(total_rows, 1)
        losses.append(epoch_loss)
        print(f"    epoch {epoch + 1:02d}/{settings.epochs}: standardized MSE={epoch_loss:.5f}", flush=True)
    return model, losses


def predict_model(model, frame: pd.DataFrame, preprocessor: NeuralPreprocessor,
                  batch_size: int, device: str) -> np.ndarray:
    numeric, categorical = preprocessor.transform(frame)
    dataset = TensorDataset(torch.from_numpy(numeric), torch.from_numpy(categorical))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch_numeric, batch_categorical in loader:
            member_prediction = model(batch_numeric.to(device), batch_categorical.to(device))
            predictions.append(member_prediction.mean(dim=1).cpu().numpy())
    return preprocessor.inverse_target(np.concatenate(predictions))


def fit_neural(fitting: pd.DataFrame, prediction: pd.DataFrame,
               settings: TrainSettings, device: str) -> tuple[np.ndarray, list[float], dict]:
    fit_features, pred_features = neural_feature_frames(fitting, prediction)
    preprocessor = NeuralPreprocessor.fit(fit_features)
    model, losses = train_model(fit_features, preprocessor, settings, device)
    predicted = predict_model(model, pred_features, preprocessor, settings.batch_size, device)
    details = {
        "categorical_cardinalities": dict(zip(CATEGORICAL_FEATURES, preprocessor.cardinalities)),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    return predicted, losses, details


def evaluate(frame: pd.DataFrame, settings: TrainSettings, device: str,
             max_folds: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    scores, monthly, predictions, training_details = [], [], [], []
    folds = FOLDS[:max_folds] if max_folds else FOLDS
    for start_text, end_text in folds:
        start, end = pd.Period(start_text, freq="M"), pd.Period(end_text, freq="M")
        fitting = frame[frame[MONTH] < start].copy()
        validation = frame[(frame[MONTH] >= start) & (frame[MONTH] < end)].copy()
        actual = validation[TARGET].to_numpy(float)
        print(f"Fold {start_text}: fitting historical tree", flush=True)
        _, tree_prediction = fit_row_model(fitting, validation, with_history=True)
        print(f"Fold {start_text}: fitting TabM-style model on {device}", flush=True)
        neural_prediction, losses, details = fit_neural(fitting, validation, settings, device)
        fold_predictions = {
            "historical_tree": tree_prediction,
            "tabm": neural_prediction,
            "blend_50": .5 * tree_prediction + .5 * neural_prediction,
        }
        training_details.append({
            "validation_start": start_text, "epoch_losses": losses, **details,
        })
        for variant, predicted in fold_predictions.items():
            score = {
                "validation_start": start_text, "validation_end_exclusive": end_text,
                "variant": variant, "n_fit": len(fitting), "n_validation": len(validation),
                "horizon_months": validation[MONTH].nunique(), **metrics(actual, predicted),
            }
            scores.append(score)
            rows = pd.DataFrame({
                "validation_start": start_text,
                "approval_month": validation[MONTH].astype(str).to_numpy(),
                "variant": variant, "actual": actual, "predicted": predicted,
            })
            predictions.append(rows)
            for month, part in rows.groupby("approval_month", sort=True):
                monthly.append({
                    "validation_start": start_text, "approval_month": month,
                    "horizon": pd.Period(month, freq="M").ordinal - start.ordinal + 1,
                    "variant": variant, "n": len(part), **metrics(part["actual"], part["predicted"]),
                })
            print(f"  {variant}: RMSE={score['rmse_sgd']:.2f}; bias={score['bias_sgd']:.2f}", flush=True)
    return pd.DataFrame(scores), pd.DataFrame(monthly), pd.concat(predictions, ignore_index=True), training_details


def add_selected_blend(predictions: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Add an OOF-selected blend for final-model selection; report its reuse explicitly."""
    tree = predictions[predictions["variant"].eq("historical_tree")].reset_index(drop=True)
    neural = predictions[predictions["variant"].eq("tabm")].reset_index(drop=True)
    if not tree[["validation_start", "approval_month", "actual"]].equals(
        neural[["validation_start", "approval_month", "actual"]]
    ):
        raise AssertionError("Tree and neural out-of-fold rows do not align")
    weight, score = select_blend_weight(
        tree["actual"].to_numpy(), tree["predicted"].to_numpy(), neural["predicted"].to_numpy()
    )
    selected = tree.copy()
    selected["variant"] = "blend_selected_oof"
    selected["predicted"] = weight * tree["predicted"] + (1 - weight) * neural["predicted"]
    return pd.concat([predictions, selected], ignore_index=True), weight, score


def score_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled, folds = [], []
    for variant, part in predictions.groupby("variant", sort=False):
        pooled.append({"variant": variant, "n": len(part), **metrics(part["actual"], part["predicted"])})
        for start, fold in part.groupby("validation_start", sort=True):
            folds.append({"validation_start": start, "variant": variant, "n": len(fold),
                          **metrics(fold["actual"], fold["predicted"])})
    return pd.DataFrame(pooled).sort_values("rmse_sgd"), pd.DataFrame(folds)


def train_final(train: pd.DataFrame, test: pd.DataFrame, settings: TrainSettings,
                device: str, tree_weight: float) -> tuple[pd.DataFrame, dict]:
    print("Final training: historical tree", flush=True)
    _, tree_prediction = fit_row_model(train, test, with_history=True)
    print("Final training: TabM-style model", flush=True)
    neural_prediction, losses, details = fit_neural(train, test, settings, device)
    blended = tree_weight * tree_prediction + (1 - tree_weight) * neural_prediction
    output = pd.DataFrame({
        "Id": np.arange(len(test)), "RENT_APPROVAL_DATE": test[MONTH].astype(str).to_numpy(),
        "HistoricalTree": tree_prediction, "TabM": neural_prediction, "Predicted": blended,
    })
    return output, {"selected_tree_weight": tree_weight, "epoch_losses": losses, **details}


def plot_scores(pooled: pd.DataFrame, output_dir: Path) -> None:
    ordered = pooled.sort_values("rmse_sgd")
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.bar(ordered["variant"], ordered["rmse_sgd"], color="#147D91")
    ax.set(title="Neural and tree ensemble | fixed-origin pooled validation",
           xlabel="Variant", ylabel="RMSE (SGD)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=.25)
    for index, value in enumerate(ordered["rmse_sgd"]):
        ax.text(index, value + 1, f"{value:.1f}", ha="center")
    fig.savefig(output_dir / "model_comparison.png", dpi=160, facecolor="white")
    plt.close(fig)


def write_report(pooled: pd.DataFrame, fold_scores: pd.DataFrame, tree_weight: float,
                 output_dir: Path) -> None:
    best = pooled.iloc[0]
    tree = pooled[pooled["variant"].eq("historical_tree")].iloc[0]
    lines = [
        "# TabM-style neural rental model", "",
        "Evaluation uses the same expanding fixed-origin folds as the spatiotemporal experiment. Historical target features are frozen at each origin.", "",
        f"The lowest pooled RMSE is **S${best['rmse_sgd']:.2f}** from **{best['variant']}**. The historical tree scores S${tree['rmse_sgd']:.2f}.", "",
        f"The final blend weight selected from pooled out-of-fold predictions assigns **{tree_weight:.2f}** to the tree and **{1-tree_weight:.2f}** to the neural model. Because the same out-of-fold predictions select and report this weight, `blend_selected_oof` is a model-selection estimate rather than an unbiased performance estimate.", "",
        "```text", pooled.to_string(index=False, float_format=lambda value: f"{value:.2f}"), "```", "",
        "## Fold scores", "", "```text",
        fold_scores.to_string(index=False, float_format=lambda value: f"{value:.2f}"), "```", "",
        "The neural model uses embeddings for property/location categories and a shared MLP with per-member rank-one adapters. Member predictions are averaged at inference.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TrainSettings.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainSettings.batch_size)
    parser.add_argument("--ensemble-size", type=int, default=TrainSettings.ensemble_size)
    parser.add_argument("--hidden-dim", type=int, default=TrainSettings.hidden_dim)
    parser.add_argument("--depth", type=int, default=TrainSettings.depth)
    parser.add_argument("--dropout", type=float, default=TrainSettings.dropout)
    parser.add_argument("--learning-rate", type=float, default=TrainSettings.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=TrainSettings.weight_decay)
    parser.add_argument("--seed", type=int, default=TrainSettings.seed)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-folds", type=int, choices=[1, 2, 3])
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_torch()
    settings = TrainSettings(
        epochs=args.epochs, batch_size=args.batch_size, ensemble_size=args.ensemble_size,
        hidden_dim=args.hidden_dim, depth=args.depth, dropout=args.dropout,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=args.seed,
    )
    device = resolve_device(args.device)
    if device == "cpu":
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    source_paths = (DATA / "train.csv", DATA / "test.csv")
    source_hashes = {path.name: sha256(path) for path in source_paths}
    train = spatial_frame(pd.read_csv(DATA / "train.csv"))
    test = spatial_frame(pd.read_csv(DATA / "test.csv"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores, monthly, predictions, training_details = evaluate(
        train, settings, device, max_folds=args.max_folds
    )
    predictions, tree_weight, selected_rmse = add_selected_blend(predictions)
    pooled, fold_scores = score_predictions(predictions)
    scores.to_csv(args.output_dir / "base_fold_metrics.csv", index=False, float_format="%.4f")
    fold_scores.to_csv(args.output_dir / "fold_metrics.csv", index=False, float_format="%.4f")
    pooled.to_csv(args.output_dir / "pooled_metrics.csv", index=False, float_format="%.4f")
    monthly.to_csv(args.output_dir / "monthly_errors.csv", index=False, float_format="%.4f")
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False, float_format="%.4f")
    plot_scores(pooled, args.output_dir)
    final_details = None
    if not args.skip_final:
        final_predictions, final_details = train_final(train, test, settings, device, tree_weight)
        final_predictions.to_csv(args.output_dir / "test_predictions.csv", index=False, float_format="%.4f")
        final_predictions[["Id", "Predicted"]].to_csv(
            args.output_dir / "submission.csv", index=False, float_format="%.4f"
        )
    if {path.name: sha256(path) for path in source_paths} != source_hashes:
        raise AssertionError("An input CSV changed during the neural experiment")
    write_report(pooled, fold_scores, tree_weight, args.output_dir)
    results = {
        "model": "Local TabM-style categorical-embedding MLP with rank-one member adapters",
        "device": device, "settings": asdict(settings),
        "folds": FOLDS[:args.max_folds] if args.max_folds else FOLDS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "selected_tree_weight": tree_weight,
        "selected_blend_oof_rmse_sgd": selected_rmse,
        "pooled_scores": pooled.to_dict(orient="records"),
        "training_details": training_details,
        "final_training": final_details,
        "source_sha256": source_hashes,
        "notes": [
            "No validation target is used in a feature or fed back within a fold.",
            "The selected blend weight is tuned on the same pooled OOF predictions used to report it.",
            "Use multiple seeds before treating a small neural improvement as stable.",
        ],
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved neural evaluation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
