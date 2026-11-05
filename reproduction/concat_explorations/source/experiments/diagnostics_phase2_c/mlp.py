"""MLP control on the same protocol as the Transformer.

Required by the plan: "Minimum MLP 2-layer 256 with explicit optimizer/budget/inner epoch
choice." So the architecture, optimizer and budget are declared here as constants, and the
only thing chosen on A/holdout is the epoch — the same kind of inner selection the
Transformer's trials use, which keeps the comparison honest.

Declared, not tuned:

    width            256 hidden, 2 layers
    activation       GELU
    dropout          0.05 (same as the Transformer)
    optimiser        AdamW, lr 1e-3, weight decay 1e-4
    batch            512
    schedule         cosine over the full 60-epoch budget, identical in every arm
    max epochs       60
    selection        best A/holdout RMSE over epochs 5..60 (epochs 1-4 ineligible, as in v4)
    patience         10

The input is the frozen protocol table: 56 numeric + one-hot categoricals, no anchor as a
feature, residual target, anchor added back at scoring time.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import protocol as P

OUT = Path(__file__).resolve().parents[2] / "reports" / "diagnostics_phase2_c"

WIDTH = 256
LAYERS = 2
DROPOUT = 0.05
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 512
MAX_EPOCHS = 60
PATIENCE = 10
ELIGIBLE_AFTER = 5
SEED = 42


class MLP(nn.Module):
    """2 x 256 GELU with a scalar head; one output, the residual in kSGD."""

    def __init__(self, n_features: int):
        super().__init__()
        blocks: list[nn.Module] = []
        width = n_features
        for _ in range(LAYERS):
            blocks += [nn.Linear(width, WIDTH), nn.GELU(), nn.Dropout(DROPOUT)]
            width = WIDTH
        blocks.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class MLPResult:
    curve: list[dict] = field(default_factory=list)
    selected_epoch: int | None = None
    selected_rmse: float | None = None
    ineligible_best: dict | None = None
    epochs_run: int = 0
    seconds: float = 0.0


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_mlp(train: P.Table, hold: P.Table, log_path: Path | None = None) -> tuple[MLP, MLPResult]:
    _set_seed(SEED)
    X_fit, names = train.design_matrix(one_hot=True)
    X_hold, _ = hold.design_matrix(one_hot=True)
    X_fit_t = torch.from_numpy(X_fit.astype(np.float32))
    y_fit_t = torch.from_numpy(train.target.astype(np.float32))
    X_hold_t = torch.from_numpy(X_hold.astype(np.float32))

    model = MLP(X_fit.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=MAX_EPOCHS)
    loss_fn = nn.MSELoss()

    result = MLPResult()
    best_state, best_any, best_any_epoch, best_any_rmse = None, None, None, float("inf")
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.perf_counter()
        model.train(True)
        order = np.random.default_rng(SEED * 1000 + epoch).permutation(len(X_fit_t))
        total, seen = 0.0, 0
        for start in range(0, len(order), BATCH):
            index = torch.from_numpy(order[start:start + BATCH])
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(X_fit_t[index]), y_fit_t[index])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += float(loss.item()) * len(index)
            seen += len(index)
        current_lr = float(optimiser.param_groups[0]["lr"])
        scheduler.step()

        model.eval()
        with torch.no_grad():
            hold_residual = model(X_hold_t).numpy().astype(np.float64)
        hold_metrics = P.score(hold, hold_residual)
        record = {"epoch": epoch, "train_mse": total / max(seen, 1),
                  "monitor_rmse_sgd": hold_metrics["rmse_sgd"],
                  "seconds": round(time.perf_counter() - epoch_started, 2),
                  "learning_rate": current_lr}
        result.curve.append(record)
        if log_path is not None:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

        if hold_metrics["rmse_sgd"] < best_any_rmse - 1e-9:
            best_any_rmse, best_any_epoch = hold_metrics["rmse_sgd"], epoch
            best_any = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch >= ELIGIBLE_AFTER:
            if result.selected_rmse is None or hold_metrics["rmse_sgd"] < result.selected_rmse - 1e-9:
                result.selected_rmse = hold_metrics["rmse_sgd"]
                result.selected_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    elif best_any is not None:
        model.load_state_dict(best_any)

    if best_any_epoch is not None and (result.selected_epoch is None
                                       or best_any_epoch < result.selected_epoch):
        result.ineligible_best = {"epoch": best_any_epoch, "rmse_sgd": best_any_rmse,
                                  "note": "before the eligibility floor; reported, not used"}
    result.epochs_run = len(result.curve)
    result.seconds = time.perf_counter() - started
    return model, result


def run_mlp(train: P.Table, hold: P.Table, name: str = "mlp") -> dict:
    out = OUT / "stageA"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{name}_epochs.jsonl"
    if log_path.exists():
        log_path.unlink()

    model, result = train_mlp(train, hold, log_path)
    X_hold, names = hold.design_matrix(one_hot=True)
    model.eval()
    with torch.no_grad():
        residual = model(torch.from_numpy(X_hold.astype(np.float32))).numpy().astype(np.float64)
    predicted = (residual + hold.anchor_ksgd) * P.TARGET_SCALE

    P.export_predictions(out / f"{name}_holdout.csv", hold, predicted, name)
    checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                  "n_features": int(X_hold.shape[1]),
                  "declared": {"width": WIDTH, "layers": LAYERS, "dropout": DROPOUT,
                               "lr": LR, "weight_decay": WEIGHT_DECAY, "batch": BATCH,
                               "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                               "eligible_after": ELIGIBLE_AFTER, "seed": SEED}}
    torch.save(checkpoint, out / f"{name}_checkpoint.pt")

    summary = {
        "method": name, "stage": "A", "n_fit": len(train), "n_holdout": len(hold),
        "anchor_as_feature": False,
        "protocol_label": "main protocol: 56 numeric + 4 categorical, no anchor input",
        "n_features": int(X_hold.shape[1]), "feature_names": names,
        "declared": checkpoint["declared"],
        "selected_epoch": result.selected_epoch, "epochs_run": result.epochs_run,
        "ineligible_best": result.ineligible_best,
        "holdout": P.score(hold, residual), "curve": result.curve,
        "anchor_only_holdout": P.anchor_only(hold),
        "anchor_only_fit": P.anchor_only(train),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "seconds": round(result.seconds, 2),
        "protocol": train.describe(),
    }
    (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="mlp")
    args = parser.parse_args(argv)

    torch.set_num_threads(2)
    train, hold = P.load_tables()
    print(f"[{args.name}] {train.numeric.shape[1]} numeric + one-hot, "
          f"{len(train)} fit / {len(hold)} holdout, MSE on residual", flush=True)
    summary = run_mlp(train, hold, args.name)
    h = summary["holdout"]
    print(f"    epoch {summary['selected_epoch']} of {summary['epochs_run']} | "
          f"holdout RMSE {h['rmse_sgd']:.4f} | MAE {h['mae_sgd']:.4f} | "
          f"bias {h['bias_sgd']:.4f} | {summary['parameters']} params | "
          f"{summary['seconds']}s", flush=True)
    print(f"    anchor-only holdout RMSE "
          f"{summary['anchor_only_holdout']['rmse_sgd']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
