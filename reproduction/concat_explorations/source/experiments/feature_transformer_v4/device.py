"""Device-aware training: identical algorithm, explicit device, CPU-portable checkpoints.

Why this is a separate module rather than a parameter on `fit`: `models.py`, `data.py` and
`training.py` are frozen — a live CPU trial is training against them, and the CPU results
already on disk must stay reproducible. `fit` already accepts any object exposing `batches`,
`to_sgd`, `actual_sgd` and `parity_signature`, so a device-moving wrapper reuses the whole
training loop without editing it.

Two rules this enforces:

* **One batch never straddles devices.** Features and targets are moved together, and any
  mask created here is created on the model's device.
* **Checkpoints stay CPU-portable.** Weights are saved with `map_location="cpu"` semantics
  (tensors are moved to CPU before serialisation), so a model trained on MPS reloads on a
  CPU-only host and reproduces its predictions.

CPU and MPS are **not** bitwise identical: the measured evaluation difference is ~3.7e-8
kSGD. Every manifest written here records the backend and the observed per-epoch time.
"""
from __future__ import annotations

import sys as _sys
import time
from dataclasses import dataclass, field
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import numpy as np
import torch

from .training import TrainConfig, metrics

SUPPORTED_DEVICES = ("cpu", "mps")


class DeviceError(RuntimeError):
    pass


def resolve_device(requested: str) -> torch.device:
    """Validate a device string; refuse MPS when the host cannot provide it."""
    if requested not in SUPPORTED_DEVICES:
        raise DeviceError(f"unknown device {requested!r}; available {SUPPORTED_DEVICES}")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise DeviceError("MPS was requested but torch.backends.mps.is_available() is False")
    return torch.device(requested)


def device_report(device: torch.device) -> dict:
    payload = {
        "device": str(device),
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "bitwise_identical_across_backends": False,
        "note": ("CPU and MPS kernels differ; the measured eval gap at batch 512 / 60 "
                 "tokens is ~3.7e-8 kSGD. Never compare a CPU number and an MPS number as "
                 "if the backend were not part of the setting."),
    }
    if device.type == "mps":
        payload["mps_macos_build"] = getattr(torch.backends.mps, "is_built", lambda: None)()
    return payload


class DeviceTensors:
    """`FeatureTensors` whose batches arrive already on the model's device.

    Everything else — batching order, shuffling, the anchor reconstruction — is delegated to
    the frozen implementation, so a device run and a CPU run see byte-identical inputs.
    """

    def __init__(self, base, device: torch.device, keep_cpu: bool = False):
        self.base = base
        self.device = device
        # metrics need CPU numpy; the parent tensors are already numpy so no copy arises
        self._keep_cpu = keep_cpu

    def __len__(self) -> int:
        return len(self.base)

    @property
    def actual_sgd(self) -> np.ndarray:
        return self.base.actual_sgd

    @property
    def anchor_ksgd(self) -> np.ndarray:
        return self.base.anchor_ksgd

    @property
    def target(self) -> np.ndarray:
        return self.base.target

    @property
    def row_id(self) -> np.ndarray:
        return self.base.row_id

    @property
    def horizon(self) -> np.ndarray:
        return self.base.horizon

    @property
    def numeric(self) -> np.ndarray:
        return self.base.numeric

    @property
    def categorical(self) -> np.ndarray:
        return self.base.categorical

    @property
    def query_context(self) -> np.ndarray:
        return self.base.query_context

    def to_sgd(self, raw: np.ndarray) -> np.ndarray:
        return self.base.to_sgd(raw)

    def subset(self, index) -> "DeviceTensors":
        return DeviceTensors(self.base.subset(index), self.device, self._keep_cpu)

    def parity_signature(self) -> str:
        return self.base.parity_signature()

    def batches(self, batch_size: int, shuffle: bool = False, seed: int = 0,
                with_target: bool = False):
        for item in self.base.batches(batch_size, shuffle=shuffle, seed=seed,
                                      with_target=with_target):
            if with_target:
                features, target = item
                yield ({k: v.to(self.device, non_blocking=False) for k, v in features.items()},
                       target.to(self.device, non_blocking=False))
            else:
                yield {k: v.to(self.device, non_blocking=False) for k, v in item.items()}


def move_model(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    return model.to(device)


def _score(actual: np.ndarray, predicted: np.ndarray) -> dict:
    residual = np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return {"rmse_sgd": float(np.sqrt(np.mean(residual ** 2))),
            "mae_sgd": float(np.mean(np.abs(residual))),
            "bias_sgd": float(np.mean(residual)), "n": int(residual.size)}


def predict_on(model: torch.nn.Module, tensors: DeviceTensors,
               batch_size: int = 4096) -> np.ndarray:
    """Predict in numpy float64, moving each batch's output back to CPU."""
    model.eval()
    out = []
    with torch.no_grad():
        for batch in tensors.batches(batch_size, shuffle=False):
            out.append(model(batch).detach().to("cpu").numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def cpu_state_dict(model: torch.nn.Module) -> dict:
    """A checkpoint dict that reloads anywhere: every tensor forced to CPU."""
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


def run_trial_on_device(entry: dict, config, state, a_fit, a_hold, run_dir, log_path,
                        device=None) -> dict:
    """One search trial on an explicit device, mirroring `search.run_trial` exactly."""
    if device is None:
        raise DeviceError("run_trial_on_device needs an explicit device")
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    from .training import build_for
    model = build_for(state.numeric_names, state.cardinalities(),
                      context_dim=a_fit.query_context.shape[1], config=config)

    def log(record):
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"trial": entry["id"], **record}) + "\n")

    dev_fit = DeviceTensors(a_fit, device)
    dev_hold = DeviceTensors(a_hold, device)
    result = fit_on(device, model, dev_fit, dev_hold, config, log=log)

    torch.save({"config": config.to_dict(), "state_dict": cpu_state_dict(result.model),
                "state": state.to_dict(), "pooling": config.pooling,
                "selected_epoch": result.selected_epoch, "device": str(device),
                "description": result.model.to("cpu").describe()},
               run_dir / "checkpoint.pt")
    # put the model back where it was so the caller can still predict on it
    result.model.to(device)

    inner_predictions = a_hold.to_sgd(predict_on(result.model, dev_hold))
    np.savez_compressed(run_dir / "inner_predictions.npz",
                        predicted_sgd=inner_predictions, actual_sgd=a_hold.actual_sgd,
                        row_id=a_hold.row_id, horizon=a_hold.horizon)

    summary = {
        "id": entry["id"], "pooling": config.pooling, "changes": entry["changes"],
        "selected_epoch": result.selected_epoch, "selected_rmse_sgd": result.selected_rmse,
        "inner_predictions": _score(a_hold.actual_sgd, inner_predictions),
        "ineligible_best": result.ineligible_best, "epochs_run": result.epochs_run,
        "seconds": round(time.perf_counter() - started, 2),
        "parameters": result.model.describe()["parameters"],
        "config": config.to_dict(), "description": result.model.describe(),
        "device": result.device_report, "curve": result.curve,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


@dataclass
class DeviceFitResult:
    model: torch.nn.Module
    curve: list[dict]
    epochs_run: int
    selected_epoch: int | None
    selected_rmse: float | None
    ineligible_best: dict | None
    seconds: float
    config: TrainConfig
    device_report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"config": self.config.to_dict(), "epochs_run": self.epochs_run,
                "selected_epoch": self.selected_epoch,
                "selected_rmse_sgd": self.selected_rmse,
                "ineligible_best": self.ineligible_best, "curve": self.curve,
                "seconds": round(self.seconds, 2), "device": self.device_report}


def fit_on(device: torch.device, model: torch.nn.Module, train: DeviceTensors,
           monitor: DeviceTensors, config: TrainConfig, log=None) -> DeviceFitResult:
    """The frozen training loop, run on `device`.

    Mirrors `training.fit` step for step — same optimiser, same cosine `T_max`, same
    eligibility floor, same patience — so the only difference from a CPU run is the backend.
    """
    from .training import _epoch, set_seed

    set_seed(config.seed)
    model = move_model(model, device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.schedule_length())

    curve: list[dict] = []
    best_state, best_epoch, best_rmse = None, None, float("inf")
    best_any_state, best_any_epoch, best_any_rmse = None, None, float("inf")
    stale = 0
    started = time.perf_counter()
    actual = monitor.actual_sgd

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        train_mse = _epoch(model, train, config, optimiser, epoch)
        current_lr = float(optimiser.param_groups[0]["lr"])
        predicted = monitor.to_sgd(predict_on(model, monitor))
        monitor_rmse = metrics(actual, predicted)["rmse_sgd"]
        scheduler.step()
        record = {"epoch": epoch, "train_mse": train_mse,
                  "monitor_rmse_sgd": monitor_rmse,
                  "seconds": round(time.perf_counter() - epoch_started, 2),
                  "learning_rate": current_lr, "device": str(device)}
        curve.append(record)
        if log is not None:
            log(record)

        if monitor_rmse < best_any_rmse - 1e-9:
            best_any_rmse, best_any_epoch = monitor_rmse, epoch
            best_any_state = cpu_state_dict(model)

        if epoch >= config.eligible_after:
            if monitor_rmse < best_rmse - 1e-9:
                best_rmse, best_epoch = monitor_rmse, epoch
                best_state = cpu_state_dict(model)
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    elif best_any_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_any_state.items()})

    ineligible = None
    if best_any_epoch is not None and (best_epoch is None or best_any_epoch < best_epoch):
        ineligible = {"epoch": best_any_epoch, "rmse_sgd": best_any_rmse,
                      "note": "better than the selected epoch but before the eligibility floor"}
        if best_epoch is None and best_any_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_any_state.items()})

    return DeviceFitResult(model=model, curve=curve, epochs_run=len(curve),
                           selected_epoch=best_epoch,
                           selected_rmse=None if best_epoch is None else best_rmse,
                           ineligible_best=ineligible,
                           seconds=time.perf_counter() - started, config=config,
                           device_report=device_report(device))


def reload_and_predict(checkpoint_path: _Path, numeric_names, cardinalities, context_dim: int,
                       tensors, config: TrainConfig, device: torch.device = torch.device("cpu")):
    """Load a checkpoint on CPU (or any device) and re-predict, never refitting."""
    from .models import build_model

    stored = torch.load(_Path(checkpoint_path), map_location="cpu", weights_only=False)
    model = build_model(numeric_names, cardinalities, context_dim=context_dim,
                        pooling=config.pooling, d_model=config.d_model,
                        layers=config.layers, heads=config.heads,
                        ff_multiplier=config.ff_multiplier, dropout=config.dropout)
    model.load_state_dict(stored["state_dict"])       # saved on CPU, loads anywhere
    model = move_model(model, device)
    model.eval()
    return predict_on(model, tensors), stored
