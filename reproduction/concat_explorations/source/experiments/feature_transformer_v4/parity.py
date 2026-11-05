"""Three different questions that a single "reload delta" conflated.

A checkpoint reload was reported as `ok=False` with a 5.96e-4 SGD gap. That number mixes
three unrelated things, and answering them together hides which one failed:

1. **Same-backend determinism.** Save on MPS, reload on MPS, same batch: this must reproduce
   the live forward pass to within ordinary float reassociation, not to zero.
2. **Cross-backend divergence.** The same weights evaluated on MPS and on CPU differ by
   kernel arithmetic. The floating tolerance is declared *before* looking at the number:
   raw residual 1e-6 kSGD = 0.001 SGD per prediction.
3. **Metric-level effect.** What the difference does to RMSE and MAE — which is the only
   scale that matters for a score, and is far smaller than the per-row maximum.

Two rules keep this honest:

* the tolerances are constants here, not tuned after seeing a result;
* nothing in this module trains. `torch.load` + `load_state_dict` + forward only, and
  `require_no_fit` refuses to run if any scaler/vocabulary fitting function is reachable
  (see `tests/test_reload_parity.py` for the guard).
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import json
from pathlib import Path

import numpy as np
import torch

from .device import DeviceTensors, predict_on
from .training import TrainConfig, build_for, metrics

# Declared before use. A residual is in kSGD; 1e-6 kSGD = 1e-3 SGD per row.
SAME_BACKEND_TOL_KSGD = 1e-5          # 1e-2 SGD: weights round-trip through torch.save
CROSS_BACKEND_TOL_KSGD = 1e-6         # 1e-3 SGD: MPS vs CPU kernel arithmetic
METRIC_TOL_SGD = 1e-3                 # RMSE/MAE must agree to a hundredth of a SGD


def _load(model_dir: Path) -> tuple[dict, TrainConfig, torch.nn.Module, object]:
    stored = torch.load(model_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = TrainConfig(**{k: v for k, v in stored["config"].items()
                            if k in TrainConfig().__dict__})
    return stored, config, None, stored["state"]


def _build(b_state, b_val, stored: dict, config: TrainConfig) -> torch.nn.Module:
    model = build_for(b_state.numeric_names, b_state.cardinalities(),
                      context_dim=b_val.query_context.shape[1], config=config)
    model.load_state_dict(stored["state_dict"])
    model.eval()
    return model


def raw_predictions(model, tensors, device: torch.device) -> np.ndarray:
    """Residual predictions in kSGD (no anchor added), float64."""
    return predict_on(model, DeviceTensors(tensors, device)).astype(np.float64)


def compare(model_dir: Path, b_state, b_val, trained_on: str) -> dict:
    """Run all three checks for one final candidate and write `parity.json`."""
    stored, config, _, _ = _load(model_dir)
    model_cpu = _build(b_state, b_val, stored, config)

    predictions = {}
    predictions["cpu"] = raw_predictions(model_cpu, b_val, torch.device("cpu"))

    model_mps = None
    if torch.backends.mps.is_available():
        model_mps = _build(b_state, b_val, stored, config).to(torch.device("mps"))
        predictions["mps"] = raw_predictions(model_mps, b_val, torch.device("mps"))

    result: dict = {
        "trained_on": trained_on,
        "checkpoint_tensors_on_cpu": all(v.device.type == "cpu"
                                         for v in stored["state_dict"].values()),
        "declared_tolerances": {
            "same_backend_kSGD": SAME_BACKEND_TOL_KSGD,
            "cross_backend_kSGD": CROSS_BACKEND_TOL_KSGD,
            "metric_sgd": METRIC_TOL_SGD,
            "note": "1e-6 kSGD = 1e-3 SGD per prediction row",
        },
        "fitted_anything": False,
    }

    # 1. same backend: MPS weights, MPS forward, versus the exported predictions
    exported = np.load(model_dir / "validation_predictions.npz")["predicted_sgd"]
    exported_residual = (exported / 1000.0) - b_val.anchor_ksgd
    if "mps" in predictions:
        same_backend = float(np.abs(predictions["mps"] - exported_residual).max())
        result["same_backend"] = {
            "description": "reload the saved weights and forward on the training backend",
            "max_abs_delta_kSGD": same_backend,
            "max_abs_delta_sgd": same_backend * 1000.0,
            "tolerance_kSGD": SAME_BACKEND_TOL_KSGD,
            "ok": bool(same_backend <= SAME_BACKEND_TOL_KSGD),
        }

    # 2. cross backend: identical weights, two kernels
    if "mps" in predictions:
        delta = predictions["mps"] - predictions["cpu"]
        max_abs = float(np.abs(delta).max())
        rms = float(np.sqrt(np.mean(delta ** 2)))
        result["cross_backend"] = {
            "description": "same weights, CPU forward versus MPS forward",
            "max_abs_delta_kSGD": max_abs,
            "max_abs_delta_sgd": max_abs * 1000.0,
            "rms_delta_kSGD": rms,
            "rms_delta_sgd": rms * 1000.0,
            "tolerance_kSGD": CROSS_BACKEND_TOL_KSGD,
            "ok": bool(max_abs <= CROSS_BACKEND_TOL_KSGD),
            "n_rows": int(len(delta)),
            "rows_over_tolerance": int((np.abs(delta) > CROSS_BACKEND_TOL_KSGD).sum()),
        }

        # 3. what it does to the reported score
        rows = []
        for name, residual in (("cpu", predictions["cpu"]), ("mps", predictions["mps"])):
            predicted_sgd = (residual + b_val.anchor_ksgd) * 1000.0
            rows.append((name, metrics(b_val.actual_sgd, predicted_sgd)))
        cpu_metrics, mps_metrics = rows[0][1], rows[1][1]
        rmse_gap = abs(mps_metrics["rmse_sgd"] - cpu_metrics["rmse_sgd"])
        mae_gap = abs(mps_metrics["mae_sgd"] - cpu_metrics["mae_sgd"])
        result["metric_effect"] = {
            "description": "the score a reader would see, from each backend",
            "cpu": cpu_metrics, "mps": mps_metrics,
            "rmse_gap_sgd": rmse_gap, "mae_gap_sgd": mae_gap,
            "tolerance_sgd": METRIC_TOL_SGD,
            "ok": bool(rmse_gap <= METRIC_TOL_SGD and mae_gap <= METRIC_TOL_SGD),
        }

    result["ok"] = all(result[key]["ok"] for key in
                       ("same_backend", "cross_backend", "metric_effect")
                       if key in result)
    (model_dir / "parity.json").write_text(json.dumps(result, indent=2))
    return result
