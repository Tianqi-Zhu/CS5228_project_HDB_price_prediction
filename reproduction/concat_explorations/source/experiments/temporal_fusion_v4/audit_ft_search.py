"""Load-only audit of finished feature_transformer_v4 search trials.

Reads two trial checkpoints, rebuilds the model on CPU, restores the saved
`StageState`, re-predicts the full A/holdout population and compares against the
trial's own saved inner predictions.

Guarantees, enforced in code rather than promised:

* **No fit of any kind.** `data.fit_stage_state` and `ScalerState.fit` are replaced by
  traps that raise; the model is only ever built from the checkpoint's own recorded
  config and state. The audit fails loudly if anything tries to fit.
* **CPU only.** `map_location="cpu"`. The trials ran on MPS, so the measured delta
  contains a backend difference as well as reconstruction error; this script reports
  both the raw delta and the backend note recorded in the trial, and never claims the
  two are bitwise comparable.
* **Row-level agreement.** `SOURCE_ROW_ID`, label and horizon are compared element-wise
  against the saved arrays and against the v2 contract.

Writes nothing except its own JSON report under
`reports/temporal_fusion_v4/v4_search_review/`.

Run:
    .venv/bin/python -m experiments.temporal_fusion_v4.audit_ft_search
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import torch  # noqa: E402

V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
SEARCH = ROOT / "reports" / "feature_transformer_v4" / "runs" / "v4-search-mps-s42"
OUT = ROOT / "reports" / "temporal_fusion_v4" / "v4_search_review"

PASSED: list[str] = []
FAILED: list[str] = []
NOTES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"PASS  {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"FAIL  {name}  {detail}")


def note(text: str) -> None:
    NOTES.append(text)
    print(f"NOTE  {text}")


@contextmanager
def forbid_fitting():
    """Make every fitting entry point raise for the duration of the block."""
    from neural_fusion_v2 import preprocessing
    from experiments.feature_transformer_v4 import data as ft_data

    originals = {
        "scaler_fit": preprocessing.ScalerState.fit,
        "stage_fit": ft_data.fit_stage_state,
    }

    def trap(*args, **kwargs):
        raise AssertionError("audit attempted to FIT; load-only is required")

    preprocessing.ScalerState.fit = staticmethod(trap)      # type: ignore[assignment]
    ft_data.fit_stage_state = trap                          # type: ignore[assignment]
    try:
        yield
    finally:
        preprocessing.ScalerState.fit = originals["scaler_fit"]  # type: ignore[assignment]
        ft_data.fit_stage_state = originals["stage_fit"]         # type: ignore[assignment]


def decompose(actual, predicted) -> dict:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    bias = float(error.mean())
    mse = float(np.mean(error ** 2))
    return {
        "rmse_sgd": float(np.sqrt(mse)),
        "mae_sgd": float(np.mean(np.abs(error))),
        "bias_sgd": bias,
        "error_sd_sgd": float(error.std()),
        "bias_share_of_mse": float(bias ** 2 / mse) if mse else None,
        "n": int(error.size),
    }


def audit_trial(trial: str) -> dict:
    from experiments.feature_transformer_v4.data import StageState, build_tensors
    from experiments.feature_transformer_v4.models import build_model
    from experiments.feature_transformer_v4.training import predict
    from neural_fusion_v2 import contract as C

    directory = SEARCH / trial
    print(f"\n{'=' * 70}\n{trial}\n{'=' * 70}")

    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu",
                            weights_only=False)
    saved = np.load(directory / "inner_predictions.npz")
    summary = json.loads((directory / "summary.json").read_text())

    # ---------------------------------------------------------------- integrity
    print("--- 1. checkpoint is self-describing")
    for key in ("config", "state_dict", "state", "pooling", "selected_epoch", "device"):
        check(f"{trial}: checkpoint carries {key}", key in checkpoint)
    check(f"{trial}: pooling agrees between checkpoint and summary",
          checkpoint["pooling"] == summary["pooling"])
    check(f"{trial}: selected_epoch agrees",
          int(checkpoint["selected_epoch"]) == int(summary["selected_epoch"]))
    state = StageState.from_dict(checkpoint["state"])
    check(f"{trial}: restored state was fitted on a fit population",
          "fit" in str(state.fitted_population),
          str(state.fitted_population))
    print(f"      fitted_population={state.fitted_population} n_rows={state.n_rows} "
          f"numeric_positions={len(state.numeric_names)} "
          f"cardinalities={state.cardinalities()}")
    check(f"{trial}: state records 56 numeric positions",
          len(state.numeric_names) == 56, str(len(state.numeric_names)))

    # ---------------------------------------------------------------- rebuild
    print("\n--- 2. rebuild from the recorded config only (fitting forbidden)")
    with forbid_fitting():
        model = build_model(state.numeric_names, state.cardinalities(),
                            context_dim=_context_dim(state),
                            pooling=checkpoint["config"]["pooling"],
                            d_model=checkpoint["config"]["d_model"],
                            layers=checkpoint["config"]["layers"],
                            heads=checkpoint["config"]["heads"],
                            ff_multiplier=checkpoint["config"]["ff_multiplier"],
                            dropout=checkpoint["config"]["dropout"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        check(f"{trial}: state_dict loads strictly", True)

        population = C.load_population(V2_DIR, C.load_manifest(V2_DIR), "orig", "A", "holdout")
        tensors = build_tensors(population, state)

    n_model = sum(p.numel() for p in model.parameters())
    print(f"      model parameters {n_model:,} (summary records {summary['parameters']:,})")
    check(f"{trial}: parameter count matches the summary",
          n_model == int(summary["parameters"]), f"{n_model} vs {summary['parameters']}")

    # ---------------------------------------------------------------- row agreement
    print("\n--- 3. row-level agreement with the saved inner predictions")
    check(f"{trial}: rebuilt population has 35,882 rows", len(tensors) == 35882,
          str(len(tensors)))
    check(f"{trial}: saved predictions have 35,882 rows",
          saved["predicted_sgd"].shape == (35882,))
    check(f"{trial}: SOURCE_ROW_ID identical to the saved array",
          np.array_equal(tensors.row_id, saved["row_id"].astype(np.int64)))
    check(f"{trial}: labels identical to the saved array",
          np.array_equal(tensors.actual_sgd, saved["actual_sgd"].astype(np.float64)))
    check(f"{trial}: horizons identical to the saved array",
          np.array_equal(tensors.horizon, saved["horizon"].astype(np.int32)))
    check(f"{trial}: row ids are unique",
          len(np.unique(tensors.row_id)) == len(tensors.row_id))

    # ---------------------------------------------------------------- re-predict
    print("\n--- 4. CPU re-prediction vs the trial's own numbers")
    raw = predict(model, tensors)
    predicted = tensors.to_sgd(raw)
    check(f"{trial}: predictions are finite", bool(np.isfinite(predicted).all()))

    rebuilt_metrics = decompose(tensors.actual_sgd, predicted)
    saved_metrics = decompose(saved["actual_sgd"], saved["predicted_sgd"])
    print(f"      saved      RMSE {saved_metrics['rmse_sgd']:.9f}  "
          f"MAE {saved_metrics['mae_sgd']:.9f}  bias {saved_metrics['bias_sgd']:+.6f}")
    print(f"      rebuilt    RMSE {rebuilt_metrics['rmse_sgd']:.9f}  "
          f"MAE {rebuilt_metrics['mae_sgd']:.9f}  bias {rebuilt_metrics['bias_sgd']:+.6f}")
    print(f"      summary    RMSE {summary['inner_predictions']['rmse_sgd']:.9f}  "
          f"MAE {summary['inner_predictions']['mae_sgd']:.9f}  "
          f"bias {summary['inner_predictions']['bias_sgd']:+.6f}")

    check(f"{trial}: saved-array metrics equal the summary's metrics",
          abs(saved_metrics["rmse_sgd"] - summary["inner_predictions"]["rmse_sgd"]) < 1e-9
          and abs(saved_metrics["mae_sgd"] - summary["inner_predictions"]["mae_sgd"]) < 1e-9)

    delta = float(np.max(np.abs(predicted - saved["predicted_sgd"].astype(np.float64))))
    rms = float(np.sqrt(np.mean((predicted - saved["predicted_sgd"]) ** 2)))
    check(f"{trial}: per-row predictions reproduce (backend-tolerant tolerance)",
          delta < 1e-2, f"max |delta| {delta:.3e} SGD")
    note(f"{trial}: max per-row |CPU rebuilt - saved MPS| = {delta:.6e} SGD, "
         f"RMS = {rms:.6e}; RMSE delta {rebuilt_metrics['rmse_sgd'] - saved_metrics['rmse_sgd']:+.3e}. "
         f"The trial ran on {checkpoint['device']}; {summary['device']['note']}")

    return {
        "trial": trial,
        "pooling": checkpoint["pooling"],
        "selected_epoch": int(checkpoint["selected_epoch"]),
        "params": n_model,
        "fitted_population": state.fitted_population,
        "cardinalities": list(state.cardinalities()),
        "saved_metrics": saved_metrics,
        "rebuilt_metrics": rebuilt_metrics,
        "summary_metrics": summary["inner_predictions"],
        "max_abs_row_delta_sgd": delta,
        "rms_row_delta_sgd": rms,
        "device_of_record": checkpoint["device"],
    }


def _context_dim(state) -> int:
    """`query_context` width the model was built with: context columns minus the anchor."""
    return len(state.scaler.context_columns) - 1


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = json.loads((SEARCH / "search_ledger.json").read_text())
    trials = {t["trial_id"]: t for t in ledger["trials"]}

    finished = {}
    for trial_id in sorted(trials):
        summary_path = SEARCH / f"trial{trial_id}" / "summary.json"
        if summary_path.exists():
            finished[trial_id] = json.loads(summary_path.read_text())["selected_rmse_sgd"]
    ranked = sorted(finished.items(), key=lambda item: item[1])
    print("finished trials ranked by inner RMSE:")
    for trial_id, rmse in ranked:
        print(f"  trial{trial_id}: {rmse:.6f}")
    top_two = [f"trial{trial_id}" for trial_id, _ in ranked[:2]]
    print(f"top two: {top_two}")

    check("the two audited trials are the finished top two",
          set(top_two) == {"trial6", "trial2"}, str(top_two))

    results = [audit_trial(name) for name in ("trial6", "trial2")]

    print("\n" + "=" * 70)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for item in FAILED:
        print(f"  - {item}")
    (OUT / "checkpoint_audit.json").write_text(json.dumps({
        "search_run": str(SEARCH.relative_to(ROOT)),
        "finished_ranking": [{"trial": f"trial{i}", "inner_rmse_sgd": v}
                             for i, v in ranked],
        "audited": results,
        "notes": NOTES,
        "passed": len(PASSED), "failed": FAILED,
    }, indent=2, default=str) + "\n")
    print(f"wrote {OUT / 'checkpoint_audit.json'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
