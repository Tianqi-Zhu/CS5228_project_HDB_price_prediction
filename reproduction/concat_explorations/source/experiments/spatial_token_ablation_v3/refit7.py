"""Fixed 7-epoch B refit diagnostic for the per-feature tokenisation arm.

Purpose, stated before the run so it cannot be chosen after seeing results:

  The existing `spatial-probe-perfeature-none-s42-h1` run selected 1 stage-B refit
  epoch because its inner early-stopping monitor peaked at epoch 1, while the block
  reference selected 7. Selecting each arm by its own inner rule is a legitimate
  pipeline comparison. This diagnostic does **not** re-litigate that. It asks a
  narrower question: holding the B refit at the block reference's 7 epochs, how much
  of the gap is attributable to the refit-length difference and how much survives?

  7 is pre-declared here because it is the block arm's already-recorded inner
  selection. It is not chosen from the outer validation RMSE, no epoch grid is run,
  and no "best outer epoch" is picked. This is a one-point diagnostic, not a search.

Everything else is held identical to the existing probe run:
  - same v2 contract and B/fit population (88,192 rows)
  - the **saved** fit-only stage-B scaler from the probe run, loaded from disk;
    never fitted on validation labels (asserted against a fresh B/fit fit)
  - `set_seed(42)` immediately before model construction, as `run_one` does
  - same optimiser (AdamW lr 1e-3, wd 1e-4, clip 1.0), same batch size 512, same
    batch-order seed (`seed * 1000 + epoch`), same `_epoch` implementation imported
    from the shared trainer
  - no shared source is modified; this script only reads it

Two phases:

  Phase 1 — initialisation equivalence. Rebuild the model and train exactly **1**
  epoch with `T_max=1` (what the existing run did), then compare the resulting
  `state_dict` bit-for-bit against the existing checkpoint. If they match, the
  initialisation, data order and optimiser are proven identical, so phase 2 starts
  from the same weights the existing run did.

  Phase 2 — the 7-epoch refit with `T_max=7` (what the block arm's refit did), with a
  real per-epoch training curve recorded.

Only the training loss is recorded per epoch. Per-epoch validation RMSE is
deliberately **not** recorded: it is the outer evaluation set, and storing it would
create the option of selecting an epoch by outer score.

Run:
    .venv/bin/python -m experiments.spatial_token_ablation_v3.refit7
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
RUNS = ROOT / "reports" / "neural_fusion_v3" / "runs"
PROBE_RUN = "spatial-probe-perfeature-none-s42-h1"
BLOCK_RUN = "macro-probe-block-none-s42"
OUT = ROOT / "reports" / "spatial_token_ablation_v3" / "refit7"

FIXED_EPOCHS = 7          # pre-declared: the block arm's recorded inner selection
CONTROL = "lstm_fusion"
TOKEN_MODE = "per_feature"
SPLIT = "orig"
SEED = 42

FAILURES: list[str] = []
NOTES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def note(text: str) -> None:
    NOTES.append(text)
    print(f"NOTE  {text}")


def decompose(actual, predicted) -> dict:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    bias = float(error.mean())
    mse = float(np.mean(error ** 2))
    return {
        "rmse_sgd": float(np.sqrt(mse)),
        "mae_sgd": float(np.mean(np.abs(error))),
        "bias_sgd": bias,
        "error_sd_sgd": float(error.std()),
        "bias_share_of_mse": float(bias ** 2 / mse) if mse else float("nan"),
    }


def build_model(shapes_b, columns):
    """Exactly `run_one`'s construction: set_seed then build, no expected_numeric."""
    from neural_fusion_v2.training import set_seed
    from neural_fusion_v3.models import build_control

    set_seed(SEED)
    return build_control(CONTROL, shapes_b, columns, token_mode=TOKEN_MODE)


def fit_loop(model, tensors, config, epochs: int) -> list[dict]:
    """`fit_fixed_epochs` verbatim (same optimiser/scheduler/`_epoch`), plus a curve.

    `_epoch` is imported from the shared trainer rather than reimplemented, so the
    batch order, clipping and scheduler step are literally the same code the existing
    runs used.
    """
    from neural_fusion_v2.training import _epoch

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                          T_max=max(epochs, 1))
    curve = []
    for epoch in range(1, epochs + 1):
        loss = _epoch(model, tensors, config, optimiser, scheduler, epoch)
        curve.append({"epoch": epoch, "train_mse_ksgd2": float(loss)})
    return curve


def main() -> int:
    from neural_fusion_v2 import contract as C
    from neural_fusion_v2.adapter import build_tensors
    from neural_fusion_v2.preprocessing import ScalerState
    from neural_fusion_v2.training import FitConfig, metrics, predict, predictions_frame
    from neural_fusion_v3.main import shapes_of
    from neural_fusion_v3.models import parameter_count

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = C.load_manifest(V2_DIR)
    encoding_b = C.load_encoding(V2_DIR, manifest, SPLIT, "B")

    a_fit = C.load_population(V2_DIR, manifest, SPLIT, "A", "fit")
    a_hold = C.load_population(V2_DIR, manifest, SPLIT, "A", "holdout")
    b_fit = C.load_population(V2_DIR, manifest, SPLIT, "B", "fit")
    b_val = C.load_population(V2_DIR, manifest, SPLIT, "B", "validation")
    print(f"B/fit {len(b_fit.arrays.row_id):,} rows, "
          f"B/validation {len(b_val.arrays.row_id):,} rows\n")

    # ---------------------------------------------------------------- 1
    print("--- 1. reuse the saved scaler; prove it was fitted on B/fit only")
    probe_dir = RUNS / PROBE_RUN
    saved_path = probe_dir / "preprocessing" / SPLIT / f"{CONTROL}__{TOKEN_MODE}_seed{SEED}_stageB.json"
    check("saved stage-B scaler file exists", saved_path.exists(), str(saved_path))
    state_b = ScalerState.from_dict(json.loads(saved_path.read_text()))

    fresh = ScalerState.fit(b_fit.arrays, population="B/fit")
    # Compare the full serialised state: any dependence on validation labels would
    # show up as a difference from a scaler seen only B/fit rows.
    saved_state, fresh_state = state_b.to_dict(), fresh.to_dict()
    differing = []
    for key in sorted(set(saved_state) | set(fresh_state)):
        left, right = saved_state.get(key), fresh_state.get(key)
        if json.dumps(left, sort_keys=True, default=str) != json.dumps(right, sort_keys=True, default=str):
            differing.append(key)
    check("loaded scaler equals a fresh B/fit fit (so it is fit-only, not label-fitted)",
          not differing, f"differing keys: {differing}")
    print(f"  fitted_population: {getattr(state_b, 'fitted_population', '?')}, "
          f"fitted_rows: {getattr(state_b, 'fitted_rows', '?')}")
    check("loaded scaler's static_columns match the contract order",
          list(getattr(state_b, "static_columns", [])) == list(fresh.static_columns))

    use_sequence = True
    t_b_fit = build_tensors(b_fit, state_b, use_sequence=use_sequence)
    t_b_val = build_tensors(b_val, state_b, use_sequence=use_sequence)
    shapes_b = shapes_of(b_fit, use_sequence, encoding_b)
    columns = state_b.static_columns
    config = FitConfig(control=CONTROL, seed=SEED, epochs=20, patience=5, batch_size=512)

    # ---------------------------------------------------------------- 2
    print("\n--- 2. phase 1: does my rebuild reproduce the existing run's init exactly?")
    reference = torch.load(probe_dir / "checkpoints" / SPLIT / f"{CONTROL}__{TOKEN_MODE}_seed{SEED}.pt",
                           map_location="cpu", weights_only=False)
    model = build_model(shapes_b, columns)
    n_params = parameter_count(model)
    # The run summary reports the STAGE-A model's count (56,011); this is the stage-B
    # model, whose third categorical cardinality is 20 rather than 19 in the A/fit
    # encoding, so one embedding row differs -> +8 parameters. Same architecture, so
    # this is a reporting-scope difference, not a structural one.
    print(f"  stage-B parameters: {n_params:,} "
          f"(run summary reports stage-A's {56011:,}; +8 from cardinality 20 vs 19)")
    check("stage-B parameter count is the stage-A count plus one embedding row",
          n_params == 56011 + 8, str(n_params))

    started = time.perf_counter()
    one_epoch_curve = fit_loop(model, t_b_fit, config, epochs=1)
    one_epoch_seconds = time.perf_counter() - started

    ref_state = reference["state_dict"]
    mine = model.state_dict()
    mismatched = [name for name in ref_state
                  if name not in mine
                  or not torch.equal(ref_state[name].detach().cpu(), mine[name].detach().cpu())]
    check("1-epoch refit reproduces the existing checkpoint bit-for-bit",
          not mismatched, f"{len(mismatched)} tensors differ, e.g. {mismatched[:4]}")
    if not mismatched:
        print("  -> initialisation, B/fit batch order, optimiser and scaler all proven "
              "identical to the existing runner")
    else:
        note("initialisation differs from the existing run; the 7-epoch diagnostic is "
             "then still a valid standalone run but is not traceable to the probe's "
             "starting weights")
    print(f"  phase-1 train loss: {one_epoch_curve[0]['train_mse_ksgd2']:.6f} "
          f"({one_epoch_seconds:.1f}s)")

    # ---------------------------------------------------------------- 3
    print(f"\n--- 3. phase 2: fixed {FIXED_EPOCHS}-epoch B refit from the same init")
    model7 = build_model(shapes_b, columns)
    started = time.perf_counter()
    curve = fit_loop(model7, t_b_fit, config, epochs=FIXED_EPOCHS)
    seconds = time.perf_counter() - started
    for row in curve:
        print(f"  epoch {row['epoch']}: train MSE (kSGD^2) {row['train_mse_ksgd2']:.6f}")

    # ---------------------------------------------------------------- 4
    print("\n--- 4. predict and save the new run")
    predicted = t_b_val.to_sgd(predict(model7, t_b_val))
    check("predictions are finite", bool(np.isfinite(predicted).all()))
    frame = predictions_frame(b_val, predicted, control=CONTROL, seed=SEED)
    frame.insert(8, "token_mode", TOKEN_MODE)
    frame.insert(9, "macro_mode", "none")
    frame["refit_epochs"] = FIXED_EPOCHS
    overall = metrics(frame["actual_rent_sgd"].to_numpy(),
                      frame["predicted_rent_sgd"].to_numpy())
    print(f"  RMSE {overall['rmse_sgd']:.6f}  MAE {overall['mae_sgd']:.6f}  "
          f"bias {overall['bias_sgd']:+.4f}  n={overall['n']:,}")
    note(f"fixed {FIXED_EPOCHS}-epoch refit used {seconds:.1f}s "
         f"(the 1-epoch probe's stage-B step took 22.5s)")

    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "predictions" / f"{CONTROL}__{TOKEN_MODE}_refit7_seed{SEED}.csv",
                 index=False)
    state_b.save(OUT / "checkpoints" / "stageB_scaler_reused.json")

    checkpoint = {
        "control": CONTROL, "token_mode": TOKEN_MODE, "macro_mode": "none",
        "split": SPLIT, "seed": SEED, "uses_sequence": use_sequence,
        "state_dict": model7.state_dict(), "shapes": shapes_b.__dict__,
        "numeric_columns": list(columns), "stage_b_state": state_b.to_dict(),
        "refit_epochs": FIXED_EPOCHS,
        "epochs_predeclared": True,
        "epochs_provenance": ("pre-declared as the block reference's recorded inner "
                              "early-stopping selection; not chosen by outer score"),
        "training_curve": curve,
        "provenance": {
            "contract_id": manifest["contract_id"],
            "contract_dir": str(V2_DIR.relative_to(ROOT)),
            "b_scaler_reused_from": str(saved_path.relative_to(ROOT)),
            "init_matches_probe_checkpoint": not mismatched,
            "derived_from": [PROBE_RUN, BLOCK_RUN],
        },
        "metrics": overall,
    }
    torch.save(checkpoint, OUT / "checkpoints" / f"{CONTROL}__{TOKEN_MODE}_refit7_seed{SEED}.pt")

    # ---------------------------------------------------------------- 5
    print("\n--- 5. reload without refitting and re-predict")
    stored = torch.load(OUT / "checkpoints" / f"{CONTROL}__{TOKEN_MODE}_refit7_seed{SEED}.pt",
                        map_location="cpu", weights_only=False)
    from neural_fusion_v3.models import Shapes, build_control
    recorded = dict(stored["shapes"])
    recorded["cardinalities"] = tuple(recorded["cardinalities"])
    restored_shapes = Shapes(**recorded)
    rebuilt = build_control(stored["control"], restored_shapes,
                            list(stored["numeric_columns"]), token_mode=stored["token_mode"])
    rebuilt.load_state_dict(stored["state_dict"], strict=True)
    restored_state = ScalerState.from_dict(stored["stage_b_state"])
    restored_tensors = build_tensors(b_val, restored_state, use_sequence=stored["uses_sequence"])
    rebuilt_sgd = restored_tensors.to_sgd(predict(rebuilt, restored_tensors))
    delta = float(np.max(np.abs(np.asarray(rebuilt_sgd) - predicted)))
    print(f"  max |original - reloaded| = {delta:.3e} SGD")
    check("checkpoint reload reproduces the predictions without refitting", delta < 1e-4)
    check("reload reproduces the recorded RMSE",
          abs(metrics(frame['actual_rent_sgd'].to_numpy(),
                      np.asarray(rebuilt_sgd))['rmse_sgd'] - overall["rmse_sgd"]) < 1e-9)

    # ---------------------------------------------------------------- 6
    print("\n--- 6. compare the three B refit lengths")
    block_frame = pd.read_csv(next((RUNS / BLOCK_RUN / "predictions" / SPLIT).glob("*.csv")))
    probe_frame = pd.read_csv(next((RUNS / PROBE_RUN / "predictions" / SPLIT).glob("*.csv")))

    rows = []
    for label, frame_in, epochs in (
            ("block / 7 ep", block_frame, 7),
            ("per_feature / 1 ep", probe_frame, 1),
            ("per_feature / 7 ep", frame, FIXED_EPOCHS)):
        stats = decompose(frame_in["actual_rent_sgd"], frame_in["predicted_rent_sgd"])
        stats["arm"] = label
        stats["refit_epochs"] = epochs
        rows.append(stats)
    table = pd.DataFrame(rows)[["arm", "refit_epochs", "rmse_sgd", "mae_sgd",
                                "bias_sgd", "error_sd_sgd"]]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    block_stats, probe_stats, seven_stats = rows
    gap_total = probe_stats["rmse_sgd"] - block_stats["rmse_sgd"]
    gap_after = seven_stats["rmse_sgd"] - block_stats["rmse_sgd"]
    recovered = gap_total - gap_after
    print(f"\n  block -> per_feature@1ep  : {gap_total:+.3f} SGD")
    print(f"  block -> per_feature@7ep  : {gap_after:+.3f} SGD")
    print(f"  recovered by 1->7 epochs  : {recovered:+.3f} SGD "
          f"({recovered / gap_total * 100:.0f}% of the original gap)")
    print(f"  bias  1ep {probe_stats['bias_sgd']:+.2f} -> 7ep {seven_stats['bias_sgd']:+.2f} "
          f"(block {block_stats['bias_sgd']:+.2f})")
    print(f"  errSD 1ep {probe_stats['error_sd_sgd']:.2f} -> 7ep "
          f"{seven_stats['error_sd_sgd']:.2f} (block {block_stats['error_sd_sgd']:.2f})")

    # row identity against both references
    for label, other in (("block", block_frame), ("per_feature@1ep", probe_frame)):
        check(f"row ids and labels identical to {label}",
              bool((other["SOURCE_ROW_ID"].to_numpy() == frame["SOURCE_ROW_ID"].to_numpy()).all())
              and float(np.max(np.abs(other["actual_rent_sgd"].to_numpy()
                                      - frame["actual_rent_sgd"].to_numpy()))) == 0.0)

    summary = {
        "run": f"{CONTROL}__{TOKEN_MODE}_refit7_seed{SEED}",
        "fixed_epochs": FIXED_EPOCHS,
        "epochs_predeclared_not_searched": True,
        "init_matches_probe_checkpoint": not mismatched,
        "scaler_reused_from": str(saved_path.relative_to(ROOT)),
        "scaler_equals_fresh_Bfit": not differing,
        "training_curve": curve,
        "refit_seconds": seconds,
        "metrics": overall,
        "reload_max_abs_delta_sgd": delta,
        "n_parameters_stage_b": n_params,
        "n_parameters_stage_a_reported": 56011,
        "parameter_count_note": (
            "the run summary reports the stage-A model (56,011); the saved stage-B "
            "checkpoint has 56,019 because the B/fit encoding has one extra category "
            "(cardinality 20 vs 19). Same architecture, one embedding row."),
        "arms": rows,
        "gap_block_to_1ep": gap_total,
        "gap_block_to_7ep": gap_after,
        "gap_recovered_by_7ep": recovered,
        "gap_recovered_fraction": recovered / gap_total if gap_total else None,
        "notes": NOTES,
        "failures": FAILURES,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    table.to_csv(OUT / "arm_comparison.csv", index=False)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print(f"fixed {FIXED_EPOCHS}-epoch diagnostic complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
