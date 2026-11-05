"""Gate 1 CLI: four controls on the producer's real v2 contract.

    env -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
      -m experiments.neural_fusion_v2.main --contract-id <id> --split orig \
      --controls all --mode smoke

Population roles (fixed by the producer, not chosen here):

    A/fit        -> fit stage-A scalers
    A/holdout    -> early stopping only
    B/fit        -> refit, fixed epoch count, last weights kept
    B/validation -> outer evaluation

Writes only under ``reports/neural_fusion_v2/runs/<run_id>/``. Exits 3 when the
contract is absent; never fabricates inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import contract as C
from .adapter import build_tensors
from .models import Shapes, build_control
from .preprocessing import ScalerState
from .settings import CAPACITY, CONTROLS, MODES, PACKAGE, REPORT_ROOT, TRAIN
from .training import (FitConfig, band_report, build, fit_early_stopping, fit_fixed_epochs,
                       metrics, monthly_report, predict, predictions_frame, save_json, set_seed)

EXIT_CONTRACT_NOT_READY = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint() -> dict:
    return {str(p.relative_to(PACKAGE)): sha256_file(p)
            for p in sorted(PACKAGE.rglob("*.py"))}


def environment() -> dict:
    return {
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "torch": torch.__version__, "platform": platform.platform(), "host": socket.gethostname(),
        "torch_threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
        "device": "cpu", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }


def shapes_of(population, use_sequence: bool, encoding: dict) -> Shapes:
    a = population.arrays
    cards = C.encoding_cardinalities(encoding, a)
    channels = a.n_channels if use_sequence else 0
    length = a.n_months * a.n_levels if use_sequence else 0
    return Shapes(
        n_categorical=len(a.categorical_columns), static_dim=len(a.numeric_columns),
        history_dim=len(a.history_order), context_dim=len(a.context_order) - 1,
        cardinalities=tuple(cards), sequence_channels=channels, sequence_length=length)


def load_split(contract_dir: Path, manifest: dict, split: str):
    """Load the v2.1 population set: A/fit, A/holdout, B/fit, B/validation.

    Stage A deliberately has no validation population; asking for one would be a copy of
    B/validation and is not part of this contract.
    """
    a_fit = C.load_population(contract_dir, manifest, split, "A", "fit")
    a_hold = C.load_population(contract_dir, manifest, split, "A", "holdout")
    b_fit = C.load_population(contract_dir, manifest, split, "B", "fit")
    b_val = C.load_population(contract_dir, manifest, split, "B", "validation")
    report = C.validate_boundaries(manifest, split, a_fit, a_hold, b_fit, b_val)
    return a_fit, a_hold, b_fit, b_val, report


def run_control(control: str, split: str, contract_dir: Path, manifest: dict, pops,
                encoding_a: dict, encoding_b: dict, mode: str, seed: int,
                run_dir: Path, command: str) -> dict:
    a_fit, a_hold, b_fit, b_val, bounds = pops
    spec = MODES[mode]
    use_sequence = control in ("lstm_concat", "lstm_fusion")
    started = time.perf_counter()

    row_cap = spec["row_cap"]
    if row_cap is not None:
        a_fit = a_fit.subset(np.arange(min(row_cap, len(a_fit))))
        a_hold = a_hold.subset(np.arange(min(row_cap, len(a_hold))))
        b_fit = b_fit.subset(np.arange(min(row_cap, len(b_fit))))

    # stage-A state: fitted on A/fit only
    state_a = ScalerState.fit(a_fit.arrays, population="A/fit")
    tensors_a_fit = build_tensors(a_fit, state_a, use_sequence=use_sequence)
    tensors_a_hold = build_tensors(a_hold, state_a, use_sequence=use_sequence)

    shapes = shapes_of(a_fit, use_sequence, encoding_a)
    config = FitConfig(control=control, seed=seed, epochs=spec["epochs"],
                       patience=spec["patience"], batch_size=spec["batch_size"])

    set_seed(seed)
    model_a, n_params = build(control, shapes)
    stage_a = fit_early_stopping(model_a, tensors_a_fit, tensors_a_hold, config)
    chosen = max(stage_a.best_epoch, 1)

    # stage-B state: fitted on B/fit only; A/validation is not used at all
    state_b = ScalerState.fit(b_fit.arrays, population="B/fit")
    tensors_b_fit = build_tensors(b_fit, state_b, use_sequence=use_sequence)
    tensors_b_val = build_tensors(b_val, state_b, use_sequence=use_sequence)

    shapes_b = shapes_of(b_fit, use_sequence, encoding_b)
    # Stage A and B legitimately have different vocabularies, so the categorical
    # cardinalities may differ; the numeric widths must not.
    if (shapes_b.static_dim, shapes_b.history_dim, shapes_b.context_dim,
            shapes_b.n_categorical, shapes_b.sequence_channels) != (
            shapes.static_dim, shapes.history_dim, shapes.context_dim,
            shapes.n_categorical, shapes.sequence_channels):
        raise AssertionError(
            f"A and B feature shapes disagree: {shapes} vs {shapes_b} (only cardinalities "
            "may differ between stages)")

    set_seed(seed)   # seed BEFORE construction: reproducible stage-B initialisation
    model_b, _ = build(control, shapes_b)
    stage_b_started = time.perf_counter()
    fit_fixed_epochs(model_b, tensors_b_fit, config, chosen)
    stage_b_seconds = time.perf_counter() - stage_b_started

    predicted = tensors_b_val.to_sgd(predict(model_b, tensors_b_val))
    if not np.isfinite(predicted).all():
        raise FloatingPointError(f"{control}: non-finite predictions")
    frame = predictions_frame(b_val, predicted, control=control, seed=seed)
    overall = metrics(frame["actual_rent_sgd"].to_numpy(), frame["predicted_rent_sgd"].to_numpy())

    paths = {name: run_dir / name / split for name in ("predictions", "checkpoints",
                                                       "metrics", "preprocessing")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths["predictions"] / f"{control}_seed{seed}.csv", index=False)
    state_a.save(paths["preprocessing"] / f"{control}_seed{seed}_stageA.json")
    state_b.save(paths["preprocessing"] / f"{control}_seed{seed}_stageB.json")
    torch.save({
        "control": control, "state_dict": model_b.state_dict(),
        "shapes": shapes_b.__dict__, "capacity": dict(CAPACITY), "seed": seed, "split": split,
        "contract_id": manifest["contract_id"], "contract_version": manifest["contract_version"],
        "chosen_epochs": chosen, "uses_sequence": use_sequence,
        "stage_a_state": state_a.to_dict(), "stage_b_state": state_b.to_dict(),
        "cardinalities_A": list(shapes.cardinalities),
        "cardinalities_B": list(shapes_b.cardinalities),
        "columns": tensors_b_val.columns,
    }, paths["checkpoints"] / f"{control}_seed{seed}.pt")

    result = {
        "run_id": run_dir.name, "mode": mode, "contract_id": manifest["contract_id"],
        "split": split, "control": control, "seed": seed, "uses_sequence": use_sequence,
        "n_parameters": n_params,
        "n_A_fit": len(a_fit), "n_A_holdout": len(a_hold), "n_B_fit": len(b_fit),
        "n_B_validation": len(b_val),
        "chosen_epochs": chosen, "stage_a_epochs_run": stage_a.epochs_run,
        "stage_a_best_holdout_rmse_sgd": stage_a.best_inner_rmse,
        "stage_a_seconds": round(stage_a.seconds, 2), "stage_b_seconds": round(stage_b_seconds, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
        "overall": overall, "by_horizon_band": band_report(frame),
        "by_month": monthly_report(frame).to_dict(orient="records"),
        "parity_signature_B_fit": tensors_b_fit.parity_signature(),
        "parity_signature_B_validation": tensors_b_val.parity_signature(),
        "boundaries": bounds,
        "stage_a_scaler_fitted_on": "A/fit",
        "stage_b_scaler_fitted_on": "B/fit",
        "actual_manifest": config.actual_manifest(shapes_b),
        "command": command,
        "stage_a_curve": stage_a.history,
    }
    save_json(paths["metrics"] / f"{control}_seed{seed}.json", result)
    return {k: v for k, v in result.items() if k not in ("by_month", "stage_a_curve")}


def verify_checkpoint(run_dir: Path, split: str, control: str, seed: int,
                      contract_dir: Path, manifest: dict) -> dict:
    stored = torch.load(run_dir / "checkpoints" / split / f"{control}_seed{seed}.pt",
                        map_location="cpu", weights_only=False)
    model = build_control(stored["control"], Shapes(**stored["shapes"]))
    model.load_state_dict(stored["state_dict"])
    b_val = C.load_population(contract_dir, manifest, split, "B", "validation")
    state_b = ScalerState.from_dict(stored["stage_b_state"])
    tensors = build_tensors(b_val, state_b, use_sequence=stored["uses_sequence"])
    rebuilt = tensors.to_sgd(predict(model, tensors))
    exported = pd.read_csv(run_dir / "predictions" / split / f"{control}_seed{seed}.csv")
    if len(exported) != len(rebuilt):
        return {"ok": False, "reason": f"row mismatch {len(exported)} vs {len(rebuilt)}"}
    delta = float(np.abs(exported["predicted_rent_sgd"].to_numpy() - rebuilt).max())
    return {"ok": delta < 1e-4, "max_abs_delta_sgd": delta, "n": int(len(rebuilt))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gate 1 neural controls on the v2 contract")
    parser.add_argument("--contract-id", required=True)
    parser.add_argument("--contract-root", default=None)
    parser.add_argument("--split", default="orig")
    parser.add_argument("--controls", default="all")
    parser.add_argument("--seeds", default=str(TRAIN["seed"]))
    parser.add_argument("--mode", default="full", choices=list(MODES))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args(argv)

    controls = CONTROLS if args.controls == "all" else [c.strip() for c in args.controls.split(",") if c.strip()]
    unknown = [c for c in controls if c not in CONTROLS]
    if unknown:
        raise SystemExit(f"unknown controls {unknown}; available {CONTROLS}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    torch.set_num_threads(TRAIN["threads"])
    root = Path(args.contract_root) if args.contract_root else (
        PACKAGE.parents[1] / "reports" / "fusion_contract_v2")
    contract_dir = root / args.contract_id
    try:
        manifest = C.load_manifest(contract_dir)
        encoding_a = C.load_encoding(contract_dir, manifest, args.split, "A")
        encoding_b = C.load_encoding(contract_dir, manifest, args.split, "B")
        pops = load_split(contract_dir, manifest, args.split)
    except (C.ContractNotReady, C.ContractInvalid) as error:
        print(f"CONTRACT NOT READY: {error}", file=sys.stderr)
        print("Nothing was written.", file=sys.stderr)
        return EXIT_CONTRACT_NOT_READY

    run_id = args.run_id or f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = REPORT_ROOT / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"run dir already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    command = " ".join([".venv/bin/python", "-m", "experiments.neural_fusion_v2.main"] +
                       (argv if argv is not None else sys.argv[1:]))
    framing = {
        "run_id": run_id, "mode": args.mode, "contract_id": manifest["contract_id"],
        "contract_version": manifest["contract_version"], "contract_dir": str(contract_dir),
        "contract_source_sha256": manifest.get("source_sha256"),
        "contract_fingerprint": C.contract_fingerprint(contract_dir),
        "code_fingerprint": code_fingerprint(), "environment": environment(),
        "capacity": dict(CAPACITY), "args": vars(args), "controls": controls, "seeds": seeds,
        "split_metadata": manifest["splits"][args.split]["split"],
        "feature_order": manifest["splits"][args.split]["feature_order"],
        "encoding": {
            "A": {"path": "features/%s/encodingA.json" % args.split,
                  "fitted_on": encoding_a.get("fitted_on"),
                  "n_fit_rows": encoding_a.get("n_fit_rows"),
                  "unknown_code": encoding_a.get("unknown_code")},
            "B": {"path": "features/%s/encodingB.json" % args.split,
                  "fitted_on": encoding_b.get("fitted_on"),
                  "n_fit_rows": encoding_b.get("n_fit_rows"),
                  "unknown_code": encoding_b.get("unknown_code")},
        },
        "population_boundaries": pops[-1],
        "contract_deviations": pops[-1].get("deviations", []),
        "command": command, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "result_status": ("REAL_CONTRACT_PENDING_FIX" if pops[-1].get("deviations")
                          else "REAL_CONTRACT"),
        "result_disclaimer": (
            "Contract deviations recorded in contract_deviations. These numbers are "
            "diagnostic only and are NOT Gate 1 comparison results; the contract owner is "
            "regenerating the affected populations."
            if pops[-1].get("deviations") else
            "Real contract inputs with all boundary checks satisfied."),
    }
    save_json(run_dir / "run_manifest.json", framing)

    results, checks = [], []
    for control in controls:
        for seed in seeds:
            print(f"[{args.split}] {control} seed={seed} mode={args.mode} ...", flush=True)
            result = run_control(control, args.split, contract_dir, manifest, pops,
                                 encoding_a, encoding_b, args.mode, seed, run_dir, command)
            results.append(result)
            print(f"    RMSE {result['overall']['rmse_sgd']:.2f} SGD | "
                  f"epochs {result['chosen_epochs']} | {result['n_parameters']} params | "
                  f"{result['total_seconds']:.1f}s", flush=True)
            if args.verify_checkpoints:
                check = verify_checkpoint(run_dir, args.split, control, seed, contract_dir,
                                          manifest)
                checks.append({"control": control, "seed": seed, **check})
                print(f"    checkpoint reload: {check}", flush=True)

    summary = {**framing, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "results": results, "checkpoint_verification": checks}
    save_json(run_dir / "run_summary.json", summary)
    with open(REPORT_ROOT / "commands.log", "a", encoding="utf-8") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [{run_id}] {command} (exit 0)\n")
    print(f"saved {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
