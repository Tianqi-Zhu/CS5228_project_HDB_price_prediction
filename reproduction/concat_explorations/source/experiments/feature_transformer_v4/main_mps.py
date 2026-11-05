"""CLI for the MPS run of the same eight candidates.

    # gate: real rental-loss training on MPS, then a CPU-only reload that must match
    ... -m experiments.feature_transformer_v4.main_mps --stage smoke --device mps \
        --run-id v4-mps-smoke-s42

    # all eight candidates on one backend
    ... --stage trials --device mps --run-id v4-search-mps-s42 --resume

Why a separate entry point: `main.py` writes CPU results under `v4-search-s42`, and those
runs are the reference. Two results produced on different backends must never share a run id
or a ledger, because CPU and MPS differ by ~3.7e-8 kSGD in evaluation and cannot be promised
bitwise equal. Keeping the backend in the run id makes the comparison honest.

The algorithm is not duplicated: `device.fit_on` mirrors `training.fit` step for step on
whichever device it is given, and both import the same frozen `models`/`data`/`training`.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from neural_fusion_v2 import contract as C

from . import device as DV
from . import orchestrate as O
from . import search as S
from .data import StageState, build_tensors, fit_stage_state
from .training import TrainConfig, build_for

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
REPORT_ROOT = ROOT / "reports" / "feature_transformer_v4"
CONTRACT_ROOT = ROOT / "reports" / "fusion_contract_v2"
CONTRACT_ID = "fusion-contract-v2.1.0__43867d0ba111"
SPLIT = "orig"

EXIT_CONTRACT_NOT_READY = 3
EXIT_DEVICE_UNAVAILABLE = 5


def code_fingerprint() -> dict:
    import hashlib

    payload = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        payload[str(path.relative_to(PACKAGE))] = hashlib.sha256(path.read_bytes()).hexdigest()
    v2 = PACKAGE.parent / "neural_fusion_v2"
    for path in sorted(v2.glob("*.py")):
        payload[f"../neural_fusion_v2/{path.name}"] = hashlib.sha256(
            path.read_bytes()).hexdigest()
    return payload


def load_a(contract_dir: Path, manifest: dict):
    a_fit = C.load_population(contract_dir, manifest, SPLIT, "A", "fit")
    a_hold = C.load_population(contract_dir, manifest, SPLIT, "A", "holdout")
    return a_fit, a_hold


def run_smoke(run_dir: Path, contract_dir: Path, manifest: dict, seed: int,
              device: torch.device) -> dict:
    """Train briefly on `device`, then prove the checkpoint reloads and predicts on CPU."""
    started = time.perf_counter()
    a_fit_pop, a_hold_pop = load_a(contract_dir, manifest)
    state = fit_stage_state(a_fit_pop, "A/fit")

    cap = 2000
    fit_pop = a_fit_pop.subset(np.arange(min(cap, len(a_fit_pop))))
    hold_pop = a_hold_pop.subset(np.arange(min(cap, len(a_hold_pop))))
    fit_tensors = build_tensors(fit_pop, state)
    hold_tensors = build_tensors(hold_pop, state)

    config = replace(S.trial_config({"id": 0, "pooling": "mean", "changes": {}}),
                     max_epochs=3, patience=3, eligible_after=1, seed=seed, batch_size=512)
    model = build_for(state.numeric_names, state.cardinalities(),
                      context_dim=fit_tensors.query_context.shape[1], config=config)

    dev_fit = DV.DeviceTensors(fit_tensors, device)
    dev_hold = DV.DeviceTensors(hold_tensors, device)
    result = DV.fit_on(device, model, dev_fit, dev_hold, config)

    predicted_on_device = hold_tensors.to_sgd(DV.predict_on(result.model, dev_hold))
    overall = {"rmse_sgd": float(np.sqrt(np.mean(
        (predicted_on_device - hold_tensors.actual_sgd) ** 2))),
        "n": int(len(hold_tensors))}

    run_dir.mkdir(parents=True, exist_ok=True)
    # describe() must not move the live model off its device: temporary CPU twin instead
    cpu_state = DV.cpu_state_dict(result.model)
    description = build_for(state.numeric_names, state.cardinalities(),
                            context_dim=fit_tensors.query_context.shape[1],
                            config=config)
    description.load_state_dict(cpu_state)
    description = description.describe()
    checkpoint = {"config": config.to_dict(), "state_dict": cpu_state,
                  "state": state.to_dict(), "device": str(device),
                  "description": description}
    torch.save(checkpoint, run_dir / "smoke_checkpoint.pt")

    # the gate: reload with map_location="cpu" and re-predict on CPU only
    stored = torch.load(run_dir / "smoke_checkpoint.pt", map_location="cpu",
                        weights_only=False)
    cpu_model = build_for(state.numeric_names, state.cardinalities(),
                          context_dim=fit_tensors.query_context.shape[1], config=config)
    cpu_model.load_state_dict(stored["state_dict"])
    cpu_model.eval()
    cpu_tensors = build_tensors(hold_pop, StageState.from_dict(stored["state"]))
    cpu_predicted = cpu_tensors.to_sgd(DV.predict_on(cpu_model,
                                                     DV.DeviceTensors(cpu_tensors,
                                                                      torch.device("cpu"))))
    cross_device_delta = float(np.abs(cpu_predicted - predicted_on_device).max())
    # the same rows on both backends, to report the actual numeric gap
    raw_device_pred = DV.predict_on(result.model, DV.DeviceTensors(hold_tensors, device))
    raw_cpu_pred = DV.predict_on(cpu_model, DV.DeviceTensors(hold_tensors,
                                                            torch.device("cpu")))

    summary = {
        "stage": "smoke", "device": str(device), "seed": seed,
        "n_fit": len(fit_pop), "n_hold": len(hold_pop),
        "parameters": result.model.describe()["parameters"],
        "selected_epoch": result.selected_epoch, "epochs_run": result.epochs_run,
        "overall": overall, "device_report": result.device_report,
        "checkpoint_tensors_on_cpu": all(v.device.type == "cpu"
                                        for v in stored["state_dict"].values()),
        "cpu_reload_max_abs_delta_sgd": cross_device_delta,
        "cpu_reload_max_abs_delta_ksgd": float(np.abs(raw_cpu_pred - raw_device_pred).max()),
        "reload_ok": cross_device_delta < 1e-4,
        "curve": result.curve,
        "seconds": round(time.perf_counter() - started, 2),
    }
    (run_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2))

    # keep the ledger honest: no comparable CPU result may live in this run
    (run_dir / "device_manifest.json").write_text(json.dumps({
        "backend": str(device), "device_report": result.device_report,
        "purpose": "all candidates in this run share one backend; a CPU number from "
                   "another run is a different setting and is never merged into it",
    }, indent=2))
    return summary


def run_trials(run_dir: Path, contract_dir: Path, manifest: dict, seed: int,
               device: torch.device, only: list[int] | None, resume: bool) -> dict:
    """Drive the eight trials on `device`, reusing the frozen orchestration rules."""
    a_fit_pop, a_hold_pop = load_a(contract_dir, manifest)
    state = fit_stage_state(a_fit_pop, "A/fit")
    a_fit = build_tensors(a_fit_pop, state)
    a_hold = build_tensors(a_hold_pop, state)

    if resume:
        (run_dir / f"resume_manifest_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
            json.dumps({"resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "device": str(device), "trials_requested": only,
                        "code_fingerprint": code_fingerprint(), "sequence": sys.argv},
                       indent=2))

    return O.run_search(run_dir, state, a_fit, a_hold, seed, only=only, resume=resume,
                        populations=(a_fit_pop, a_hold_pop),
                        device=device, fit_on=DV.fit_on,
                        make_device_tensors=DV.DeviceTensors,
                        predict_fn=DV.predict_on,
                        state_dict_fn=DV.cpu_state_dict)


def run_final(run_dir: Path, contract_dir: Path, manifest: dict,
              device: torch.device) -> int:
    """First and only read of B: refit the frozen top two, then score validation once."""
    from . import final_eval as FE

    registration = run_dir / "registered_candidates.json"
    if not registration.exists():
        print(f"NO REGISTRATION: {registration} missing; finish --stage trials first",
              file=sys.stderr)
        return 4
    payload = json.loads(registration.read_text())
    if not payload.get("registered_before_loading_B"):
        raise SystemExit("registration was not frozen before B; refusing to evaluate")
    top = payload["candidates"]

    b_fit_pop = C.load_population(contract_dir, manifest, SPLIT, "B", "fit")
    b_val_pop = C.load_population(contract_dir, manifest, SPLIT, "B", "validation")
    b_state = fit_stage_state(b_fit_pop, "B/fit")
    b_fit = build_tensors(b_fit_pop, b_state)
    b_val = build_tensors(b_val_pop, b_state)

    finals = []
    for candidate in top:
        print(f"[final] rank {candidate['rank']} trial {candidate['trial_id']} "
              f"({candidate['pooling']}) epochs={candidate['selected_epoch']} "
              f"device={device} ...", flush=True)
        summary = FE.evaluate_final_on_device(candidate, b_state, b_fit, b_val, run_dir, device)
        out_dir = run_dir / f"final_trial{candidate['trial_id']}"
        # three separate questions, three declared tolerances (see parity.py). The single
        # 1e-4 SGD threshold used earlier sat below normal MPS-vs-CPU divergence.
        from . import parity as PA

        reload = PA.compare(out_dir, b_state, b_val, str(device))
        summary["parity"] = reload
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"    B/validation RMSE {summary['overall']['rmse_sgd']:.4f} | "
              f"MAE {summary['overall']['mae_sgd']:.4f} | bias "
              f"{summary['overall']['bias_sgd']:.4f} | n={summary['overall']['n']}", flush=True)
        sb, cb, me = reload["same_backend"], reload["cross_backend"], reload["metric_effect"]
        print(f"    parity same-backend {sb['max_abs_delta_sgd']:.3e} SGD ok={sb['ok']}", flush=True)
        print(f"    parity cross-backend max {cb['max_abs_delta_sgd']:.3e} SGD rms "
              f"{cb['rms_delta_sgd']:.3e}, over-tol rows {cb['rows_over_tolerance']}"
              f"/{cb['n_rows']} ok={cb['ok']}", flush=True)
        print(f"    parity metric RMSE gap {me['rmse_gap_sgd']:.3e} SGD ok={me['ok']}",
              flush=True)
        finals.append(summary)

    finals_path = run_dir / "final_summary.json"
    existing = json.loads(finals_path.read_text()) if finals_path.exists() else []
    finals_path.write_text(json.dumps(existing + finals, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v4 feature Transformer on an explicit device")
    parser.add_argument("--stage", default="smoke", choices=["smoke", "trials", "final"])
    parser.add_argument("--device", default="mps", choices=list(DV.SUPPORTED_DEVICES))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contract-id", default=CONTRACT_ID)
    parser.add_argument("--trials", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(2)
    try:
        device = DV.resolve_device(args.device)
    except DV.DeviceError as error:
        print(f"DEVICE UNAVAILABLE: {error}", file=sys.stderr)
        return EXIT_DEVICE_UNAVAILABLE

    contract_dir = CONTRACT_ROOT / args.contract_id
    try:
        manifest = C.load_manifest(contract_dir)
    except (C.ContractNotReady, C.ContractInvalid) as error:
        print(f"CONTRACT NOT READY: {error}", file=sys.stderr)
        return EXIT_CONTRACT_NOT_READY

    run_id = args.run_id or f"v4-{args.stage}-{args.device}-s{args.seed}"
    run_dir = REPORT_ROOT / "runs" / run_id
    exists = run_dir.exists() and any(run_dir.iterdir())
    if exists and not (args.resume or args.stage == "final"):
        raise SystemExit(f"run dir already exists: {run_dir}; pass --resume or a new --run-id")

    framing = {
        "run_id": run_id, "stage": args.stage, "device": str(device),
        "split": SPLIT, "seed": args.seed,
        "contract_id": manifest["contract_id"],
        "contract_version": manifest["contract_version"],
        "contract_fingerprint": C.contract_fingerprint(contract_dir),
        "code_fingerprint": code_fingerprint(),
        "device_report": DV.device_report(device),
        "command": " ".join(sys.argv), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "search_space": str(REPORT_ROOT / "SEARCH_SPACE.json"),
        "b_loaded": args.stage == "final",
        "cross_backend_comparison": "not permitted as a like-for-like comparison",
    }
    if args.stage != "final":
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(framing, indent=2))
        elif not args.resume:
            raise SystemExit(f"refusing to overwrite {manifest_path}; pass --resume")

    if args.stage == "smoke":
        summary = run_smoke(run_dir, contract_dir, manifest, args.seed, device)
        print(json.dumps({k: summary[k] for k in
                          ("device", "overall", "parameters", "selected_epoch",
                           "epochs_run", "checkpoint_tensors_on_cpu", "reload_ok",
                           "cpu_reload_max_abs_delta_sgd", "cpu_reload_max_abs_delta_ksgd",
                           "seconds")}, indent=2))
        return 0 if summary["reload_ok"] else 1

    if args.stage == "trials":
        only = [int(t) for t in args.trials.split(",")] if args.trials else None
        run_trials(run_dir, contract_dir, manifest, args.seed, device, only, args.resume)
        return 0

    return run_final(run_dir, contract_dir, manifest, device)


if __name__ == "__main__":
    raise SystemExit(main())
