"""CLI for the v4 feature Transformer.

    # real-data smoke + checkpoint reconstruction (the gate before any full run)
    env -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
      -m experiments.feature_transformer_v4.main --stage smoke --run-id v4-smoke-s42

    # the eight-trial inner-only search
    ... --stage trials --run-id v4-search-s42

    # after registration: refit the frozen top two on B and score once
    ... --stage final --run-id v4-search-s42

`smoke` and `trials` never load B. `final` requires an existing
`registered_candidates.json` written by `freeze_top_two`.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from neural_fusion_v2 import contract as C

from . import orchestrate as O
from . import search as S
from .data import (FeatureDataError, StageState, build_tensors, fit_stage_state)
from .training import (TRAIN_THREADS, TrainConfig, build_for, fit, metrics, predict)

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
REPORT_ROOT = ROOT / "reports" / "feature_transformer_v4"
CONTRACT_ROOT = ROOT / "reports" / "fusion_contract_v2"
CONTRACT_ID = "fusion-contract-v2.1.0__43867d0ba111"
SPLIT = "orig"

EXIT_CONTRACT_NOT_READY = 3
EXIT_NO_REGISTRATION = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint() -> dict:
    payload = {str(p.relative_to(PACKAGE)): sha256_file(p) for p in sorted(PACKAGE.rglob("*.py"))}
    sibling = PACKAGE.parent / "neural_fusion_v2"
    payload.update({f"../neural_fusion_v2/{p.relative_to(sibling)}": sha256_file(p)
                    for p in sorted(sibling.rglob("*.py"))})
    return payload


def environment() -> dict:
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "torch": torch.__version__, "platform": platform.platform(),
            "host": socket.gethostname(), "torch_threads": torch.get_num_threads(),
            "device": "cpu", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}


def load_a_populations(contract_dir: Path, manifest: dict):
    a_fit = C.load_population(contract_dir, manifest, SPLIT, "A", "fit")
    a_hold = C.load_population(contract_dir, manifest, SPLIT, "A", "holdout")
    return a_fit, a_hold


def load_b_populations(contract_dir: Path, manifest: dict):
    b_fit = C.load_population(contract_dir, manifest, SPLIT, "B", "fit")
    b_val = C.load_population(contract_dir, manifest, SPLIT, "B", "validation")
    return b_fit, b_val


def run_smoke(run_dir: Path, contract_dir: Path, manifest: dict, seed: int) -> dict:
    """Real rental-loss training on real rows, then a load-only reconstruction."""
    started = time.perf_counter()
    a_fit, a_hold = load_a_populations(contract_dir, manifest)
    state = fit_stage_state(a_fit, "A/fit")

    cap = 1500
    a_fit_small = a_fit.subset(np.arange(min(cap, len(a_fit))))
    a_hold_small = a_hold.subset(np.arange(min(cap, len(a_hold))))
    t_fit = build_tensors(a_fit_small, state)
    t_hold = build_tensors(a_hold_small, state)

    config = replace(S.trial_config({"id": 0, "pooling": "mean", "changes": {}}),
                     max_epochs=3, patience=3, eligible_after=1, seed=seed,
                     batch_size=256)
    model = build_for(state.numeric_names, state.cardinalities(),
                      context_dim=t_fit.query_context.shape[1], config=config)
    n_params = model.describe()["parameters"]
    result = fit(t_fit, t_hold, config, model=model)
    predicted = t_hold.to_sgd(predict(result.model, t_hold))
    overall = metrics(t_hold.actual_sgd, predicted)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"config": config.to_dict(), "state_dict": result.model.state_dict(),
                "state": state.to_dict(), "description": result.model.describe()},
               run_dir / "smoke_checkpoint.pt")
    np.savez_compressed(run_dir / "smoke_predictions.npz",
                        predicted_sgd=predicted, actual_sgd=t_hold.actual_sgd,
                        row_id=t_hold.row_id, horizon=t_hold.horizon)

    # load-only reconstruction: rebuild from the file and re-predict, never refitting
    stored = torch.load(run_dir / "smoke_checkpoint.pt", map_location="cpu",
                        weights_only=False)
    restored_state = StageState.from_dict(stored["state"])
    rebuilt_config = TrainConfig(**{k: v for k, v in stored["config"].items()
                                    if k in TrainConfig().__dict__})
    rebuilt = build_for(restored_state.numeric_names, restored_state.cardinalities(),
                        context_dim=t_fit.query_context.shape[1], config=rebuilt_config)
    rebuilt.load_state_dict(stored["state_dict"])
    rebuilt_tensors = build_tensors(a_hold_small, restored_state)
    rebuilt_predicted = rebuilt_tensors.to_sgd(predict(rebuilt, rebuilt_tensors))
    delta = float(np.abs(rebuilt_predicted - predicted).max())
    state_round_trip = bool(np.array_equal(rebuilt_tensors.numeric, t_hold.numeric))

    summary = {
        "stage": "smoke", "seed": seed, "n_fit": len(a_fit_small), "n_hold": len(a_hold_small),
        "parameters": n_params, "epochs_run": result.epochs_run,
        "selected_epoch": result.selected_epoch, "overall": overall,
        "config": config.to_dict(), "description": result.model.describe(),
        "checkpoint_reload_max_abs_delta_sgd": delta,
        "reload_ok": delta < 1e-4,
        "state_round_trip_matches": state_round_trip,
        "curve": [r.__dict__ for r in result.curve],
        "seconds": round(time.perf_counter() - started, 2),
    }
    (run_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_trials(run_dir: Path, contract_dir: Path, manifest: dict, seed: int,
               only: list[int] | None = None, resume: bool = False) -> dict:
    """Run or resume trials through `orchestrate`, which never loads B."""
    a_fit_pop, a_hold_pop = load_a_populations(contract_dir, manifest)
    state = fit_stage_state(a_fit_pop, "A/fit")
    a_fit = build_tensors(a_fit_pop, state)
    a_hold = build_tensors(a_hold_pop, state)

    if resume:
        # keep the original manifest untouched and record what this invocation did
        (run_dir / f"resume_manifest_{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(
            json.dumps({"resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "trials_requested": only, "seed": seed,
                        "code_fingerprint": code_fingerprint(),
                        "contract_fingerprint": C.contract_fingerprint(contract_dir),
                        "sequence": sys.argv}, indent=2))

    return O.run_search(run_dir, state, a_fit, a_hold, seed, only=only, resume=resume,
                        populations=(a_fit_pop, a_hold_pop))


def run_final(run_dir: Path, contract_dir: Path, manifest: dict) -> dict:
    registration = run_dir / "registered_candidates.json"
    if not registration.exists():
        print(f"NO REGISTRATION: {registration} missing; run --stage trials first",
              file=sys.stderr)
        return {"exit": EXIT_NO_REGISTRATION}
    top = json.loads(registration.read_text())["candidates"]

    b_fit_pop, b_val_pop = load_b_populations(contract_dir, manifest)
    b_state = fit_stage_state(b_fit_pop, "B/fit")
    b_fit = build_tensors(b_fit_pop, b_state)
    b_val = build_tensors(b_val_pop, b_state)

    finals = []
    for candidate in top:
        print(f"[final] trial {candidate['trial_id']} epochs={candidate['selected_epoch']}",
              flush=True)
        summary = S.evaluate_final(candidate, b_state, b_fit, b_val, run_dir)
        print(f"    B/validation RMSE {summary['overall']['rmse_sgd']:.4f} "
              f"MAE {summary['overall']['mae_sgd']:.4f} n={summary['overall']['n']}", flush=True)
        finals.append(summary)
    (run_dir / "final_summary.json").write_text(json.dumps(finals, indent=2))
    return {"finals": finals}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v4 feature Transformer")
    parser.add_argument("--stage", default="smoke", choices=["smoke", "trials", "final"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contract-id", default=CONTRACT_ID)
    parser.add_argument("--trials", default=None, help="comma-separated trial ids to run")
    parser.add_argument("--resume", action="store_true",
                        help="reuse completed trials after verifying their inputs match")
    args = parser.parse_args(argv)

    torch.set_num_threads(TRAIN_THREADS)
    contract_dir = CONTRACT_ROOT / args.contract_id
    try:
        manifest = C.load_manifest(contract_dir)
    except (C.ContractNotReady, C.ContractInvalid) as error:
        print(f"CONTRACT NOT READY: {error}", file=sys.stderr)
        return EXIT_CONTRACT_NOT_READY

    run_id = args.run_id or f"v4-{args.stage}-s{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = REPORT_ROOT / "runs" / run_id
    exists = run_dir.exists() and any(run_dir.iterdir())
    if exists and not (args.resume or args.stage == "final"):
        raise SystemExit(
            f"run dir already exists: {run_dir}\n"
            "Pass --resume to continue it (completed trials are reused after their inputs "
            "are verified), or choose a new --run-id.")

    framing = {
        "run_id": run_id, "stage": args.stage, "split": SPLIT, "seed": args.seed,
        "contract_id": manifest["contract_id"],
        "contract_version": manifest["contract_version"],
        "contract_fingerprint": C.contract_fingerprint(contract_dir),
        "code_fingerprint": code_fingerprint(), "environment": environment(),
        "command": " ".join(sys.argv), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "search_space": str(REPORT_ROOT / "SEARCH_SPACE.json"),
        "b_loaded": args.stage == "final",
    }
    if args.stage != "final":
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            manifest_path.write_text(json.dumps(framing, indent=2))
        elif not args.resume:
            raise SystemExit(f"refusing to overwrite {manifest_path}; pass --resume")

    if args.stage == "smoke":
        summary = run_smoke(run_dir, contract_dir, manifest, args.seed)
        print(json.dumps({k: summary[k] for k in
                          ("overall", "parameters", "selected_epoch", "epochs_run",
                           "reload_ok", "checkpoint_reload_max_abs_delta_sgd",
                           "state_round_trip_matches", "seconds")}, indent=2))
        return 0 if summary["reload_ok"] else 1

    if args.stage == "trials":
        only = [int(t) for t in args.trials.split(",")] if args.trials else None
        run_trials(run_dir, contract_dir, manifest, args.seed, only, resume=args.resume)
        return 0

    outcome = run_final(run_dir, contract_dir, manifest)
    return outcome.get("exit", 0)


if __name__ == "__main__":
    raise SystemExit(main())
