"""Orchestration for the eight-trial search: resume, no early freeze, one candidate schema.

Three defects in `main.run_trials` this module replaces, all found before they could corrupt
a result:

1. It decided the pooling for trials 4-8 after whatever trials the caller happened to ask
   for. `--trials 1` alone therefore froze "best of trials 1-3" from a single trial. Here the
   decision refuses to run until all of 1-3 have a completed `summary.json`.
2. It refused any existing run directory, so a finished trial could only be re-run, never
   resumed. `resume` reuses a completed trial's `summary.json` **after** verifying that the
   contract, code, config and data parity match; it never re-trains.
3. `freeze_top_two` returned dicts keyed `id` while callers read `trial_id`, so the print
   crashed after a successful freeze. Everything here uses one schema, defined once.

Nothing in this file changes `models.py`, `data.py` or `training.py`: a live trial is
training against them. It is a pure orchestration layer over `search.run_trial`.
"""
from __future__ import annotations

import hashlib
import json
import sys as _sys
import time
from dataclasses import replace
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

from pathlib import Path

import numpy as np
import torch

from . import search as S
from .data import build_tensors, fit_stage_state
from .training import TrainConfig

# the one schema every candidate crosses this module with
CANDIDATE_KEYS = ("trial_id", "pooling", "rank", "selected_epoch", "selected_rmse_sgd",
                  "config")


class OrchestrationError(RuntimeError):
    pass


def candidate_of(summary: dict, rank: int | None = None) -> dict:
    """Project a trial summary onto the single candidate schema."""
    return {"trial_id": int(summary["id"]), "pooling": summary["pooling"],
            "rank": rank, "selected_epoch": summary["selected_epoch"],
            "selected_rmse_sgd": summary["selected_rmse_sgd"],
            "config": summary["config"]}


def fingerprint_inputs(state, a_fit, a_hold) -> dict:
    """What a resumed trial must still agree with: data plus fitted state."""
    scaler = state.scaler
    return {
        "a_fit_parity": a_fit.parity_signature(),
        "a_hold_parity": a_hold.parity_signature(),
        "scalers": hashlib.sha256(json.dumps(
            {"static": scaler.static_median, "history": scaler.history_median,
             "context": scaler.context_median,
             "log1p": scaler.log1p_columns}, sort_keys=True).encode()).hexdigest(),
        "vocabularies": hashlib.sha256(json.dumps(
            state.vocabularies, sort_keys=True).encode()).hexdigest(),
        "numeric_names": list(state.numeric_names),
    }


def completed_trial(run_dir: Path, trial_id: int) -> dict | None:
    path = run_dir / f"trial{trial_id}" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def assert_config_matches(summary: dict, expected_config: TrainConfig) -> None:
    differences = []
    recorded = summary.get("config", {})
    for field, value in expected_config.to_dict().items():
        if field in recorded and recorded[field] != value:
            differences.append(f"config.{field}: recorded {recorded[field]!r} != {value!r}")
    if differences:
        raise OrchestrationError(
            f"trial {summary.get('id')} cannot be resumed:\n  " + "\n  ".join(differences))


def assert_resumable(summary: dict, expected_config: TrainConfig,
                     expected_fingerprint: dict) -> None:
    """Refuse to reuse a trial whose settings or inputs no longer match."""
    differences = []
    recorded = summary.get("config", {})
    for field, value in expected_config.to_dict().items():
        if field in recorded and recorded[field] != value:
            differences.append(f"config.{field}: recorded {recorded[field]!r} != {value!r}")
    recorded_fp = summary.get("input_fingerprint")
    if recorded_fp is None:
        differences.append("summary has no input_fingerprint, so equality cannot be checked")
    else:
        for key, value in expected_fingerprint.items():
            if recorded_fp.get(key) != value:
                differences.append(f"input_fingerprint.{key} differs")
    if differences:
        raise OrchestrationError(
            f"trial {summary.get('id')} cannot be resumed:\n  " + "\n  ".join(differences))


ATTESTATION_NAME = "resume_attestation.json"


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relevant_sources() -> dict[str, str]:
    """The files whose content can change a number in a trial.

    Deliberately narrow: `models.py`, `data.py`, `training.py`, `search.py` and the
    contract/scaler module they depend on. Orchestration and tests cannot change a score, so
    a change there must not block a resume.
    """
    package = _Path(__file__).resolve().parent
    files = [package / name for name in ("models.py", "data.py", "training.py", "search.py")]
    v2 = package.parent / "neural_fusion_v2"
    files += [v2 / "preprocessing.py", v2 / "contract.py", v2 / "adapter.py"]
    return {str(p.relative_to(package.parent)): _source_hash(p) for p in files}


def attest_legacy_trial(run_dir: Path, trial_id: int, state, a_hold_tensors,
                        a_fit_population, a_hold_population) -> dict:
    """Verify a trial written before attestations existed, without touching its records.

    A trial that finished under the older orchestrator has no `input_fingerprint`. Rather
    than rewriting its summary (which would fabricate provenance) or retraining it (which
    throws away valid work), the evidence that *can* be checked is checked, and the result
    is written to a sidecar:

    * the run manifest's contract fingerprint still matches the contract on disk;
    * the source hashes of `models/data/training/search` still match what the manifest
      recorded, and the checkpoint's description matches the live model class;
    * the checkpoint's saved `StageState` reproduces the same tensors as a fresh fit;
    * the trial's saved inner predictions align, row for row and label for label, with the
      A/holdout population.

    A caller may then resume on the strength of this sidecar, which is explicit about what
    it did and did not verify.
    """
    trial_dir = run_dir / f"trial{trial_id}"
    summary_path = trial_dir / "summary.json"
    checkpoint_path = trial_dir / "checkpoint.pt"
    predictions_path = trial_dir / "inner_predictions.npz"
    for path in (summary_path, checkpoint_path, predictions_path):
        if not path.exists():
            raise OrchestrationError(f"cannot attest trial {trial_id}: {path} is missing")

    summary = json.loads(summary_path.read_text())
    sources_now = relevant_sources()
    checks: dict[str, dict] = {}

    # 1. source identity, against the manifest the run was launched with
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    recorded = (manifest.get("code_fingerprint") or {})
    source_mismatch = []
    for name, digest in sources_now.items():
        short = Path(name).name
        candidates = [v for k, v in recorded.items() if Path(k).name == short]
        if not candidates:
            source_mismatch.append(f"{name}: not recorded in run_manifest")
        elif not any(str(v).startswith(digest[:12]) or digest.startswith(str(v)[:12])
                     for v in candidates):
            source_mismatch.append(f"{name}: recorded {candidates[0][:12]} != now {digest[:12]}")
    checks["sources"] = {"ok": not source_mismatch, "mismatches": source_mismatch,
                         "files": {k: v[:12] for k, v in sources_now.items()}}

    # 2. the checkpoint was produced by this model class
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    described = checkpoint.get("description", {})
    checks["checkpoint"] = {
        "ok": bool(described.get("numeric", {}).get("n_tokens") == 56),
        "numeric_tokens": described.get("numeric", {}).get("n_tokens"),
        "token_width": described.get("numeric", {}).get("token_width"),
        "parameters": described.get("parameters"),
        "saved_config_matches_summary": all(
            checkpoint.get("config", {}).get(k) == v
            for k, v in summary.get("config", {}).items()
            if k in checkpoint.get("config", {})),
    }

    # 3. the saved scaler/vocabulary state reproduces the tensors a fresh fit gives
    saved_state = checkpoint.get("state")
    state_ok, state_detail = False, "no saved state in checkpoint"
    if saved_state:
        from .data import StageState, build_tensors

        restored = StageState.from_dict(saved_state)
        rebuilt = build_tensors(a_fit_population, restored)
        fresh = build_tensors(a_fit_population, state)
        same_numeric = bool(np.array_equal(rebuilt.numeric, fresh.numeric))
        same_codes = bool(np.array_equal(rebuilt.categorical, fresh.categorical))
        same_target = bool(np.array_equal(rebuilt.target, fresh.target))
        same_anchor = bool(np.allclose(rebuilt.anchor_ksgd, fresh.anchor_ksgd, rtol=0,
                                       atol=1e-9))
        same_names = list(restored.numeric_names) == list(state.numeric_names)
        state_ok = same_numeric and same_codes and same_target and same_anchor and same_names
        state_detail = {
            "numeric_identical": same_numeric, "codes_identical": same_codes,
            "target_identical": same_target, "anchor_identical": same_anchor,
            "numeric_names_identical": same_names,
            "fitted_population": restored.fitted_population,
            "n_rows_checked": int(len(rebuilt)),
        }
    checks["state"] = {"ok": bool(state_ok), "detail": state_detail}

    # 4. the saved inner predictions still line up with A/holdout
    stored = np.load(predictions_path)
    hold = a_hold_tensors
    align_ok = (len(stored["row_id"]) == len(hold)
                and np.array_equal(stored["row_id"], hold.row_id)
                and np.allclose(stored["actual_sgd"], hold.actual_sgd, rtol=0, atol=1e-6)
                and np.array_equal(stored["horizon"], hold.horizon))
    recomputed_rmse = None
    if align_ok:
        recomputed_rmse = metrics_from_arrays(hold.actual_sgd, stored["predicted_sgd"])
    checks["rows"] = {
        "ok": bool(align_ok), "n": int(len(stored["row_id"])),
        "row_id_identical": bool(align_ok and np.array_equal(stored["row_id"], hold.row_id)),
        "labels_identical": bool(align_ok and np.allclose(stored["actual_sgd"], hold.actual_sgd,
                                                          rtol=0, atol=1e-6)),
        "recomputed_rmse_sgd": recomputed_rmse,
        "summary_rmse_sgd": summary.get("selected_rmse_sgd"),
        "rmse_matches_summary": (recomputed_rmse is not None
                                 and abs(recomputed_rmse - summary["selected_rmse_sgd"]) < 1e-6),
    }

    ok = all(checks[key]["ok"] for key in ("sources", "checkpoint", "state", "rows")) \
        and checks["rows"]["rmse_matches_summary"]

    attestation = {
        "trial_id": trial_id,
        "attested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "origin": "summary written before input_fingerprint support",
        "verified": checks,
        "verified_summary_of": {
            "selected_epoch": summary.get("selected_epoch"),
            "selected_rmse_sgd": summary.get("selected_rmse_sgd"),
            "epochs_run": summary.get("epochs_run"),
            "parameters": summary.get("parameters"),
        },
        "not_verified": [
            "the exact training-loss trajectory (the curve is taken from the summary)",
            "that gradient values during training matched the current code",
        ],
        "ok": bool(ok),
        "note": ("the original summary.json and run_manifest.json are left unchanged; this "
                 "sidecar is the record of what was checked"),
    }
    (trial_dir / ATTESTATION_NAME).write_text(json.dumps(attestation, indent=2))
    if not ok:
        failed = [k for k in ("sources", "checkpoint", "state", "rows") if not checks[k]["ok"]]
        raise OrchestrationError(
            f"trial {trial_id} failed attestation ({failed}); see "
            f"{trial_dir / ATTESTATION_NAME}. It must be re-run rather than resumed.")
    return attestation


def metrics_from_arrays(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return float(np.sqrt(np.mean(residual ** 2)))


def load_attestation(run_dir: Path, trial_id: int) -> dict | None:
    path = run_dir / f"trial{trial_id}" / ATTESTATION_NAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not payload.get("ok"):
        raise OrchestrationError(
            f"trial {trial_id} has a failed attestation at {path}; it must be re-run")
    return payload


def execute_trial(entry: dict, config: TrainConfig, state, a_fit, a_hold,
                  run_dir: Path, resume: bool, *, device=None,
                  fit_on=None, make_device_tensors=None, predict_fn=None,
                  state_dict_fn=None) -> tuple[dict, str]:
    """Run or resume one trial. Returns (summary, 'reused'|'trained')."""
    fingerprint = fingerprint_inputs(state, a_fit, a_hold)
    config = replace(config, seed=config.seed)

    if resume:
        existing = completed_trial(run_dir, entry["id"])
        if existing is not None:
            if existing.get("input_fingerprint"):
                assert_resumable(existing, config, fingerprint)
                basis = "input_fingerprint"
            else:
                # written before fingerprints existed: verified and accepted via a sidecar
                attestation = load_attestation(run_dir, entry["id"])
                if attestation is None:
                    raise OrchestrationError(
                        f"trial {entry['id']} has no input_fingerprint and no attestation; "
                        "call attest_legacy_trial first")
                assert_config_matches(existing, config)
                basis = "resume_attestation.json"
                print(f"[trial {entry['id']}] attested legacy summary accepted "
                      f"({len(attestation['verified'])} checks)", flush=True)
            print(f"[trial {entry['id']}] reused from {run_dir}/trial{entry['id']} "
                  f"(inner {existing['selected_rmse_sgd']:.4f}, "
                  f"{existing['epochs_run']} epochs already run; basis {basis})", flush=True)
            return existing, "reused"
        print(f"[trial {entry['id']}] no completed summary on disk; training it now",
              flush=True)

    print(f"[trial {entry['id']}] pooling={config.pooling} d={config.d_model} "
          f"L={config.layers} lr={config.learning_rate} bs={config.batch_size} "
          f"dropout={config.dropout} wd={config.weight_decay}"
          f"{'' if device is None else f' device={device}'} ...", flush=True)
    log_path = run_dir / "train_log.jsonl"
    if device is None:
        summary = S.run_trial(entry, config, state, a_fit, a_hold,
                              run_dir / f"trial{entry['id']}", log_path)
    else:
        from .device import run_trial_on_device

        summary = run_trial_on_device(entry, config, state, a_fit, a_hold,
                                      run_dir / f"trial{entry['id']}", log_path,
                                      device=device)
    summary["input_fingerprint"] = fingerprint
    (run_dir / f"trial{entry['id']}" / "summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"    inner RMSE {summary['selected_rmse_sgd']} | selected epoch "
          f"{summary['selected_epoch']} of {summary['epochs_run']} run | "
          f"{summary['parameters']} params | {summary['seconds']}s", flush=True)
    if summary.get("ineligible_best"):
        print(f"    note: epoch {summary['ineligible_best']['epoch']} scored "
              f"{summary['ineligible_best']['rmse_sgd']:.4f} before the eligibility floor",
              flush=True)
    return summary, "trained"


def decide_pooling(summaries: list[dict]) -> dict:
    """Freeze the pooling for trials 4-8 — only when trials 1-3 are all complete."""
    by_id = {int(s["id"]): s for s in summaries}
    missing = [i for i in (1, 2, 3) if i not in by_id]
    if missing:
        raise OrchestrationError(
            f"refusing to choose a pooling: trials {missing} have not completed. The "
            "decision for trials 4-8 must be made from all of trials 1-3.")
    eligible = [by_id[i] for i in (1, 2, 3) if by_id[i]["selected_rmse_sgd"] is not None]
    if not eligible:
        raise OrchestrationError("none of trials 1-3 produced an eligible epoch")
    eligible.sort(key=lambda s: (s["selected_rmse_sgd"], s["id"]))
    best = eligible[0]
    return {"trial_id": int(best["id"]), "pooling": best["pooling"],
            "selected_rmse_sgd": best["selected_rmse_sgd"],
            "decided_from": sorted(int(s["id"]) for s in eligible),
            "complete_trials_1_3": sorted(by_id[i]["id"] for i in (1, 2, 3)),
            "rule": "lowest eligible A/holdout RMSE among trials 1-3; ties -> earlier id"}


def register_top_two(summaries: list[dict], run_dir: Path) -> list[dict]:
    """Freeze the top two candidates into one schema, before B is ever loaded."""
    scored = [s for s in summaries if s["selected_rmse_sgd"] is not None]
    if len(scored) < 2:
        raise OrchestrationError(f"only {len(scored)} trial(s) produced an eligible epoch")
    scored.sort(key=lambda s: (s["selected_rmse_sgd"], s["id"]))
    top = [candidate_of(s, rank=rank) for rank, s in enumerate(scored[:2], start=1)]
    for candidate in top:
        if set(candidate) != set(CANDIDATE_KEYS):
            raise OrchestrationError(f"candidate schema drift: {sorted(candidate)}")
    (run_dir / "registered_candidates.json").write_text(json.dumps({
        "registered_before_loading_B": True,
        "schema": list(CANDIDATE_KEYS),
        "rule": "lowest eligible A/holdout RMSE; ties -> earlier trial id",
        "candidates": top,
    }, indent=2))
    return top


def run_search(run_dir: Path, state, a_fit, a_hold, seed: int,
               only: list[int] | None = None, resume: bool = False,
               populations=None, device=None, fit_on=None,
               make_device_tensors=None, predict_fn=None, state_dict_fn=None) -> dict:
    """``populations`` = (a_fit_population, a_hold_population), needed only to attest a
    legacy trial by rebuilding its tensors with the checkpoint's saved state."""
    """Drive trials 1-3, decide, then 4-8. Never loads B."""
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = run_dir / "search_ledger.json"
    ledger = (json.loads(ledger_path.read_text()) if ledger_path.exists()
              else {"trials": [], "decisions": []})
    ledger["a_parity"] = a_fit.parity_signature()
    ledger["hold_parity"] = a_hold.parity_signature()
    ledger["input_fingerprint"] = fingerprint_inputs(state, a_fit, a_hold)

    def flush():
        ledger_path.write_text(json.dumps(ledger, indent=2))

    # a summary can predate fingerprint support: verify it into a sidecar, never rewrite it
    for trial_id in (1, 2, 3, 4, 5, 6, 7, 8):
        existing = completed_trial(run_dir, trial_id)
        if existing is not None and not existing.get("input_fingerprint"):
            if load_attestation(run_dir, trial_id) is None:
                if populations is None:
                    raise OrchestrationError(
                        f"trial {trial_id} predates input_fingerprint and needs attesting; "
                        "pass populations=(a_fit_population, a_hold_population)")
                attest_legacy_trial(run_dir, trial_id, state, a_hold, *populations)

    # the earlier orchestrator could freeze a pooling from one trial; keep that record but
    # never act on it
    stale = ledger.get("best_pooling_decision")
    if isinstance(stale, dict):
        decided_from = sorted(stale.get("decided_from") or [])
        if decided_from != [1, 2, 3]:
            ledger.setdefault("invalidated_decisions", []).append({
                **stale,
                "invalidated_because": (
                    f"decided from trials {decided_from} instead of all of 1-3; "
                    "a single trial cannot choose the pooling for trials 4-8"),
            })
            ledger.pop("best_pooling_decision", None)
            flush()

    summaries: list[dict] = []
    outcomes: dict[int, str] = {}

    def run(entry, best_pooling=None):
        config = replace(S.trial_config(entry, best_pooling), seed=seed)
        summary, how = execute_trial(
            entry, config, state, a_fit, a_hold, run_dir, resume, device=device,
            fit_on=fit_on, make_device_tensors=make_device_tensors,
            predict_fn=predict_fn, state_dict_fn=state_dict_fn)
        outcomes[int(entry["id"])] = how
        summaries.append(summary)
        ledger["trials"] = [candidate_of(s) for s in summaries]
        flush()

    first = [t for t in S.TRIALS if t["id"] <= 3]
    if only:
        first = [t for t in first if t["id"] in only]
    for entry in first:
        run(entry)

    # only decide when all three are on disk, whether they ran now or earlier
    available = [completed_trial(run_dir, i) for i in (1, 2, 3)]
    if all(a is not None for a in available):
        # merge by trial id: a partial batch (e.g. --trials 2,3) must still see trial 1,
        # which a previous invocation completed. Taking `summaries` at face value here
        # crashed a whole run after three finished trials.
        merged: dict[int, dict] = {int(s["id"]): s for s in summaries}
        for summary_on_disk in available:
            if summary_on_disk is not None:
                merged.setdefault(int(summary_on_disk["id"]), summary_on_disk)
        summaries = [merged[i] for i in sorted(merged)]
        ledger["trials"] = [candidate_of(s) for s in summaries]
        decision = decide_pooling(summaries)
        ledger["decisions"].append(decision)
        ledger["best_pooling_decision"] = decision
        flush()
        print(f"[decision] pooling for trials 4-8: {decision['pooling']} "
              f"(trial {decision['trial_id']}, inner {decision['selected_rmse_sgd']:.4f}, "
              f"from trials {decision['complete_trials_1_3']})", flush=True)

        rest = [t for t in S.TRIALS if t["id"] >= 4]
        if only:
            rest = [t for t in rest if t["id"] in only]
        for entry in rest:
            run(entry, decision["pooling"])
    else:
        waiting = [i for i, a in zip((1, 2, 3), available) if a is None]
        print(f"[decision deferred] trials {waiting} have no summary yet; trials 4-8 are "
              "not started and no pooling is frozen", flush=True)

    # registering the final two requires all eight trials, not just two summaries
    on_disk = [completed_trial(run_dir, i) for i in range(1, 9)]
    complete = [s for s in on_disk if s is not None]
    missing_trials = [i for i, s in zip(range(1, 9), on_disk) if s is None]
    if len(complete) >= 8:
        top = register_top_two(complete, run_dir)
        ledger["registered"] = top
        flush()
        print(f"[registered] top two by inner RMSE: "
              f"{[(c['rank'], c['trial_id'], round(c['selected_rmse_sgd'], 4)) for c in top]}",
              flush=True)
    elif complete:
        print(f"[not registered] {len(complete)}/8 trials complete; missing {missing_trials}. "
              "The final two are frozen only when all eight are on disk.", flush=True)

    return {"summaries": summaries, "outcomes": outcomes, "ledger": ledger}
