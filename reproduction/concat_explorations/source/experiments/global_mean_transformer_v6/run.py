"""Persistent runner: v4 Feature Transformer, base 56 vs +grm_latest_log 57 columns.

Frozen configuration (the plan's, unchanged): cls / d=48 / L=3 / heads=4 / ff=2x / dropout .05 /
lr 1e-3 / wd 1e-4 / bs 512 / max 60 / patience 10 / eligible >= 5 / cosine T_max 60 / seed 42.

Two arms, identical in every respect except the numeric column count:

* `base56`  — the frozen 56 columns. Equals the existing trial6 baseline if and only if the
              attestation below passes; when it does, the recorded 581.0164 is reused instead of
              retraining, and the arm is marked `REUSED`.
* `mean57`  — the same 56 columns plus `grm_latest_log`.

No DBS/COE anywhere: the whole point is that the tree and this model see the same tabular
information.

Scheduling: this script holds an exclusive GPU lock file. It waits for the lock rather than
failing, so it can be queued behind the active macro job and will start on its own when that job
exits. The waiting state is durable — the lock is an OS file, not a pane.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from experiments.global_mean_transformer_v6 import data as GD
from experiments.global_mean_transformer_v6 import initialise as GI

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "global_mean_comparison_v6" / "transformer"
TRIAL6 = ROOT / "reports" / "feature_transformer_v4" / "runs" / "v4-search-mps-s42" / "trial6"
LOCK = ROOT / "reports" / "global_mean_comparison_v6" / "transformer" / ".gpu.lock"

FROZEN = {
    "pooling": "cls", "d_model": 48, "layers": 3, "heads": 4, "ff_multiplier": 2,
    "dropout": 0.05, "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 512,
    "clip_grad": 1.0, "max_epochs": 60, "patience": 10, "eligible_after": 5,
    "seed": 42, "schedule_epochs": 60,
}

# the baseline this comparison is allowed to reuse, and the condition for doing so
BASELINE_RMSE = 581.0164302081428


def sha_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_hashes() -> dict:
    out = {}
    for path in sorted((ROOT / "experiments" / "global_mean_transformer_v6").glob("*.py")):
        out[path.name] = sha_of(path)
    return out


def attest_baseline(bundle, fit_tensors, hold_tensors) -> dict:
    """Check whether the recorded 581.0164 is usable for this cohort.

    Reuse is only legitimate if the baseline saw the same rows, the same preprocessing and the
    same targets. Each condition is measured; a failure means the arm must be retrained rather
    than quoted.
    """
    checks = {}
    if not (TRIAL6 / "summary.json").exists():
        return {"reusable": False, "reason": "trial6 summary missing", "checks": {}}

    summary = json.loads((TRIAL6 / "summary.json").read_text())
    checkpoint = torch.load(TRIAL6 / "checkpoint.pt", map_location="cpu", weights_only=False)
    predictions = np.load(TRIAL6 / "inner_predictions.npz")

    checks["cohort_rows"] = {
        "trial6": int(len(predictions["row_id"])),
        "ours": int(len(hold_tensors)),
        "equal": int(len(predictions["row_id"])) == int(len(hold_tensors)),
    }
    checks["row_ids"] = {
        "equal": bool(np.array_equal(predictions["row_id"],
                                     np.asarray(hold_tensors.row_id, dtype=np.int64))),
    }
    # trial6 stored predicted residual; the shared table is the authority for the target, so
    # assert the two describe the same problem by reconstructing the baseline's own numbers
    # trial6 stores the reconstructed prediction; recomputing RMSE from it against the shared
    # table's actual rent is the real test that both describe the same cohort and target
    predicted = np.asarray(predictions["predicted_sgd"], dtype=np.float64)
    actual = np.asarray(hold_tensors.actual_sgd, dtype=np.float64)
    recomputed = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    checks["targets"] = {
        "npz_fields": sorted(predictions.files),
        "baseline_rmse_recomputed_from_predicted": recomputed,
        "recorded_rmse": BASELINE_RMSE,
        "matches_recorded": bool(abs(recomputed - BASELINE_RMSE) < 1e-6),
        "actual_matches_shared_table": bool(np.allclose(
            np.asarray(predictions["actual_sgd"], dtype=np.float64), actual, atol=1e-9)),
        "horizon_matches_shared_table": bool(np.array_equal(
            np.asarray(predictions["horizon"]), np.asarray(hold_tensors.horizon))),
    }
    # the 56 base columns must be the ones this comparison carries, in BOTH splits
    split_checks = {}
    ours = {"fit": fit_tensors, "holdout": hold_tensors}
    for split, npz_name in (("fit", "afit.npz"), ("holdout", "ahold.npz")):
        frozen = np.load(ROOT / "reports" / "diagnostics_phase2_c" / "frozen" / npz_name)
        mine = np.asarray(ours[split].numeric)
        split_checks[split] = {
            "rows": int(mine.shape[0]),
            "base_numeric_equal": bool(np.array_equal(mine[:, :56],
                                                      frozen["numeric"].astype(np.float32))),
            "categorical_equal": bool(np.array_equal(np.asarray(ours[split].categorical),
                                                     frozen["categorical"])),
            "anchor_equal": bool(np.array_equal(np.asarray(ours[split].anchor_ksgd),
                                                frozen["anchor_ksgd"])),
            "target_equal": bool(np.array_equal(np.asarray(ours[split].target),
                                                frozen["target"].astype(np.float32))),
        }
    checks["splits"] = split_checks
    # trial6 recorded `schedule_epochs: null` plus the effective value it actually used, so the
    # comparison has to read the effective field rather than the raw one
    trial6_config = dict(checkpoint["config"])
    trial6_schedule = trial6_config.get("schedule_epochs")
    if trial6_schedule is None:
        # the recorded value was null and the harness resolved it to the epoch budget; match
        # that semantic instead of flagging a false config difference
        trial6_schedule = trial6_config.get("schedule_epochs_effective",
                                            trial6_config.get("max_epochs"))
    checks["config"] = {
        "trial6": {k: trial6_config.get(k) for k in FROZEN},
        "trial6_schedule_effective": trial6_schedule,
        "ours": FROZEN,
        "equal": all(trial6_config.get(k) == v for k, v in FROZEN.items()
                     if k != "schedule_epochs") and trial6_schedule == FROZEN["schedule_epochs"],
    }
    checks["backend"] = {
        "trial6_device": checkpoint.get("device"),
        "note": "the recorded 581.0164 was produced on MPS; a CPU rerun is not the same "
                "backend and is not interchangeable",
        "comparable": checkpoint.get("device") == "mps",
    }
    checks["epoch_selection"] = {
        "trial6_selected_epoch": checkpoint.get("selected_epoch"),
        "trial6_inner_rmse": summary.get("selected_rmse_sgd"),
        "matches_recorded": abs(float(summary.get("selected_rmse_sgd", -1))
                                - BASELINE_RMSE) < 1e-6,
    }

    reusable = all([
        checks["cohort_rows"]["equal"], checks["row_ids"]["equal"],
        checks["config"]["equal"],
        checks["targets"].get("matches_recorded", False),
        checks["targets"].get("actual_matches_shared_table", False),
        checks["targets"].get("horizon_matches_shared_table", False),
        checks["epoch_selection"]["matches_recorded"],
        checks["backend"]["comparable"],
        all(v["base_numeric_equal"] and v["categorical_equal"] and v["anchor_equal"]
            and v["target_equal"] for v in split_checks.values()),
    ])
    return {"reusable": bool(reusable), "checks": checks,
            "baseline_rmse_sgd": BASELINE_RMSE,
            "note": ("reused only when every protocol condition matches; otherwise the base arm "
                     "is trained here too")}


def acquire_lock(lock: Path, wait: bool, poll_seconds: int = 30) -> bool:
    """Exclusive GPU lock. `wait=True` blocks until the current holder exits."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    announced = False
    while True:
        try:
            handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(handle, str(os.getpid()).encode())
            os.close(handle)
            return True
        except FileExistsError:
            holder = ""
            try:
                holder = lock.read_text().strip()
            except OSError:
                pass
            alive = False
            if holder.isdigit():
                try:
                    os.kill(int(holder), 0)
                    alive = True
                except OSError:
                    alive = False
            if not alive:
                # stale lock: the holder is gone, so the queue may proceed
                print(f"[queue] stale lock from pid {holder!r}; taking it", flush=True)
                try:
                    lock.unlink()
                except OSError:
                    pass
                continue
            if not wait:
                return False
            if not announced:
                print(f"[queue] waiting for pid {holder} to finish on the GPU", flush=True)
                announced = True
            time.sleep(poll_seconds)


def wait_for_external_pids(pids: list[int], poll_seconds: int = 30) -> None:
    """Block until every pid in `pids` has exited.

    The private lock file only coordinates runs that use this script; it cannot see an unrelated
    training process such as the active macro job. Waiting on the actual pid is the only way a
    queued job starts *after* that job rather than alongside it.
    """
    pending = [int(p) for p in pids if int(p) > 0]
    if not pending:
        return
    announced = False
    while True:
        alive = []
        for pid in pending:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except OSError:
                pass
        if not alive:
            if announced:
                print(f"[queue] external pids {pending} have exited; starting", flush=True)
            return
        if not announced:
            print(f"[queue] waiting for external pid(s) {alive}", flush=True)
            announced = True
        time.sleep(poll_seconds)


def release_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except OSError:
        pass


def train_arm(arm: str, device: torch.device, out_dir: Path, config: dict,
              bundle, fit_tensors, hold_tensors, numeric_names, cardinalities,
              parity: dict, base_state_record: dict) -> dict:
    from feature_transformer_v4.training import TrainConfig, build_for, set_seed
    from feature_transformer_v4 import device as device_module
    from diagnostics_phase2_c import protocol as protocol

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "epochs.jsonl"
    if log_path.exists():
        log_path.unlink()

    torch.manual_seed(config["seed"])
    model = build_for(numeric_names, cardinalities,
                      context_dim=fit_tensors.query_context.shape[1],
                      config=TrainConfig(**config))

    # apply the paired initialisation: the 57-column arm starts from the 56-column one's values
    if arm == "mean57":
        model.load_state_dict(parity["enlarged_state"])

    model = model.to(device)
    # Reset the RNG exactly as the frozen trainer does before fitting. `manual_seed` alone
    # leaves `random`, numpy and the determinism flag untouched, so dropout masks and batch
    # shuffling would start from a different RNG position than the reference run.
    set_seed(config["seed"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=config["schedule_epochs"])
    loss_fn = torch.nn.MSELoss()

    dev_fit = device_module.DeviceTensors(fit_tensors, device)
    dev_hold = device_module.DeviceTensors(hold_tensors, device)

    curve, best_state, best_rmse, best_epoch, stale = [], None, None, None, 0
    started = time.perf_counter()

    for epoch in range(1, config["max_epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train(True)
        total, seen = 0.0, 0
        for features, target in dev_fit.batches(config["batch_size"], shuffle=True,
                                                seed=config["seed"] * 1000 + epoch,
                                                with_target=True):
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            optimiser.step()
            total += float(loss.item()) * len(target)
            seen += len(target)
        learning_rate = float(optimiser.param_groups[0]["lr"])
        scheduler.step()

        model.eval()
        with torch.no_grad():
            residual = torch.cat([model(features) for features, _ in
                                  dev_hold.batches(config["batch_size"], shuffle=False,
                                                   with_target=True)]).cpu().numpy()
        residual = residual.astype(np.float64)
        metrics = protocol.score(hold_tensors, residual)
        record = {"epoch": epoch, "train_mse": total / max(seen, 1),
                  "inner_rmse_sgd": metrics["rmse_sgd"], "inner_mae_sgd": metrics["mae_sgd"],
                  "inner_bias_sgd": metrics["bias_sgd"],
                  "seconds": round(time.perf_counter() - epoch_started, 2),
                  "learning_rate": learning_rate}
        curve.append(record)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"    epoch {epoch:2d} train {record['train_mse']:.5f} "
              f"inner {metrics['rmse_sgd']:8.2f} ({record['seconds']}s)", flush=True)

        if epoch >= config["eligible_after"]:
            if best_rmse is None or metrics["rmse_sgd"] < best_rmse - 1e-9:
                best_rmse, best_epoch = metrics["rmse_sgd"], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= config["patience"]:
                    print(f"    early stop at epoch {epoch}", flush=True)
                    break

    if best_state is None:
        raise RuntimeError(f"{arm}: no eligible epoch; raise max_epochs")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        residual = torch.cat([model(features) for features, _ in
                              dev_hold.batches(config["batch_size"], shuffle=False,
                                               with_target=True)]).cpu().numpy()
    residual = residual.astype(np.float64)
    final = protocol.score(hold_tensors, residual)
    predicted = (residual + np.asarray(hold_tensors.anchor_ksgd)) * 1000.0

    np.savez_compressed(out_dir / "inner_predictions.npz",
                        row_id=hold_tensors.row_id, origin=bundle.holdout.origin,
                        horizon=hold_tensors.horizon, actual_sgd=hold_tensors.actual_sgd,
                        predicted_sgd=predicted, residual_ksgd=residual)

    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arm": arm, "config": config, "seed": config["seed"],
        "numeric_names": numeric_names,
        "n_numeric": len(numeric_names),
        "description": model.describe() if hasattr(model, "describe") else None,
        "feature_state": base_state_record,
        "extra_feature_state": bundle.extra_state,
        "extra_feature_schema": {"name": GD.NEW_COLUMN,
                                 "position": len(numeric_names) - 1,
                                 "base_numeric": GD.BASE_NUMERIC},
        "vocabularies": bundle.vocabularies,
        "categorical_order": bundle.categorical_order,
        "contract_id": GD.CONTRACT_ID,
        "shared_table_sha256": {"npz": sha_of(GD.SHARED_NPZ),
                                "manifest": sha_of(GD.SHARED_MANIFEST)},
        "source_hashes": source_hashes(),
        "init_parity": {"base56_sha": parity["parity"]["base56_sha"],
                        "enlarged57_sha": parity["parity"]["enlarged57_sha"],
                        "all_checks_pass": parity["parity"]["all_checks_pass"]},
    }, out_dir / "checkpoint.pt")

    summary = {
        "arm": arm, "stage": "A", "split": "orig",
        "result_status": "INNER_ONLY_NO_B",
        "n_fit": int(len(fit_tensors)), "n_holdout": int(len(hold_tensors)),
        "n_numeric": len(numeric_names), "numeric_names": numeric_names,
        "config": config, "selected_epoch": best_epoch, "epochs_run": len(curve),
        "inner": final, "curve": curve,
        "seconds": round(time.perf_counter() - started, 2),
        "device": str(device),
        "inner_origin_count": int(len(np.unique(bundle.holdout.origin))),
        "fit_origin_count": int(len(np.unique(bundle.fit.origin))),
        "source_hashes": source_hashes(),
        "contract_id": GD.CONTRACT_ID,
        "shared_table_sha256": {"npz": sha_of(GD.SHARED_NPZ),
                                "manifest": sha_of(GD.SHARED_MANIFEST)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="v6-origA-s42")
    parser.add_argument("--arms", default="mean57",
                        help="default trains only the +mean arm; base56 is reused when the "
                             "attestation passes, and is added explicitly if it does not")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--wait-for-gpu", action="store_true",
                        help="queue behind another job holding the lock file")
    parser.add_argument("--wait-pid", default="",
                        help="comma-separated pids to wait for before starting, e.g. 66357")
    parser.add_argument("--lock-poll", type=int, default=30)
    parser.add_argument("--preflight-only", action="store_true",
                        help="run the real entry path up to the paired initialisation, then "
                             "exit without training; used to prove the queue cannot fail late")
    args = parser.parse_args(argv)

    config = dict(FROZEN)
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
        config["schedule_epochs"] = max(args.max_epochs, 1)
        config["eligible_after"] = min(config["eligible_after"], max(1, args.max_epochs // 2))

    torch.set_num_threads(2)
    device = torch.device(args.device)
    out_root = REPORTS / "runs" / args.run_id
    if (out_root / "comparison.json").exists():
        raise SystemExit(f"{out_root} already holds a completed comparison; pick a new run id")

    external = [p.strip() for p in args.wait_pid.split(",") if p.strip()]
    if external:
        wait_for_external_pids([int(p) for p in external], poll_seconds=args.lock_poll)

    if args.wait_for_gpu:
        if not acquire_lock(LOCK, wait=True, poll_seconds=args.lock_poll):
            raise SystemExit("could not acquire the GPU lock")
        print(f"[queue] acquired the GPU lock as pid {os.getpid()}", flush=True)
    else:
        if not acquire_lock(LOCK, wait=False):
            # Another run holds the GPU. Continuing would train two models on one device and
            # the `finally` below would delete the holder's lock.
            raise SystemExit(
                f"{LOCK} is held by another run; use --wait-for-gpu to queue, or --wait-pid "
                f"to wait for a specific process")

    try:
        bundle = GD.load_shared_bundle()
        base_state_record = bundle.base_state
        contexts, state = GD.load_query_context(base_state_record)
        fit_tensors, hold_tensors = GD.bundle_feature_tensors(bundle, contexts)
        cardinalities = tuple(int(bundle.vocabularies[name])
                              for name in bundle.categorical_order)
        baseline_attestation = attest_baseline(bundle, fit_tensors, hold_tensors)
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "baseline_attestation.json").write_text(
            json.dumps(baseline_attestation, indent=2))

        from feature_transformer_v4.training import TrainConfig, build_for
        # `build_for` reads `config.seed` etc. as attributes, so it needs the dataclass rather
        # than the plain dict the CLI works with
        train_config = TrainConfig(**config)
        base_model, enlarged_model, parity = GI.build_enlarged_initial(
            config["seed"], build_for, bundle.numeric_names, cardinalities,
            context_dim=fit_tensors.query_context.shape[1], config=train_config)
        (out_root / "init_parity.json").write_text(json.dumps(parity, indent=2))
        print(f"[init] 56-col params {parity['base56_parameters']} | "
              f"57-col {parity['enlarged57_parameters']} | "
              f"parity {parity['all_checks_pass']}", flush=True)

        initial = {"parity": parity,
                   "enlarged_state": {k: v for k, v in enlarged_model.state_dict().items()}}

        if args.preflight_only:
            print(f"[preflight] entry path complete: bundle, context, tensors, attestation "
                  f"(reusable={baseline_attestation.get('reusable')}), initialisation "
                  f"(parity={parity['all_checks_pass']}). No training performed.", flush=True)
            return 0

        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        if "base56" not in arms and not baseline_attestation.get("reusable"):
            # training +mean without a matched baseline would leave nothing to compare against
            raise SystemExit(
                "the recorded baseline failed attestation, so a matched base56 arm is required. "
                "Re-run with --arms base56,mean57.\nFailed checks: "
                + json.dumps({k: v for k, v in baseline_attestation.get("checks", {}).items()
                              if isinstance(v, dict) and
                              (v.get("equal") is False or v.get("matches_recorded") is False
                               or v.get("comparable") is False
                               or v.get("actual_matches_shared_table") is False
                               or v.get("horizon_matches_shared_table") is False)},
                             indent=2, default=str))

        results = {}
        for arm in arms:
            print(f"[{arm}] device={args.device} seed={config['seed']}", flush=True)
            if arm == "base56" and baseline_attestation.get("reusable"):
                results[arm] = {"arm": arm, "result_status": "REUSED_BASELINE",
                                "inner": {"rmse_sgd": BASELINE_RMSE},
                                "source": str(TRIAL6)}
                print(f"    reused the attested baseline: {BASELINE_RMSE:.4f}", flush=True)
                continue
            numeric_names = (bundle.numeric_names if arm == "mean57"
                             else bundle.numeric_names[:GD.BASE_NUMERIC])
            prepared = {"parity": parity,
                        "enlarged_state": (initial["enlarged_state"] if arm == "mean57"
                                           else None)}
            summary = train_arm(arm, device, out_root / arm, config, bundle,
                                fit_tensors, hold_tensors, numeric_names, cardinalities,
                                prepared, base_state_record)
            print(f"    DONE {arm}: epoch {summary['selected_epoch']} "
                  f"inner {summary['inner']['rmse_sgd']:.4f}", flush=True)
            results[arm] = summary

        if "base56" in results and "mean57" in results:
            base = results["base56"]["inner"]["rmse_sgd"]
            mean = results["mean57"]["inner"]["rmse_sgd"]
            gap = base - mean
            combined = {
                "run_id": args.run_id, "stage": "A", "split": "orig",
                "result_status": "INNER_ONLY_NO_B",
                "base56_rmse_sgd": base, "mean57_rmse_sgd": mean,
                "base_minus_mean_sgd": gap, "tie": abs(gap) < 0.1,
                "base_reused": results["base56"].get("result_status") == "REUSED_BASELINE",
                "baseline_attestation": baseline_attestation,
                "inner_origin_count": int(len(np.unique(bundle.holdout.origin))),
                "interpretation_limit": (
                    "A/holdout is one origin spanning twelve target months, so the new column "
                    "is constant there; a difference cannot be attributed to rent-level "
                    "information by permuting holdout rows. Confirmation needs several inner "
                    "origins."),
                "source_hashes": source_hashes(),
            }
            (out_root / "comparison.json").write_text(json.dumps(combined, indent=2))
            print(f"\nbase56 - mean57 = {gap:+.4f} SGD "
                  f"({'tie' if abs(gap) < 0.1 else 'difference'})", flush=True)
        print(f"saved {out_root}", flush=True)
    finally:
        if args.wait_for_gpu:
            release_lock(LOCK)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
