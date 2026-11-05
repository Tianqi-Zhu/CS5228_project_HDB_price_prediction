"""Formal history arm: v8 Bi-LSTM model + 50 rent-history feature tokens.

The baseline is the already-accepted v8 `bilstm` run (606.7975547) and is **reused, not retrained**,
when the attestation below confirms the data, config and source are identical.

Training configuration is v8's frozen set: seed 42, AdamW lr 1e-3 / wd 1e-4 / batch 512,
max 60 / patience 10 / eligible >= 5, cosine T_max 60, grad clip 1.0, MSE on `MONTHLY_RENT/1000`.
No multi-seed grid, no hyper-parameter search.

The market branch, calendar inputs, target, and row set are untouched; the only change is the 50
history columns entering the house encoder as 50 additional tokens.
"""
from __future__ import annotations

import argparse
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

from experiments.history_fusion_v9 import data as HD
from experiments.history_fusion_v9 import models as HM

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "reports" / "history_fusion_v9" / "neural"
DATA_DIR = ROOT / "reports" / "history_fusion_v9" / "data"
V8_DATA = ROOT / "reports" / "target_asof_v8" / "data"
V8_RUN = ROOT / "reports" / "target_asof_v8" / "runs" / "v8-dev-orig-s42"
LOCK = OUT_ROOT / ".gpu.lock"

FROZEN = {
    "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 512, "clip_grad": 1.0,
    "max_epochs": 60, "patience": 10, "eligible_after": 5, "schedule_epochs": 60, "seed": 42,
}
BASELINE_RMSE = 606.7975547


def sha_of(path: Path) -> str:
    return HD.sha_of(path)


def source_hashes() -> dict:
    out = {}
    for path in sorted((ROOT / "experiments" / "history_fusion_v9").glob("*.py")):
        out[path.name] = sha_of(path)
    return out


# ---------------------------------------------------------------- v8 base arrays
def load_v8_base():
    """The v8 house/market inputs, unchanged. Read-only reuse of the frozen export."""
    from experiments.target_asof_fusion_v8 import run as V8R

    return V8R.load_split("train", V8_DATA, "orig"), V8R.load_split("val", V8_DATA, "orig")


def attest_baseline() -> dict:
    """Check that the recorded v8 bilstm result is still a valid comparison point."""
    checks = {}
    summary_path = V8_RUN / "bilstm" / "summary.json"
    if not summary_path.exists():
        return {"reusable": False, "reason": "v8 bilstm summary missing", "checks": {}}
    summary = json.loads(summary_path.read_text())
    checkpoint = torch.load(V8_RUN / "bilstm" / "checkpoint.pt", map_location="cpu",
                            weights_only=False)

    checks["recorded_rmse"] = summary["validation"]["rmse_sgd"]
    checks["matches_expected"] = abs(summary["validation"]["rmse_sgd"] - BASELINE_RMSE) < 1e-6
    checks["n_val"] = summary["validation"]["n"]
    checks["n_val_matches"] = summary["validation"]["n"] == 35882
    checks["config_equal"] = all(checkpoint["config"].get(k) == v for k, v in FROZEN.items()
                                 if k != "schedule_epochs")
    checks["seed"] = checkpoint["config"].get("seed")
    checks["fold"] = checkpoint.get("fold")
    checks["data_hashes"] = checkpoint.get("data_hashes")
    checks["baseline_data_unchanged"] = all(
        sha_of(V8_DATA / name) == declared
        for name, declared in (checkpoint.get("data_hashes") or {}).items())
    checks["baseline_arm"] = checkpoint.get("arm")
    reusable = bool(checks["matches_expected"] and checks["n_val_matches"]
                    and checks["config_equal"] and checks["baseline_data_unchanged"]
                    and checkpoint.get("arm") == "bilstm")
    return {"reusable": reusable, "baseline_rmse_sgd": BASELINE_RMSE, "checks": checks,
            "note": ("the v8 bilstm run is reused as the no-history baseline only when its "
                     "data, config and arm still match; it is never warm-started from")}


# ---------------------------------------------------------------------- training
def set_seed(seed: int) -> None:
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def batches(base_split: dict, history_split, batch_size: int, shuffle: bool, seed: int = 0,
            with_target: bool = True):
    count = len(base_split["house_numeric"])
    if len(history_split) != count:
        raise HD.HistoryDataError(f"history rows {len(history_split)} != base rows {count}")
    order = np.arange(count)
    if shuffle:
        order = np.random.default_rng(seed).permutation(count)
    tensor = torch.from_numpy
    for start in range(0, count, batch_size):
        index = order[start:start + batch_size]
        batch = {
            "house_numeric": tensor(base_split["house_numeric"][index]),
            "house_cat": tensor(base_split["house_cat"][index]),
            "calendar": tensor(base_split["calendar"][index]),
            "market_values": tensor(base_split["market_values"][index]),
            "market_observed": tensor(base_split["market_observed"][index]),
            "market_step_valid": tensor(base_split["market_step_valid"][index]),
            "history": tensor(history_split.history_numeric[index]),
        }
        yield (batch, tensor(base_split["target_ksgd"][index])) if with_target else (batch, None)


def metrics(actual, predicted) -> dict:
    residual = np.asarray(predicted) - np.asarray(actual)
    return {"rmse_sgd": float(np.sqrt(np.mean(residual ** 2))),
            "mae_sgd": float(np.mean(np.abs(residual))),
            "bias_sgd": float(np.mean(residual)), "n": int(residual.size)}


def per_month(months, actual, predicted) -> dict:
    return {str(int(m)): metrics(actual[months == m], predicted[months == m])
            for m in np.unique(months)}


def per_support(history_raw, actual, predicted) -> dict:
    """RMSE by how much history a row had: no group history, one record, two or more."""
    count_col = 0                                    # `count` of the first group
    counts = np.asarray(history_raw)[:, count_col]
    buckets = {"zero": counts == 0, "one": counts == 1, "two_or_more": counts >= 2}
    out = {}
    for name, mask in buckets.items():
        if mask.sum():
            out[name] = metrics(actual[mask], predicted[mask])
    return out


def train(history_split_train, history_split_val, base_train, base_val, config, device,
          out_dir: Path, parity: dict) -> dict:
    from experiments.target_asof_fusion_v8 import models as V8

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "epochs.jsonl"
    if log_path.exists():
        log_path.unlink()

    numeric_names = list(base_train["schema"]["house_numeric_order"])
    cardinalities = tuple(int(base_train["house_cat"][:, i].max()) + 1
                          for i in range(base_train["house_cat"].shape[1]))
    channels = int(base_train["market_values"].shape[-1])

    model = HM.build_history_model(config["seed"], numeric_names, cardinalities, channels)
    model.load_state_dict(parity["state_dict"])
    model = model.to(device)

    set_seed(config["seed"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=config["schedule_epochs"])
    loss_fn = torch.nn.MSELoss()

    curve, best_state, best_rmse, best_epoch, stale = [], None, None, None, 0
    started = time.perf_counter()

    for epoch in range(1, config["max_epochs"] + 1):
        epoch_started = time.perf_counter()
        model.train(True)
        total, seen = 0.0, 0
        for batch, target in batches(base_train, history_split_train, config["batch_size"],
                                     shuffle=True, seed=config["seed"] * 1000 + epoch):
            batch = {k: v.to(device) for k, v in batch.items()}
            target = target.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            optimiser.step()
            total += float(loss.item()) * len(target)
            seen += len(target)
        learning_rate = float(optimiser.param_groups[0]["lr"])
        scheduler.step()

        model.eval()
        chunks = []
        with torch.no_grad():
            for batch, _ in batches(base_val, history_split_val, config["batch_size"],
                                    shuffle=False, with_target=False):
                chunks.append(model({k: v.to(device) for k, v in batch.items()}).cpu())
        predicted = torch.cat(chunks).numpy().astype(np.float64) * 1000.0
        scores = metrics(history_split_val.actual_sgd, predicted)
        record = {"epoch": epoch, "train_mse": total / max(seen, 1),
                  "val_rmse_sgd": scores["rmse_sgd"], "val_mae_sgd": scores["mae_sgd"],
                  "val_bias_sgd": scores["bias_sgd"],
                  "seconds": round(time.perf_counter() - epoch_started, 2),
                  "learning_rate": learning_rate}
        curve.append(record)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"    epoch {epoch:2d} train {record['train_mse']:.5f} "
              f"val {scores['rmse_sgd']:8.2f} ({record['seconds']}s)", flush=True)

        if epoch >= config["eligible_after"]:
            if best_rmse is None or scores["rmse_sgd"] < best_rmse - 1e-9:
                best_rmse, best_epoch = scores["rmse_sgd"], epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= config["patience"]:
                    print(f"    early stop at epoch {epoch}", flush=True)
                    break

    if best_state is None:
        raise RuntimeError("history arm: no eligible epoch")
    model.load_state_dict(best_state)
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch, _ in batches(base_val, history_split_val, config["batch_size"],
                                shuffle=False, with_target=False):
            chunks.append(model({k: v.to(device) for k, v in batch.items()}).cpu())
    predicted = torch.cat(chunks).numpy().astype(np.float64) * 1000.0
    final = metrics(history_split_val.actual_sgd, predicted)

    np.savez_compressed(out_dir / "predictions.npz",
                        row_id=history_split_val.row_id,
                        target_month=history_split_val.target_month,
                        actual_sgd=history_split_val.actual_sgd,
                        predicted_sgd=predicted,
                        predicted_ksgd=predicted / 1000.0,
                        cutoff_month=history_split_val.cutoff_month)

    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arm": "bilstm_history", "config": config, "seed": config["seed"],
        "description": {"history_columns": HM.HISTORY_COLUMNS,
                        "parameters": int(sum(p.numel() for p in model.parameters())),
                        "cls_type_id": int(model.house.query_type_id),
                        "n_feature_tokens": int(model.house.n_feature_tokens)},
        "numeric_names": numeric_names, "cardinalities": list(cardinalities),
        "market_channels": channels,
        "schema": base_train["schema"],
        "fold_manifest": base_train["manifest"],
        "fold": "orig",
        "data_dir": str(DATA_DIR),
        "v8_data_dir": str(V8_DATA),
        "data_hashes": {"v8_" + name: sha_of(V8_DATA / name)
                        for name in ("orig_dataset.npz", "orig_manifest.json", "schema.json")},
        "history_hashes": {"orig_history.npz": sha_of(DATA_DIR / "orig_history.npz"),
                           "schema.json": sha_of(DATA_DIR / "schema.json")},
        "preprocessing": {
            "v8_house_scaler": base_train["manifest"]["house_scaler"],
            "v8_market_scaler": base_train["manifest"]["market_scaler"],
            "history_numeric_note": ("history_numeric arrives already fit-only transformed by "
                                     "the data producer; the model never fits it"),
        },
        "init_parity": {"cls_move": parity["cls_move"],
                        "all_checks_pass": parity["all_checks_pass"],
                        "base_params": parity["base_params"],
                        "history_params": parity["history_params"]},
        "source_hashes": source_hashes(),
        "no_warm_start": "initialised from a fresh seed-42 draw, not from trained v8 weights",
    }, out_dir / "checkpoint.pt")

    summary = {
        "arm": "bilstm_history",
        "result_status": ("SMOKE" if config["max_epochs"] <= 5 else "DEV_FOLD_A_ONLY"),
        "fold": "orig", "n_epochs_budget": config["max_epochs"],
        "n_train": int(len(history_split_train)), "n_val": int(len(history_split_val)),
        "config": config, "selected_epoch": best_epoch, "epochs_run": len(curve),
        "validation": final,
        "validation_per_month": per_month(history_split_val.target_month,
                                          history_split_val.actual_sgd, predicted),
        "validation_by_history_support": per_support(history_split_val.history_raw,
                                                     history_split_val.actual_sgd, predicted),
        "curve": curve,
        "seconds": round(time.perf_counter() - started, 2), "device": str(device),
        "description": {"history_columns": HM.HISTORY_COLUMNS,
                        "parameters": int(sum(p.numel() for p in model.parameters()))},
        "source_hashes": source_hashes(),
        "target": "MONTHLY_RENT/1000 predicted directly; no anchor, no log, no clipping",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def acquire_lock(lock: Path, wait: bool, poll: int = 30) -> bool:
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
                print(f"[queue] stale lock from pid {holder!r}; taking it", flush=True)
                try:
                    lock.unlink()
                except OSError:
                    pass
                continue
            if not wait:
                return False
            if not announced:
                print(f"[queue] waiting for pid {holder}", flush=True)
                announced = True
            time.sleep(poll)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="v9-history-orig-s42")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    parser.add_argument("--wait-pid", default="")
    parser.add_argument("--lock-poll", type=int, default=30)
    args = parser.parse_args(argv)

    config = dict(FROZEN)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
        config["schedule_epochs"] = max(args.max_epochs, 1)
        config["eligible_after"] = min(config["eligible_after"], max(1, args.max_epochs // 2))

    torch.set_num_threads(2)
    data_dir = Path(args.data_dir)
    run_root = OUT_ROOT / "runs" / args.run_id
    if (run_root / "summary.json").exists():
        raise SystemExit(f"{run_root} already holds a run; pick a new --run-id")

    for pid in [int(p) for p in args.wait_pid.split(",") if p.strip()]:
        while True:
            try:
                os.kill(pid, 0)
            except OSError:
                break
            print(f"[queue] waiting for external pid {pid}", flush=True)
            time.sleep(args.lock_poll)

    if args.wait_for_gpu:
        if not acquire_lock(LOCK, wait=True, poll=args.lock_poll):
            raise SystemExit("could not acquire the GPU lock")
        release = True
    else:
        if not acquire_lock(LOCK, wait=False):
            raise SystemExit(f"{LOCK} is held by another run; use --wait-for-gpu")
        release = False

    try:
        ready = HD.require_ready(data_dir)
        print(f"[data] DATA_READY verified: {json.dumps(ready.get('files', {}))[:200]}",
              flush=True)
        schema = HD.load_schema(data_dir)
        history_train = HD.load_split("train", data_dir)
        history_val = HD.load_split("validation", data_dir)

        leak_train = HD.check_leakage(history_train)
        leak_val = HD.check_leakage(history_val)
        print(f"[leak] train {leak_train}\n[leak] val {leak_val}", flush=True)
        if not (leak_train["leak_free"] and leak_val["leak_free"]):
            raise HD.HistoryDataError(f"history cutoff leaks: train {leak_train}, "
                                      f"val {leak_val}")

        base_train, base_val = load_v8_base()
        if not np.array_equal(history_train.row_id, np.asarray(base_train["row_id"])):
            raise HD.HistoryDataError("history train row_ids do not match the v8 rows")
        if not np.array_equal(history_val.row_id, np.asarray(base_val["row_id"])):
            raise HD.HistoryDataError("history val row_ids do not match the v8 rows")
        if not np.allclose(history_train.actual_sgd, np.asarray(base_train["actual_sgd"]),
                           atol=1e-9):
            raise HD.HistoryDataError("history train labels differ from v8")
        print(f"[data] train {len(history_train)} val {len(history_val)} "
              f"| history columns {HM.HISTORY_COLUMNS}", flush=True)

        attestation = attest_baseline()
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "baseline_attestation.json").write_text(json.dumps(attestation, indent=2))
        print(f"[baseline] reusable={attestation['reusable']} "
              f"rmse={attestation['checks'].get('recorded_rmse')}", flush=True)

        numeric_names = list(base_train["schema"]["house_numeric_order"])
        cardinalities = tuple(int(base_train["house_cat"][:, i].max()) + 1
                              for i in range(base_train["house_cat"].shape[1]))
        channels = int(base_train["market_values"].shape[-1])
        base_model = HM.build_base_model(config["seed"], numeric_names, cardinalities,
                                        channels)
        history_model = HM.build_history_model(config["seed"], numeric_names, cardinalities,
                                              channels)
        parity = HM.copy_common_parameters(base_model, history_model)
        parity["state_dict"] = {k: v.detach().clone()
                                for k, v in history_model.state_dict().items()}
        (run_root / "init_parity.json").write_text(
            json.dumps({k: v for k, v in parity.items() if k != "state_dict"}, indent=2))
        print(f"[init] CLS {parity['cls_move']['old']} -> {parity['cls_move']['new']} | "
              f"parity {parity['all_checks_pass']} | params "
              f"{parity['base_params']} -> {parity['history_params']}", flush=True)
        if not parity["all_checks_pass"]:
            raise HM.HistoryFusionError("initialisation parity failed")

        if args.preflight_only:
            print("[preflight] data, leakage, attestation and init parity complete; "
                  "no training performed", flush=True)
            return 0

        summary = train(history_train, history_val, base_train, base_val, config,
                        torch.device(args.device), run_root, parity)
        gap = BASELINE_RMSE - summary["validation"]["rmse_sgd"]
        combined = {
            "run_id": args.run_id, "fold": "orig", "seed": config["seed"],
            "result_status": summary["result_status"],
            "baseline_v8_bilstm_rmse_sgd": BASELINE_RMSE,
            "baseline_reused": attestation["reusable"],
            "history_arm_rmse_sgd": summary["validation"]["rmse_sgd"],
            "history_gain_sgd": gap,
            "tie": abs(gap) < 0.1,
            "selected_epoch": summary["selected_epoch"],
            "scale_note": ("single seed, single fold; a gain here is a candidate, not a "
                           "robust result"),
        }
        (run_root / "comparison.json").write_text(json.dumps(combined, indent=2))
        print(f"\nhistory - baseline = {-gap:+.4f} SGD "
              f"({'tie' if abs(gap) < 0.1 else 'difference'})", flush=True)
        print(f"saved {run_root}", flush=True)
    finally:
        if release:
            try:
                LOCK.unlink()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
