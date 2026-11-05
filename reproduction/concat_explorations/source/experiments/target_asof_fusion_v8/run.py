"""Runner for the v8 target-as-of fusion experiment.

One transaction per sample; the market window is `[t-12, ..., t-1]`. Three arms (`house`,
`stats`, `bilstm`) share the house encoder and head initialisation exactly.

Frozen protocol (from the plan): seed 42, AdamW lr 1e-3 / wd 1e-4 / batch 512,
max 60 epochs / patience 10 / eligible >= 5, cosine T_max 60, grad clip 1.0, MSE on
`MONTHLY_RENT/1000`. Validation is eval-only and never updates the model or the scalers.

There is **no stage B**: this experiment reads only the `orig` development fold's A populations.
`--preflight-only` exercises the whole entry path without training.

Data contract (`experiments/target_asof_data_v8`, owned by the data producer):

```
train/val NPZ:
  house_numeric (n, F)   house_cat (n, 3)   calendar (n, 2)
  target_ksgd (n)        actual_sgd (n)     row_id (n)     target_month (n)
  window_id (n)
market NPZ:
  market_values (W, 12, C)  market_observed (W, 12, C)  market_step_valid (W, 12)
  market_month_index (W, 12)
json:
  schema.json / manifest.json (feature order, scalers, vocabularies, hashes, date assumptions)
```
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

from experiments.target_asof_fusion_v8 import models as M

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "reports" / "target_asof_v8" / "data"
REPORTS = ROOT / "reports" / "target_asof_v8"
LOCK = REPORTS / ".gpu.lock"

FROZEN = {
    "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 512, "clip_grad": 1.0,
    "max_epochs": 60, "patience": 10, "eligible_after": 5, "schedule_epochs": 60,
    "seed": 42,
}
WINDOW_MONTHS = 12


class DataNotReady(RuntimeError):
    pass


def sha_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_hashes() -> dict:
    out = {}
    for path in sorted((ROOT / "experiments" / "target_asof_fusion_v8").glob("*.py")):
        out[path.name] = sha_of(path)
    return out


# ---------------------------------------------------------------------------- data
FOLD = "orig"
# the producer's populations are named `train` and `validation` inside one NPZ per fold
POPULATIONS = {"train": "train", "val": "validation"}


def dataset_path(fold: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"{fold}_dataset.npz"


def manifest_path(fold: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"{fold}_manifest.json"


def load_fold_manifest(fold: str, data_dir: Path = DATA_DIR) -> dict:
    """The producer's per-fold manifest: vocabulary, house scaler, market scaler, hashes.

    This — not `schema.json` — is where the frozen preprocessing state lives. A checkpoint that
    stored only the schema could not rebuild its inputs without refitting.
    """
    path = manifest_path(fold, data_dir)
    if not path.exists():
        raise DataNotReady(f"missing {path}; it carries the fitted scalers and vocabularies")
    manifest = json.loads(path.read_text())
    for key in ("house_scaler", "market_scaler", "vocabularies", "arrays_sha256"):
        if key not in manifest:
            raise DataNotReady(f"{path.name} lacks {key!r}")
    declared = manifest["arrays_sha256"]
    actual = sha_of(dataset_path(fold, data_dir))
    if declared != actual:
        raise DataNotReady(
            f"{path.name} declares arrays_sha256 {declared[:16]} but the file hashes to "
            f"{actual[:16]}; the arrays changed after the manifest was written")
    return manifest


def load_split(split: str, data_dir: Path = DATA_DIR, fold: str = FOLD) -> dict:
    """Load one population plus its market windows, joined by `window_id`.

    The producer writes one NPZ per fold holding both populations and one shared market block,
    so this reads the same file twice and indexes the market arrays by `window_id`. Nothing is
    fitted here: `house_numeric` is already standardised with training statistics, and the file
    is the authority for that.
    """
    path = dataset_path(fold, data_dir)
    if not path.exists():
        raise DataNotReady(
            f"missing {path}; the data producer owns this file. Build it with "
            f"experiments/target_asof_data_v8 before running the neural arms.")
    schema_path = data_dir / "schema.json"
    if not schema_path.exists():
        raise DataNotReady(f"missing {schema_path}")
    schema = json.loads(schema_path.read_text())
    manifest = load_fold_manifest(fold, data_dir)
    if list(manifest.get("house_numeric_order", [])) != list(schema["house_numeric_order"]):
        raise DataNotReady(f"{fold}: manifest and schema disagree on the numeric order")
    if list(manifest.get("market_channel_order", [])) != list(schema["market_channel_order"]):
        raise DataNotReady(f"{fold}: manifest and schema disagree on the market channels")

    population = POPULATIONS[split]
    payload = np.load(path)
    required = [f"{population}__{name}" for name in
                ("house_numeric", "house_cat", "calendar", "target_ksgd", "actual_sgd",
                 "row_id", "target_month", "window_id")]
    missing = [key for key in required if key not in payload.files]
    if missing:
        raise DataNotReady(f"{path.name}: missing arrays {missing}")

    values = payload["market_values"].astype(np.float32)
    observed = payload["market_observed"].astype(np.float32)
    step_valid = payload["market_step_valid"].astype(np.float32)
    if values.shape[1] != WINDOW_MONTHS:
        raise DataNotReady(f"{fold}: market window is {values.shape[1]} months, "
                           f"expected {WINDOW_MONTHS}")
    channels = values.shape[2]
    if channels != int(schema["market_channel_count"]):
        raise DataNotReady(f"{fold}: {channels} market channels but the schema declares "
                           f"{schema['market_channel_count']}")

    window_id = payload[f"{population}__window_id"].astype(np.int64)
    if window_id.size and (window_id.min() < 0 or window_id.max() >= values.shape[0]):
        raise DataNotReady(f"{fold}/{population}: window_id outside [0, {values.shape[0]})")

    return {
        "house_numeric": payload[f"{population}__house_numeric"].astype(np.float32),
        "house_cat": payload[f"{population}__house_cat"].astype(np.int64),
        "calendar": payload[f"{population}__calendar"].astype(np.float32),
        "target_ksgd": payload[f"{population}__target_ksgd"].astype(np.float32),
        "actual_sgd": payload[f"{population}__actual_sgd"].astype(np.float64),
        "row_id": payload[f"{population}__row_id"].astype(np.int64),
        "target_month": payload[f"{population}__target_month"].astype(np.int64),
        "market_values": values[window_id],
        "market_observed": observed[window_id],
        "market_step_valid": step_valid[window_id],
        "schema": schema,
        "manifest": manifest,
        "fold": fold,
        "n_windows": int(values.shape[0]),
    }


def batches(split: dict, batch_size: int, shuffle: bool, seed: int = 0, with_target=True):
    """Yield dicts of tensors. Month order inside each window is never altered."""
    count = len(split["house_numeric"])
    order = np.arange(count)
    if shuffle:
        order = np.random.default_rng(seed).permutation(count)
    for start in range(0, count, batch_size):
        index = order[start:start + batch_size]
        tensor = torch.from_numpy
        batch = {
            "house_numeric": tensor(split["house_numeric"][index]),
            "house_cat": tensor(split["house_cat"][index]),
            "calendar": tensor(split["calendar"][index]),
            "market_values": tensor(split["market_values"][index]),
            "market_observed": tensor(split["market_observed"][index]),
            "market_step_valid": tensor(split["market_step_valid"][index]),
        }
        if with_target:
            yield batch, tensor(split["target_ksgd"][index])
        else:
            yield batch, None


# ------------------------------------------------------------------------- metrics
def metrics(actual_sgd: np.ndarray, predicted_sgd: np.ndarray) -> dict:
    residual = np.asarray(predicted_sgd) - np.asarray(actual_sgd)
    return {"rmse_sgd": float(np.sqrt(np.mean(residual ** 2))),
            "mae_sgd": float(np.mean(np.abs(residual))),
            "bias_sgd": float(np.mean(residual)), "n": int(residual.size)}


def per_month_metrics(months: np.ndarray, actual: np.ndarray, predicted: np.ndarray) -> dict:
    out = {}
    for month in np.unique(months):
        mask = months == month
        out[str(int(month))] = metrics(actual[mask], predicted[mask])
    return out


# ------------------------------------------------------------------------ training
def set_seed(seed: int) -> None:
    """Match the frozen trainer: python, numpy, torch and the determinism flag."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_arm(arm: str, models: dict, split_train: dict, split_val: dict,
              config: dict, device: torch.device, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "epochs.jsonl"
    if log_path.exists():
        log_path.unlink()

    model = models[arm].to(device)
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
        for batch, target in batches(split_train, config["batch_size"], shuffle=True,
                                     seed=config["seed"] * 1000 + epoch):
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
        predictions = []
        with torch.no_grad():
            for batch, _ in batches(split_val, config["batch_size"], shuffle=False,
                                    with_target=False):
                predictions.append(model({k: v.to(device) for k, v in batch.items()}).cpu())
        predicted_ksgd = torch.cat(predictions).numpy().astype(np.float64)
        predicted_sgd = predicted_ksgd * 1000.0
        scores = metrics(split_val["actual_sgd"], predicted_sgd)
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
        raise RuntimeError(f"{arm}: no eligible epoch")

    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch, _ in batches(split_val, config["batch_size"], shuffle=False,
                                with_target=False):
            predictions.append(model({k: v.to(device) for k, v in batch.items()}).cpu())
    predicted_ksgd = torch.cat(predictions).numpy().astype(np.float64)
    predicted_sgd = predicted_ksgd * 1000.0
    final = metrics(split_val["actual_sgd"], predicted_sgd)

    np.savez_compressed(
        out_dir / "predictions.npz",
        row_id=split_val["row_id"], target_month=split_val["target_month"],
        actual_sgd=split_val["actual_sgd"], predicted_sgd=predicted_sgd,
        predicted_ksgd=predicted_ksgd)

    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arm": arm, "config": config, "seed": config["seed"],
        "description": model.describe(),
        "numeric_names": model.house.numeric_names,
        "cardinalities": list(model.house.cardinalities),
        "market_channels": model.market_channels,
        "schema": split_train["schema"],
        "fold_manifest": split_train["manifest"],
        "fold": split_train["fold"],
        "data_dir": str(DATA_DIR),
        "data_hashes": {
            "schema.json": sha_of(DATA_DIR / "schema.json"),
            f"{split_train['fold']}_dataset.npz": sha_of(dataset_path(split_train["fold"])),
            f"{split_train['fold']}_manifest.json": sha_of(
                manifest_path(split_train["fold"])),
        },
        # the frozen preprocessing state, so a reload never has to fit anything
        "preprocessing": {
            "house_scaler": split_train["manifest"]["house_scaler"],
            "market_scaler": split_train["manifest"]["market_scaler"],
            "vocabularies": split_train["manifest"]["vocabularies"],
            "house_numeric_already_standardised": True,
            "month_index_origin": split_train["schema"].get("month_index_origin"),
        },
        "source_hashes": source_hashes(),
        "date_assumption": split_train["schema"].get("date_assumption"),
        "no_rent_history": True, "no_anchor": True,
    }, out_dir / "checkpoint.pt")

    summary = {
        "arm": arm,
        "result_status": ("SMOKE_3_EPOCHS" if config["max_epochs"] <= 5
                          else "DEV_FOLD_A_ONLY"),
        "fold": split_train["fold"],
        "n_epochs_budget": config["max_epochs"],
        "n_train": int(len(split_train["house_numeric"])),
        "n_val": int(len(split_val["house_numeric"])),
        "n_market_windows": split_val["n_windows"],
        "config": config, "selected_epoch": best_epoch, "epochs_run": len(curve),
        "validation": final,
        "validation_per_month": per_month_metrics(split_val["target_month"],
                                                  split_val["actual_sgd"], predicted_sgd),
        "curve": curve,
        "seconds": round(time.perf_counter() - started, 2),
        "device": str(device),
        "description": model.describe(),
        "source_hashes": source_hashes(),
        "target": "MONTHLY_RENT/1000 predicted directly; no anchor, no log, no clipping",
        "scale_note": "single seed, single development fold; not a robustness claim",
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
    parser.add_argument("--arm", choices=M.ARMS + ("all",), default="all")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default="v8-dev-orig-s42")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--fold", default=FOLD, choices=("orig", "early"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
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
    run_root = REPORTS / "runs" / args.run_id
    if (run_root / "comparison.json").exists():
        raise SystemExit(f"{run_root} already has a comparison; pick a new --run-id")

    external = [int(p) for p in args.wait_pid.split(",") if p.strip()]
    for pid in external:
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
    else:
        if not acquire_lock(LOCK, wait=False):
            raise SystemExit(f"{LOCK} is held by another run; use --wait-for-gpu")
    release = args.wait_for_gpu

    try:
        split_train = load_split("train", data_dir, args.fold)
        split_val = load_split("val", data_dir, args.fold)
        schema = split_train["schema"]
        numeric_names = list(schema["house_numeric_order"])
        # the schema declares the categorical order but not the vocabulary sizes; derive them
        # from the training codes so a new category cannot be silently dropped
        cardinalities = tuple(int(split_train["house_cat"][:, i].max()) + 1
                             for i in range(split_train["house_cat"].shape[1]))
        declared = schema.get("house_categorical_order", [])
        if declared and len(declared) != len(cardinalities):
            raise DataNotReady(f"schema declares {len(declared)} categorical columns but the "
                               f"data carries {len(cardinalities)}")
        if split_val["house_cat"].max(initial=0) >= max(cardinalities):
            raise DataNotReady("validation carries a category code outside the train vocabulary")
        channels = int(split_train["market_values"].shape[-1])
        print(f"[data] train {len(split_train['house_numeric'])} val "
              f"{len(split_val['house_numeric'])} | numeric {len(numeric_names)} "
              f"cat {len(cardinalities)} market channels {channels} "
              f"| windows {split_train['n_windows']}/{split_val['n_windows']}", flush=True)

        models = M.build_three_arms(config["seed"], numeric_names, cardinalities, channels)
        for arm, model in models.items():
            d = model.describe()
            print(f"[init] {arm:7s} params {d['parameters']:6d} "
                  f"(house {d['house_parameters']}, market {d['market_parameters']}, "
                  f"head {d['head_parameters']})", flush=True)

        if args.preflight_only:
            print("[preflight] data, model construction and init parity complete; "
                  "no training performed", flush=True)
            return 0

        arms = list(M.ARMS) if args.arm == "all" else [args.arm]
        results = {}
        for arm in arms:
            print(f"[{arm}] device={args.device} seed={config['seed']}", flush=True)
            results[arm] = train_arm(arm, models, split_train, split_val, config,
                                     torch.device(args.device), run_root / arm)
            print(f"    DONE {arm}: epoch {results[arm]['selected_epoch']} "
                  f"val RMSE {results[arm]['validation']['rmse_sgd']:.4f}", flush=True)

        if len(results) > 1:
            comparison = {
                "run_id": args.run_id,
                "result_status": ("SMOKE_3_EPOCHS" if config["max_epochs"] <= 5
                                  else "DEV_FOLD_A_ONLY"),
                "fold": split_train["fold"], "seed": config["seed"], "arms": {a: r["validation"] for a, r in results.items()},
                "note": "same rows, same windows, same budget; house arm is the ablation floor",
                "scale_note": "single seed / single fold; screening only",
            }
            (run_root / "comparison.json").write_text(json.dumps(comparison, indent=2))
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
