"""Persistent CLI runner: `summary2` vs `blocks7` on orig/A only.

Frozen configuration, from the plan:

    pooling cls | d_model 48 | layers 3 | heads 4 | ff 96 (2x) | dropout 0.05
    lr 1e-3 | wd 1e-4 | bs 512 | max 60 epochs | patience 10 | eligible >= 5
    cosine T_max 60 | seed 42

Stage A only. B is never loaded — not for selection, not for reporting. The run stops after
A/holdout.

Usage::

    python -m experiments.macro_time_tokens_v5.run --device mps --arm both
    python -m experiments.macro_time_tokens_v5.run --device cpu --arm blocks7 --max-epochs 2

Each arm writes `summary.json`, `checkpoint.pt`, `inner_predictions.npz` and `epochs.jsonl`
under `reports/macro_time_tokens_v5/runs/<run-id>/<arm>/`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from experiments.macro_time_tokens_v5 import model as M
from experiments.macro_time_tokens_v5 import sources as S

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "macro_time_tokens_v5"
CONTRACT_ROOT = ROOT / "reports" / "fusion_contract_v2"
CONTRACT_ID = "fusion-contract-v2.1.0__43867d0ba111"
# Pinned, not `latest`: a run must be reproducible from its own metadata. The id is also
# written into every summary and checkpoint, so a mismatch is visible rather than silent.
MACRO_CONTRACT_ID = "fusion-contract-v3.0.0__43867d0ba111__c52fef9f33cf__e32fccd39da3"
SPLIT = "orig"
POLICY = "origin_asof"

# the plan's frozen configuration, in one place
FROZEN = {
    "pooling": "cls", "d_model": 48, "layers": 3, "heads": 4, "ff_multiplier": 2,
    "dropout": 0.05, "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": 512,
    "clip_grad": 1.0, "max_epochs": 60, "patience": 10, "eligible_after": 5,
    "seed": 42, "schedule_epochs": 60,
}


def _hash_file(path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def source_hashes() -> dict:
    hashes = {}
    for path in sorted((ROOT / "experiments" / "macro_time_tokens_v5").glob("*.py")):
        hashes[path.name] = _hash_file(path)
    return hashes


def load_data():
    from neural_fusion_v2 import contract as contract
    from feature_transformer_v4 import data as feature_data

    contract_dir = CONTRACT_ROOT / CONTRACT_ID
    manifest = contract.load_manifest(contract_dir)
    a_fit = contract.load_population(contract_dir, manifest, SPLIT, "A", "fit")
    a_hold = contract.load_population(contract_dir, manifest, SPLIT, "A", "holdout")
    state = feature_data.fit_stage_state(a_fit, "A/fit")
    # `FeatureTensors` carries no origin; keep the contract arrays for the record
    origins = {"fit": np.asarray(a_fit.arrays.origin, dtype=np.int32),
               "holdout": np.asarray(a_hold.arrays.origin, dtype=np.int32)}
    return (manifest, state, feature_data.build_tensors(a_fit, state),
            feature_data.build_tensors(a_hold, state), origins)


def load_macro():
    from experiments.macro_encoders_v3 import api as macro_api
    from experiments.macro_encoders_v3 import contract as macro_contract
    from experiments.macro_encoders_v3 import data as macro_data
    from neural_fusion_v3.macro_bridge import build_stage_macro

    contract_dir = macro_contract.latest_v3_contract_dir()
    if contract_dir.name != MACRO_CONTRACT_ID:
        raise RuntimeError(
            f"macro contract mismatch: MACRO_CONTRACT_ID pins {MACRO_CONTRACT_ID!r} but the "
            f"resolved directory is {contract_dir.name!r}. Update the pin deliberately "
            f"rather than silently training on a different contract.")
    manifest = macro_contract.load_manifest(contract_dir)
    stage = build_stage_macro(manifest, SPLIT, "A")
    windows = macro_data.load_windows(manifest, SPLIT, "A")
    tensors = macro_api.build_window_tensors(windows, stage.scaler)
    return manifest, stage, tensors


def train_arm(arm: str, device: torch.device, out_dir: Path, config: dict,
              state, a_fit, a_hold, macro_manifest, macro_stage, window_tensors,
              initial, origins_fit, origins_hold, feature_state: dict,
              v2_contract_id: str) -> dict:
    from feature_transformer_v4.macro_ablation import MacroDeviceTensors
    from feature_transformer_v4.training import TrainConfig, build_for
    from diagnostics_phase2_c import protocol as P

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "epochs.jsonl"
    if log_path.exists():
        log_path.unlink()

    torch.manual_seed(config["seed"])
    base = build_for(state.numeric_names, state.cardinalities(),
                     context_dim=a_fit.query_context.shape[1],
                     config=TrainConfig(**config))

    summary_state, block_state, _ = initial
    if arm == "summary2":
        source = S.SummarySource(window_tensors, macro_stage.window_index,
                                 macro_manifest["contract_id"], macro_stage.policy,
                                 summary_state=summary_state)
    else:
        source = S.BlockEndpointSource(window_tensors, macro_stage.window_index,
                                       macro_manifest["contract_id"], macro_stage.policy,
                                       dbs_state=block_state["dbs"],
                                       coe_state=block_state["coe"])
    model = M.MacroBlocksOnFeatureTransformer(base, source, arm).to(device)

    train_cfg = TrainConfig(**config)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                           T_max=config["schedule_epochs"])
    loss_fn = torch.nn.MSELoss()

    dev_fit = MacroDeviceTensors(a_fit, device, macro_stage)
    dev_hold = MacroDeviceTensors(a_hold, device, macro_stage)

    curve = []
    best_state, best_rmse, best_epoch, stale = None, None, None, 0
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
        metrics = P.score(a_hold, residual)
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
                    print(f"    early stop at epoch {epoch} (patience {config['patience']})",
                          flush=True)
                    break

    if best_state is None:
        raise RuntimeError(f"{arm}: no eligible epoch; raise max_epochs above "
                           f"eligible_after={config['eligible_after']}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        residual = torch.cat([model(features) for features, _ in
                              dev_hold.batches(config["batch_size"], shuffle=False,
                                               with_target=True)]).cpu().numpy()
    residual = residual.astype(np.float64)
    final = P.score(a_hold, residual)
    predicted = (residual + a_hold.anchor_ksgd) * 1000.0

    np.savez_compressed(out_dir / "inner_predictions.npz",
                        row_id=a_hold.row_id, origin=origins_hold,
                        horizon=a_hold.horizon, actual_sgd=a_hold.actual_sgd,
                        predicted_sgd=predicted, residual_ksgd=residual)

    description = model.describe()
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "arm": arm, "config": config, "seed": config["seed"],
        "description": description,
        # the FT preprocessing state, so a reload never has to refit anything.
        # Fields: scaler (per-column mean/scale for 56 numeric columns, including the log1p
        # handling of the eight history counts), vocabularies, categorical_names,
        # fitted_population, n_rows.
        "feature_state": feature_state,
        "feature_state_format": "feature_transformer_v4.data.StageState.to_dict",
        "numeric_columns": list(state.numeric_names),
        "categorical_columns": list(state.categorical_names),
        "fit_metadata": {"fitted_population": state.fitted_population,
                         "n_rows": state.n_rows,
                         "n_numeric": len(state.numeric_names),
                         "n_categorical": len(state.categorical_names)},
        "v2_contract_id": v2_contract_id,
        "macro_scalers": macro_stage.scaler,
        "macro_scaler_fitted_on": macro_stage.scaler_fitted_on,
        "contract_id": macro_manifest["contract_id"],
        "policy": macro_stage.policy,
        "row_window_index": np.asarray(macro_stage.window_index),
        "row_id": np.asarray(macro_stage.row_id),
        "age_mask": {"dbs_block_rows": 21, "dbs_blocks": 3,
                     "coe_block_months": 3, "coe_blocks": 4},
        "source_hashes": source_hashes(),
        "macro_source_hashes": macro_manifest.get("code_hashes"),
    }, out_dir / "checkpoint.pt")

    summary = {
        "arm": arm, "stage": "A", "split": SPLIT,
        "result_status": "INNER_ONLY_NO_B",
        "note": ("orig/A only. B was never loaded. 26 distinct origins carry the macro "
                 "signal, so this is a representation result, not evidence of forecasting "
                 "skill until repeated across inner origins."),
        "n_fit": len(a_fit), "n_holdout": len(a_hold),
        "distinct_origin_count": int(len(np.unique(origins_fit))),
        "config": config,
        "selected_epoch": best_epoch, "epochs_run": len(curve),
        "inner": final,
        "anchor_only_inner": P.anchor_only(a_hold),
        "description": description,
        "curve": curve,
        "seconds": round(time.perf_counter() - started, 2),
        "device": str(device),
        "source_hashes": source_hashes(),
        "macro_contract_id": macro_manifest["contract_id"],
        "v2_contract_id": v2_contract_id,
        "policy": macro_stage.policy,
        "feature_state_fitted_on": state.fitted_population,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="v5-origA-s42")
    parser.add_argument("--arm", choices=("summary2", "blocks7", "both"), default="both")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="override for a short check; the frozen value is 60")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    config = dict(FROZEN)
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
        config["schedule_epochs"] = max(args.max_epochs, 1)
        # a short check must still be able to select; the frozen values are untouched otherwise
        config["eligible_after"] = min(config["eligible_after"], max(1, args.max_epochs // 2))
    if args.seed is not None:
        config["seed"] = args.seed

    torch.set_num_threads(2)
    device = torch.device(args.device)
    out_root = REPORTS / "runs" / args.run_id
    completed = out_root / "comparison.json"
    if completed.exists():
        raise SystemExit(
            f"{out_root} already holds a completed comparison; refusing to overwrite. "
            f"Pick a new --run-id.")
    for arm in ("summary2", "blocks7"):
        artifact = out_root / arm / "summary.json"
        if artifact.exists() and args.arm in ("both", arm):
            raise SystemExit(
                f"{artifact} already exists; refusing to overwrite a completed arm. "
                f"Pick a new --run-id.")
    out_root.mkdir(parents=True, exist_ok=True)

    macro_manifest, macro_stage, window_tensors = load_macro()
    manifest, state, a_fit, a_hold, origins = load_data()
    initial = S.shared_initial_state(config["seed"])

    # record the initialisation parity next to the run, so it is auditable later
    (out_root / "init_parity.json").write_text(json.dumps(initial[2], indent=2))

    arms = ("summary2", "blocks7") if args.arm == "both" else (args.arm,)
    results = {}
    for arm in arms:
        print(f"[{arm}] device={args.device} seed={config['seed']} "
              f"max_epochs={config['max_epochs']}", flush=True)
        summary = train_arm(arm, device, out_root / arm, config, state, a_fit, a_hold,
                            macro_manifest, macro_stage, window_tensors, initial,
                            origins["fit"], origins["holdout"],
                            feature_state=state.to_dict(),
                            v2_contract_id=manifest["contract_id"])
        inner = summary["inner"]
        print(f"    DONE {arm}: epoch {summary['selected_epoch']} "
              f"inner RMSE {inner['rmse_sgd']:.4f} MAE {inner['mae_sgd']:.4f} "
              f"bias {inner['bias_sgd']:.4f}", flush=True)
        results[arm] = {k: summary[k] for k in
                        ("selected_epoch", "epochs_run", "inner", "description")}

    if len(results) == 2:
        gap = results["summary2"]["inner"]["rmse_sgd"] - results["blocks7"]["inner"]["rmse_sgd"]
        combined = {
            "run_id": args.run_id, "stage": "A", "split": SPLIT,
            "result_status": "INNER_ONLY_NO_B",
            "comparison": "blocks7 minus summary2; negative means blocks7 better",
            "summary2_minus_blocks7_sgd": gap,
            "ties": abs(gap) < 0.1,
            "arms": results,
            "distinct_origin_count": int(len(np.unique(origins["fit"]))),
            "interpretation_limit": (
                "A/fit has 26 distinct origin indices; its rows share those 26 macro windows. "
                "A gap measured on one inner holdout cannot establish economic forecasting "
                "skill — that needs the sign to hold across several inner origins."),
            "source_hashes": source_hashes(),
        }
        (out_root / "comparison.json").write_text(json.dumps(combined, indent=2))
        print(f"\nsummary2 - blocks7 = {gap:+.4f} SGD "
              f"({'tie' if abs(gap) < 0.1 else 'difference'})", flush=True)

    print(f"saved {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
