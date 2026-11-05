"""Primary-candidate macro ablation: `none` versus DBS+COE, on one architecture.

This is **one** ablation, not another search. The architecture and every hyperparameter come
from the registered rank-1 candidate; the only thing that changes between the two runs is
whether the DBS/COE memory tokens are present.

Protocol, identical for both arms:

1. fit on A/fit, monitor A/holdout, select the eligible epoch by inner RMSE (same rule,
   same eligibility floor, same patience);
2. refit on B/fit for exactly that many epochs, with the stage-A cosine schedule;
3. score the complete 35,158-row B/validation once.

Both arms are evaluated on B, which is a second look at that split inside this workspace, so
the result is reported as an ablation on an already-viewed validation set — not as an
independent estimate.

The macro artifacts come from the producer package read-only:
`experiments/macro_encoders_v3` for the encoders and scaling, and the v3 contract for the
deduplicated windows and the row mapping. Nothing here re-derives a stock or COE statistic.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:
    _sys.path.insert(0, str(_EXPERIMENTS))

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from neural_fusion_v2 import contract as C

from . import device as DV
from . import macro_v4 as MV
from .data import StageState, build_tensors, fit_stage_state
from .training import TrainConfig, build_for, metrics

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
REPORT_ROOT = ROOT / "reports" / "feature_transformer_v4"
CONTRACT_ROOT = ROOT / "reports" / "fusion_contract_v2"
CONTRACT_ID = "fusion-contract-v2.1.0__43867d0ba111"
SPLIT = "orig"
POLICY = "origin_asof"


def _source_factory(stage_macro, macro_manifest, mode, d_model):
    """DBS/COE encoders from the producer package, exposed as a token source."""
    from experiments.macro_encoders_v3 import api as macro_api
    from experiments.macro_encoders_v3 import data as macro_data
    from experiments.macro_encoders_v3 import encoders as macro_encoders

    windows = macro_data.load_windows(macro_manifest, stage_macro.split, stage_macro.stage)
    window_tensors = macro_api.build_window_tensors(windows, stage_macro.scaler)
    modules = macro_encoders.build_encoders()

    class Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoders = modules
            self.tensors = window_tensors
            self.mode = mode
            self.n_tokens = ((1 if mode in ("dbs", "both") else 0)
                             + (1 if mode in ("coe", "both") else 0))
            self.d_model = 48                      # the producer's token width
            self.register_buffer(
                "row_window_index",
                torch.as_tensor(np.asarray(stage_macro.window_index), dtype=torch.long),
                persistent=False)

        def _tensors_on(self, t, device):
            """Move a window-tensor bundle to `device`.

            `encode_windows` runs `pack_padded_sequence` inside the encoder, which needs the
            sequence and the recurrent weights on the same device. The producer builds them
            on CPU, so the batch's slice is moved after selection.
            """
            from experiments.macro_encoders_v3.api import WindowTensors

            return WindowTensors(
                dbs_seq=t.dbs_seq.to(device), dbs_mask=t.dbs_mask.to(device),
                coe_seq=t.coe_seq.to(device), coe_mask=t.coe_mask.to(device),
                coe_channel_mask=t.coe_channel_mask.to(device),
                window_cutoff_month=t.window_cutoff_month,
                dbs_asof_date=t.dbs_asof_date, coe_month_ids=t.coe_month_ids,
                policy=t.policy, contract_id=t.contract_id)

        def forward(self, batch):
            from experiments.macro_encoders_v3.api import (
                batch_indices, encode_windows, gather_to_rows, select_window_tensors)

            device = next(self.encoders.parameters()).device
            rows = batch["macro_row"].to("cpu").numpy()
            # the buffer follows the model to MPS, so copy before converting
            row_window = self.row_window_index.detach().cpu().numpy()
            unique, inverse = batch_indices(row_window, rows, self.tensors.n_windows)
            # select on CPU (where the producer built them), then move only the batch's
            # windows to the model's device: index_select requires both sides co-located
            subset = select_window_tensors(self.tensors, unique)
            subset = self._tensors_on(subset, device)
            per_window = encode_windows(self.encoders, subset)
            # gather here rather than via `gather_to_rows`: that helper builds its index on
            # CPU, and the encoded tokens live on the model's device
            index = torch.as_tensor(np.asarray(inverse, dtype=np.int64), device=device)
            per_row = {k: v.index_select(0, index) for k, v in per_window.items()}

            tokens = []
            if self.mode in ("dbs", "both"):
                tokens.append(per_row["dbs_token"])
            if self.mode in ("coe", "both"):
                tokens.append(per_row["coe_token"])
            return torch.stack(tokens, dim=1)

        def describe(self):
            payload = {"package": "experiments/macro_encoders_v3", "mode": self.mode,
                       "tokens": self.n_tokens,
                       "encoder_parameters": macro_encoders.parameter_count(self.encoders),
                       "policy": POLICY,
                       "semantics": "financial-state embedding learned through the rental "
                                    "loss; not a validated measure of economic health"}
            if self.mode in ("dbs", "both"):
                payload["dbs"] = {"symbol": "D05.SI", "window": "63 trading days",
                                  "axis": "trading_day"}
            if self.mode in ("coe", "both"):
                payload["coe"] = {"categories": ["a", "b", "c", "e"],
                                  "window": "12 calendar months",
                                  "axis": "calendar_month",
                                  "categories_are_not_timesteps": True}
            return payload

    return Source()


class MacroDeviceTensors(DV.DeviceTensors):
    """`DeviceTensors` that adds the macro keys to every batch.

    The prediction paths (`predict_on`, `_epoch`) only know how to move a batch to a device;
    they know nothing about macro. Completing the batch here keeps those shared modules
    untouched. This class is stateless: it recomputes the row order for every `batches` call
    from the same seed the base implementation uses, so it can never drift out of step with
    the base iterator.
    """

    def __init__(self, base, device, stage_macro, shuffle=False, seed=0):
        super().__init__(base, device)
        self.stage_macro = stage_macro
        self._shuffle = shuffle
        self._seed = seed

    def subset(self, index):
        index = np.asarray(index)
        return MacroDeviceTensors(self.base.subset(index), self.device, self.stage_macro,
                                  self._shuffle, self._seed)

    def batches(self, batch_size, shuffle=False, seed=0, with_target=False):
        order = np.arange(len(self.base))
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, len(self.base), batch_size):
            rows = order[start:start + batch_size]
            inner = self.base.subset(rows)
            item = next(iter(inner.batches(len(rows), shuffle=False,
                                           with_target=with_target)))
            features = item[0] if with_target else item
            moved = {k: v.to(self.device) for k, v in features.items()}
            moved.update(self.stage_macro.select(inner.row_id))
            if with_target:
                yield moved, item[1].to(self.device)
            else:
                yield moved


def run_arm(mode: str, config: TrainConfig, device: torch.device, out_dir: Path,
            a_fit, a_hold, b_fit, b_val, state_a, state_b, macro_manifest,
            stage_macro_a, stage_macro_b, build_stage_macro) -> dict:
    from .training import _epoch, set_seed

    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_model(state, stage_macro, context_dim):
        base = build_for(state.numeric_names, state.cardinalities(),
                         context_dim=context_dim, config=config)
        return MV.build_with_macro(
            base, lambda m: _source_factory(stage_macro, macro_manifest, m, config.d_model),
            mode)

    # ---- stage A: inner selection
    set_seed(config.seed)
    model_a = make_model(state_a, stage_macro_a, a_fit.query_context.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model_a.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.schedule_length())

    curve, best_state, best_epoch, best_rmse, stale = [], None, None, float("inf"), 0
    dev_a_fit = DV.DeviceTensors(a_fit, device)
    dev_a_hold = MacroDeviceTensors(a_hold, device, stage_macro_a)
    for epoch in range(1, config.max_epochs + 1):
        batches = MacroDeviceTensors(a_fit, device, stage_macro_a).batches(
            config.batch_size, shuffle=True, seed=config.seed * 1000 + epoch,
            with_target=True)
        total, seen = 0.0, 0
        model_a.train(True)
        for features, target in batches:
            optimiser.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model_a(features), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), config.clip_grad)
            optimiser.step()
            total += float(loss.item()) * len(target)
            seen += len(target)
        current_lr = float(optimiser.param_groups[0]["lr"])
        scheduler.step()
        hold_pred = a_hold.to_sgd(DV.predict_on(model_a, dev_a_hold))
        hold_rmse = metrics(a_hold.actual_sgd, hold_pred)["rmse_sgd"]
        curve.append({"epoch": epoch, "train_mse": total / max(seen, 1),
                      "monitor_rmse_sgd": hold_rmse, "learning_rate": current_lr})
        if epoch >= config.eligible_after:
            if hold_rmse < best_rmse - 1e-9:
                best_rmse, best_epoch = hold_rmse, epoch
                best_state = DV.cpu_state_dict(model_a)
                stale = 0
            else:
                stale += 1
                if stale >= config.patience:
                    break
    if best_state is None:
        raise RuntimeError(f"{mode}: no eligible epoch improved on A/holdout")
    epochs = int(best_epoch)
    model_a.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # ---- stage B: refit for the inner-selected count, same schedule
    set_seed(config.seed)
    model_b = make_model(state_b, stage_macro_b, b_fit.query_context.shape[1]).to(device)
    optimiser = torch.optim.AdamW(model_b.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.schedule_length())
    b_curve = []
    for epoch in range(1, epochs + 1):
        total, seen = 0.0, 0
        model_b.train(True)
        for features, target in MacroDeviceTensors(b_fit, device, stage_macro_b).batches(
                config.batch_size, shuffle=True, seed=config.seed * 1000 + epoch,
                with_target=True):
            optimiser.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model_b(features), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_b.parameters(), config.clip_grad)
            optimiser.step()
            total += float(loss.item()) * len(target)
            seen += len(target)
        scheduler.step()
        b_curve.append({"epoch": epoch, "train_mse": total / max(seen, 1)})

    # ---- evaluate once, then prove the checkpoint reloads on CPU
    dev_b_val = MacroDeviceTensors(b_val, device, stage_macro_b)
    predicted = b_val.to_sgd(DV.predict_on(model_b, dev_b_val))
    overall = metrics(b_val.actual_sgd, predicted)

    description = _describe(model_b)
    # save EVERYTHING a reload needs, so the checkpoint is self-sufficient by construction
    # rather than recoverable afterwards (the s42 arms needed a recovery sidecar)
    checkpoint_payload = {
        "config": config.to_dict(), "state_dict": DV.cpu_state_dict(model_b),
        "state_a": state_a.to_dict(), "state_b": state_b.to_dict(),
        "macro_mode": mode, "selected_epoch": epochs, "device": str(device),
        "description": description,
        "macro_contract": {
            "contract_id": macro_manifest.get("contract_id"),
            "contract_version": macro_manifest.get("contract_version"),
            "policy": macro_manifest.get("policy"),
            "source_sha256": macro_manifest.get("source_sha256"),
        },
    }
    if mode != "none":
        checkpoint_payload["macro_scalers"] = {
            stage: {name: getattr(sm.scaler, name).to_dict()
                    for name in ("dbs", "dbs_scalar", "coe", "coe_scalar")}
            for stage, sm in (("A", stage_macro_a), ("B", stage_macro_b))}
        checkpoint_payload["macro_scalers_meta"] = {
            stage: {"contract_id": sm.scaler.contract_id, "split_id": sm.scaler.split_id,
                    "stage": sm.scaler.stage, "policy": sm.scaler.policy}
            for stage, sm in (("A", stage_macro_a), ("B", stage_macro_b))}
        checkpoint_payload["macro_row_window_index"] = {
            stage: np.asarray(sm.window_index, dtype=np.int64).tolist()
            for stage, sm in (("A", stage_macro_a), ("B", stage_macro_b))}
    torch.save(checkpoint_payload, out_dir / "checkpoint.pt")
    np.savez_compressed(out_dir / "validation_predictions.npz",
                        predicted_sgd=predicted, actual_sgd=b_val.actual_sgd,
                        row_id=b_val.row_id, horizon=b_val.horizon)

    stored = torch.load(out_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    reload_stage_macro_b = stage_macro_b
    if mode != "none" and "macro_scalers" in stored:
        # rebuild the macro preprocessing from the checkpoint, never from live artifacts
        from experiments.macro_encoders_v3.scaling import MacroScalers, ScalerState

        states = {n: ScalerState.from_dict(stored["macro_scalers"]["B"][n])
                  for n in ("dbs", "dbs_scalar", "coe", "coe_scalar")}
        meta = stored["macro_scalers_meta"]["B"]
        saved_scalers = MacroScalers(dbs=states["dbs"], dbs_scalar=states["dbs_scalar"],
                                     coe=states["coe"], coe_scalar=states["coe_scalar"],
                                     contract_id=meta["contract_id"],
                                     split_id=meta["split_id"], stage=meta["stage"],
                                     policy=meta["policy"])
        reload_stage_macro_b = build_stage_macro(macro_manifest, SPLIT, "B",
                                                scalers=saved_scalers)
    reload_model = make_model(state_b, reload_stage_macro_b, b_fit.query_context.shape[1])
    reload_model.load_state_dict(stored["state_dict"])
    reload_model.eval()
    reload_pred = b_val.to_sgd(DV.predict_on(
        reload_model, MacroDeviceTensors(b_val, torch.device("cpu"),
                                          reload_stage_macro_b)))
    reload_delta = float(np.abs(reload_pred - predicted).max())

    summary = {
        "macro_mode": mode, "pooling": config.pooling,
        "selected_epoch": epochs, "schedule_epochs_effective": config.schedule_length(),
        "inner_curve": curve, "b_curve": b_curve,
        "overall": overall,
        "description": description,
        "device": str(device),
        "cpu_reload_max_abs_delta_sgd": reload_delta,
        "cpu_reload_ok": reload_delta < 1e-2,
        "n_validation": int(len(b_val)),
        "seconds": round(time.perf_counter() - started, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _describe(model) -> dict:
    return MV.describe(model)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--run-id", default="v4-macro-primary-s42")
    parser.add_argument("--candidate", default=None,
                        help="registered_candidates.json to take rank 1 from")
    parser.add_argument("--modes", default="none,both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from experiments.macro_encoders_v3 import contract as macro_contract
    from neural_fusion_v3.macro_bridge import build_stage_macro, StageMacroTensors

    device = DV.resolve_device(args.device)
    torch.set_num_threads(2)
    contract_dir = CONTRACT_ROOT / CONTRACT_ID
    manifest = C.load_manifest(contract_dir)

    source_run = (REPORT_ROOT / "runs" / "v4-search-mps-s42")
    registration = Path(args.candidate) if args.candidate else \
        source_run / "registered_candidates.json"
    candidate = json.loads(Path(registration).read_text())["candidates"][0]

    config = TrainConfig(**{k: v for k, v in candidate["config"].items()
                            if k in TrainConfig().__dict__})
    config.schedule_epochs = config.max_epochs
    # the CLI seed must actually reach training; an earlier version recorded it in the
    # manifest only, so a "seed 43" run silently reused seed 42
    config.seed = int(args.seed)

    macro_dir = macro_contract.latest_v3_contract_dir()
    macro_manifest = macro_contract.load_manifest(macro_dir)

    a_fit_pop = C.load_population(contract_dir, manifest, SPLIT, "A", "fit")
    a_hold_pop = C.load_population(contract_dir, manifest, SPLIT, "A", "holdout")
    b_fit_pop = C.load_population(contract_dir, manifest, SPLIT, "B", "fit")
    b_val_pop = C.load_population(contract_dir, manifest, SPLIT, "B", "validation")

    state_a = fit_stage_state(a_fit_pop, "A/fit")
    state_b = fit_stage_state(b_fit_pop, "B/fit")
    a_fit = build_tensors(a_fit_pop, state_a)
    a_hold = build_tensors(a_hold_pop, state_a)
    b_fit = build_tensors(b_fit_pop, state_b)
    b_val = build_tensors(b_val_pop, state_b)

    stage_macro_a = build_stage_macro(macro_manifest, SPLIT, "A")
    stage_macro_b = build_stage_macro(macro_manifest, SPLIT, "B")

    run_dir = REPORT_ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "run_id": args.run_id, "device": str(device), "candidate": candidate,
        "macro_contract": macro_manifest.get("contract_id"),
        "policy": macro_manifest.get("policy"), "seed": args.seed,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "one macro ablation on the registered rank-1 architecture; not a search",
    }, indent=2))

    results = []
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"[macro={mode}] rank1 trial {candidate['trial_id']} "
              f"pooling={config.pooling} device={device} ...", flush=True)
        summary = run_arm(mode, config, device, run_dir / f"arm_{mode}",
                          a_fit, a_hold, b_fit, b_val, state_a, state_b,
                          macro_manifest, stage_macro_a, stage_macro_b, build_stage_macro)
        print(f"    inner epoch {summary['selected_epoch']} | B/validation RMSE "
              f"{summary['overall']['rmse_sgd']:.4f} | MAE {summary['overall']['mae_sgd']:.4f}"
              f" | params {summary['description']['parameters']}"
              f" | {summary['seconds']}s", flush=True)
        print(f"    cpu reload delta {summary['cpu_reload_max_abs_delta_sgd']:.3e} "
              f"ok={summary['cpu_reload_ok']}", flush=True)
        results.append(summary)

    (run_dir / "summary.json").write_text(json.dumps(results, indent=2))
    if len(results) == 2:
        gap = results[0]["overall"]["rmse_sgd"] - results[1]["overall"]["rmse_sgd"]
        print(f"[comparison] {results[0]['macro_mode']} minus {results[1]['macro_mode']}: "
              f"{gap:+.4f} SGD", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
