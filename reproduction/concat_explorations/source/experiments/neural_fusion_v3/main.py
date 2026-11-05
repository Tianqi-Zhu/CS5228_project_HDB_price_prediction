"""Tokenisation and macro-branch ablation on the frozen v2.1 contract.

    # tokenisation only (numeric block vs per-feature tokens)
    env -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \\
      -m experiments.neural_fusion_v3.main --split orig --controls lstm_fusion \\
      --token-modes block --seeds 42 --mode probe --macro none

    # macro branches (needs reports/fusion_contract_v3/)
    ... --macro dbs|coe|both --macro-policy origin_asof

Everything except the numeric representation and the macro memory tokens is reused from
`neural_fusion_v2`: same contract reader, same per-stage scaler, same training loop, same
metrics, same prediction exporter.

`--macro none` builds v2's own model class (see `fusion_macro.build_fusion`), so a `none`
run is the reference the enabled runs are compared against, not a look-alike of it.

Writes only under ``reports/neural_fusion_v3/runs/<run_id>/``. Existing run directories are
never deleted or reused: a failure keeps its partial artifacts and gets a new run id.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_EXPERIMENTS = _Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in _sys.path:      # so `neural_fusion_v2` is importable
    _sys.path.insert(0, str(_EXPERIMENTS))

import argparse
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neural_fusion_v2 import contract as C
from neural_fusion_v2.adapter import build_tensors
from neural_fusion_v2.preprocessing import ScalerState
from neural_fusion_v2.settings import CAPACITY, TRAIN
from neural_fusion_v2.training import (FitConfig, band_report, fit_early_stopping,
                                      fit_fixed_epochs, metrics, monthly_report, predict,
                                      predictions_frame, save_json, set_seed)

from . import macro
from .fusion_macro import MacroFusionUnsupported, build_fusion, describe
from .macro_bridge import (MacroAlignmentError, MacroTensors, align_population,
                           build_stage_macro)
from .models import Shapes, build_control, parameter_count
from .tokens import FEATURE_TOKEN_MODES

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
REPORT_ROOT = ROOT / "reports" / "neural_fusion_v3"
DEFAULT_CONTRACT_ROOT = ROOT / "reports" / "fusion_contract_v2"
DEFAULT_CONTRACT_ID = "fusion-contract-v2.1.0__43867d0ba111"
MACRO_CONTRACT_ROOT = ROOT / "reports" / "fusion_contract_v3"

CONTROLS = ["mlp", "static_transformer", "lstm_concat", "lstm_fusion"]
MACRO_MODES = ("none", "dbs", "coe", "both")
MODES = {
    "macro-smoke": {"epochs": 2, "patience": 2, "batch_size": 128, "row_cap": 512},
    "smoke": {"epochs": 3, "patience": 2, "batch_size": 256, "row_cap": 4000},
    "probe": {"epochs": 20, "patience": 5, "batch_size": 512, "row_cap": None},
    "full": {"epochs": 60, "patience": 8, "batch_size": 512, "row_cap": None},
}
EXIT_CONTRACT_NOT_READY = 3
EXIT_MACRO_NOT_READY = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint() -> dict:
    """Every file that can change a number in this run, including the macro package."""
    payload = {str(p.relative_to(PACKAGE)): sha256_file(p) for p in sorted(PACKAGE.rglob("*.py"))}
    for name in ("neural_fusion_v2", "macro_encoders_v3"):
        sibling = PACKAGE.parent / name
        if sibling.exists():
            payload.update({f"../{name}/{p.relative_to(sibling)}": sha256_file(p)
                            for p in sorted(sibling.rglob("*.py"))})
    return payload


def environment() -> dict:
    return {
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "torch": torch.__version__, "platform": platform.platform(), "host": socket.gethostname(),
        "torch_threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
        "device": "cpu", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
    }


def shapes_of(population, use_sequence: bool, encoding: dict) -> Shapes:
    """Dimensions only. The token mode is a constructor argument, not a shape field."""
    a = population.arrays
    cards = C.encoding_cardinalities(encoding, a)
    return Shapes(
        n_categorical=len(a.categorical_columns), numeric_dim=len(a.numeric_columns),
        history_dim=len(a.history_order), context_dim=len(a.context_order) - 1,
        cardinalities=tuple(cards),
        sequence_channels=a.n_channels if use_sequence else 0,
        sequence_length=a.n_months * a.n_levels if use_sequence else 0)


def load_split(contract_dir: Path, manifest: dict, split: str):
    a_fit = C.load_population(contract_dir, manifest, split, "A", "fit")
    a_hold = C.load_population(contract_dir, manifest, split, "A", "holdout")
    b_fit = C.load_population(contract_dir, manifest, split, "B", "fit")
    b_val = C.load_population(contract_dir, manifest, split, "B", "validation")
    report = C.validate_boundaries(manifest, split, a_fit, a_hold, b_fit, b_val)
    return a_fit, a_hold, b_fit, b_val, report


def actual_manifest(config, shapes, macro_mode: str) -> dict:
    """The settings this run really used, written from v3's own Shapes fields."""
    return {
        "control": config.control, "macro_mode": macro_mode,
        "capacity": dict(CAPACITY),
        "shapes": {
            "n_categorical": shapes.n_categorical, "numeric_dim": shapes.numeric_dim,
            "history_dim": shapes.history_dim, "context_dim": shapes.context_dim,
            "cardinalities": list(shapes.cardinalities),
            "sequence_channels": shapes.sequence_channels,
            "sequence_length": shapes.sequence_length,
        },
        "optimiser": {"name": "AdamW", "lr": config.learning_rate,
                      "weight_decay": config.weight_decay, "clip_grad": config.clip_grad,
                      "scheduler": "CosineAnnealingLR"},
        "batch_size": config.batch_size, "seed": config.seed,
        "max_epochs": config.epochs, "patience": config.patience,
        "threads": torch.get_num_threads(),
        "loss": "MSE on kSGD residual from the contract anchor (anchor kept in SGD)",
        "early_stopping": "A/fit train, A/holdout monitor",
        "refit": "B/fit, fixed epoch count, last weights kept",
    }


def _tag(control: str, token_mode: str, macro_mode: str) -> str:
    base = f"{control}__{token_mode}"
    return base if macro_mode == "none" else f"{base}__macro-{macro_mode}"


def _macro_source(stage_macro, macro_manifest: dict, mode: str):
    """Real DBS/COE encoders from the producer package, exposed as a token source.

    Rows go through the producer's own `encode_batch`, which encodes only the unique
    windows a batch needs and gathers back to rows while keeping the graph. Nothing is
    re-derived here: the values are the contract's, scaled by the stage's fit population.
    """
    if stage_macro is None:
        raise MacroFusionUnsupported(f"macro mode {mode!r} needs a stage's macro tensors")
    from experiments.macro_encoders_v3 import api as macro_api
    from experiments.macro_encoders_v3 import data as macro_data
    from experiments.macro_encoders_v3 import encoders as macro_encoders

    windows = macro_data.load_windows(macro_manifest, stage_macro.split, stage_macro.stage)
    window_tensors = macro_api.build_window_tensors(windows, stage_macro.scaler)
    modules = macro_encoders.build_encoders()
    row_window_index = np.asarray(stage_macro.window_index, dtype=np.int64)

    class _Source(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # NOT `self.modules`: that name is nn.Module.modules(), a property. Assigning
            # to it would shadow the registry and `self.modules` would later return the
            # bound method, so .dbs / .parameters() would break.
            self.encoders = modules
            self.tensors = window_tensors
            self.mode = mode
            self.n_tokens = ((1 if mode in ("dbs", "both") else 0)
                             + (1 if mode in ("coe", "both") else 0))
            self.register_buffer("row_window_index",
                                 torch.as_tensor(row_window_index, dtype=torch.long),
                                 persistent=False)

        def forward(self, batch):
            rows = batch["macro_row"]
            per_row = macro_api.encode_batch(self.encoders, self.tensors,
                                             self.row_window_index.numpy(), rows.numpy())
            tokens = []
            if self.mode in ("dbs", "both"):
                tokens.append(per_row["dbs_token"])
            if self.mode in ("coe", "both"):
                tokens.append(per_row["coe_token"])
            return torch.stack(tokens, dim=1)

        def describe(self):
            payload = {
                "package": "experiments/macro_encoders_v3", "mode": self.mode,
                "tokens": self.n_tokens,
                "encoder_parameters": macro_encoders.parameter_count(self.encoders),
                "semantics": "financial-state embedding learned through the rental loss; "
                             "not a validated measure of economic health",
            }
            if self.mode in ("dbs", "both"):
                payload["dbs"] = {"symbol": "D05.SI", "window": "63 trading days",
                                  "axis": "trading_day",
                                  "hidden": self.encoders.dbs.hidden_dim,
                                  "token_dim": self.encoders.dbs.token_dim}
            if self.mode in ("coe", "both"):
                payload["coe"] = {"categories": ["a", "b", "c", "e"],
                                  "window": "12 calendar months", "axis": "calendar_month",
                                  "categories_are_not_timesteps": True,
                                  "hidden": self.encoders.coe.hidden_dim,
                                  "token_dim": self.encoders.coe.token_dim}
            return payload

    return _Source()


def _scalers_from_checkpoint(stored: dict, stage: str):
    """Rebuild the saved MacroScalers; a reload must not refit them."""
    from experiments.macro_encoders_v3.scaling import MacroScalers, ScalerState

    payload = stored["macro"]["scalers_" + stage.lower()]
    states = {name: ScalerState.from_dict(payload[name])
              for name in ("dbs", "dbs_scalar", "coe", "coe_scalar")}
    meta = stored["macro"]["stage_" + stage.lower()]
    return MacroScalers(dbs=states["dbs"], dbs_scalar=states["dbs_scalar"],
                        coe=states["coe"], coe_scalar=states["coe_scalar"],
                        contract_id=str(stored["macro"]["contract_id"]),
                        split_id=str(meta["split"]), stage=str(meta["stage"]),
                        policy=str(stored["macro"]["policy"]))


def _wrap_macro(base_tensors, population, stage_macro) -> MacroTensors:
    if stage_macro is None:
        raise MacroAlignmentError("macro tensors were not built for this stage")
    return MacroTensors(base_tensors, population.arrays.row_id, stage_macro)


def run_one(control: str, token_mode: str, split: str, pops, encoding_a: dict,
            encoding_b: dict, mode: str, seed: int, run_dir: Path, command: str,
            macro_mode: str = "none", macro_manifest: dict | None = None) -> dict:
    a_fit, a_hold, b_fit, b_val, bounds = pops
    spec = MODES[mode]
    use_sequence = control in ("lstm_concat", "lstm_fusion")
    started = time.perf_counter()

    row_cap = spec["row_cap"]
    if row_cap is not None:
        a_fit = a_fit.subset(np.arange(min(row_cap, len(a_fit))))
        a_hold = a_hold.subset(np.arange(min(row_cap, len(a_hold))))
        b_fit = b_fit.subset(np.arange(min(row_cap, len(b_fit))))
        # B/validation is the evaluation set: never cap it, or the exported predictions
        # stop covering the population a checkpoint reload has to reproduce

    state_a = ScalerState.fit(a_fit.arrays, population="A/fit")
    t_a_fit = build_tensors(a_fit, state_a, use_sequence=use_sequence)
    t_a_hold = build_tensors(a_hold, state_a, use_sequence=use_sequence)

    stage_macro_a = stage_macro_b = None
    if macro_mode != "none":
        stage_macro_a = build_stage_macro(macro_manifest, split, "A")
        stage_macro_b = build_stage_macro(macro_manifest, split, "B")
        # a positional mismatch would train on another row's stock window: fail loudly
        for population in (a_fit, a_hold):
            align_population(stage_macro_a, population)
        for population in (b_fit, b_val):
            align_population(stage_macro_b, population)

    numeric_columns = state_a.static_columns
    shapes = shapes_of(a_fit, use_sequence, encoding_a)
    config = FitConfig(control=control, seed=seed, epochs=spec["epochs"],
                       patience=spec["patience"], batch_size=spec["batch_size"])

    def make_model(shapes_for_control, columns, stage_macro):
        def base_factory(name, sh):
            return build_control(name, sh, columns, token_mode=token_mode)

        return build_fusion(base_factory, control, shapes_for_control, macro_mode,
                            lambda mode_name: _macro_source(stage_macro, macro_manifest,
                                                            mode_name))

    set_seed(seed)
    model_a = make_model(shapes, numeric_columns, stage_macro_a)
    n_params = parameter_count(model_a)
    tensors_a_fit = (_wrap_macro(t_a_fit, a_fit, stage_macro_a) if macro_mode != "none"
                     else t_a_fit)
    tensors_a_hold = (_wrap_macro(t_a_hold, a_hold, stage_macro_a) if macro_mode != "none"
                      else t_a_hold)
    stage_a = fit_early_stopping(model_a, tensors_a_fit, tensors_a_hold, config)
    chosen = max(stage_a.best_epoch, 1)

    state_b = ScalerState.fit(b_fit.arrays, population="B/fit")
    t_b_fit = build_tensors(b_fit, state_b, use_sequence=use_sequence)
    t_b_val = build_tensors(b_val, state_b, use_sequence=use_sequence)
    shapes_b = shapes_of(b_fit, use_sequence, encoding_b)
    if (shapes_b.numeric_dim, shapes_b.history_dim, shapes_b.context_dim,
            shapes_b.sequence_channels) != (shapes.numeric_dim, shapes.history_dim,
                                            shapes.context_dim, shapes.sequence_channels):
        raise AssertionError(f"A/B feature shapes disagree: {shapes} vs {shapes_b}")

    set_seed(seed)
    model_b = make_model(shapes_b, state_b.static_columns, stage_macro_b)
    tensors_b_fit = (_wrap_macro(t_b_fit, b_fit, stage_macro_b) if macro_mode != "none"
                     else t_b_fit)
    tensors_b_val = (_wrap_macro(t_b_val, b_val, stage_macro_b) if macro_mode != "none"
                     else t_b_val)
    stage_b_started = time.perf_counter()
    fit_fixed_epochs(model_b, tensors_b_fit, config, chosen)
    stage_b_seconds = time.perf_counter() - stage_b_started

    predicted = t_b_val.to_sgd(predict(model_b, tensors_b_val))
    if not np.isfinite(predicted).all():
        raise FloatingPointError(f"{control}/{token_mode}/{macro_mode}: non-finite predictions")
    frame = predictions_frame(b_val, predicted, control=control, seed=seed)
    frame.insert(8, "token_mode", token_mode)
    frame.insert(9, "macro_mode", macro_mode)
    overall = metrics(frame["actual_rent_sgd"].to_numpy(), frame["predicted_rent_sgd"].to_numpy())

    tag = _tag(control, token_mode, macro_mode)
    paths = {name: run_dir / name / split for name in ("predictions", "checkpoints",
                                                       "metrics", "preprocessing")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths["predictions"] / f"{tag}_seed{seed}.csv", index=False)
    state_a.save(paths["preprocessing"] / f"{tag}_seed{seed}_stageA.json")
    state_b.save(paths["preprocessing"] / f"{tag}_seed{seed}_stageB.json")

    description = describe(model_b)
    checkpoint = {
        "control": control, "token_mode": token_mode, "macro_mode": macro_mode,
        "state_dict": model_b.state_dict(),
        "shapes": shapes_b.__dict__, "capacity": dict(CAPACITY), "seed": seed, "split": split,
        "chosen_epochs": chosen, "uses_sequence": use_sequence,
        "stage_a_state": state_a.to_dict(), "stage_b_state": state_b.to_dict(),
        "numeric_columns": list(state_b.static_columns), "description": description,
    }
    if macro_mode != "none":
        macro_payload = {
            "contract_id": macro_manifest.get("contract_id"),
            "contract_version": macro_manifest.get("contract_version"),
            "policy": macro_manifest.get("policy"),
            "mode": macro_mode, "stage_a": stage_macro_a.describe(),
            "stage_b": stage_macro_b.describe(),
            "scalers_a": {name: getattr(stage_macro_a.scaler, name).to_dict()
                          for name in ("dbs", "dbs_scalar", "coe", "coe_scalar")},
            "scalers_b": {name: getattr(stage_macro_b.scaler, name).to_dict()
                          for name in ("dbs", "dbs_scalar", "coe", "coe_scalar")},
        }
        checkpoint["macro"] = macro_payload
        (paths["preprocessing"] / f"{tag}_seed{seed}_macroScalers.json").write_text(
            json.dumps(macro_payload, indent=2), encoding="utf-8")
    torch.save(checkpoint, paths["checkpoints"] / f"{tag}_seed{seed}.pt")

    result = {
        "run_id": run_dir.name, "mode": mode, "split": split, "control": control,
        "token_mode": token_mode, "macro_mode": macro_mode, "seed": seed,
        "uses_sequence": use_sequence, "n_parameters": n_params, "description": description,
        "n_A_fit": len(a_fit), "n_A_holdout": len(a_hold), "n_B_fit": len(b_fit),
        "n_B_validation": len(b_val),
        "chosen_epochs": chosen, "stage_a_epochs_run": stage_a.epochs_run,
        "stage_a_best_holdout_rmse_sgd": stage_a.best_inner_rmse,
        "stage_a_seconds": round(stage_a.seconds, 2),
        "stage_b_seconds": round(stage_b_seconds, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
        "overall": overall, "by_horizon_band": band_report(frame),
        "by_month": monthly_report(frame).to_dict(orient="records"),
        "parity_signature_B_fit": tensors_b_fit.parity_signature(),
        "parity_signature_B_validation": tensors_b_val.parity_signature(),
        "boundaries": bounds,
        "stage_a_scaler_fitted_on": "A/fit", "stage_b_scaler_fitted_on": "B/fit",
        "actual_manifest": actual_manifest(config, shapes_b, macro_mode),
        "command": command, "stage_a_curve": stage_a.history,
    }
    if macro_mode != "none":
        result["macro"] = {
            "mode": macro_mode, "policy": macro_manifest.get("policy"),
            "contract_id": macro_manifest.get("contract_id"),
            "stage_a": stage_macro_a.describe(), "stage_b": stage_macro_b.describe(),
            "tokens": description["macro_tokens"], "role": description["macro_role"],
        }
    save_json(paths["metrics"] / f"{tag}_seed{seed}.json", result)
    return {k: v for k, v in result.items() if k not in ("by_month", "stage_a_curve")}


def verify_checkpoint(run_dir: Path, split: str, control: str, token_mode: str, macro_mode: str,
                      seed: int, pops, macro_manifest: dict | None = None) -> dict:
    """Reload from disk and re-predict; proves the saved weights fully determine the output."""
    a_fit, a_hold, b_fit, b_val, _ = pops
    tag = _tag(control, token_mode, macro_mode)
    stored = torch.load(run_dir / "checkpoints" / split / f"{tag}_seed{seed}.pt",
                        map_location="cpu", weights_only=False)

    def base_factory(name, sh):
        return build_control(name, sh, stored["numeric_columns"],
                             token_mode=stored["token_mode"])

    stage_macro_b = None
    if macro_mode != "none":
        # the saved scalers are authoritative: refitting here would silently change the
        # scaling and make the comparison meaningless
        saved = _scalers_from_checkpoint(stored, "B")
        stage_macro_b = build_stage_macro(macro_manifest, split, "B", scalers=saved)
        align_population(stage_macro_b, b_val)
    model = build_fusion(base_factory, stored["control"], Shapes(**stored["shapes"]),
                         stored["macro_mode"],
                         lambda mode_name: _macro_source(stage_macro_b, macro_manifest,
                                                         mode_name))
    model.load_state_dict(stored["state_dict"])
    state_b = ScalerState.from_dict(stored["stage_b_state"])
    tensors = build_tensors(b_val, state_b, use_sequence=stored["uses_sequence"])
    if macro_mode != "none":
        tensors = _wrap_macro(tensors, b_val, stage_macro_b)
    rebuilt = tensors.to_sgd(predict(model, tensors))
    exported = pd.read_csv(run_dir / "predictions" / split / f"{tag}_seed{seed}.csv")
    if len(exported) != len(rebuilt):
        return {"ok": False, "reason": f"rows {len(exported)} vs {len(rebuilt)}"}
    delta = float(np.abs(exported["predicted_rent_sgd"].to_numpy() - rebuilt).max())
    return {"ok": delta < 1e-4, "max_abs_delta_sgd": delta, "n": int(len(rebuilt)),
            "n_state_dict_keys": len(stored["state_dict"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tokenisation and macro ablation on the frozen v2.1 contract")
    parser.add_argument("--contract-id", default=DEFAULT_CONTRACT_ID)
    parser.add_argument("--contract-root", default=str(DEFAULT_CONTRACT_ROOT))
    parser.add_argument("--split", default="orig")
    parser.add_argument("--controls", default="all")
    parser.add_argument("--token-modes", default="block,per_feature")
    parser.add_argument("--seeds", default=str(TRAIN["seed"]))
    parser.add_argument("--mode", default="probe", choices=list(MODES))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--verify-checkpoints", action="store_true")
    parser.add_argument("--macro", default="none", choices=list(MACRO_MODES),
                        help="none | dbs | coe | both")
    parser.add_argument("--macro-policy", default=macro.POLICIES[0],
                        choices=list(macro.POLICIES))
    parser.add_argument("--macro-contract-root", default=str(MACRO_CONTRACT_ROOT))
    args = parser.parse_args(argv)

    controls = (CONTROLS if args.controls == "all"
                else [c.strip() for c in args.controls.split(",") if c.strip()])
    unknown = [c for c in controls if c not in CONTROLS]
    if unknown:
        raise SystemExit(f"unknown controls {unknown}; available {CONTROLS}")
    token_modes = [m.strip() for m in args.token_modes.split(",") if m.strip()]
    bad = [m for m in token_modes if m not in FEATURE_TOKEN_MODES]
    if bad:
        raise SystemExit(f"unknown token modes {bad}; available {FEATURE_TOKEN_MODES}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    torch.set_num_threads(TRAIN["threads"])
    contract_dir = Path(args.contract_root) / args.contract_id
    try:
        manifest = C.load_manifest(contract_dir)
        encoding_a = C.load_encoding(contract_dir, manifest, args.split, "A")
        encoding_b = C.load_encoding(contract_dir, manifest, args.split, "B")
        pops = load_split(contract_dir, manifest, args.split)
    except (C.ContractNotReady, C.ContractInvalid) as error:
        print(f"CONTRACT NOT READY: {error}", file=sys.stderr)
        return EXIT_CONTRACT_NOT_READY

    macro_manifest = None
    macro_framing = {"enabled": False, "mode": args.macro, "policy": args.macro_policy}
    if args.macro != "none":
        from experiments.macro_encoders_v3 import contract as macro_contract
        from experiments.macro_encoders_v3 import data as macro_data

        from .macro import MacroInvalid, MacroNotReady, validate_policy
        try:
            macro_dir = macro_contract.latest_v3_contract_dir()
            macro_manifest = macro_contract.load_manifest(macro_dir)
            policy = validate_policy(macro_manifest, args.macro_policy)
            for stage in ("A", "B"):
                macro_data.load_windows(macro_manifest, args.split, stage)
            macro_framing = {
                "enabled": True, "mode": args.macro, "policy": policy["policy"],
                "strict_fixed_cutoff": policy["strict_fixed_cutoff"],
                "contract_id": macro_manifest.get("contract_id"),
                "contract_version": macro_manifest.get("contract_version"),
                "contract_dir": str(macro_dir),
                "policy_report": macro.policy_report(macro_manifest),
            }
        except (MacroNotReady, MacroInvalid, ValueError, KeyError) as error:
            print(f"MACRO NOT READY: {error}", file=sys.stderr)
            return EXIT_MACRO_NOT_READY

    run_id = args.run_id or f"tok-{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = REPORT_ROOT / "runs" / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"run dir already exists: {run_dir}")

    command = " ".join([".venv/bin/python", "-m", "experiments.neural_fusion_v3.main"] +
                       (argv if argv is not None else sys.argv[1:]))
    framing = {
        "run_id": run_id, "mode": args.mode, "split": args.split,
        "contract_id": manifest["contract_id"], "contract_version": manifest["contract_version"],
        "contract_dir": str(contract_dir),
        "contract_source_sha256": manifest.get("source_sha256"),
        "contract_fingerprint": C.contract_fingerprint(contract_dir),
        "code_fingerprint": code_fingerprint(), "environment": environment(),
        "capacity": dict(CAPACITY), "args": vars(args),
        "controls": controls, "token_modes": token_modes, "seeds": seeds,
        "macro": macro_framing,
        "split_metadata": manifest["splits"][args.split]["split"],
        "population_boundaries": pops[-1],
        "command": command, "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "result_status": ("REAL_CONTRACT_MACRO_ABLATION" if args.macro != "none"
                          else "REAL_CONTRACT_TOKEN_ABLATION"),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "run_manifest.json", framing)

    results, checks = [], []
    for control in controls:
        for token_mode in token_modes:
            for seed in seeds:
                print(f"[{args.split}] {control} token={token_mode} macro={args.macro} "
                      f"seed={seed} ...", flush=True)
                result = run_one(control, token_mode, args.split, pops, encoding_a, encoding_b,
                                 args.mode, seed, run_dir, command, args.macro, macro_manifest)
                results.append(result)
                print(f"    RMSE {result['overall']['rmse_sgd']:.2f} SGD | "
                      f"epochs {result['chosen_epochs']} | {result['n_parameters']} params | "
                      f"macro tokens {result['description']['macro_tokens']} | "
                      f"{result['total_seconds']:.1f}s", flush=True)
                if args.verify_checkpoints:
                    check = verify_checkpoint(run_dir, args.split, control, token_mode,
                                              args.macro, seed, pops, macro_manifest)
                    checks.append({"control": control, "token_mode": token_mode,
                                   "macro_mode": args.macro, "seed": seed, **check})
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
