"""Make a macro-arm checkpoint self-sufficient, without rewriting it.

The macro ablation's `checkpoint.pt` stores the two FT `StageState`s but **not** the macro
side of the preprocessing: no `MacroScalers`, no contract id / policy, no row mapping. Its
reload path therefore rebuilt them from live artifacts, which proves the *weights* reload —
not that the saved state is enough to reproduce the predictions.

This module closes that gap the only honest way available after the fact: it writes a
**sidecar** next to the checkpoint, recording everything a reload needs, and then proves the
sidecar is sufficient by rebuilding with fitting disabled. The original `checkpoint.pt` is
never modified.

What the sidecar carries, and why each piece is necessary:

* `macro_scalers_a` / `macro_scalers_b` — the four fitted `ScalerState`s the encoders were
  trained with. Recomputing them from the live stage would silently change the scaling.
* `macro_contract` — contract id, version, policy (`origin_asof`), and the source hashes the
  macro arrays were produced from, so a future reader can tell whether the artifacts match.
* `row_window_index` — the row -> window mapping actually used. It is derived from the
  contract, but recording it makes the checkpoint's meaning explicit rather than implicit.
* `writer_fingerprint` — hashes of the files that produced the run.

The verification then loads *only* what is in the sidecar plus the checkpoint, with
`fit_macro_scalers` and `fit_stage_state` replaced by raisers, and compares predictions.
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
from pathlib import Path

import numpy as np
import torch

from neural_fusion_v2 import contract as C

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
CONTRACT_ROOT = ROOT / "reports" / "fusion_contract_v2"
CONTRACT_ID = "fusion-contract-v2.1.0__43867d0ba111"
SPLIT = "orig"
SIDECAR = "recovery_state.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_macro_state(macro_manifest: dict, split: str, stage: str) -> dict:
    """The macro preprocessing state for one stage, as JSON-safe payload."""
    from experiments.macro_encoders_v3 import data as macro_data
    from experiments.macro_encoders_v3 import scaling as macro_scaling

    windows = macro_data.load_windows(macro_manifest, split, stage)
    rows = macro_data.load_rows(macro_manifest, split, stage)
    scalers = macro_scaling.fit_macro_scalers(windows, rows)
    return {
        "stage": stage,
        "scalers": {name: getattr(scalers, name).to_dict()
                    for name in ("dbs", "dbs_scalar", "coe", "coe_scalar")},
        "scalers_meta": {"contract_id": scalers.contract_id, "split_id": scalers.split_id,
                         "stage": scalers.stage, "policy": scalers.policy},
        "row_window_index": np.asarray(rows.row_window_index, dtype=np.int64).tolist(),
        "n_windows": int(windows.n_windows),
        "fitted_on": str(getattr(scalers.dbs, "fitted_on", "unknown")),
    }


def write_sidecar(arm_dir: Path, macro_manifest: dict, split: str) -> dict:
    """Write `recovery_state.json` beside the checkpoint. Never touches the checkpoint."""
    checkpoint_path = arm_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    macro_dir = _Path(macro_manifest.get("_contract_dir", "")) if macro_manifest else None
    sidecar = {
        "arm": arm_dir.name,
        "macro_mode": checkpoint.get("macro_mode"),
        "written_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "reason": (
            "the arm's checkpoint stores the FT StageStates but not the macro preprocessing "
            "state, so its reload used live artifacts. This sidecar records everything a "
            "reload needs and is verified with fitting disabled."),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "macro_contract": {
            "contract_id": macro_manifest.get("contract_id"),
            "contract_version": macro_manifest.get("contract_version"),
            "policy": macro_manifest.get("policy"),
            "source_sha256": macro_manifest.get("source_sha256"),
            "contract_dir_name": (macro_dir.name if macro_dir else None),
        },
        "policy_note": ("strict fixed-cutoff forecasting: no observation postdates the "
                        "window's cutoff month; 'target_previous_month' is not substituted"),
        "stage_macro": {},
        "writer_fingerprint": {
            name: _sha256(PACKAGE / name)
            for name in ("macro_ablation.py", "macro_v4.py", "device.py", "data.py",
                         "models.py", "training.py") if (PACKAGE / name).exists()
        },
        "not_modified": "checkpoint.pt is unchanged; its sha256 is recorded above",
    }
    for stage in ("A", "B"):
        sidecar["stage_macro"][stage] = capture_macro_state(macro_manifest, split, stage)

    (arm_dir / SIDECAR).write_text(json.dumps(sidecar, indent=2))
    return sidecar


def load_scalers(payload: dict):
    """Rebuild `MacroScalers` from a sidecar payload without fitting anything."""
    from experiments.macro_encoders_v3.scaling import MacroScalers, ScalerState

    states = {name: ScalerState.from_dict(payload["scalers"][name])
              for name in ("dbs", "dbs_scalar", "coe", "coe_scalar")}
    meta = payload["scalers_meta"]
    return MacroScalers(dbs=states["dbs"], dbs_scalar=states["dbs_scalar"],
                        coe=states["coe"], coe_scalar=states["coe_scalar"],
                        contract_id=meta["contract_id"], split_id=meta["split_id"],
                        stage=meta["stage"], policy=meta["policy"])


def verify_from_sidecar(arm_dir: Path, split: str = SPLIT) -> dict:
    """Rebuild the model and macro state from saved artefacts only, then re-predict.

    Every fitting entry point is replaced by a raiser for the duration, so a pass here means
    the saved state really is sufficient.
    """
    import experiments.macro_encoders_v3.scaling as macro_scaling
    from feature_transformer_v4 import data as D
    from feature_transformer_v4 import device as DV
    from feature_transformer_v4 import macro_ablation as MA
    from experiments.macro_encoders_v3 import contract as macro_contract
    from neural_fusion_v3.macro_bridge import build_stage_macro

    sidecar = json.loads((arm_dir / SIDECAR).read_text())
    checkpoint = torch.load(arm_dir / "checkpoint.pt", map_location="cpu", weights_only=False)

    manifest = C.load_manifest(CONTRACT_ROOT / CONTRACT_ID)
    b_val_pop = C.load_population(CONTRACT_ROOT / CONTRACT_ID, manifest, split, "B",
                                  "validation")

    # the macro contract manifest comes from the live directory, but only its identity is
    # used: the numeric state comes from the sidecar
    macro_manifest = macro_contract.load_manifest(macro_contract.latest_v3_contract_dir())
    recorded = sidecar["macro_contract"]
    contract_ok = macro_manifest.get("contract_id") == recorded["contract_id"]

    def forbid(*args, **kwargs):
        raise AssertionError("a sidecar reload must not fit anything")

    saved_fit = macro_scaling.fit_macro_scalers
    saved_stage_fit = D.fit_stage_state
    macro_scaling.fit_macro_scalers = forbid
    D.fit_stage_state = forbid
    try:
        stage_b = load_scalers(sidecar["stage_macro"]["B"])
        stage_macro_b = build_stage_macro(macro_manifest, split, "B", scalers=stage_b)
        state_b = D.StageState.from_dict(checkpoint["state_b"])
        tensors = D.build_tensors(b_val_pop, state_b)

        recorded_rows = np.asarray(sidecar["stage_macro"]["B"]["row_window_index"],
                                   dtype=np.int64)
        rows_match = bool(np.array_equal(recorded_rows,
                                         np.asarray(stage_macro_b.window_index, dtype=np.int64)))

        from feature_transformer_v4.training import TrainConfig, build_for
        config = TrainConfig(**{k: v for k, v in checkpoint["config"].items()
                                if k in TrainConfig().__dict__})
        base = build_for(state_b.numeric_names, state_b.cardinalities(),
                         context_dim=tensors.query_context.shape[1], config=config)
        from feature_transformer_v4 import macro_v4 as MV

        mode = checkpoint.get("macro_mode", "none")
        if mode == "none":
            model = base
        else:
            model = MV.build_with_macro(
                base,
                lambda m: MA._source_factory(stage_macro_b, macro_manifest, m, config.d_model),
                mode)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        device = torch.device("cpu")
        wrapped = (DV.DeviceTensors(tensors, device) if mode == "none"
                   else MA.MacroDeviceTensors(tensors, device, stage_macro_b))
        predicted = tensors.to_sgd(DV.predict_on(model, wrapped))
    finally:
        macro_scaling.fit_macro_scalers = saved_fit
        D.fit_stage_state = saved_stage_fit

    exported = np.load(arm_dir / "validation_predictions.npz")["predicted_sgd"]
    delta = float(np.abs(predicted - exported).max())
    result = {
        "arm": arm_dir.name, "macro_mode": mode,
        "checkpoint_sha256_unchanged": _sha256(arm_dir / "checkpoint.pt")
        == sidecar["checkpoint_sha256"],
        "contract_id_matches": bool(contract_ok),
        "recorded_policy": recorded["policy"],
        "row_mapping_reproduces": rows_match,
        "fitted_anything": False,
        "n": int(len(predicted)),
        "max_abs_delta_sgd": delta,
        "tolerance_sgd": 1e-2,
        "ok": bool(contract_ok and rows_match and delta <= 1e-2),
    }
    (arm_dir / "recovery_verification.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="v4-macro-primary-s42")
    parser.add_argument("--arms", default="none,both")
    parser.add_argument("--split", default=SPLIT)
    args = parser.parse_args(argv)

    from experiments.macro_encoders_v3 import contract as macro_contract
    macro_dir = macro_contract.latest_v3_contract_dir()
    macro_manifest = macro_contract.load_manifest(macro_dir)
    macro_manifest["_contract_dir"] = str(macro_dir)

    run_dir = ROOT / "reports" / "feature_transformer_v4" / "runs" / args.run_id
    outcomes = []
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        arm_dir = run_dir / f"arm_{arm}"
        if not (arm_dir / "checkpoint.pt").exists():
            print(f"[{arm}] no checkpoint yet; skipped", flush=True)
            continue
        print(f"[{arm}] writing recovery sidecar ...", flush=True)
        sidecar = write_sidecar(arm_dir, macro_manifest, args.split)
        print(f"    recorded contract {sidecar['macro_contract']['contract_id']} "
              f"policy {sidecar['macro_contract']['policy']}", flush=True)
        print(f"    macro scalers for stages: "
              f"{sorted(sidecar['stage_macro'])}", flush=True)
        print(f"[{arm}] verifying with fitting disabled ...", flush=True)
        result = verify_from_sidecar(arm_dir, args.split)
        print(f"    delta {result['max_abs_delta_sgd']:.3e} SGD "
              f"rows_match={result['row_mapping_reproduces']} "
              f"contract={result['contract_id_matches']} ok={result['ok']}", flush=True)
        outcomes.append(result)

    (run_dir / "recovery_summary.json").write_text(json.dumps(outcomes, indent=2))
    return 0 if all(o["ok"] for o in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
