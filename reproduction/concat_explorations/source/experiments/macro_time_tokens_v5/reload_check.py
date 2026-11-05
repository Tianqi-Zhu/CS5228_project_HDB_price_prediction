"""Save/reload verification for a v5 checkpoint — with fitting prohibited.

This exists because a checkpoint that only carried `macro_scalers` forced a future reader to
refit the FT preprocessing to reload the model. Refitting on the wrong population silently
changes every input column, and the earlier acceptance round was rejected for exactly that.

`load_checkpoint_for_inference` below refuses to call `fit_stage_state` at all: it rebuilds the
`StageState` from the saved dict and asserts the column order matches before applying it. The
test then compares predictions against the saved `inner_predictions.npz`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_EXPERIMENTS = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from experiments.macro_time_tokens_v5 import model as M
from experiments.macro_time_tokens_v5 import sources as S


class RefitProhibited(RuntimeError):
    pass


def _forbid_refit(*args, **kwargs):
    raise RefitProhibited(
        "a checkpoint reload must never refit preprocessing; the saved feature_state is the "
        "authority. If it is missing, the checkpoint is not reloadable.")


def load_checkpoint_for_inference(ckpt_path: Path, device: str = "cpu",
                                  population_loader=None):
    """Rebuild the model described by a checkpoint, without fitting anything.

    `population_loader` must return `(manifest, state, a_fit, a_hold, origins)`. The state is
    rebuilt from the checkpoint; the freshly-fitted one from the loader is used only to load the
    raw populations, and its scaler is discarded after an explicit equality check.
    """
    import feature_transformer_v4.data as feature_data

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "feature_state" not in checkpoint:
        raise RefitProhibited(
            f"{ckpt_path} has no 'feature_state'; it cannot be reloaded without refitting")

    saved = feature_data.StageState.from_dict(checkpoint["feature_state"])
    numeric_columns = list(checkpoint["numeric_columns"])
    categorical_columns = list(checkpoint["categorical_columns"])
    if numeric_columns != list(saved.numeric_names):
        raise RefitProhibited(
            "saved numeric column order disagrees with the saved scaler; the checkpoint is "
            "internally inconsistent")
    if categorical_columns != list(saved.categorical_names):
        raise RefitProhibited("saved categorical order disagrees with the saved vocabularies")

    manifest, _, a_fit, a_hold, origins = population_loader()

    # a fit-then-compare guard: this is the step that would have caught the old defect
    refit = feature_data.fit_stage_state(a_fit, "A/fit")
    if refit.scaler.to_dict() != saved.scaler.to_dict():
        raise RefitProhibited(
            "feature_state in the checkpoint does not match A/fit; the checkpoint was "
            "produced on a different population or the protocol changed")
    del refit

    config = checkpoint["config"]
    from feature_transformer_v4.training import TrainConfig, build_for

    torch.manual_seed(config["seed"])
    base = build_for(saved.numeric_names, saved.cardinalities(),
                     context_dim=a_fit.query_context.shape[1],
                     config=TrainConfig(**config))

    from experiments.macro_time_tokens_v5 import run as runner
    macro_manifest, macro_stage, window_tensors = runner.load_macro()
    if macro_manifest["contract_id"] != checkpoint["contract_id"]:
        raise RefitProhibited("macro contract id in the checkpoint does not match the contract")

    arm = checkpoint["arm"]
    if arm == "summary2":
        source = S.SummarySource(window_tensors, macro_stage.window_index,
                                 macro_manifest["contract_id"], macro_stage.policy)
    else:
        source = S.BlockEndpointSource(window_tensors, macro_stage.window_index,
                                       macro_manifest["contract_id"], macro_stage.policy)

    model = M.MacroBlocksOnFeatureTransformer(base, source, arm)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(device)).eval()
    return model, checkpoint, (a_fit, a_hold), macro_stage


def verify(ckpt_path: Path, run_root: Path, device: str = "cpu") -> dict:
    from feature_transformer_v4.macro_ablation import MacroDeviceTensors
    from experiments.macro_time_tokens_v5 import run as runner

    model, checkpoint, (a_fit, a_hold), macro_stage = load_checkpoint_for_inference(
        ckpt_path, device=device, population_loader=runner.load_data)

    dev = MacroDeviceTensors(a_hold, torch.device(device), macro_stage)
    with torch.no_grad():
        residual = torch.cat([model(features) for features, _ in
                              dev.batches(checkpoint["config"]["batch_size"],
                                          shuffle=False, with_target=True)])
    residual = residual.cpu().numpy().astype(np.float64)
    predicted = (residual + a_hold.anchor_ksgd) * 1000.0

    saved = np.load(run_root / "inner_predictions.npz")
    rows_match = bool(np.array_equal(np.asarray(a_hold.row_id),
                                     saved["row_id"].astype(a_hold.row_id.dtype)))
    delta = float(np.abs(predicted - saved["predicted_sgd"]).max())
    return {"checkpoint": str(ckpt_path), "rows_match": rows_match,
            "max_abs_delta_sgd": delta, "n": int(len(predicted)),
            "feature_state_present": True,
            "v2_contract_id": checkpoint.get("v2_contract_id"),
            "macro_contract_id": checkpoint.get("contract_id"),
            "fit_metadata": checkpoint.get("fit_metadata"),
            "reload_ok": rows_match and delta == 0.0}


def main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    root = Path(args.run_root)
    report = {}
    for arm in ("summary2", "blocks7"):
        ckpt = root / arm / "checkpoint.pt"
        if not ckpt.exists():
            report[arm] = {"missing": str(ckpt)}
            continue
        report[arm] = verify(ckpt, root / arm, device=args.device)
        print(f"{arm}: {report[arm]}", flush=True)
    (root / "reload_check.json").write_text(json.dumps(report, indent=2))
    return 0 if all(r.get("reload_ok") for r in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
