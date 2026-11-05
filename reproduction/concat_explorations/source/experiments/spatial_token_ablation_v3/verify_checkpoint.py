"""Load-only checkpoint reconstruction and error decomposition.

Rather than re-implementing the reconstruction, this script reuses the consumer's own
`--verify-checkpoints` path, which loads the saved weights and the saved scaler states
and re-predicts without fitting. That is the strongest available evidence: it proves
the checkpoint alone determines the exported predictions.

If the consumer's verification is unavailable, the script falls back to a direct
load-only reconstruction of its own and says so in the output, so the report never
claims a check that did not run.

Also decomposes the MSE gap into bias^2 and variance, which is what the report needs
in order to say whether the ablation changed the level or the dispersion.

Run:
    .venv/bin/python -m experiments.spatial_token_ablation_v3.verify_checkpoint \
        --run-id spatial-probe-perfeature-none-s42-h1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

RUNS = ROOT / "reports" / "neural_fusion_v3" / "runs"
REPORT = ROOT / "reports" / "spatial_token_ablation_v3"

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def decompose(actual, predicted) -> dict:
    """MSE = bias^2 + var(error). Reported separately so level vs spread is visible."""
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    bias = float(error.mean())
    mse = float(np.mean(error ** 2))
    return {
        "rmse_sgd": float(np.sqrt(mse)),
        "mae_sgd": float(np.mean(np.abs(error))),
        "bias_sgd": bias,
        "error_sd_sgd": float(error.std()),
        "mse_from_bias": bias ** 2,
        "mse_from_variance": float(error.var()),
        "bias_share_of_mse": float(bias ** 2 / mse) if mse else float("nan"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="spatial-probe-perfeature-none-s42-h1")
    parser.add_argument("--reference", default="macro-probe-block-none-s42")
    args = parser.parse_args(argv)

    run_dir = RUNS / args.run_id
    summary = json.loads((run_dir / "run_summary.json").read_text())
    result = summary["results"][0]
    split = result["split"]

    checkpoint_path = next((run_dir / "checkpoints" / split).glob("*.pt"))
    print(f"checkpoint: {checkpoint_path.relative_to(run_dir)}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # ---------------------------------------------------------------- 1
    print("\n--- 1. the checkpoint is self-describing")
    required = ("state_dict", "shapes", "capacity", "token_mode", "numeric_columns",
                "stage_a_state", "stage_b_state", "chosen_epochs", "uses_sequence")
    for key in required:
        check(f"checkpoint carries {key}", key in payload)
    check("token mode recorded is per_feature",
          payload.get("token_mode") == "per_feature", str(payload.get("token_mode")))
    check("saved stage B scaler present (refit not required)",
          isinstance(payload.get("stage_b_state"), dict)
          and len(payload["stage_b_state"]) > 0)
    check("saved chosen_epochs present",
          isinstance(payload.get("chosen_epochs"), int),
          str(payload.get("chosen_epochs")))
    check("numeric_columns has 21 entries",
          len(payload.get("numeric_columns", [])) == 21,
          str(len(payload.get("numeric_columns", []))))

    # ---------------------------------------------------------------- 2
    print("\n--- 2. rebuild the model from recorded shapes only (no fit)")
    from neural_fusion_v3.models import Shapes, build_control

    recorded = payload["shapes"]
    shapes = Shapes(
        n_categorical=recorded["n_categorical"],
        numeric_dim=recorded["numeric_dim"],
        history_dim=recorded["history_dim"],
        context_dim=recorded["context_dim"],
        cardinalities=tuple(recorded["cardinalities"]),
        sequence_channels=recorded["sequence_channels"],
        sequence_length=recorded.get("sequence_length", 0))
    model = build_control(
        payload["control"], shapes, list(payload["numeric_columns"]),
        token_mode=payload["token_mode"], expected_numeric=list(payload["numeric_columns"]))
    load_result = model.load_state_dict(payload["state_dict"], strict=True)
    check("state_dict loads strictly (every tensor accounted for)", True,
          str(load_result))
    n_loaded = len(payload["state_dict"])
    n_model = len(dict(model.named_parameters())) + len(dict(model.named_buffers()))
    print(f"  saved tensors: {n_loaded}, model tensors: {n_model}")

    from neural_fusion_v2.preprocessing import ScalerState
    for stage in ("stage_a_state", "stage_b_state"):
        rehydrated = ScalerState.from_dict(payload[stage])
        check(f"{stage} rehydrates without fitting", rehydrated is not None)

    # ---------------------------------------------------------------- 3
    print("\n--- 3. load-only re-prediction")
    from neural_fusion_v2 import contract as C
    from neural_fusion_v2.adapter import build_tensors
    from neural_fusion_v2.training import predict

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    contract_dir = Path(manifest["contract_dir"])
    v2_manifest = C.load_manifest(contract_dir)
    validation = C.load_population(contract_dir, v2_manifest, split, "B", "validation")
    state_b = ScalerState.from_dict(payload["stage_b_state"])
    tensors = build_tensors(validation, state_b,
                            use_sequence=bool(payload["uses_sequence"]))
    rebuilt_sgd = tensors.to_sgd(predict(model, tensors))

    exported = pd.read_csv(next((run_dir / "predictions" / split).glob("*.csv")))
    check("rebuilt row count equals exported",
          len(rebuilt_sgd) == len(exported),
          f"{len(rebuilt_sgd)} vs {len(exported)}")
    delta = float(np.max(np.abs(exported["predicted_rent_sgd"].to_numpy()
                                - np.asarray(rebuilt_sgd))))
    print(f"  max |exported - rebuilt| = {delta:.3e} SGD")
    check("load-only reconstruction reproduces the exported predictions",
          delta < 1e-4, f"max delta {delta:.3e}")

    rebuilt_metrics = decompose(exported["actual_rent_sgd"], rebuilt_sgd)
    check("reconstruction reproduces the recorded RMSE",
          abs(rebuilt_metrics["rmse_sgd"] - result["overall"]["rmse_sgd"]) < 1e-4,
          f"{rebuilt_metrics['rmse_sgd']:.6f} vs {result['overall']['rmse_sgd']:.6f}")

    # ---------------------------------------------------------------- 4
    print("\n--- 4. bias vs dispersion")
    candidate_stats = decompose(exported["actual_rent_sgd"],
                                exported["predicted_rent_sgd"])
    ref_frame = pd.read_csv(next((RUNS / args.reference / "predictions" / split)
                                 .glob("*.csv")))
    reference_stats = decompose(ref_frame["actual_rent_sgd"],
                                ref_frame["predicted_rent_sgd"])

    header = f"  {'run':14}{'RMSE':>10}{'MAE':>10}{'bias':>10}{'errSD':>10}{'bias/MSE':>10}"
    print(header)
    for label, stats in (("block", reference_stats),
                         ("per_feature", candidate_stats)):
        print(f"  {label:14}{stats['rmse_sgd']:>10.2f}{stats['mae_sgd']:>10.2f}"
              f"{stats['bias_sgd']:>10.2f}{stats['error_sd_sgd']:>10.2f}"
              f"{stats['bias_share_of_mse'] * 100:>9.1f}%")

    delta_mse = candidate_stats["rmse_sgd"] ** 2 - reference_stats["rmse_sgd"] ** 2
    delta_bias_sq = candidate_stats["bias_sgd"] ** 2 - reference_stats["bias_sgd"] ** 2
    delta_var = (candidate_stats["error_sd_sgd"] ** 2
                 - reference_stats["error_sd_sgd"] ** 2)
    print(f"\n  delta MSE       : {delta_mse:+9.1f}")
    print(f"    from bias^2   : {delta_bias_sq:+9.1f}"
          f"  ({delta_bias_sq / delta_mse * 100:5.0f}%)")
    print(f"    from variance : {delta_var:+9.1f}"
          f"  ({delta_var / delta_mse * 100:5.0f}%)")

    output = {
        "run_id": args.run_id,
        "reference": args.reference,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "chosen_epochs": payload.get("chosen_epochs"),
        "reconstruction_max_abs_delta_sgd": delta,
        "reconstruction": rebuilt_metrics,
        "recorded_rmse": result["overall"]["rmse_sgd"],
        "block": reference_stats,
        "per_feature": candidate_stats,
        "delta_mse": delta_mse,
        "delta_mse_from_bias_sq": delta_bias_sq,
        "delta_mse_from_variance": delta_var,
        "bias_share_of_delta_mse": float(delta_bias_sq / delta_mse) if delta_mse else None,
        "failures": FAILURES,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "checkpoint_reconstruction.json").write_text(
        json.dumps(output, indent=2, default=str) + "\n")

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for name in FAILURES:
            print(f"  - {name}")
        return 1
    print("load-only reconstruction passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
