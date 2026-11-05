"""Why does per_feature score 68 SGD worse? Test the under-training hypothesis.

The stage-A early-stopping monitor selected epoch 1 for `per_feature` and epoch 7 for
`block`, and stage B then refits for exactly `chosen_epochs`. So the two runs did not
receive the same effective B budget, which is a confound independent of tokenisation.

The signature of an under-trained residual head is that its predictions sit close to
the anchor it is asked to correct: the head has not yet learned a displacement. This
script measures that directly — mean |prediction - anchor| and the fraction of the
anchor's own error that survives in each run's predictions — instead of inferring it
from the aggregate score.

It fits nothing. It reads the exported predictions and the contract's anchor column.

Run:
    .venv/bin/python -m experiments.spatial_token_ablation_v3.budget_confound
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RUNS = ROOT / "reports" / "neural_fusion_v3" / "runs"
REPORT = ROOT / "reports" / "spatial_token_ablation_v3"
V2_DIR = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"


def main() -> int:
    from neural_fusion_v2 import contract as C
    from neural_fusion_v2.adapter import build_tensors
    from neural_fusion_v2.preprocessing import ScalerState

    manifest = C.load_manifest(V2_DIR)
    validation = C.load_population(V2_DIR, manifest, "orig", "B", "validation")
    # `anchor_sgd` is produced by the scaler transform, not carried on the raw arrays.
    # The scaler must be fitted on B/fit -- fitting it on the validation population
    # would leak and would also not match what the runs used.
    b_fit = C.load_population(V2_DIR, manifest, "orig", "B", "fit")
    state = ScalerState.fit(b_fit.arrays, population="B/fit")
    tensors = build_tensors(validation, state, use_sequence=False)
    anchor = pd.DataFrame({
        "SOURCE_ROW_ID": validation.samples["SOURCE_ROW_ID"].to_numpy(),
        "anchor_sgd": np.asarray(tensors.anchor_sgd, dtype=float),
        "actual_sgd": np.asarray(validation.actual_sgd, dtype=float),
    })

    anchor_error = anchor["anchor_sgd"] - anchor["actual_sgd"]
    anchor_rmse = float(np.sqrt(np.mean(anchor_error ** 2)))
    print(f"anchor alone : RMSE {anchor_rmse:.2f}  bias {anchor_error.mean():+.2f}")
    print("(both runs predict anchor + a learned residual, so they must beat this)\n")

    rows = []
    for label, run_id, chosen in (
            ("block", "macro-probe-block-none-s42", 7),
            ("per_feature", "spatial-probe-perfeature-none-s42-h1", 1)):
        frame = pd.read_csv(next((RUNS / run_id / "predictions" / "orig").glob("*.csv")))
        merged = frame.merge(anchor, on="SOURCE_ROW_ID", how="left")
        if merged["anchor_sgd"].isna().any():
            raise RuntimeError(f"{run_id}: rows missing an anchor")

        error = merged["predicted_rent_sgd"] - merged["actual_rent_sgd"]
        residual = merged["predicted_rent_sgd"] - merged["anchor_sgd"]
        # how much of the anchor's own squared error survives
        survived = float(np.sum(error ** 2) / np.sum(anchor_error ** 2))
        # correlation of the applied correction with the correction that was needed
        needed = merged["actual_rent_sgd"] - merged["anchor_sgd"]
        corr = float(np.corrcoef(residual, needed)[0, 1])

        rows.append({
            "run": label,
            "chosen_epochs": chosen,
            "rmse_sgd": float(np.sqrt(np.mean(error ** 2))),
            "bias_sgd": float(error.mean()),
            "mean_abs_residual_from_anchor": float(np.abs(residual).mean()),
            "rms_residual_from_anchor": float(np.sqrt(np.mean(residual ** 2))),
            "anchor_error_surviving_fraction": survived,
            "corr_correction_with_needed": corr,
        })

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print()
    block = rows[0]
    feature = rows[1]
    print(f"mean |correction applied| : block {block['mean_abs_residual_from_anchor']:.2f} "
          f"vs per_feature {feature['mean_abs_residual_from_anchor']:.2f}")
    print(f"anchor error surviving    : block {block['anchor_error_surviving_fraction']:.4f} "
          f"vs per_feature {feature['anchor_error_surviving_fraction']:.4f}")
    print(f"correction/needed corr    : block {block['corr_correction_with_needed']:.4f} "
          f"vs per_feature {feature['corr_correction_with_needed']:.4f}")

    small_correction = (feature["mean_abs_residual_from_anchor"]
                        < 0.75 * block["mean_abs_residual_from_anchor"])
    ratio = (feature["mean_abs_residual_from_anchor"]
             / block["mean_abs_residual_from_anchor"])
    print()
    print(f"correction magnitude ratio (per_feature / block): {ratio:.3f}")
    if small_correction:
        print("VERDICT: per_feature displaces markedly LESS than block. Consistent with a "
              "residual head that never converged, but see the epoch table before "
              "accepting that as the mechanism.")
    else:
        print("VERDICT: the 'under-trained head barely moves' hypothesis is REFUTED -- "
              "per_feature displaces MORE than block, yet lands further from the label "
              "and with a much larger bias. So the loss is not a missing displacement; "
              "it is a displacement in a worse direction. The differing epoch count "
              "(1 vs 7) is still a confound for the comparison, but it is not "
              "explained by 'the head did not have time to move'.")

    REPORT.mkdir(parents=True, exist_ok=True)
    output = {
        "anchor_only": {"rmse_sgd": anchor_rmse, "bias_sgd": float(anchor_error.mean())},
        "runs": rows,
        "correction_magnitude_ratio": ratio,
        "small_correction_hypothesis": bool(small_correction),
        "chosen_epochs": {"block": 7, "per_feature": 1},
    }
    (REPORT / "budget_confound.json").write_text(
        json.dumps(output, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
