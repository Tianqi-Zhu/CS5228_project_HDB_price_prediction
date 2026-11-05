"""Rank 15: horizon-dependent T0 and temporal-transformer ensemble weights."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.config import REPORTS
from analysis.data import sha256

SOURCE = REPORTS / "spatial_temporal_transformer"
OUT = REPORTS / "transformer_ranked_experiments/15_context_ensemble"
VARIANTS = ("T1", "T2", "T3")


def group(horizon):
    h = np.asarray(horizon)
    return np.select([h <= 3, h <= 6, h <= 10], [0,1,2], default=3)


def frame(fold, seed=42):
    pieces = []
    for name in ("T0", *VARIANTS):
        tag = f"calibration/{name}/{fold}" if fold != "outer" else f"outer/{name}/seed{seed}"
        part = pd.read_csv(SOURCE / tag / "predictions.csv")
        if pieces and (not np.array_equal(part.row_id, pieces[0].row_id) or
                       not np.allclose(part.actual, pieces[0].actual)):
            raise ValueError(f"Prediction mismatch: {name}/{fold}")
        pieces.append(part)
    output = pieces[0].copy()
    for name, part in zip(VARIANTS, pieces[1:]):
        output[name] = part.predicted
    output["group"] = group(output.horizon)
    return output


def rmse(frame, pred):
    return float(np.sqrt(np.average((frame.actual.to_numpy()-pred)**2,
                                    weights=frame.weight.to_numpy())))


def fit_weights(frame, variant, method):
    baseline = frame.predicted.to_numpy()
    difference = frame[variant].to_numpy()-baseline
    target = frame.actual.to_numpy()-baseline
    weight = frame.weight.to_numpy()
    def coefficient(mask):
        d, t, w = difference[mask], target[mask], weight[mask]
        return float(np.clip(np.dot(w*d, t)/(np.dot(w*d,d)+10000.), 0., 1.))
    global_weight = coefficient(np.ones(len(frame), bool))
    result = np.full(4, global_weight)
    if method == "horizon":
        groups = frame.group.to_numpy()
        for k in range(4):
            mask = groups == k
            if mask.sum() >= 50:
                local = coefficient(mask)
                share = mask.sum()/(mask.sum()+250.)
                result[k] = share*local+(1-share)*global_weight
    return result


def blended(frame, variant, weights):
    baseline = frame.predicted.to_numpy()
    return baseline+weights[frame.group.to_numpy()]*(frame[variant].to_numpy()-baseline)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [SOURCE / f"{tag}/predictions.csv" for name in ("T0", *VARIANTS)
             for tag in (f"calibration/{name}/early_2022",
                         f"calibration/{name}/early_2023",
                         *(f"outer/{name}/seed{s}" for s in (42,43,44)))]
    before = {str(p): sha256(p) for p in paths}
    first, second = frame("early_2022"), frame("early_2023")
    rows = [{"variant": "T0", "method": "none", "earlier_rmse_sgd":
             rmse(second, second.predicted)}]
    for variant in VARIANTS:
        for method in ("global", "horizon"):
            weights = fit_weights(first, variant, method)
            rows.append({"variant": variant, "method": method,
                         "earlier_rmse_sgd": rmse(second, blended(second, variant, weights)),
                         "early_weights": json.dumps(weights.tolist())})
    selection = pd.DataFrame(rows).sort_values(["earlier_rmse_sgd", "variant"])
    selection.to_csv(OUT / "earlier_selection.csv", index=False)
    variant, method = str(selection.iloc[0].variant), str(selection.iloc[0].method)
    historical = pd.concat([first, second], ignore_index=True)
    weights = np.zeros(4) if variant == "T0" else fit_weights(historical, variant, method)
    outer_rows = []
    for seed in (42,43,44):
        data = frame("outer", seed)
        pred = data.predicted.to_numpy() if variant == "T0" else blended(data, variant, weights)
        data["ensemble_prediction"] = pred
        data.to_csv(OUT / f"outer_seed{seed}.csv", index=False, float_format="%.10f")
        outer_rows.append({"seed": seed, "variant": variant, "method": method,
                           "base_rmse_sgd": rmse(data, data.predicted),
                           "ensemble_rmse_sgd": rmse(data, pred)})
    pd.DataFrame(outer_rows).to_csv(OUT / "scores.csv", index=False)
    if {str(p): sha256(Path(p)) for p in before} != before:
        raise ValueError("Original forecast changed")
    (OUT / "verification.json").write_text(json.dumps({
        "source_hashes": before, "chosen_variant": variant, "chosen_method": method,
        "final_horizon_weights": weights.tolist(),
        "outer_labels_used_for_selection": False}, indent=2), encoding="utf-8")
    print(selection.to_string(index=False), flush=True)
    print(pd.DataFrame(outer_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
