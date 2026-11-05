"""Rank 23: fixed forecast-horizon decay of the saved temporal-model delta.

This is a prediction-level proxy for hard temporal-branch decay. T0 and T1
were independently trained, so their difference is not the exact T1 branch.
"""
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
OUT = REPORTS / "transformer_ranked_experiments/23_forced_decay"


def frame(fold, seed=42):
    tag = f"calibration/{{name}}/{fold}" if fold != "outer" else f"outer/{{name}}/seed{seed}"
    base = pd.read_csv(SOURCE / tag.format(name="T0") / "predictions.csv")
    temporal = pd.read_csv(SOURCE / tag.format(name="T1") / "predictions.csv")
    if not np.array_equal(base.row_id, temporal.row_id) or not np.allclose(base.actual, temporal.actual):
        raise ValueError("T0/T1 row mismatch")
    base["temporal"] = temporal.predicted
    return base


def predict(data, alpha, tau):
    amplitude = alpha*np.exp(-(data.horizon.to_numpy()-1)/tau)
    return data.predicted.to_numpy()+amplitude*(data.temporal.to_numpy()-data.predicted.to_numpy())


def score(data, prediction):
    return float(np.sqrt(np.average((prediction-data.actual.to_numpy())**2,
                                    weights=data.weight.to_numpy())))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [SOURCE / f"{tag}/predictions.csv" for name in ("T0", "T1")
             for tag in (f"calibration/{name}/early_2022",
                         f"calibration/{name}/early_2023",
                         *(f"outer/{name}/seed{s}" for s in (42,43,44)))]
    paths += [SOURCE / f"long16/{name}/seed42/predictions.csv" for name in ("T0", "T1")]
    before = {str(p): sha256(p) for p in paths}
    first, second = frame("early_2022"), frame("early_2023")
    candidates = [{"alpha": 0., "tau": 0., "early_2022_rmse_sgd": score(first, first.predicted),
                   "early_2023_rmse_sgd": score(second, second.predicted)}]
    for alpha in (.25, .5, 1.):
        for tau in (3., 6., 12.):
            candidates.append({"alpha": alpha, "tau": tau,
                "early_2022_rmse_sgd": score(first, predict(first, alpha, tau)),
                "early_2023_rmse_sgd": score(second, predict(second, alpha, tau))})
    selection = pd.DataFrame(candidates).sort_values(["early_2023_rmse_sgd", "alpha", "tau"])
    selection.to_csv(OUT / "earlier_selection.csv", index=False)
    alpha, tau = float(selection.iloc[0].alpha), float(selection.iloc[0].tau)
    scores = []
    for seed in (42,43,44):
        outer = frame("outer", seed)
        corrected = outer.predicted.to_numpy() if alpha == 0 else predict(outer, alpha, tau)
        outer["decayed_prediction"] = corrected
        outer.to_csv(OUT / f"outer_seed{seed}.csv", index=False, float_format="%.10f")
        scores.append({"seed": seed, "alpha": alpha, "tau": tau,
            "base_rmse_sgd": score(outer, outer.predicted),
            "decayed_rmse_sgd": score(outer, corrected)})
    pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)
    diagnostics = []
    rng = np.random.default_rng(20261103)
    for seed in (42,43,44):
        data = frame("outer", seed)
        decayed = data.predicted.to_numpy() if alpha == 0 else predict(data, alpha, tau)
        diagnostics.extend({"seed": seed, "month": month,
            "base_rmse_sgd": score(part, part.predicted.to_numpy()),
            "decayed_rmse_sgd": score(part, decayed[part.index.to_numpy()]),
            "rows": len(part)} for month, part in data.groupby("month", sort=True))
        months = sorted(data.month.unique())
        n = len(months)
        sq_base, sq_decay, counts = [], [], []
        for month in months:
            mask = data.month.eq(month).to_numpy()
            y = data.actual.to_numpy()[mask]
            sq_base.append(float(np.sum((data.predicted.to_numpy()[mask]-y)**2)))
            sq_decay.append(float(np.sum((decayed[mask]-y)**2)))
            counts.append(int(mask.sum()))
        starts = rng.integers(0, n, size=(10000, (n+2)//3))
        sampled = np.stack([(starts+j)%n for j in range(3)], axis=-1).reshape(10000,-1)[:,:n]
        b = np.sqrt(np.sum(np.asarray(sq_base)[sampled], axis=1)/np.sum(np.asarray(counts)[sampled], axis=1))
        d = np.sqrt(np.sum(np.asarray(sq_decay)[sampled], axis=1)/np.sum(np.asarray(counts)[sampled], axis=1))
        np.save(OUT / f"bootstrap_delta_seed{seed}.npy", d-b)
    pd.DataFrame(diagnostics).to_csv(OUT / "monthly_scores.csv", index=False)
    long_base = pd.read_csv(SOURCE / "long16/T0/seed42/predictions.csv")
    long_temporal = pd.read_csv(SOURCE / "long16/T1/seed42/predictions.csv")
    if not np.array_equal(long_base.row_id, long_temporal.row_id) or not np.allclose(long_base.actual, long_temporal.actual):
        raise ValueError("Long16 T0/T1 mismatch")
    long_base["temporal"] = long_temporal.predicted
    long_base["decayed_prediction"] = long_base.predicted if alpha == 0 else predict(long_base, alpha, tau)
    long_base.to_csv(OUT / "long16_seed42.csv", index=False, float_format="%.10f")
    long_scores = [{"group": label, "rows": len(part),
        "base_rmse_sgd": score(part, part.predicted.to_numpy()),
        "decayed_rmse_sgd": score(part, part.decayed_prediction.to_numpy())}
        for label, part in (("all", long_base), ("1-10", long_base[long_base.horizon.le(10)]),
                            ("11-16", long_base[long_base.horizon.ge(11)]))]
    pd.DataFrame(long_scores).to_csv(OUT / "long16_scores.csv", index=False)
    if {str(p): sha256(Path(p)) for p in before} != before:
        raise ValueError("Source changed")
    (OUT / "verification.json").write_text(json.dumps({
        "source_hashes": before, "chosen_alpha": alpha, "chosen_tau": tau,
        "outer_labels_used_for_selection": False,
        "scope": "prediction-level proxy, not exact internal temporal branch",
    }, indent=2), encoding="utf-8")
    print(selection.to_string(index=False), flush=True)
    print(pd.DataFrame(scores).to_string(index=False), flush=True)
    print(pd.DataFrame(long_scores).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
