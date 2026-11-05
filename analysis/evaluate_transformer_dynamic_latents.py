"""Rank 11: causal static and AR(1) submarket latent residuals on frozen T0."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from analysis.config import ROOT, REPORTS
from analysis.data import sha256
from analysis.experiments.hierarchical_latent_residual import (
    DynamicHierarchicalResidual, HierarchicalResidual,
)

SOURCE = REPORTS / "spatial_temporal_transformer"
PREPARED = ROOT / ".cache/spatial_temporal_transformer/prepared.joblib"
OUT = REPORTS / "transformer_ranked_experiments/11_dynamic_latents"


def frame(prepared, fold, seed=42):
    tag = f"calibration/T0/{fold}" if fold != "outer" else f"outer/T0/seed{seed}"
    data = pd.read_csv(SOURCE / tag / "predictions.csv")
    x = prepared["folds"][f"{fold}_d0"]["X_query"]
    if not np.array_equal(data.row_id, x.index.to_numpy()):
        raise ValueError("Query IDs changed")
    data["building"] = x.BLOCK_ID.astype(str).to_numpy()
    data["submarket"] = (x.TOWN_CLEAN.astype(str)+"|"+x.FLAT_TYPE_CLEAN.astype(str)).to_numpy()
    data["residual"] = data.actual-data.predicted
    return data


def score(data, prediction):
    return float(np.sqrt(np.average((data.actual.to_numpy()-prediction)**2,
                                    weights=data.weight.to_numpy())))


def model_for(name):
    if name == "static":
        return HierarchicalResidual(block_penalty=100., group_penalty=100.)
    if name == "dynamic_05":
        return DynamicHierarchicalResidual(block_penalty=100., group_penalty=100.,
                                           rho=.5, state_penalty=100.)
    if name == "dynamic_09":
        return DynamicHierarchicalResidual(block_penalty=100., group_penalty=100.,
                                           rho=.9, state_penalty=100.)
    raise ValueError(name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / "prepared.json").read_text())
    if sha256(PREPARED) != manifest["sha256"]:
        raise ValueError("Prepared cache changed")
    sources = [PREPARED]
    sources += [SOURCE / f"{name}/predictions.csv" for name in (
        "calibration/T0/early_2022", "calibration/T0/early_2023",
        "outer/T0/seed42", "outer/T0/seed43", "outer/T0/seed44")]
    before = {str(p): sha256(p) for p in sources}
    prepared = joblib.load(PREPARED)
    first = frame(prepared, "early_2022")
    second = frame(prepared, "early_2023")
    candidates = [{"model": "none", "alpha": 0., "earlier_rmse_sgd": score(second, second.predicted)}]
    for name in ("static", "dynamic_05", "dynamic_09"):
        model = model_for(name).fit(first, "2023-07")
        correction = model.predict_components(second).total_correction.to_numpy()
        for alpha in (.25, .5, 1.):
            candidates.append({"model": name, "alpha": alpha,
                "earlier_rmse_sgd": score(second, second.predicted+alpha*correction),
                "fit_rows": len(first), "solver_iterations": model.solver_iterations_})
    grid = pd.DataFrame(candidates).sort_values(["earlier_rmse_sgd", "model", "alpha"])
    grid.to_csv(OUT / "earlier_selection.csv", index=False)
    winner, alpha = str(grid.iloc[0]["model"]), float(grid.iloc[0]["alpha"])
    historical = pd.concat([first, second], ignore_index=True)
    fitted = None if winner == "none" else model_for(winner).fit(historical, "2024-06")
    rows = []
    for seed in (42,43,44):
        outer = frame(prepared, "outer", seed)
        correction = np.zeros(len(outer)) if fitted is None else fitted.predict_components(outer).total_correction.to_numpy()
        outer["corrected"] = outer.predicted+alpha*correction
        outer["correction_sgd"] = alpha*correction
        outer.to_csv(OUT / f"outer_seed{seed}.csv", index=False, float_format="%.10f")
        rows.append({"seed": seed, "chosen": winner, "alpha": alpha,
            "base_rmse_sgd": score(outer, outer.predicted),
            "corrected_rmse_sgd": score(outer, outer.corrected),
            "mean_correction_sgd": float(outer.correction_sgd.mean())})
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)
    if {str(p): sha256(Path(p)) for p in before} != before:
        raise ValueError("Source artifact changed")
    (OUT / "verification.json").write_text(json.dumps({
        "source_hashes": before, "chosen_model": winner, "chosen_alpha": alpha,
        "historical_fit_cutoff": "2024-05", "outer_labels_used_for_selection": False,
        "scope": "posthoc latent residual on frozen T0 forecasts",
    }, indent=2), encoding="utf-8")
    print(grid.to_string(index=False), flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
