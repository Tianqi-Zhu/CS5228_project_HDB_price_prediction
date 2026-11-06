"""Causal block-comparable control on saved T0 forecasts; no model refits."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analysis.config import AUX, DATA, ROOT, REPORTS
from analysis.data import sha256
from analysis.experiments.local_comparable_graph import (
    _features_at_anchor, _history, _tag_rows, block_graph,
)
from analysis.experiments.spatiotemporal_hybrid import spatial_frame


SOURCE = REPORTS / "spatial_temporal_transformer"
OUT = REPORTS / "transformer_ranked_experiments" / "03_comparable_control"
ORIGINS = {"early_2022": "2022-09", "early_2023": "2023-07", "outer": "2024-06"}
FEATURES = {
    "raw": [f"LOCAL_RENT_MEAN_{w}M" for w in (3, 6, 12)],
    "premium": [f"LOCAL_PREMIUM_MEAN_{w}M" for w in (3, 6, 12)],
    "both": [f"LOCAL_{kind}_MEAN_{w}M" for kind in ("RENT", "PREMIUM") for w in (3, 6, 12)],
}
COUNTS = [f"LOCAL_RENT_COUNT_{w}M" for w in (3, 6, 12)]


def inputs(frame, columns):
    x = frame[columns + COUNTS].copy()
    for col in COUNTS:
        x[col] = np.log1p(x[col])
    return x.fillna(0).to_numpy(float)


def comparable_frame(raw, index, edges, fold, seed=42):
    tag = f"calibration/T0/{fold}" if fold != "outer" else f"outer/T0/seed{seed}"
    source = pd.read_csv(SOURCE / tag / "predictions.csv")
    start = pd.Period(ORIGINS[fold], "M")
    past = raw[raw._MONTH < start]
    query = raw.loc[source.row_id.to_numpy()]
    if len(query) != len(source) or not np.array_equal(
            query._MONTH.astype(str).to_numpy(), source.month.to_numpy()):
        raise ValueError("Comparable query alignment changed")
    if past._MONTH.max() >= start or query._MONTH.min() < start:
        raise ValueError("Comparable information cutoff violated")
    historical = _history(_tag_rows(past, index))
    tagged = _tag_rows(query, index)
    values = _features_at_anchor(tagged, historical, edges, start.ordinal)
    output = source.copy()
    for col in FEATURES["both"] + COUNTS:
        output[col] = values[col].to_numpy()
    output["comparable_origin"] = str(start - 1)
    return output


def fit(table, mode, alpha):
    x = inputs(table, FEATURES[mode])
    scaler = StandardScaler().fit(x)
    ridge = Ridge(alpha=alpha, fit_intercept=False).fit(
        scaler.transform(x), table.actual.to_numpy() - table.predicted.to_numpy())
    return scaler, ridge


def correction(model, table, mode):
    scaler, ridge = model
    return ridge.predict(scaler.transform(inputs(table, FEATURES[mode])))


def rmse(frame, correction_values=None):
    pred = frame.predicted.to_numpy()
    if correction_values is not None:
        pred = pred + correction_values
    return float(np.sqrt(np.average((pred-frame.actual.to_numpy())**2, weights=frame.weight)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = [DATA / "train.csv", AUX / "sg-hdb-block.csv",
                ROOT / "analysis/experiments/local_comparable_graph.py",
                SOURCE / "calibration/T0/early_2022/predictions.csv",
                SOURCE / "calibration/T0/early_2023/predictions.csv",
                *(SOURCE / f"outer/T0/seed{s}/predictions.csv" for s in (42,43,44))]
    before = {str(p): sha256(p) for p in original}
    raw = spatial_frame(pd.read_csv(DATA / "train.csv"))
    index, edges = block_graph(pd.read_csv(AUX / "sg-hdb-block.csv"))
    first = comparable_frame(raw, index, edges, "early_2022")
    second = comparable_frame(raw, index, edges, "early_2023")
    selection = [{"mode": "none", "alpha": 0., "earlier_rmse_sgd": rmse(second),
                  "earlier_delta_sgd": 0.}]
    for mode in FEATURES:
        for alpha in (10., 100., 1000.):
            model = fit(first, mode, alpha)
            value = rmse(second, correction(model, second, mode))
            selection.append({"mode": mode, "alpha": alpha, "earlier_rmse_sgd": value,
                              "earlier_delta_sgd": value-rmse(second)})
    grid = pd.DataFrame(selection).sort_values(["earlier_rmse_sgd", "mode", "alpha"])
    grid.to_csv(OUT / "earlier_selection.csv", index=False)
    best = grid.iloc[0]
    mode, alpha = str(best["mode"]), float(best["alpha"])
    fit_rows = pd.concat([first, second], ignore_index=True)
    model = None if mode == "none" else fit(fit_rows, mode, alpha)
    report, coverage = [], []
    for seed in (42,43,44):
        outer = comparable_frame(raw, index, edges, "outer", seed)
        corr = np.zeros(len(outer)) if model is None else correction(model, outer, mode)
        outer["comparable_correction_sgd"] = corr
        outer["comparable_prediction"] = outer.predicted + corr
        outer.to_csv(OUT / f"outer_seed{seed}.csv", index=False, float_format="%.10f")
        report.append({"seed": seed, "base_rmse_sgd": rmse(outer),
                       "comparable_rmse_sgd": rmse(outer, corr),
                       "delta_rmse_sgd": rmse(outer, corr)-rmse(outer),
                       "mean_correction_sgd": float(corr.mean())})
        coverage.append({"seed": seed, **{f"available_{w}m": float(outer[f"LOCAL_RENT_COUNT_{w}M"].gt(0).mean())
                                           for w in (3,6,12)}})
    pd.DataFrame(report).to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "coverage.csv", index=False)
    if {str(p): sha256(p) for p in original} != before:
        raise ValueError("Original source changed")
    (OUT / "verification.json").write_text(json.dumps({
        "original_sources_sha256": before, "earlier_selected_mode": mode,
        "earlier_selected_alpha": alpha, "calibration_label_max": "2024-04",
        "outer_labels_used_for_selection": False, "causal_cutoffs_checked": True,
        "previous_fixed_graph_study_was_negative": True}, indent=2), encoding="utf-8")
    print(pd.DataFrame(report).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
