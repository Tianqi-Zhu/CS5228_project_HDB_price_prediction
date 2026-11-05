"""Earlier-forecast residual calibration for the frozen T0 rental transformer."""
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
from analysis.evaluation import metrics


SOURCE = REPORTS / "spatial_temporal_transformer"
SOURCE_CACHE = ROOT / ".cache/spatial_temporal_transformer"
OUT = REPORTS / "transformer_ranked_experiments" / "02_hierarchical_calibration"
COLS = ("TOWN_CLEAN", "PLANNING_AREA", "FLAT_TYPE_CLEAN", "BLOCK_ID")


def source_frame(prepared, fold, seed=42):
    tag = f"calibration/T0/{fold}" if fold != "outer" else f"outer/T0/seed{seed}"
    predictions = pd.read_csv(SOURCE / tag / "predictions.csv")
    features = prepared["folds"][f"{fold}_d0"]["X_query"]
    if not np.array_equal(predictions.row_id, features.index.to_numpy()):
        raise ValueError(f"Prediction/feature row mismatch: {tag}")
    for col in COLS:
        predictions[col] = features[col].astype(str).to_numpy()
    return predictions


def normalize_weights(frame):
    output = frame.copy()
    # Each sampled query month represents one unit of time; the scale of its
    # population weights must not redefine shrinkage pseudo-counts.
    average = output.groupby("month").weight.transform("mean")
    output["cal_weight"] = output.weight / average
    return output


def fit(table, *, hierarchy, shrink, half_life, latest_month):
    if hierarchy is None:
        return {"global": 0., "maps": [], "shrink": 0.,
                "half_life": 0., "hierarchy": None}
    data = normalize_weights(table)
    if half_life:
        age = pd.Period(latest_month, "M").ordinal - pd.PeriodIndex(data.month, freq="M").asi8
        data["cal_weight"] *= np.exp2(-age / half_life)
    data["residual"] = data.actual - data.predicted
    maps = []
    global_n = data.cal_weight.sum()
    global_mean = float(np.dot(data.cal_weight, data.residual) / (global_n + 500.))
    prior = np.full(len(data), global_mean)
    for keys in hierarchy:
        group = data.groupby(list(keys), observed=True, dropna=False)
        numerator = group.apply(lambda x: float(np.dot(x.cal_weight, x.residual)),
                                include_groups=False)
        denominator = group.cal_weight.sum()
        old_prior = group.apply(lambda x: float(np.average(prior[x.index], weights=x.cal_weight)),
                                include_groups=False)
        value = (numerator + shrink * old_prior) / (denominator + shrink)
        lookup = {(key if isinstance(key, tuple) else (key,)): float(number)
                  for key, number in value.items()}
        maps.append((keys, lookup))
        labels = list(map(tuple, data[list(keys)].itertuples(index=False, name=None)))
        prior = np.asarray([lookup[t] for t in labels])
    return {"global": global_mean, "maps": maps, "shrink": shrink,
            "half_life": half_life, "hierarchy": hierarchy}


def correction(model, table):
    out = np.full(len(table), model["global"])
    for keys, mapping in model["maps"]:
        labels = table[list(keys)].itertuples(index=False, name=None)
        out = np.asarray([mapping.get(tuple(label), prior) for label, prior in zip(labels, out)])
    return out


def candidate_grid():
    yield {"name": "none", "hierarchy": None, "shrink": 0., "half_life": 0.}
    tiers = {
        "global": (),
        "town": (("TOWN_CLEAN",),),
        "area_type": (("TOWN_CLEAN",), ("PLANNING_AREA", "FLAT_TYPE_CLEAN")),
        "building": (("TOWN_CLEAN",), ("PLANNING_AREA", "FLAT_TYPE_CLEAN"),
                     ("BLOCK_ID", "FLAT_TYPE_CLEAN")),
    }
    for name, hierarchy in tiers.items():
        for shrink in (30., 100., 300.):
            for half_life in (0., 6., 12.):
                yield {"name": name, "hierarchy": hierarchy, "shrink": shrink,
                       "half_life": half_life}


def score(actual, predicted, weights):
    return float(np.sqrt(np.average((actual-predicted)**2, weights=weights)))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SOURCE / "prepared.json").read_text())
    if sha256(SOURCE_CACHE / "prepared.joblib") != manifest["sha256"]:
        raise ValueError("Original prepared fold changed")
    before = {str(SOURCE / tag / "predictions.csv"): sha256(SOURCE / tag / "predictions.csv")
              for tag in ("calibration/T0/early_2022", "calibration/T0/early_2023",
                          "outer/T0/seed42", "outer/T0/seed43", "outer/T0/seed44")}
    prepared = joblib.load(SOURCE_CACHE / "prepared.joblib")
    first = source_frame(prepared, "early_2022")
    second = source_frame(prepared, "early_2023")
    selection = []
    for candidate in candidate_grid():
        model = fit(first, hierarchy=candidate["hierarchy"], shrink=candidate["shrink"],
                    half_life=candidate["half_life"], latest_month="2023-06")
        pred = second.predicted.to_numpy() + correction(model, second)
        selection.append({**{k: v for k, v in candidate.items() if k != "hierarchy"},
                          "weighted_rmse_sgd": score(second.actual, pred, second.weight),
                          "delta_rmse_sgd": score(second.actual, pred, second.weight) -
                                            score(second.actual, second.predicted, second.weight)})
    grid = pd.DataFrame(selection).sort_values(["weighted_rmse_sgd", "name", "shrink", "half_life"])
    grid.to_csv(OUT / "earlier_selection.csv", index=False)
    winner = grid.iloc[0]
    candidate = next(c for c in candidate_grid() if c["name"] == winner["name"] and
                     c["shrink"] == winner["shrink"] and c["half_life"] == winner["half_life"])
    # Prior residuals are only from query months ending April 2024.
    calibration = pd.concat([first, second], ignore_index=True)
    model = fit(calibration, hierarchy=candidate["hierarchy"], shrink=candidate["shrink"],
                half_life=candidate["half_life"], latest_month="2024-04")
    write_json(OUT / "selection.json", {"chosen": {k: v for k, v in candidate.items() if k != "hierarchy"},
                                        "selection_query_months": ["2023-07", "2024-04"],
                                        "calibration_label_max": "2024-04",
                                        "outer_labels_used_for_selection": False,
                                        "source_predictions_sha256": before})
    scores, town_scores = [], []
    for seed in (42, 43, 44):
        outer = source_frame(prepared, "outer", seed)
        calibrated = outer.predicted.to_numpy() + correction(model, outer)
        result = outer.copy()
        result["calibrated"] = calibrated
        result.to_csv(OUT / f"outer_seed{seed}.csv", index=False, float_format="%.10f")
        scores.append({"seed": seed, "base_rmse_sgd": score(outer.actual, outer.predicted, outer.weight),
                       "calibrated_rmse_sgd": score(outer.actual, calibrated, outer.weight),
                       "delta_rmse_sgd": score(outer.actual, calibrated, outer.weight) -
                                         score(outer.actual, outer.predicted, outer.weight),
                       "base_bias_sgd": float((outer.predicted-outer.actual).mean()),
                       "calibrated_bias_sgd": float((calibrated-outer.actual).mean()),
                       "mean_correction_sgd": float((calibrated-outer.predicted).mean())})
        for town, group in result.groupby("TOWN_CLEAN"):
            town_scores.append({"seed": seed, "town": town, "rows": len(group),
                                "base_bias_sgd": float((group.predicted-group.actual).mean()),
                                "calibrated_bias_sgd": float((group.calibrated-group.actual).mean())})
    pd.DataFrame(scores).to_csv(OUT / "scores.csv", index=False)
    pd.DataFrame(town_scores).to_csv(OUT / "town_scores.csv", index=False)
    if any(sha256(Path(p)) != h for p, h in before.items()):
        raise ValueError("Original prediction artifact changed")
    write_json(OUT / "verification.json", {"original_predictions_unchanged": True,
                                           "outer_labels_used_for_selection": False,
                                           "seed42_only_earlier_calibration": True,
                                           "outer_seeds": [42,43,44],
                                           "candidate_count": len(selection)})
    print(pd.DataFrame(scores).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
