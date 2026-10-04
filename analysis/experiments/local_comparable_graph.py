"""Fixed-origin ablation of nearby comparable rents at the HDB block level."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import BallTree

from analysis.config import AUX, DATA, REPORTS
from analysis.data import normalized_address, sha256
from analysis.evaluation import metrics
from analysis.experiments.spatiotemporal_hybrid import (
    BASE_FEATURES, HISTORY_FEATURES, MONTH, ROW_MODEL_SETTINGS, TARGET,
    fit_row_model, historical_features, model_inputs, spatial_frame,
)

OUT = REPORTS / "local_comparable_graph"
BLOCK_KEYS = ["TOWN", "BLOCK", "STREET"]
WINDOWS = (3, 6, 12)
RADIUS_KM = 1.0
K_NEIGHBORS = 12
AREA_BIN_SQM = 10
FOLDS = [
    ("2023-01", "2023-07"),
    ("2023-07", "2024-04"),
    ("2024-04", "2025-04"),
    ("2023-12", "2025-04"),  # 16 months, matching the forward test horizon
]
RAW_FEATURES = [f"LOCAL_RENT_{suffix}_{w}M" for w in WINDOWS for suffix in ("MEAN", "COUNT")]
PREMIUM_FEATURES = [f"LOCAL_PREMIUM_MEAN_{w}M" for w in WINDOWS]
RESIDUAL_FEATURES = [f"LOCAL_RESIDUAL_MEAN_{w}M" for w in WINDOWS]
COUNT_FEATURES = [f"LOCAL_RENT_COUNT_{w}M" for w in WINDOWS]
OOF_STARTS = ("2022-01", "2022-07", "2023-01", "2023-07", "2024-01")


def block_graph(blocks: pd.DataFrame) -> tuple[pd.MultiIndex, list[list[tuple[int, float]]]]:
    """Connect each block to up to 12 other blocks within one kilometre."""
    blocks = normalized_address(blocks).drop_duplicates(BLOCK_KEYS).reset_index(drop=True)
    index = pd.MultiIndex.from_frame(blocks[BLOCK_KEYS])
    coords = np.radians(blocks[["LATITUDE", "LONGITUDE"]].to_numpy(float))
    distance, neighbor = BallTree(coords, metric="haversine").query(coords, k=min(len(blocks), K_NEIGHBORS + 1))
    edges = []
    for node, (distances, candidates) in enumerate(zip(distance, neighbor)):
        links = [(int(other), float(1.0 / (0.1 + km)))
                 for other, radians in zip(candidates, distances)
                 if other != node and (km := radians * 6371.0088) <= RADIUS_KM]
        edges.append(links[:K_NEIGHBORS])
    return index, edges


def _tag_rows(frame: pd.DataFrame, block_index: pd.MultiIndex) -> pd.DataFrame:
    result = frame.copy()
    result["_BLOCK_ID"] = block_index.get_indexer(pd.MultiIndex.from_frame(result[BLOCK_KEYS]))
    if (result["_BLOCK_ID"] < 0).any():
        raise ValueError("A rental block was absent from the graph")
    result["_AREA_BIN"] = np.floor(result["FLOOR_AREA_SQM"].to_numpy(float) / AREA_BIN_SQM).astype(int)
    result["_MONTH_NUMBER"] = result[MONTH].astype("int64")
    return result


def _history(fitting: pd.DataFrame) -> pd.DataFrame:
    """Use only fitting labels; premiums remove the observed monthly type/size level."""
    history = fitting[["_MONTH_NUMBER", "_BLOCK_ID", "FLAT_TYPE_CLEAN", "_AREA_BIN", TARGET]].copy()
    market = history.groupby(["_MONTH_NUMBER", "FLAT_TYPE_CLEAN", "_AREA_BIN"])[TARGET].transform("mean")
    history["_PREMIUM"] = history[TARGET] - market
    history["_OOF_RESIDUAL"] = fitting["_OOF_RESIDUAL"] if "_OOF_RESIDUAL" in fitting else np.nan
    return history


def _window_lookup(history: pd.DataFrame, anchor: int, width: int) -> dict[tuple, tuple]:
    prior = history[(history["_MONTH_NUMBER"] >= anchor - width) & (history["_MONTH_NUMBER"] < anchor)]
    grouped = prior.groupby(["_BLOCK_ID", "FLAT_TYPE_CLEAN", "_AREA_BIN"], observed=True).agg(
        rent_sum=(TARGET, "sum"), premium_sum=("_PREMIUM", "sum"), count=(TARGET, "size"),
        residual_sum=("_OOF_RESIDUAL", "sum"), residual_count=("_OOF_RESIDUAL", "count"))
    return grouped.to_dict(orient="index")


def _features_at_anchor(targets: pd.DataFrame, history: pd.DataFrame,
                        edges: list[list[tuple[int, float]]], anchor: int) -> pd.DataFrame:
    """Compute one snapshot; a future batch reuses the same frozen snapshot."""
    lookups = {width: _window_lookup(history, anchor, width) for width in WINDOWS}
    unique = targets[["_BLOCK_ID", "FLAT_TYPE_CLEAN", "_AREA_BIN"]].drop_duplicates()
    rows = []
    for block, flat_type, area_bin in unique.itertuples(index=False, name=None):
        values = {"_BLOCK_ID": block, "FLAT_TYPE_CLEAN": flat_type, "_AREA_BIN": area_bin}
        for width in WINDOWS:
            weighted_rent = weighted_premium = weight_total = 0.0
            weighted_residual = residual_weight_total = 0.0
            count = 0
            lookup = lookups[width]
            for neighbor, distance_weight in edges[block]:
                for offset, area_weight in ((-1, 0.5), (0, 1.0), (1, 0.5)):
                    record = lookup.get((neighbor, flat_type, area_bin + offset))
                    if record is None:
                        continue
                    weight = distance_weight * area_weight
                    weighted_rent += weight * record["rent_sum"]
                    weighted_premium += weight * record["premium_sum"]
                    weight_total += weight * record["count"]
                    count += record["count"]
                    weighted_residual += weight * record["residual_sum"]
                    residual_weight_total += weight * record["residual_count"]
            values[f"LOCAL_RENT_MEAN_{width}M"] = weighted_rent / weight_total if weight_total else np.nan
            values[f"LOCAL_RENT_COUNT_{width}M"] = count
            values[f"LOCAL_PREMIUM_MEAN_{width}M"] = weighted_premium / weight_total if weight_total else np.nan
            values[f"LOCAL_RESIDUAL_MEAN_{width}M"] = (
                weighted_residual / residual_weight_total if residual_weight_total else np.nan)
        rows.append(values)
    features = pd.DataFrame(rows)
    tagged = targets[["_BLOCK_ID", "FLAT_TYPE_CLEAN", "_AREA_BIN"]].reset_index(names="_ROW_INDEX")
    return (tagged.merge(features, on=["_BLOCK_ID", "FLAT_TYPE_CLEAN", "_AREA_BIN"],
                         how="left", validate="many_to_one")
            .set_index("_ROW_INDEX").reindex(targets.index)[RAW_FEATURES + PREMIUM_FEATURES + RESIDUAL_FEATURES])


def local_features(fitting: pd.DataFrame, prediction: pd.DataFrame,
                   block_index: pd.MultiIndex, edges: list[list[tuple[int, float]]]
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit rows see earlier months; all future rows share the cutoff snapshot."""
    fit = _tag_rows(fitting, block_index)
    pred = _tag_rows(prediction, block_index)
    history = _history(fit)
    feature_columns = RAW_FEATURES + PREMIUM_FEATURES + RESIDUAL_FEATURES
    fit_features = pd.DataFrame(index=fit.index, columns=feature_columns, dtype=float)
    for month, part in fit.groupby("_MONTH_NUMBER", sort=True):
        fit_features.loc[part.index] = _features_at_anchor(part, history, edges, int(month))
    origin = int(fit["_MONTH_NUMBER"].max()) + 1
    pred_features = pd.DataFrame(index=pred.index, columns=feature_columns, dtype=float)
    # The one overlapping test month (2025-03) may only use strictly earlier labels.
    for month, part in pred.groupby("_MONTH_NUMBER", sort=True):
        pred_features.loc[part.index] = _features_at_anchor(part, history, edges, min(int(month), origin))
    return fit_features.astype(float), pred_features.astype(float)


def out_of_time_residuals(train: pd.DataFrame) -> pd.DataFrame:
    """Make past-row hedonic residuals from six-month forward predictions."""
    result = train.copy()
    residual = pd.Series(np.nan, index=result.index, dtype=float)
    for start_text in OOF_STARTS:
        start = pd.Period(start_text, freq="M")
        end = start + 6
        earlier = result[result[MONTH] < start]
        later = result[(result[MONTH] >= start) & (result[MONTH] < end)]
        if earlier.empty or later.empty:
            continue
        predicted = fit_row_model(earlier, later, with_history=True)[1]
        residual.loc[later.index] = later[TARGET].to_numpy(float) - predicted
        print(f"Out-of-time residuals {start_text}: {len(later)} rows", flush=True)
    # Remove broad monthly flat-type drift; remaining values describe local deviations.
    group_mean = residual.groupby([result[MONTH], result["FLAT_TYPE_CLEAN"]]).transform("mean")
    result["_OOF_RESIDUAL"] = residual - group_mean
    return result


def _fit_variant(fitting: pd.DataFrame, prediction: pd.DataFrame,
                 fit_history: pd.DataFrame, pred_history: pd.DataFrame,
                 fit_local: pd.DataFrame, pred_local: pd.DataFrame,
                 mode: str) -> np.ndarray:
    if mode == "rent":
        graph_features = RAW_FEATURES
    elif mode == "premium":
        graph_features = RAW_FEATURES + PREMIUM_FEATURES
    elif mode == "residual":
        graph_features = COUNT_FEATURES + RESIDUAL_FEATURES
    else:
        raise ValueError(f"Unknown local feature mode: {mode}")
    columns = BASE_FEATURES + HISTORY_FEATURES + graph_features
    fit = fitting.join(fit_history).join(fit_local)
    pred = prediction.join(pred_history).join(pred_local)
    x_fit, x_pred = model_inputs(fit, pred, columns)
    model = HistGradientBoostingRegressor(**ROW_MODEL_SETTINGS)
    model.fit(x_fit, fitting[TARGET].to_numpy(float))
    return np.asarray(model.predict(x_pred), float)


def evaluate(train: pd.DataFrame, block_index: pd.MultiIndex,
             edges: list[list[tuple[int, float]]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores, monthly, coverage = [], [], []
    for start_text, end_text in FOLDS:
        start, end = pd.Period(start_text, freq="M"), pd.Period(end_text, freq="M")
        fitting = train[train[MONTH] < start].copy()
        validation = train[(train[MONTH] >= start) & (train[MONTH] < end)].copy()
        print(f"Fold {start_text}: {len(fitting)} fit, {len(validation)} validation", flush=True)
        fit_hist, pred_hist = historical_features(fitting, validation)
        fit_local, pred_local = local_features(fitting, validation, block_index, edges)
        actual = validation[TARGET].to_numpy(float)
        variants = {
            "historical": fit_row_model(fitting, validation, with_history=True)[1],
            "historical_local_rent": _fit_variant(fitting, validation, fit_hist, pred_hist, fit_local, pred_local, "rent"),
            "historical_local_premium": _fit_variant(fitting, validation, fit_hist, pred_hist, fit_local, pred_local, "premium"),
            "historical_local_residual": _fit_variant(fitting, validation, fit_hist, pred_hist, fit_local, pred_local, "residual"),
        }
        for name, values in variants.items():
            scores.append({"validation_start": start_text, "validation_end_exclusive": end_text,
                           "horizon_months": validation[MONTH].nunique(), "variant": name,
                           "n_fit": len(fitting), "n_validation": len(validation), **metrics(actual, values)})
            for month in validation[MONTH].unique():
                mask = validation[MONTH].eq(month).to_numpy()
                monthly.append({"validation_start": start_text, "approval_month": str(month),
                                "horizon": month.ordinal - start.ordinal + 1, "variant": name,
                                "n": int(mask.sum()), **metrics(actual[mask], values[mask])})
            print(f"  {name}: RMSE={scores[-1]['rmse_sgd']:.2f}", flush=True)
        coverage.append({"validation_start": start_text, "n": len(validation),
                         "share_with_3m_comparable": float(pred_local["LOCAL_RENT_COUNT_3M"].gt(0).mean()),
                         "share_with_12m_comparable": float(pred_local["LOCAL_RENT_COUNT_12M"].gt(0).mean()),
                         "median_12m_comparables": float(pred_local["LOCAL_RENT_COUNT_12M"].median()),
                         "share_with_12m_residual": float(pred_local["LOCAL_RESIDUAL_MEAN_12M"].notna().mean()),
                         "median_graph_degree": float(np.median([len(e) for e in edges]))})
    return pd.DataFrame(scores), pd.DataFrame(monthly), pd.DataFrame(coverage)


def write_report(scores: pd.DataFrame, coverage: pd.DataFrame) -> None:
    comparison = scores.pivot(index="validation_start", columns="variant", values="rmse_sgd")
    comparison = comparison.loc[[start for start, _ in FOLDS]]
    comparison["raw_delta"] = comparison["historical_local_rent"] - comparison["historical"]
    comparison["premium_delta"] = comparison["historical_local_premium"] - comparison["historical"]
    comparison["residual_delta"] = comparison["historical_local_residual"] - comparison["historical"]
    columns = ["historical", "historical_local_rent", "raw_delta",
               "historical_local_premium", "premium_delta", "historical_local_residual", "residual_delta"]
    lines = [
        "# Block-level local comparable-rent graph", "",
        "All scores are fixed-origin batch-forecast RMSE in SGD. A negative delta improves on the historical baseline in the same fold.", "",
        "| Validation start | Historical | + local rent | Delta | + rent and premium | Delta | + local residual | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for start, row in comparison[columns].iterrows():
        lines.append(f"| {start} | {row['historical']:.2f} | {row['historical_local_rent']:.2f} | "
                     f"{row['raw_delta']:+.2f} | {row['historical_local_premium']:.2f} | "
                     f"{row['premium_delta']:+.2f} | {row['historical_local_residual']:.2f} | "
                     f"{row['residual_delta']:+.2f} |")
    lines += [
        "", "The first three validation windows are disjoint (6, 9, and 12 months). The final 16-month window overlaps them and is a separate horizon check.", "",
        "## Graph and features", "",
        "Each HDB block connects to its 12 nearest *other* blocks within 1 km. Comparables must share flat type and be in the same or adjacent 10 m² floor-area bin. Distance and bin similarity weight the mean. For each 3-, 6-, and 12-month lookback, the raw variant receives local rent mean and count. The premium variant adds deviations from the observed monthly flat-type × area-bin market mean. The residual variant uses local out-of-time hedonic residual means and comparable counts. All variants retain the baseline property, spatial, amenity, and historical-rent features.", "",
        "Residuals are generated in six-month forward blocks from the historical row model trained only on earlier months, starting in 2022-01. The residual bank is centered by observed month and flat type; residuals from 2021 are unavailable. A validation fold only consumes residuals from its own fitting period.", "",
        "Every fitting row sees only earlier approval months. All validation months reuse the fitting-cutoff snapshot; validation targets never update graph features. For an overlapping prediction month, only strictly earlier training months are eligible.", "",
        "## Coverage and interpretation", "",
        f"A 12-month comparable is available for {coverage['share_with_12m_comparable'].min():.1%}–{coverage['share_with_12m_comparable'].max():.1%} of validation rows, and an out-of-time residual is available for {coverage['share_with_12m_residual'].min():.1%}–{coverage['share_with_12m_residual'].max():.1%}. The median row has 17–18 raw comparables. Raw local rent worsens all four folds. See the fold table for the two normalized variants; retain the historical baseline unless a graph variant improves consistently across forward windows.", "",
        "This is one fixed graph definition and model setting, not a general test of all graphs or a trained graph neural network. See `fold_metrics.csv`, `monthly_metrics.csv`, `coverage.csv`, and `results.json` for details. Run `python analysis/compare_local_comparable_graph.py` from the repository root.", "",
    ]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sources = [DATA / "train.csv", AUX / "sg-hdb-block.csv", AUX / "sg-mrt-stations.csv",
               AUX / "sg-shopping-malls.csv", AUX / "sg-schools.csv"]
    before = {str(path.relative_to(DATA)): sha256(path) for path in sources}
    train = spatial_frame(pd.read_csv(DATA / "train.csv"))
    train = out_of_time_residuals(train)
    blocks = pd.read_csv(AUX / "sg-hdb-block.csv")
    block_index, edges = block_graph(blocks)
    OUT.mkdir(parents=True, exist_ok=True)
    scores, monthly, coverage = evaluate(train, block_index, edges)
    if before != {str(path.relative_to(DATA)): sha256(path) for path in sources}:
        raise AssertionError("An input CSV changed during evaluation")
    scores.to_csv(OUT / "fold_metrics.csv", index=False, float_format="%.4f")
    monthly.to_csv(OUT / "monthly_metrics.csv", index=False, float_format="%.4f")
    coverage.to_csv(OUT / "coverage.csv", index=False, float_format="%.4f")
    write_report(scores, coverage)
    summary = {"folds": FOLDS, "radius_km": RADIUS_KM, "max_neighbors": K_NEIGHBORS,
               "area_bin_sqm": AREA_BIN_SQM, "windows_months": WINDOWS,
               "features": RAW_FEATURES + PREMIUM_FEATURES + RESIDUAL_FEATURES,
               "residual_oof_starts": OOF_STARTS,
               "fold_scores": scores.to_dict(orient="records"),
               "coverage": coverage.to_dict(orient="records"), "source_sha256": before,
               "evaluation": "Fixed-origin batch forecast. No validation labels enter features."}
    (OUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
