"""Bayesian additive HDB rent model with amenity, place, and month factors.

The Gaussian coefficients have proper, group-specific priors. Conditional on
those fixed prior precisions, an inverse-gamma variance prior yields an exact
normal-inverse-gamma posterior and Student-t predictive intervals. Monthly
effects have a random-walk prior; future effects inherit the last fitted level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import t as student_t
from sklearn.preprocessing import OneHotEncoder

from analysis.config import AUX, DATA, REPORTS
from analysis.data import sha256
from analysis.evaluation import metrics
from analysis.experiments.spatiotemporal_hybrid import (
    MONTH, TARGET, fit_row_model, historical_features, spatial_frame,
)

OUT = REPORTS / "bayesian_amenity_factors"
FOLDS = [
    ("2023-01", "2023-07"),
    ("2023-07", "2024-04"),
    ("2024-04", "2025-04"),
    ("2023-12", "2025-04"),
]
VARIANTS = ("property_location_time", "plus_amenities", "plus_town_mrt_slope", "plus_history")
MRT_KNOTS = (0.3, 0.6, 1.0, 1.5)
MALL_KNOTS = (0.5, 1.0, 1.5, 2.0)
HISTORY_COLUMNS = (
    "HIST_SIGNATURE_LAST_MEAN", "HIST_SIGNATURE_COUNT", "HIST_SIGNATURE_RECENCY_MONTHS",
    "HIST_BLOCK_FLAT_LAST_MEAN", "HIST_BLOCK_FLAT_COUNT",
    "HIST_AREA_FLAT_LAST_MEAN", "HIST_AREA_FLAT_COUNT",
)
# Prior precision is relative to the observation variance. Larger values
# imply stronger shrinkage, including for sparse towns and subzones.
PRIOR_PRECISION = {
    "intercept": 1e-4,
    "property_numeric": 3.0,
    "property_category": 8.0,
    "town": 25.0,
    "subzone": 60.0,
    "cbd": 15.0,
    "season": 20.0,
    "amenity": 25.0,
    "town_mrt_slope": 150.0,
    "history": 8.0,
    "month_anchor": 0.05,
    "month_difference": 80.0,
}
VARIANCE_PRIOR_SHAPE = 2.0
VARIANCE_PRIOR_SCALE = 300.0 ** 2


@dataclass
class Design:
    fit: sparse.csr_matrix
    prediction: sparse.csr_matrix
    precision: np.ndarray
    slices: dict[str, slice]


@dataclass
class Posterior:
    mean: np.ndarray
    covariance_unit: np.ndarray
    variance_shape: float
    variance_scale: float
    target_center: float

    def predict(self, x: sparse.csr_matrix) -> np.ndarray:
        return np.asarray(x @ self.mean).ravel() + self.target_center

    def interval(self, x: sparse.csr_matrix, probability: float = .9) -> tuple[np.ndarray, np.ndarray]:
        dense = x.toarray()
        expected = dense @ self.mean + self.target_center
        quadratic = np.einsum("ij,jk,ik->i", dense, self.covariance_unit, dense)
        scale = np.sqrt(self.variance_scale / self.variance_shape * (1 + quadratic))
        margin = student_t.ppf((1 + probability) / 2, df=2 * self.variance_shape) * scale
        return expected - margin, expected + margin


def spline_basis(distance: np.ndarray, knots: tuple[float, ...]) -> np.ndarray:
    """Piecewise linear distance effect with coefficients shrunk toward zero."""
    d = np.asarray(distance, dtype=float)
    return np.column_stack([d] + [np.maximum(d - knot, 0.0) for knot in knots])


def _numeric(fit: pd.DataFrame, prediction: pd.DataFrame,
             columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    for column in columns:
        train = pd.to_numeric(fit[column], errors="coerce").to_numpy(float)
        future = pd.to_numeric(prediction[column], errors="coerce").to_numpy(float)
        median = float(np.nanmedian(train)) if np.isfinite(train).any() else 0.0
        train = np.where(np.isfinite(train), train, median)
        future = np.where(np.isfinite(future), future, median)
        center, scale = float(train.mean()), float(train.std())
        scale = scale if scale > 1e-9 else 1.0
        a.append((train - center) / scale)
        b.append((future - center) / scale)
    return np.column_stack(a), np.column_stack(b)


def make_design(fitting: pd.DataFrame, prediction: pd.DataFrame,
                variant: str) -> Design:
    """Fit preprocessing on the fitting partition and include future month states."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    parts_fit, parts_pred, penalties, slices = [], [], [], {}
    cursor = 0

    def add(name: str, a: np.ndarray | sparse.spmatrix,
            b: np.ndarray | sparse.spmatrix, penalty: float) -> None:
        nonlocal cursor
        a, b = sparse.csr_matrix(a), sparse.csr_matrix(b)
        if a.shape[1] != b.shape[1]:
            raise ValueError(f"Feature count differs for {name}")
        parts_fit.append(a)
        parts_pred.append(b)
        penalties.extend([penalty] * a.shape[1])
        slices[name] = slice(cursor, cursor + a.shape[1])
        cursor += a.shape[1]

    add("intercept", np.ones((len(fitting), 1)), np.ones((len(prediction), 1)),
        PRIOR_PRECISION["intercept"])
    prop_a, prop_b = _numeric(fitting, prediction, ["FLOOR_AREA_SQM", "FLAT_AGE_YEARS", "MAX_FLOOR"])
    prop_a = np.column_stack([prop_a, prop_a[:, :2] ** 2])
    prop_b = np.column_stack([prop_b, prop_b[:, :2] ** 2])
    add("property_numeric", prop_a, prop_b, PRIOR_PRECISION["property_numeric"])
    for name, column, penalty in (
        ("flat_type", "FLAT_TYPE_CLEAN", "property_category"),
        ("flat_model", "FLAT_MODEL_CLEAN", "property_category"),
        ("town", "TOWN_CLEAN", "town"),
        ("subzone", "SUBZONE", "subzone"),
    ):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        a = encoder.fit_transform(fitting[[column]].astype(str))
        b = encoder.transform(prediction[[column]].astype(str))
        add(name, a, b, PRIOR_PRECISION[penalty])
        if name == "town":
            town_fit, town_pred = a, b

    cbd_a, cbd_b = _numeric(fitting, prediction, ["DISTANCE_TO_CBD_KM"])
    add("cbd", cbd_a, cbd_b, PRIOR_PRECISION["cbd"])
    fit_month = fitting[MONTH].astype("int64").to_numpy()
    pred_month = prediction[MONTH].astype("int64").to_numpy()
    season_a = np.column_stack([np.sin(2 * np.pi * (fit_month % 12) / 12),
                                np.cos(2 * np.pi * (fit_month % 12) / 12)])
    season_b = np.column_stack([np.sin(2 * np.pi * (pred_month % 12) / 12),
                                np.cos(2 * np.pi * (pred_month % 12) / 12)])
    add("season", season_a, season_b, PRIOR_PRECISION["season"])

    if variant != "property_location_time":
        for name, column, knots in (
            ("mrt", "NEAREST_MRT_KM", MRT_KNOTS),
            ("mall", "NEAREST_MALL_KM", MALL_KNOTS),
        ):
            train = fitting[column].to_numpy(float)
            future = prediction[column].to_numpy(float)
            add(name, spline_basis(train, knots), spline_basis(future, knots),
                PRIOR_PRECISION["amenity"])
        count_a, count_b = _numeric(fitting, prediction,
                                     ["MRT_COUNT_1.0KM", "MALL_COUNT_1.0KM"])
        add("amenity_counts", count_a, count_b, PRIOR_PRECISION["amenity"])

    if variant == "plus_town_mrt_slope":
        proximity_a, proximity_b = _numeric(fitting, prediction, ["NEAREST_MRT_KM"])
        add("town_mrt_slope", town_fit.multiply(proximity_a),
            town_pred.multiply(proximity_b), PRIOR_PRECISION["town_mrt_slope"])

    if variant == "plus_history":
        history_fit = fitting.copy()
        history_pred = prediction.copy()
        for column in HISTORY_COLUMNS:
            if column.endswith("_COUNT"):
                history_fit[column] = np.log1p(history_fit[column].fillna(0))
                history_pred[column] = np.log1p(history_pred[column].fillna(0))
        a, b = _numeric(history_fit, history_pred, list(HISTORY_COLUMNS))
        add("history", a, b, PRIOR_PRECISION["history"])

    first_month = int(fit_month.min())
    n_months = int(max(fit_month.max(), pred_month.max()) - first_month + 1)
    if (pred_month < first_month).any():
        raise ValueError("Prediction month precedes the fitting period")
    month_a = sparse.csr_matrix((np.ones(len(fit_month)),
                                 (np.arange(len(fit_month)), fit_month - first_month)),
                                shape=(len(fitting), n_months))
    month_b = sparse.csr_matrix((np.ones(len(pred_month)),
                                 (np.arange(len(pred_month)), pred_month - first_month)),
                                shape=(len(prediction), n_months))
    add("month", month_a, month_b, PRIOR_PRECISION["month_anchor"])
    x_fit = sparse.hstack(parts_fit, format="csr")
    x_pred = sparse.hstack(parts_pred, format="csr")
    precision = np.diag(np.asarray(penalties, dtype=float))
    month_slice = slices["month"]
    for month in range(month_slice.start + 1, month_slice.stop):
        previous = month - 1
        lam = PRIOR_PRECISION["month_difference"]
        precision[month, month] += lam
        precision[previous, previous] += lam
        precision[month, previous] -= lam
        precision[previous, month] -= lam
    return Design(x_fit, x_pred, precision, slices)


def fit_posterior(design: Design, y: np.ndarray) -> Posterior:
    """Exact conjugate Gaussian posterior with fixed group prior strengths."""
    target = np.asarray(y, dtype=float)
    center = float(target.mean())
    centered = target - center
    q = (design.fit.T @ design.fit).toarray() + design.precision
    rhs = np.asarray(design.fit.T @ centered).ravel()
    factor = cho_factor(q, lower=True, check_finite=False)
    mean = cho_solve(factor, rhs, check_finite=False)
    covariance_unit = cho_solve(factor, np.eye(q.shape[0]), check_finite=False)
    shape = VARIANCE_PRIOR_SHAPE + len(target) / 2
    scale = VARIANCE_PRIOR_SCALE + 0.5 * (centered @ centered - rhs @ mean)
    if scale <= 0:
        raise ArithmeticError("Posterior variance scale was nonpositive")
    return Posterior(mean, covariance_unit, shape, scale, center)


def amenity_contrast(posterior: Posterior, design: Design,
                     name: str, near: float = .25, far: float = 1.0) -> dict:
    knots = MRT_KNOTS if name == "mrt" else MALL_KNOTS
    difference = spline_basis(np.array([near]), knots)[0] - spline_basis(np.array([far]), knots)[0]
    positions = design.slices[name]
    mean = float(difference @ posterior.mean[positions])
    unit_variance = float(difference @ posterior.covariance_unit[positions, positions] @ difference)
    scale = np.sqrt(posterior.variance_scale / posterior.variance_shape * unit_variance)
    margin = float(student_t.ppf(.95, df=2 * posterior.variance_shape) * scale)
    return {"factor": name, "near_km": near, "far_km": far,
            "near_minus_far_sgd": mean, "lower_90_sgd": mean - margin,
            "upper_90_sgd": mean + margin}


def evaluate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores, monthly, effects = [], [], []
    rng = np.random.default_rng(42)
    for start_text, end_text in FOLDS:
        start, end = pd.Period(start_text, freq="M"), pd.Period(end_text, freq="M")
        fitting = frame[frame[MONTH] < start].copy()
        validation = frame[(frame[MONTH] >= start) & (frame[MONTH] < end)].copy()
        actual = validation[TARGET].to_numpy(float)
        print(f"Fold {start_text}: {len(fitting)} fit, {len(validation)} validation", flush=True)
        historical_pred = fit_row_model(fitting, validation, with_history=True)[1]
        fit_hist, pred_hist = historical_features(fitting, validation)
        fitting = fitting.join(fit_hist)
        validation = validation.join(pred_hist)
        scores.append({"validation_start": start_text, "variant": "historical_tree",
                       "n_validation": len(validation), "horizon_months": validation[MONTH].nunique(),
                       "interval_90_coverage": np.nan, **metrics(actual, historical_pred)})
        sample = np.sort(rng.choice(len(validation), size=min(3000, len(validation)), replace=False))
        for variant in VARIANTS:
            design = make_design(fitting, validation, variant)
            posterior = fit_posterior(design, fitting[TARGET].to_numpy(float))
            predicted = posterior.predict(design.prediction)
            lower, upper = posterior.interval(design.prediction[sample])
            coverage = float(np.mean((actual[sample] >= lower) & (actual[sample] <= upper)))
            scores.append({"validation_start": start_text, "variant": variant,
                           "n_validation": len(validation), "horizon_months": validation[MONTH].nunique(),
                           "interval_90_coverage": coverage, **metrics(actual, predicted)})
            for month in validation[MONTH].unique():
                mask = validation[MONTH].eq(month).to_numpy()
                monthly.append({"validation_start": start_text, "approval_month": str(month),
                                "variant": variant, "n": int(mask.sum()),
                                **metrics(actual[mask], predicted[mask])})
            if "mrt" in design.slices:
                effects.append({"validation_start": start_text, "variant": variant,
                                **amenity_contrast(posterior, design, "mrt")})
                effects.append({"validation_start": start_text, "variant": variant,
                                **amenity_contrast(posterior, design, "mall")})
            print(f"  {variant}: RMSE={scores[-1]['rmse_sgd']:.2f}, "
                  f"90% interval coverage={coverage:.3f}", flush=True)
    return pd.DataFrame(scores), pd.DataFrame(monthly), pd.DataFrame(effects)


def write_report(scores: pd.DataFrame, effects: pd.DataFrame) -> None:
    pivot = scores.pivot(index="validation_start", columns="variant", values="rmse_sgd")
    lines = ["# Bayesian amenity and location factors", "",
             "Gaussian hierarchical additive model with proper grouped priors, a monthly random-walk effect, and an exact conditional posterior. Lower fixed-origin RMSE is better. The 16-month fold overlaps the earlier validation windows and is a separate horizon check.", "",
             "| Validation start | Historical tree | Property/location/time | + amenities | + town MRT slope | + history |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for start, _ in FOLDS:
        row = pivot.loc[start]
        lines.append(f"| {start} | {row['historical_tree']:.2f} | {row['property_location_time']:.2f} | "
                     f"{row['plus_amenities']:.2f} | {row['plus_town_mrt_slope']:.2f} | "
                     f"{row['plus_history']:.2f} |")
    amenity_gain = pivot["plus_amenities"] - pivot["property_location_time"]
    latest_gap = pivot.loc["2024-04", "plus_history"] - pivot.loc["2024-04", "historical_tree"]
    early_coverage = scores.loc[
        scores["validation_start"].eq("2023-01") & scores["variant"].eq("plus_history"),
        "interval_90_coverage"].iloc[0]
    lines += ["", "## Model and interpretation", "",
              f"Adding amenities improves this Bayesian specification by S${-amenity_gain.min():.2f} at most across the four folds, but its best 12-month variant remains S${latest_gap:.2f} above the historical tree RMSE. The history variant's nominal 90% intervals cover only {early_coverage:.1%} of sampled rows in the 2023-01 rising-market fold, indicating poor calibration there.", "",
              "Property terms include flat type/model, floor area, age, and block height. Town and subzone intercepts have shrinkage priors. MRT and mall distance have separate piecewise-linear curves, along with 1 km facility counts. The town MRT variant adds pooled town-specific slopes. The history variant includes earlier-month rent summaries and recency.", "",
              "A monthly random-walk prior pools neighboring months and carries the last fitted market level forward. Seasonality is explicit. All preprocessing and posterior fitting use only pre-origin data. Posterior-mean rents are scored in SGD; 90% predictive interval coverage uses a reproducible sample of 3,000 validation rows per fold.", "",
              "The amenity contrast compares fitted rent at 0.25 km and 1.0 km while holding other predictors fixed; for the town-slope variant it reports the shared MRT curve without a particular town deviation. It is an association rather than a causal effect; facility locations are current snapshots without historical opening dates. The Gaussian observation model treats rows as conditionally independent, so posterior intervals may understate repeated-block dependence. Prior strengths are fixed before validation, and uncertainty does not integrate over them. The forward-fold coverage numbers take precedence over nominal interval labels when judging reliability.", "",
              "## Amenity contrasts (SGD; near minus far)", "",
              "| Validation start | Variant | Factor | Estimate | 90% interval |",
              "| --- | --- | --- | ---: | ---: |"]
    for item in effects.to_dict(orient="records"):
        lines.append(f"| {item['validation_start']} | {item['variant']} | {item['factor']} | "
                     f"{item['near_minus_far_sgd']:+.1f} | "
                     f"[{item['lower_90_sgd']:+.1f}, {item['upper_90_sgd']:+.1f}] |")
    lines += ["", "See `fold_metrics.csv`, `monthly_metrics.csv`, `amenity_effects.csv`, and `results.json` for complete results and priors. Run `python analysis/compare_bayesian_amenity_factors.py` from the repository root.", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sources = [DATA / "train.csv", AUX / "sg-hdb-block.csv", AUX / "sg-mrt-stations.csv",
               AUX / "sg-shopping-malls.csv", AUX / "sg-schools.csv"]
    before = {str(path.relative_to(DATA)): sha256(path) for path in sources}
    frame = spatial_frame(pd.read_csv(DATA / "train.csv"))
    OUT.mkdir(parents=True, exist_ok=True)
    scores, monthly, effects = evaluate(frame)
    if before != {str(path.relative_to(DATA)): sha256(path) for path in sources}:
        raise AssertionError("An input CSV changed during evaluation")
    scores.to_csv(OUT / "fold_metrics.csv", index=False, float_format="%.4f")
    monthly.to_csv(OUT / "monthly_metrics.csv", index=False, float_format="%.4f")
    effects.to_csv(OUT / "amenity_effects.csv", index=False, float_format="%.4f")
    write_report(scores, effects)
    result = {"folds": FOLDS, "variants": VARIANTS, "mrt_knots_km": MRT_KNOTS,
              "mall_knots_km": MALL_KNOTS, "history_columns": HISTORY_COLUMNS,
              "prior_precision": PRIOR_PRECISION,
              "variance_prior_shape": VARIANCE_PRIOR_SHAPE,
              "variance_prior_scale": VARIANCE_PRIOR_SCALE,
              "fold_scores": scores.astype(object).where(pd.notna(scores), None).to_dict(orient="records"),
              "amenity_effects": effects.to_dict(orient="records"),
              "source_sha256": before}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
