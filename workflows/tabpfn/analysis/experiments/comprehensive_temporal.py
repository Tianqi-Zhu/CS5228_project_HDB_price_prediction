"""Composition-adjusted hierarchical dynamic rental forecasts.

The model tests regularized multi-scale/seasonal and COE corrections against a
conservative local baseline. Every model selection uses outcomes already known
at its forecast origin. COE covariates end at the origin, never at the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from analysis.experiments.aligned_temporal_features import AREA, TYPE, LocalTemporalModel, divide
from analysis.experiments.spatiotemporal_hybrid import MONTH, TARGET


MAX_HORIZON = 17
REFERENCE_CATS = [TYPE, "FLAT_MODEL_CLEAN", "TOWN_CLEAN"]
REFERENCE_NUMS = ["FLOOR_AREA_SQM", "FLAT_AGE_YEARS"]
ALPHAS = (1.0, 10.0, 100.0)
FEATURES = [
    "DYN_PROPERTY_REFERENCE", "DYN_MARKET_LEVEL3", "DYN_MARKET_LEVEL12",
    "DYN_TREND_SHORT", "DYN_TREND_LONG", "DYN_REGIME_SHIFT",
    "DYN_SEASONAL_EFFECT", "DYN_COE_EFFECT", "DYN_MARKET_VOLATILITY",
    "DYN_VOLUME_MOMENTUM", "DYN_LOCAL_PREMIUM", "DYN_BASELINE_FORECAST",
    "DYN_FORECAST_RMSE", "DYN_CORRECTION", "DYN_ECON_ENABLED",
    "DYN_FORECAST_HORIZON", "DYN_LOCAL_WEIGHT", "DYN_HISTORY_COUNT3",
    "DYN_TARGET_MONTH_SIN", "DYN_TARGET_MONTH_COS",
]


class ComprehensiveTemporalModel:
    def __init__(self, frame, coe):
        self.frame = frame.copy()
        self.first = frame[MONTH].min()
        self.reference_end = self.first + 11
        self.raw = LocalTemporalModel(frame)
        self.months = self.raw.months
        self.coe = coe.reindex(self.months).astype(float)
        self.reference = None
        if frame[MONTH].max() >= self.reference_end:
            reference_rows = frame.loc[frame[MONTH] <= self.reference_end]
            transformer = ColumnTransformer([
                ("categorical", OneHotEncoder(handle_unknown="ignore"), REFERENCE_CATS),
                ("numeric", SimpleImputer(strategy="median"), REFERENCE_NUMS),
            ])
            self.reference = make_pipeline(transformer, Ridge(alpha=100.0))
            self.reference.fit(reference_rows[REFERENCE_CATS + REFERENCE_NUMS], reference_rows[TARGET].to_numpy(float))
            adjusted = frame.copy()
            adjusted[TARGET] = frame[TARGET].to_numpy(float) - self.reference.predict(frame[REFERENCE_CATS + REFERENCE_NUMS])
            self.market = LocalTemporalModel(adjusted)
        self.covariate_cache, self.fits, self.choices = {}, {}, {}
        self.example_cache = None

    def _stats(self, origin):
        market = self.market
        state = market.state(origin)
        means = divide(market.gsum[:origin + 1], market.gcount[:origin + 1])
        last = np.asarray([column[np.isfinite(column)][-1] if np.isfinite(column).any() else np.nan
                           for column in means.T])
        lag12 = means[origin - 12] if origin >= 12 else np.full(len(market.kinds), np.nan)
        slopes = []
        for width in (3, 12):
            values, counts = means[-width:], market.gcount[max(0, origin + 1 - width):origin + 1]
            time = np.arange(len(values), dtype=float)[:, None]
            valid = np.isfinite(values) & (counts > 0)
            weight = np.where(valid, counts, 0)
            total = weight.sum(axis=0)
            center = np.divide((weight * time).sum(axis=0), total, out=np.zeros(len(total)), where=total > 0)
            denominator = (weight * (time - center) ** 2).sum(axis=0)
            numerator = (weight * (time - center) * np.nan_to_num(values)).sum(axis=0)
            slopes.append(np.clip(np.divide(numerator, denominator, out=np.zeros(len(total)), where=denominator > 0), -100, 100))
        volatility = []
        for width in (3, 12):
            values = means[-width:]
            valid = np.isfinite(values)
            n = valid.sum(axis=0)
            average = np.divide(np.nansum(values, axis=0), n, out=np.zeros(len(n)), where=n > 0)
            variance = np.divide(np.where(valid, (values - average) ** 2, 0).sum(axis=0), n,
                                 out=np.zeros(len(n)), where=n > 0)
            volatility.append(np.sqrt(variance))
        _, long_local_n = market.window(market.lprefix, origin, 12)
        volume = np.log1p(state["local_count"] / 3) - np.log1p(long_local_n / 12)
        return {"last": np.where(np.isfinite(last), last, state["short"]),
                "yoy": np.where(np.isfinite(lag12), last - lag12, 0),
                "short_slope": slopes[0], "long_slope": slopes[1],
                "vol3": volatility[0], "vol12": volatility[1], "volume": volume}

    def baseline(self, origin, horizon):
        params = self.market.fit_at(self.months[origin])
        return self.market.local_prediction(origin, horizon, params["phi"], params["shrink"], params["decay"])

    def covariates(self, origin, horizon, economic):
        key = (origin, horizon, economic)
        if key in self.covariate_cache:
            return self.covariate_cache[key]
        market, state, stats = self.market, self.market.state(origin), self._stats(origin)
        parent = market.parent
        target = self.months[origin] + horizon
        angle, initial_angle = 2 * np.pi * target.month / 12, 2 * np.pi * self.months[origin].month / 12
        pooled_short = float(np.average(np.nan_to_num(state["short"]), weights=np.maximum(state["count"], 1)))
        values = {
            "last_minus3": (stats["last"] - state["short"])[parent],
            "short_long_gap": (state["short"] - state["long"])[parent],
            "short_slope": stats["short_slope"][parent], "long_slope": stats["long_slope"][parent],
            "acceleration": (stats["short_slope"] - stats["long_slope"])[parent],
            "year_change": stats["yoy"][parent], "volatility3": stats["vol3"][parent],
            "volatility12": stats["vol12"][parent], "local_premium": np.nan_to_num(state["premium"]),
            "local_log_count": np.log1p(state["local_count"]), "volume_momentum": stats["volume"],
            "market_type_gap": (state["short"] - pooled_short)[parent],
            "horizon": np.full(len(parent), horizon), "log_horizon": np.full(len(parent), np.log1p(horizon)),
            "season_sin": np.full(len(parent), np.sin(angle) - np.sin(initial_angle)),
            "season_cos": np.full(len(parent), np.cos(angle) - np.cos(initial_angle)),
            "season_sin2": np.full(len(parent), np.sin(2 * angle) - np.sin(2 * initial_angle)),
            "season_cos2": np.full(len(parent), np.cos(2 * angle) - np.cos(2 * initial_angle)),
            "short_slope_h": stats["short_slope"][parent] * (1 - 0.8 ** horizon) / 0.2,
            "long_slope_h": stats["long_slope"][parent] * (1 - 0.8 ** horizon) / 0.2,
        }
        if economic:
            past = self.coe.iloc[:origin + 1].to_numpy(float)
            valid = np.isfinite(past) & (past > 0)
            logged = np.log(np.where(valid, past, np.nan))
            missing = not np.isfinite(logged[-min(12, len(logged)):]).all()
            observed = logged[np.isfinite(logged)]
            if len(observed):
                change3 = observed[-1] - observed[-4] if len(observed) >= 4 else 0.0
                change12 = observed[-1] - observed[-13] if len(observed) >= 13 else 0.0
                momentum = float(observed[-3:].mean() - observed[-12:].mean())
                volatility = float(np.std(np.diff(observed[-13:]))) if len(observed) > 1 else 0.0
            else:
                change3 = change12 = momentum = volatility = 0.0
            for name, value in (("coe_change3", change3), ("coe_change12", change12),
                                ("coe_momentum", momentum), ("coe_volatility", volatility),
                                ("coe_missing", float(missing))):
                values[name] = np.full(len(parent), value)
            values["coe_momentum_h"] = np.full(len(parent), momentum * (1 - 0.8 ** horizon) / 0.2)
        for i, kind in enumerate(market.kinds):
            indicator = (parent == i).astype(float)
            values[f"type_{kind}"] = indicator
            for feature in ("season_sin", "season_cos", "short_long_gap"):
                values[f"type_{kind}_x_{feature}"] = indicator * values[feature]
            if economic:
                for feature in ("coe_change3", "coe_momentum_h"):
                    values[f"type_{kind}_x_{feature}"] = indicator * values[feature]
        names = list(values)
        result = np.column_stack([np.nan_to_num(values[name]) for name in names]), names
        self.covariate_cache[key] = result
        return result

    def examples(self):
        if self.example_cache is not None:
            return self.example_cache
        x, targets, counts, months, horizons = {False: [], True: []}, [], [], [], []
        for origin in range(11, len(self.months) - 1):
            for horizon in range(1, min(MAX_HORIZON, len(self.months) - 1 - origin) + 1):
                target = origin + horizon
                actual = divide(self.market.lsum[target], self.market.lcount[target])
                base = self.baseline(origin, horizon)
                valid = np.isfinite(actual) & np.isfinite(base) & (self.market.lcount[target] > 0)
                if not valid.any():
                    continue
                for economic in (False, True):
                    array, _ = self.covariates(origin, horizon, economic)
                    x[economic].append(array[valid])
                targets.append(actual[valid] - base[valid])
                counts.append(self.market.lcount[target, valid])
                months.extend([target] * int(valid.sum()))
                horizons.extend([horizon] * int(valid.sum()))
        self.example_cache = ({key: np.concatenate(value) for key, value in x.items()}, np.concatenate(targets),
                              np.concatenate(counts), np.asarray(months), np.asarray(horizons))
        return self.example_cache

    def fit_correction(self, cutoff, economic, alpha):
        key = (cutoff, economic, alpha)
        if key in self.fits:
            return self.fits[key]
        arrays, targets, counts, months, horizons = self.examples()
        mask = months <= cutoff
        if len(np.unique(months[mask])) < 6:
            return None
        x, y = arrays[economic][mask], targets[mask]
        weights = counts[mask] * 0.5 ** ((cutoff - months[mask]) / 12)
        for horizon in np.unique(horizons[mask]):
            selected = horizons[mask] == horizon
            weights[selected] /= weights[selected].sum()
        weights *= len(np.unique(months[mask])) / weights.sum()
        mean = np.average(x, axis=0, weights=weights)
        scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
        scale[scale < 1e-8] = 1.0
        z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
        coef = np.linalg.solve(z.T @ (weights[:, None] * z) + alpha * np.eye(z.shape[1]), z.T @ (weights * y))
        fitted = {"mean": mean, "scale": scale, "coef": coef,
                  "training_target_months": len(np.unique(months[mask])), "last_target_month": str(self.months[int(months[mask].max())])}
        self.fits[key] = fitted
        return fitted

    def correction(self, cutoff, horizon, economic, alpha, fitted=None):
        fitted = self.fit_correction(cutoff, economic, alpha) if fitted is None else fitted
        if fitted is None:
            n = len(self.market.groups)
            return np.zeros(n), np.zeros(n), np.zeros(n)
        x, names = self.covariates(cutoff, horizon, economic)
        z = (x - fitted["mean"]) / fitted["scale"]
        contribution = z * fitted["coef"][1:]
        change = np.clip(0.5 * (fitted["coef"][0] + contribution.sum(axis=1)), -250, 250)
        seasonal = 0.5 * contribution[:, [i for i, name in enumerate(names) if "season_" in name]].sum(axis=1)
        coe = 0.5 * contribution[:, [i for i, name in enumerate(names) if "coe_" in name]].sum(axis=1) if economic else np.zeros(len(x))
        return change, seasonal, coe

    def select(self, cutoff):
        if cutoff in self.choices:
            return self.choices[cutoff]
        origins = [v for v in (cutoff - 20, cutoff - 10) if v >= 17]
        configurations = [(None, None)] + [(economic, alpha) for economic in (False, True) for alpha in ALPHAS]
        scores = []
        for economic, alpha in configurations:
            errors, weights = np.zeros(MAX_HORIZON), np.zeros(MAX_HORIZON)
            for origin in origins:
                fitted = None if economic is None else self.fit_correction(origin, economic, alpha)
                for horizon in range(1, min(MAX_HORIZON, cutoff - origin) + 1):
                    target = origin + horizon
                    predicted = self.baseline(origin, horizon)
                    if economic is not None:
                        predicted = predicted + self.correction(origin, horizon, economic, alpha, fitted=fitted)[0]
                    actual = divide(self.market.lsum[target], self.market.lcount[target])
                    valid = np.isfinite(predicted) & np.isfinite(actual)
                    count = self.market.lcount[target, valid]
                    errors[horizon - 1] += np.sum(count * (predicted[valid] - actual[valid]) ** 2)
                    weights[horizon - 1] += count.sum()
            per_horizon = np.sqrt(divide(errors, weights))
            score = float(np.sqrt(np.nanmean(divide(errors, weights)))) if origins else None
            scores.append({"economic": economic, "alpha": alpha, "rmse": score,
                           "errors": [float(v) if np.isfinite(v) else None for v in per_horizon]})
        chosen = scores[0]
        if origins:
            winner = min(scores, key=lambda row: row["rmse"])
            # New corrections must beat the simpler baseline by at least 1%.
            if winner["economic"] is not None and winner["rmse"] < scores[0]["rmse"] * 0.99:
                candidates = [row for row in scores[1:]
                              if row["rmse"] <= winner["rmse"] * 1.01 and row["rmse"] < scores[0]["rmse"] * 0.99]
                chosen = next((row for row in candidates if row["economic"] is False), candidates[0])
        result = {**chosen, "origin": str(self.months[cutoff]), "inner_origins": [str(self.months[v]) for v in origins],
                  "candidates": [{k: v for k, v in row.items() if k != "errors"} for row in scores]}
        self.choices[cutoff] = result
        return result

    def features(self, query, origin, fallback):
        origin = pd.Period(origin, "M")
        index = origin.ordinal - self.first.ordinal
        result = pd.DataFrame(np.nan, index=query.index, columns=FEATURES)
        horizon = query[MONTH].astype("int64").to_numpy() - origin.ordinal
        result["DYN_FORECAST_HORIZON"] = horizon
        result["DYN_TARGET_MONTH_SIN"] = np.sin(2 * np.pi * query[MONTH].dt.month / 12)
        result["DYN_TARGET_MONTH_COS"] = np.cos(2 * np.pi * query[MONTH].dt.month / 12)
        if origin < self.reference_end or self.reference is None:
            result["DYN_BASELINE_FORECAST"] = fallback
            result["DYN_ECON_ENABLED"] = 0.0
            return result
        choice, stats, state = self.select(index), self._stats(index), self.market.state(index)
        reference = self.reference.predict(query[REFERENCE_CATS + REFERENCE_NUMS])
        values = {}
        for h in np.unique(horizon):
            if not 1 <= h <= MAX_HORIZON:
                raise ValueError("Unsupported temporal forecast horizon")
            correction, seasonal, economic = (np.zeros(len(self.market.groups)),) * 3
            if choice["economic"] is not None:
                correction, seasonal, economic = self.correction(index, int(h), choice["economic"], choice["alpha"])
            values[int(h)] = self.baseline(index, int(h)) + correction, correction, seasonal, economic
        params = self.market.fit_at(origin)
        for position, (row_id, row) in enumerate(query.iterrows()):
            h = int(horizon[position])
            group = self.market.group_index.get((str(row[AREA]), str(row[TYPE])))
            kind = self.market.kind_index.get(str(row[TYPE]))
            if group is None or kind is None:
                result.loc[row_id, "DYN_BASELINE_FORECAST"] = fallback[position]
                continue
            predicted, correction, seasonal, economic = values[h]
            count = state["local_count"][group]
            weight = 0 if params["shrink"] is None else count / (count + params["shrink"])
            error = choice["errors"][h - 1]
            if error is None:
                known = [v for v in params["rmse_by_horizon"] if v is not None]
                error = (known[min(h - 1, len(known) - 1)] * np.sqrt(max(1, h / 10))) if known else np.nan
            result.loc[row_id, FEATURES[:18]] = [
                reference[position], state["short"][kind], state["long"][kind],
                stats["short_slope"][kind], stats["long_slope"][kind],
                abs(state["short"][kind] - state["long"][kind]) / max(25, stats["vol12"][kind]),
                seasonal[group], economic[group], stats["vol12"][kind], stats["volume"][group],
                weight * state["premium"][group], max(1.0, reference[position] + predicted[group]),
                error, correction[group], float(choice["economic"] is True), float(h), weight, count,
            ]
        return result
