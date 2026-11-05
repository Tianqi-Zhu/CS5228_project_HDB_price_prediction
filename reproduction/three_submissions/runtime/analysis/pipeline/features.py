"""Canonical feature candidates and strictly prior-month rental histories.

The benchmark compares these schemas without changing historical experiments.
No economic observation or prediction-partition target is needed at inference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from analysis.pipeline.auxiliary import build_auxiliary_features
from analysis.pipeline.config import DATE, FEATURE_SETS, TARGET


MAIN = tuple(FEATURE_SETS["C_property_time_town"])
MAIN_CATEGORIES = ("FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN", "TOWN_CLEAN")
BLOCK = (
    "LATITUDE", "LONGITUDE", "MAX_FLOOR", "YEAR_COMPLETED",
    "BUILDING_AGE_YEARS", "SUBZONE_CLEAN", "PLANNING_AREA_CLEAN", "REGION_CLEAN",
)
AMENITIES = (
    "MRT_DIST_KM", "MALL_DIST_KM", "PRIMARY_SCHOOL_DIST_KM", "CBD_DIST_KM",
    "AMENITY_DENSITY_1KM",
) + tuple(
    f"{kind}_COUNT_{radius}KM"
    for kind in ("MRT", "MALL", "PRIMARY_SCHOOL")
    for radius in ("0_5", "1_0", "2_0")
)
SPATIAL_CATEGORIES = MAIN_CATEGORIES + ("SUBZONE_CLEAN", "PLANNING_AREA_CLEAN", "REGION_CLEAN")
COMPACT_CATEGORIES = MAIN_CATEGORIES + ("PLANNING_AREA_CLEAN",)
COMPACT = COMPACT_CATEGORIES + (
    "FLOOR_AREA_SQM", "FLAT_AGE_YEARS", "MAX_FLOOR", "LATITUDE", "LONGITUDE",
    "APPROVAL_MONTH_INDEX", "APPROVAL_MONTH_SIN", "APPROVAL_MONTH_COS",
    "CBD_DIST_KM", "MRT_DIST_KM", "MALL_DIST_KM", "PRIMARY_SCHOOL_DIST_KM",
    "MRT_COUNT_1_0KM", "MALL_COUNT_1_0KM", "PRIMARY_SCHOOL_COUNT_1_0KM",
)
RICH_CATEGORIES = MAIN_CATEGORIES + (
    "STREET_CLEAN", "BLOCK_CLEAN", "ADDRESS_KEY", "POSTAL_CODE_CLEAN",
    "SUBZONE_CLEAN", "PLANNING_AREA_CLEAN", "REGION_CLEAN",
)
RICH = tuple(dict.fromkeys(MAIN + BLOCK + AMENITIES + RICH_CATEGORIES + (
    "APPROVAL_YEAR", "APPROVAL_MONTH_SIN", "APPROVAL_MONTH_COS", "LEASE_COMMENCE_DATE",
    "COMPLETION_LEASE_GAP", "ROOM_COUNT", "AREA_PER_ROOM",
)))
HISTORY_GRAINS = {
    "SIGNATURE": (
        "TOWN_KEY", "BLOCK_KEY", "STREET_KEY", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN",
        "FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE",
    ),
    "BLOCK_FLAT": ("TOWN_KEY", "BLOCK_KEY", "STREET_KEY", "FLAT_TYPE_CLEAN"),
    "AREA_FLAT": ("PLANNING_AREA_CLEAN", "FLAT_TYPE_CLEAN"),
}
HISTORY_SUFFIXES = ("MEAN", "COUNT", "LAST_MEAN", "RECENCY_MONTHS")
HISTORY = tuple(f"HIST_{grain}_{suffix}" for grain in HISTORY_GRAINS for suffix in HISTORY_SUFFIXES)
MARKET_COLUMNS = (
    "MARKET_TYPE_LAST", "MARKET_TYPE_MEAN_3M", "MARKET_TYPE_MEAN_12M",
    "MARKET_TYPE_LOG_SLOPE_6M", "MARKET_TYPE_PROJECTED", "MARKET_HORIZON_MONTHS",
)


@dataclass(frozen=True)
class HistoryPolicy:
    prefix: str
    value: str = "rent"
    smoothing: float = 0.0
    half_life_months: float | None = None
    window_months: int | None = None
    coarse_signature: bool = False
    statistics: tuple[str, ...] = HISTORY_SUFFIXES

    @property
    def grains(self):
        return tuple(HISTORY_GRAINS) + (("COARSE_SIGNATURE",) if self.coarse_signature else ())

    @property
    def columns(self):
        return tuple(f"{self.prefix}_{grain}_{suffix}" for grain in self.grains for suffix in self.statistics)


HISTORY_POLICIES = {
    "smooth": HistoryPolicy("SMOOTH", smoothing=20),
    "decay6": HistoryPolicy("DECAY6", half_life_months=6),
    "decay12": HistoryPolicy("DECAY12", half_life_months=12),
    "recent12": HistoryPolicy("RECENT12", window_months=12),
    "smooth_decay12": HistoryPolicy("SMOOTH_DECAY12", smoothing=20, half_life_months=12),
    "premium": HistoryPolicy("PREMIUM", value="premium", smoothing=20),
    "premium_decay12": HistoryPolicy("PREMIUM_DECAY12", value="premium", smoothing=20, half_life_months=12),
    "ppsm": HistoryPolicy("PPSM", value="ppsm", smoothing=20),
    "coarse": HistoryPolicy("COARSE", smoothing=20, coarse_signature=True),
    "premium_values": HistoryPolicy("PREMIUM", value="premium", smoothing=20,
        statistics=("MEAN", "LAST_MEAN")),
}


@dataclass(frozen=True)
class FeatureSpec:
    columns: tuple[str, ...]
    categorical: tuple[str, ...]
    use_history: bool = False
    history_policies: tuple[str, ...] = ()
    history_grains: tuple[str, ...] = tuple(HISTORY_GRAINS)
    market_context: bool = False

    @property
    def history_columns(self):
        if not self.use_history:
            return ()
        policies = self.history_policies or ("standard",)
        columns = []
        for name in policies:
            if name == "standard":
                prefix, grains = "HIST", self.history_grains
                statistics = HISTORY_SUFFIXES
            else:
                policy = HISTORY_POLICIES[name]
                prefix = policy.prefix
                grains = tuple(grain for grain in policy.grains if grain in self.history_grains
                    or grain == "COARSE_SIGNATURE")
                statistics = policy.statistics
            columns.extend(f"{prefix}_{grain}_{suffix}" for grain in grains for suffix in statistics)
        return tuple(columns)

    @property
    def all_columns(self):
        return self.columns + self.history_columns + (MARKET_COLUMNS if self.market_context else ())


FEATURE_CANDIDATES = {
    "main_c": FeatureSpec(MAIN, MAIN_CATEGORIES),
    "spatial_29": FeatureSpec(MAIN + BLOCK + AMENITIES, SPATIAL_CATEGORIES),
    "compact_static": FeatureSpec(COMPACT, COMPACT_CATEGORIES),
    "rich_static": FeatureSpec(RICH, RICH_CATEGORIES),
    "spatial_history": FeatureSpec(MAIN + BLOCK + AMENITIES, SPATIAL_CATEGORIES, True),
    "compact_history": FeatureSpec(COMPACT, COMPACT_CATEGORIES, True),
    "rich_history": FeatureSpec(RICH, RICH_CATEGORIES, True),
    "compact_history_27": FeatureSpec(COMPACT, COMPACT_CATEGORIES, True,
        history_grains=("BLOCK_FLAT", "AREA_FLAT")),
    "compact_premium_research": FeatureSpec(COMPACT, COMPACT_CATEGORIES, True,
        history_policies=("standard", "premium")),
}


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """One deterministic, label-independent builder for all candidate schemas."""
    result = build_auxiliary_features(raw, include_macro=False)
    result.index = raw.index.copy()
    return result


def _month_numbers(frame: pd.DataFrame) -> np.ndarray:
    dates = pd.to_datetime(frame[DATE], format="%Y-%m", errors="raise")
    return (dates.dt.year * 12 + dates.dt.month).to_numpy(np.int64)


def _group_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = frame[list(columns)].copy()
    for column in ("FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"):
        if column in values:
            values[column] = pd.to_numeric(values[column], errors="coerce").astype("Float64")
    values = values.astype("string").fillna("__MISSING__")
    key = values.iloc[:, 0]
    for column in values.columns[1:]:
        key = key.str.cat(values[column], sep="\x1f")
    return key


class HistoricalRentEncoder:
    """Query a fitted history using months strictly before each query month.

    All rows in the same approval month share the same past. Future query
    months use the frozen fitting history. Queries overlapping the final
    fitting month still exclude that month's labels (present in test.csv).
    """

    def fit(self, frame: pd.DataFrame):
        target = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(float)
        if not len(frame) or not np.isfinite(target).all():
            raise ValueError("Historical rents require nonempty, finite fitting targets")
        months = _month_numbers(frame)
        self.tables_ = {}
        for grain, columns in HISTORY_GRAINS.items():
            values = pd.DataFrame({
                "_key": _group_keys(frame, columns).to_numpy(),
                "_month": months, "_rent": target,
            })
            monthly = values.groupby(["_key", "_month"], as_index=False, sort=True).agg(
                _sum=("_rent", "sum"), _count=("_rent", "size"), _last_mean=("_rent", "mean"),
            )
            grouped = monthly.groupby("_key", sort=False)
            monthly["_total_sum"] = grouped["_sum"].cumsum()
            monthly["_total_count"] = grouped["_count"].cumsum()
            self.tables_[grain] = monthly.sort_values("_month", kind="stable")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "tables_"):
            raise ValueError("Fit HistoricalRentEncoder before transforming")
        months = _month_numbers(frame)
        result = pd.DataFrame(index=frame.index)
        if frame.empty:
            return result.reindex(columns=HISTORY, fill_value=np.nan)
        for grain, columns in HISTORY_GRAINS.items():
            query = pd.DataFrame({
                "_key": _group_keys(frame, columns).to_numpy(), "_month": months,
                "_position": np.arange(len(frame)),
            }).sort_values("_month", kind="stable")
            table = self.tables_[grain].rename(columns={"_month": "_observed_month"})
            matched = pd.merge_asof(
                query, table, left_on="_month", right_on="_observed_month", by="_key",
                direction="backward", allow_exact_matches=False,
            ).sort_values("_position", kind="stable")
            prefix = f"HIST_{grain}"
            result[f"{prefix}_MEAN"] = (matched["_total_sum"] / matched["_total_count"]).to_numpy()
            result[f"{prefix}_COUNT"] = matched["_total_count"].fillna(0).to_numpy()
            result[f"{prefix}_LAST_MEAN"] = matched["_last_mean"].to_numpy()
            result[f"{prefix}_RECENCY_MONTHS"] = (matched["_month"] - matched["_observed_month"]).to_numpy()
        return result

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


class EnhancedHistoryEncoder:
    """Causal weighted/windowed histories with optional hierarchical shrinkage."""

    def __init__(self, policy: HistoryPolicy):
        self.policy = policy

    def _prepare(self, frame):
        result = frame.copy()
        result["_ALL_KEY"] = "all"
        area = pd.to_numeric(result["FLOOR_AREA_SQM"], errors="coerce")
        result["_AREA_BIN_5"] = (area / 5).round().astype("Float64")
        return result

    def fit(self, frame):
        frame = self._prepare(frame)
        target = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(float)
        if not len(frame) or not np.isfinite(target).all():
            raise ValueError("Historical rents require nonempty, finite fitting targets")
        months = _month_numbers(frame)
        self.origin_ = int(months.max()) + 1
        self.reference_ = int(months.min())
        values = target.copy()
        if self.policy.value == "premium":
            reference = frame.groupby([DATE, "FLAT_TYPE_CLEAN"])[TARGET].transform("mean")
            values -= reference.to_numpy(float)
        elif self.policy.value == "ppsm":
            area = pd.to_numeric(frame["FLOOR_AREA_SQM"], errors="coerce").to_numpy(float)
            values = np.divide(target, area, out=np.full(len(frame), np.nan), where=area > 0)
        elif self.policy.value != "rent":
            raise ValueError(f"Unknown historical value: {self.policy.value}")
        self.grains_ = {
            "GLOBAL": ("_ALL_KEY",), "TYPE": ("FLAT_TYPE_CLEAN",), **HISTORY_GRAINS,
        }
        if self.policy.coarse_signature:
            self.grains_["COARSE_SIGNATURE"] = tuple(
                "_AREA_BIN_5" if column == "FLOOR_AREA_SQM" else column
                for column in HISTORY_GRAINS["SIGNATURE"]
            )
        self.tables_ = {}
        for grain, columns in self.grains_.items():
            data = pd.DataFrame({
                "_key": _group_keys(frame, columns).to_numpy(), "_month": months, "_value": values,
            }).dropna(subset=["_value"])
            monthly = data.groupby(["_key", "_month"], as_index=False, sort=True).agg(
                _sum=("_value", "sum"), _count=("_value", "size"), _last_mean=("_value", "mean"),
            )
            weight = np.ones(len(monthly)) if self.policy.half_life_months is None else np.exp2(
                (monthly["_month"].to_numpy() - self.reference_) / self.policy.half_life_months
            )
            monthly["_weighted_sum"] = monthly["_sum"] * weight
            monthly["_weighted_count"] = monthly["_count"] * weight
            grouped = monthly.groupby("_key", sort=False)
            monthly["_total_sum"] = grouped["_weighted_sum"].cumsum()
            monthly["_total_count"] = grouped["_weighted_count"].cumsum()
            self.tables_[grain] = monthly.sort_values("_month", kind="stable")
        return self

    def _query(self, frame, grain, months):
        cutoff = np.minimum(months, self.origin_)
        query = pd.DataFrame({
            "_key": _group_keys(frame, self.grains_[grain]).to_numpy(), "_cutoff": cutoff,
            "_position": np.arange(len(frame)),
        }).sort_values("_cutoff", kind="stable")
        table = self.tables_[grain].rename(columns={"_month": "_observed_month"})
        matched = pd.merge_asof(
            query, table, left_on="_cutoff", right_on="_observed_month", by="_key",
            direction="backward", allow_exact_matches=False,
        ).sort_values("_position", kind="stable")
        numerator = matched["_total_sum"].fillna(0).to_numpy(float, copy=True)
        denominator = matched["_total_count"].fillna(0).to_numpy(float, copy=True)
        if self.policy.window_months is not None:
            lower = query.copy()
            lower["_cutoff"] -= self.policy.window_months
            lower = pd.merge_asof(
                lower, table, left_on="_cutoff", right_on="_observed_month", by="_key",
                direction="backward", allow_exact_matches=False,
            ).sort_values("_position", kind="stable")
            numerator -= lower["_total_sum"].fillna(0).to_numpy(float)
            denominator -= lower["_total_count"].fillna(0).to_numpy(float)
        denominator = np.maximum(denominator, 0)
        mean = np.divide(numerator, denominator, out=np.full(len(frame), np.nan), where=denominator > 0)
        scale = 1 if self.policy.half_life_months is None else np.exp2(
            (cutoff - self.reference_) / self.policy.half_life_months
        )
        count = denominator / scale
        last_mean = matched["_last_mean"].to_numpy(float, copy=True)
        last_count = matched["_count"].fillna(0).to_numpy(float, copy=True)
        recency = months - matched["_observed_month"].to_numpy(float)
        no_history = denominator <= 0
        last_mean[no_history] = np.nan
        last_count[no_history] = 0
        recency[no_history] = np.nan
        return {"MEAN": mean, "COUNT": count, "LAST_MEAN": last_mean,
                "RECENCY_MONTHS": recency, "_last_count": last_count}

    def transform(self, frame):
        if not hasattr(self, "tables_"):
            raise ValueError("Fit EnhancedHistoryEncoder before transforming")
        if frame.empty:
            return pd.DataFrame(index=frame.index, columns=self.policy.columns, dtype=float)
        frame = self._prepare(frame)
        months = _month_numbers(frame)
        stats = {grain: self._query(frame, grain, months) for grain in self.grains_}
        parents = {"TYPE": "GLOBAL", "AREA_FLAT": "TYPE", "BLOCK_FLAT": "AREA_FLAT",
                   "SIGNATURE": "BLOCK_FLAT", "COARSE_SIGNATURE": "BLOCK_FLAT"}
        # Parents are shrunk first; missing groups inherit the available broader history.
        if self.policy.smoothing > 0:
            alpha = self.policy.smoothing
            for grain in ("TYPE", "AREA_FLAT", "BLOCK_FLAT", "SIGNATURE", "COARSE_SIGNATURE"):
                if grain not in stats:
                    continue
                current, parent = stats[grain], stats[parents[grain]]["MEAN"]
                for suffix, n in (("MEAN", current["COUNT"]), ("LAST_MEAN", current["_last_count"])):
                    observed = current[suffix]
                    smoothed = (np.nan_to_num(observed) * n + alpha * parent) / (n + alpha)
                    current[suffix] = np.where(np.isfinite(parent), smoothed, observed)
        result = pd.DataFrame(index=frame.index)
        for grain in self.policy.grains:
            for suffix in self.policy.statistics:
                result[f"{self.policy.prefix}_{grain}_{suffix}"] = stats[grain][suffix]
        return result


class HistoryBundle:
    def __init__(self, spec: FeatureSpec):
        self.spec = spec

    def fit(self, frame):
        policies = self.spec.history_policies or ("standard",)
        self.encoders_ = [
            (HistoricalRentEncoder() if policy == "standard" else EnhancedHistoryEncoder(HISTORY_POLICIES[policy])).fit(frame)
            for policy in policies
        ]
        return self

    def transform(self, frame):
        result = pd.concat([encoder.transform(frame) for encoder in self.encoders_], axis=1)
        return result[list(self.spec.history_columns)]


class MarketContextEncoder:
    """Past-only flat-type market summaries and a fixed damped log trend.

    The slope uses at most six observed months, is capped at +/-2% in log
    units per month, and damps by .85 for each additional forecast month.
    These constants are hypotheses, not fitted using validation labels.
    """

    def fit(self, frame):
        data = pd.DataFrame({
            "_type": frame["FLAT_TYPE_CLEAN"].astype(str).to_numpy(),
            "_month": _month_numbers(frame), "_rent": frame[TARGET].to_numpy(float),
        })
        self.monthly_ = data.groupby(["_type", "_month"])["_rent"].mean().reset_index()
        self.global_ = data.groupby("_month")["_rent"].mean().reset_index()
        self.origin_ = int(data["_month"].max()) + 1
        return self

    def transform(self, frame):
        if not hasattr(self, "monthly_"):
            raise ValueError("Fit MarketContextEncoder before transforming")
        months = _month_numbers(frame)
        query = pd.DataFrame({"_type": frame["FLAT_TYPE_CLEAN"].astype(str).to_numpy(), "_month": months})
        unique = query.drop_duplicates()
        rows = []
        for kind, month in unique.itertuples(index=False, name=None):
            cutoff = min(month, self.origin_)
            past = self.monthly_.loc[(self.monthly_["_type"] == kind) & (self.monthly_["_month"] < cutoff)]
            if past.empty:
                past = self.global_.loc[self.global_["_month"] < cutoff]
            past = past.sort_values("_month")
            if past.empty:
                values = [np.nan] * len(MARKET_COLUMNS)
            else:
                latest_month = int(past["_month"].iloc[-1])
                last = float(past["_rent"].iloc[-1])
                horizon = int(month - latest_month)
                recent = past.loc[past["_month"] >= latest_month - 5]
                slope = 0.0
                if len(recent) >= 3:
                    slope = float(np.clip(np.polyfit(recent["_month"] - latest_month,
                        np.log(recent["_rent"]), 1)[0], -.02, .02))
                projected = last * np.exp(slope * (1 - .85 ** horizon) / (1 - .85))
                values = [last,
                    float(past.loc[past["_month"] >= latest_month - 2, "_rent"].mean()),
                    float(past.loc[past["_month"] >= latest_month - 11, "_rent"].mean()),
                    slope, projected, horizon]
            rows.append({"_type": kind, "_month": month, **dict(zip(MARKET_COLUMNS, values))})
        if not rows:
            return pd.DataFrame(index=frame.index, columns=MARKET_COLUMNS, dtype=float)
        matched = query.merge(pd.DataFrame(rows), on=["_type", "_month"], how="left", validate="many_to_one", sort=False)
        matched.index = frame.index
        return matched[list(MARKET_COLUMNS)]


def model_inputs(fitting: pd.DataFrame, prediction: pd.DataFrame, spec: FeatureSpec):
    """Fit category vocabularies and histories exclusively on fitting rows."""
    x_fit = fitting[list(spec.columns)].copy()
    x_pred = prediction[list(spec.columns)].copy()
    if spec.use_history:
        encoder = HistoryBundle(spec).fit(fitting)
        for x, source in ((x_fit, fitting), (x_pred, prediction)):
            histories = encoder.transform(source)
            for column in spec.history_columns:
                x[column] = histories[column].to_numpy()
    if spec.market_context:
        market = MarketContextEncoder().fit(fitting)
        for x, source in ((x_fit, fitting), (x_pred, prediction)):
            context = market.transform(source)
            for column in MARKET_COLUMNS:
                x[column] = context[column].to_numpy()
    for column in spec.categorical:
        levels = pd.Index(x_fit[column].dropna().astype(str).unique())
        dtype = pd.CategoricalDtype(categories=levels)
        x_fit[column] = x_fit[column].astype("string").astype(dtype)
        values = x_pred[column].astype("string")
        x_pred[column] = values.where(values.isin(levels)).astype(dtype)
    return x_fit, x_pred


class RentalFeatureEngineer(TransformerMixin, BaseEstimator):
    """Persist a feature policy, category vocabulary and fitted rent history.

    Use inside a sklearn Pipeline, or call fit_transform(train, y) followed by
    transform(test). Target values supplied in transform inputs are ignored.
    """

    def __init__(self, variant="compact_history_27"):
        self.variant = variant

    def _fit_frame(self, frame, y):
        if self.variant not in FEATURE_CANDIDATES:
            raise ValueError(f"Unknown feature policy: {self.variant}")
        self.spec_ = FEATURE_CANDIDATES[self.variant]
        if y is not None:
            if len(y) != len(frame):
                raise ValueError("Feature and target lengths differ")
            if isinstance(y, pd.Series) and not y.index.equals(frame.index):
                raise ValueError("Feature and target indices must match")
            frame = frame.copy()
            frame[TARGET] = np.asarray(y, dtype=float)
        self.categories_ = {
            column: pd.Index(frame[column].dropna().astype(str).unique())
            for column in self.spec_.categorical
        }
        self.history_ = HistoryBundle(self.spec_).fit(frame) if self.spec_.use_history else None
        self.market_ = MarketContextEncoder().fit(frame) if self.spec_.market_context else None
        return self

    def _transform_frame(self, frame):
        if not hasattr(self, "spec_"):
            raise ValueError("Fit RentalFeatureEngineer before transforming")
        result = frame[list(self.spec_.columns)].copy()
        if self.history_ is not None:
            histories = self.history_.transform(frame)
            for column in self.spec_.history_columns:
                result[column] = histories[column].to_numpy()
        if self.market_ is not None:
            context = self.market_.transform(frame)
            for column in MARKET_COLUMNS:
                result[column] = context[column].to_numpy()
        for column, levels in self.categories_.items():
            values = result[column].astype("string")
            result[column] = values.where(values.isin(levels)).astype(pd.CategoricalDtype(categories=levels))
        return result

    def fit(self, X, y=None):
        return self._fit_frame(build_feature_frame(X), y)

    def fit_transform(self, X, y=None, **fit_params):
        frame = build_feature_frame(X)
        self._fit_frame(frame, y)
        return self._transform_frame(frame)

    def transform(self, X):
        return self._transform_frame(build_feature_frame(X))

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "spec_"):
            raise ValueError("Fit RentalFeatureEngineer before requesting feature names")
        return np.asarray(self.spec_.all_columns, dtype=object)
