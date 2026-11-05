"""Origin-safe features and small randomly initialized rental transformers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn

from analysis.experiments.aligned_temporal_features import (
    FEATURES, HISTORY_COLUMNS, LocalTemporalModel, origin_history,
)

RENT_CHANNELS = ["national_relative", "national_change", "type_relative", "type_change",
                 "area_type_relative", "area_type_change", "national_support", "type_support",
                 "area_type_support", "national_age", "type_age", "area_type_age"]
MACRO_CHANNELS = ["coe_a_log", "coe_a_change", "coe_c_log", "coe_c_change", "stock_return"]
SEQUENCE_CHANNELS = RENT_CHANNELS + MACRO_CHANNELS


@dataclass(frozen=True)
class Experiment:
    name: str
    information: str = "rent"
    processor: str = "transformer"
    window: int = 12
    delay: int = 0

    def settings(self):
        return asdict(self)


PRIMARY = [Experiment("T0", "none", "none"), Experiment("T1"),
           Experiment("T2", "rent_coe"), Experiment("T3", "rent_coe_stock")]
CONTROLS = [Experiment("C1", "rent", "lstm"), Experiment("C2", "rent_coe", "lstm"),
            Experiment("C3", "rent", "linear"), Experiment("C4", "rent_coe", "linear")]
SENSITIVITIES = [Experiment("S1", window=6), Experiment("S2", "rent_coe", window=6),
                 Experiment("S3", delay=1), Experiment("S4", "rent_coe", delay=1)]
EXPERIMENTS = {s.name: s for s in PRIMARY + CONTROLS + SENSITIVITIES}


def episode_origins(months, row_ids, first="2021-01", max_horizon=16):
    """Deterministic, label-independent episodes; preserve a year when possible."""
    periods = pd.PeriodIndex(months, freq="M")
    available = periods.asi8 - pd.Period(first, "M").ordinal
    if (available < 0).any():
        raise ValueError("Transaction precedes history")
    maximum = np.minimum(max_horizon, np.where(available > 11, available - 11, available))
    horizon = 1 + (np.asarray(row_ids, dtype=np.int64) * 2654435761 + 42) % np.maximum(maximum, 1)
    values = periods.asi8 - horizon
    values[available == 0] = pd.NaT.value
    return pd.PeriodIndex.from_ordinals(values, freq="M")


def rebuild_histories(template, raw_frame, origins):
    """All 46 target-derived inputs share each row's own information cutoff.

    The existing forecast is held at horizon ten beyond its validated range;
    actual horizon and observation age continue increasing through month 16.
    """
    result = template.copy()
    result.loc[:, HISTORY_COLUMNS + FEATURES] = np.nan
    baseline = np.zeros(len(result), dtype=float)
    origins = pd.PeriodIndex(origins, freq="M")
    if len(origins) != len(result):
        raise ValueError("Origin alignment changed")
    distinct = sorted(origins.dropna().unique())
    for number, origin in enumerate(distinct, 1):
        positions = np.flatnonzero(origins == origin)
        ids = result.index[positions]
        past = raw_frame[raw_frame._MONTH <= origin]
        blind = raw_frame.loc[ids].drop(columns="MONTHLY_RENT")
        if past.empty or blind._MONTH.min() <= origin:
            raise ValueError("Noncausal history request")
        result.loc[ids, HISTORY_COLUMNS] = origin_history(past, blind).reindex(ids).to_numpy()
        horizon = blind._MONTH.array.asi8 - origin.ordinal
        bounded = blind.copy()
        bounded["_MONTH"] = pd.PeriodIndex.from_ordinals(
            origin.ordinal + np.minimum(horizon, 10), freq="M")
        values = LocalTemporalModel(past).features(bounded, origin).reindex(ids)
        values["TEMP_FORECAST_HORIZON"] = horizon
        values["TEMP_FLAT_LAST_OBSERVATION_AGE"] += np.maximum(horizon - 10, 0)
        result.loc[ids, FEATURES] = values.to_numpy()
        local = values.TEMP_LOCAL_FORECAST.to_numpy(float)
        # Match the existing source pipeline's neutral cold-start convention.
        baseline[positions] = np.where(np.isfinite(local), local, 0.)
        if number % 10 == 0 or number == len(distinct):
            print(f"HISTORIES {number}/{len(distinct)} through {origin}", flush=True)
    if not np.isfinite(baseline).all():
        raise ValueError("Nonfinite forecast baseline")
    return result, baseline


def adjusted_rents(frame):
    """Fixed 2021 hedonic reference, unavailable at origins before 2021-12."""
    fit = frame._MONTH <= pd.Period("2021-12", "M")
    categorical = ["BLOCK_ID", "FLAT_TYPE_CLEAN", "FLAT_MODEL_CLEAN"]
    numeric = ["FLOOR_AREA_SQM", "LEASE_COMMENCE_DATE"]
    x = frame[categorical + numeric].copy()
    x[categorical] = x[categorical].astype(str)
    transform = ColumnTransformer([
        ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric)])
    model = make_pipeline(transform, Ridge(alpha=10.))
    model.fit(x.loc[fit], frame.loc[fit, "MONTHLY_RENT"])
    property_prediction = model.predict(x)
    values = frame.MONTHLY_RENT.to_numpy(float) - property_prediction + property_prediction[fit].mean()
    return pd.Series(values, index=frame.index), model


class SequenceBank:
    """Monthly panels indexed by month, property type, and area; no future fills."""

    def __init__(self, frame, adjusted, macro):
        self.first = frame._MONTH.min()
        self.months = pd.period_range(self.first, frame._MONTH.max(), freq="M")
        self.types = sorted(frame.FLAT_TYPE_CLEAN.astype(str).unique())
        self.areas = sorted(frame.PLANNING_AREA.astype(str).unique())
        self.type_map = {v: i for i, v in enumerate(self.types)}
        self.area_map = {v: i for i, v in enumerate(self.areas)}
        f = frame.assign(_ADJUSTED=adjusted)
        self.panels = {}
        for key, group, columns in [
            ("national", [], pd.Index([0])),
            ("type", ["FLAT_TYPE_CLEAN"], pd.Index(self.types)),
            ("area_type", ["PLANNING_AREA", "FLAT_TYPE_CLEAN"],
             pd.MultiIndex.from_product([self.areas, self.types]))]:
            grouped = f.groupby(["_MONTH"] + group, observed=True)._ADJUSTED.agg(["sum", "count"])
            if group:
                total = grouped["sum"].unstack(group).reindex(index=self.months, columns=columns)
                count = grouped["count"].unstack(group).reindex(index=self.months, columns=columns).fillna(0)
            else:
                total = grouped["sum"].reindex(self.months).to_frame(0)
                count = grouped["count"].reindex(self.months).fillna(0).to_frame(0)
            self.panels[key] = (total.to_numpy(float), count.to_numpy(float))
        self.macro = macro.reindex(self.months)

    def sequences(self, query, origins, window=12, delay=0):
        """Macro has one extra publication-buffer month; rental labels do not.

        Dates, not row ordering, define every lag. An extra-delay sensitivity
        shifts the full rental information cutoff before calling this method.
        """
        origins = pd.PeriodIndex(origins, freq="M")
        n = len(query)
        if len(origins) != n:
            raise ValueError("Sequence/query index mismatch")
        out = np.full((n, window, len(SEQUENCE_CHANNELS)), np.nan, np.float32)
        end = origins.asi8 - self.first.ordinal
        end = np.where(origins.isna(), -1000, end)
        offsets = np.arange(window - 1, -1, -1)
        indices = end[:, None] - offsets
        valid = (indices >= 0) & (indices < len(self.months))
        safe = np.clip(indices, 0, len(self.months) - 1)
        type_id = query.FLAT_TYPE_CLEAN.astype(str).map(self.type_map).fillna(-1).to_numpy(int)
        area_id = query.PLANNING_AREA.astype(str).map(self.area_map).fillna(-1).to_numpy(int)
        known = (type_id >= 0) & (area_id >= 0)
        group_ids = [np.zeros(n, int), np.maximum(type_id, 0),
                     np.maximum(area_id, 0) * len(self.types) + np.maximum(type_id, 0)]
        means, counts = [], []
        for key, ids in zip(("national", "type", "area_type"), group_ids):
            total, count = self.panels[key]
            c = count[safe, ids[:, None]]
            value = total[safe, ids[:, None]] / np.maximum(c, 1)
            value[c == 0] = np.nan
            value[~valid] = np.nan
            c = np.where(valid, c, 0)
            if key != "national":
                value[~known] = np.nan
                c[~known] = 0
            means.append(value)
            counts.append(c)
        # Pool sparse local cells toward type and national monthly means.
        means[1] = (np.nan_to_num(means[1]) * counts[1] + 30 * means[0]) / (counts[1] + 30)
        means[2] = (np.nan_to_num(means[2]) * counts[2] + 30 * means[1]) / (counts[2] + 30)
        warm = end >= pd.Period("2021-12", "M").ordinal - self.first.ordinal
        for g in range(3):
            values = means[g]
            anchor = values[:, -1:]
            out[:, :, 2*g] = (values - anchor) / 1000
            # First token's change is missing; no access to a later token.
            out[:, 1:, 2*g+1] = np.diff(values, axis=1) / 1000
            out[:, :, 6+g] = np.log1p(counts[g])
            seen = np.maximum.accumulate(np.where(counts[g] > 0, indices, -10000), axis=1)
            out[:, :, 9+g] = np.where(seen > -10000, indices - seen, np.nan)
        out[~warm, :, :len(RENT_CHANNELS)] = np.nan
        macro_indices = indices - 1  # one additional macro publication month
        mvalid = (macro_indices >= 0) & (macro_indices < len(self.months))
        msafe = np.clip(macro_indices, 0, len(self.months) - 1)
        macro = self.macro[MACRO_CHANNELS].to_numpy(float)[msafe]
        macro[~mvalid] = np.nan
        out[:, :, len(RENT_CHANNELS):] = macro
        return out


def macro_panel(coe, stocks):
    c = coe.copy()
    c["month"] = pd.to_datetime(c.YEAR.astype(str) + "-" + c.MONTH,
                                   format="%Y-%b").dt.to_period("M")
    c["premium"] = pd.to_numeric(c.QUOTA_PREMIUM.astype(str).str.replace(r"[$,]", "", regex=True))
    wide = c.groupby(["month", "CATEGORY"]).premium.median().unstack()
    months = pd.period_range(wide.index.min(), wide.index.max(), freq="M")
    out = pd.DataFrame(index=months)
    for category in ("a", "c"):
        level = np.log(wide[category].reindex(months))
        out[f"coe_{category}_log"] = level
        out[f"coe_{category}_change"] = level.diff()
    s = stocks[stocks.SYMBOL.str.endswith(".SI")].copy()
    s["month"] = pd.to_datetime(s.DATE).dt.to_period("M")
    prices = s.sort_values("DATE").groupby(["month", "SYMBOL"]).ADJUSTED_CLOSE.last().unstack().reindex(months)
    # Documented equal-weight/median snapshot proxy, not official STI.
    out["stock_return"] = np.log(prices).diff().median(axis=1)
    return out


class FeaturePreprocessor:
    """Fit-only feature statistics; numeric missingness stays within each token."""

    def fit(self, frame):
        self.columns = list(frame)
        self.categories = [c for c in frame if isinstance(frame[c].dtype, pd.CategoricalDtype)]
        self.numerical = [c for c in frame if c not in self.categories]
        raw = frame[self.numerical].to_numpy(float)
        if np.isinf(raw).any():
            raise ValueError("Infinite property feature")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median = np.nanmedian(raw, axis=0)
        self.median = np.nan_to_num(self.median)
        clean = np.where(np.isfinite(raw), raw, self.median)
        self.mean, self.sd = clean.mean(0), np.maximum(clean.std(0), 1e-6)
        self.vocabularies = [{v: i+1 for i, v in enumerate(frame[c].astype("string").dropna().unique())}
                             for c in self.categories]
        self.cardinalities = [len(v) + 1 for v in self.vocabularies]
        return self

    def transform(self, frame):
        if list(frame) != self.columns:
            raise ValueError("Property feature schema changed")
        raw = frame[self.numerical].to_numpy(float)
        if np.isinf(raw).any():
            raise ValueError("Infinite property feature")
        numeric = ((np.where(np.isfinite(raw), raw, self.median) - self.mean) / self.sd).astype(np.float32)
        missing = (~np.isfinite(raw)).astype(np.float32)
        cats = np.column_stack([frame[c].astype("string").map(v).fillna(0).to_numpy(np.int64)
                                for c, v in zip(self.categories, self.vocabularies)])
        xy = frame[["LATITUDE", "LONGITUDE"]].to_numpy(float)
        xy = np.column_stack([(xy[:, 1] - 103.85) * 111.2 * np.cos(np.deg2rad(1.35)),
                              (xy[:, 0] - 1.35) * 111.2])
        angles = 2 * np.pi * xy[:, :, None] / np.asarray([.25, .5, 1., 2., 4., 8., 16.])
        spatial = np.column_stack([xy/20, np.sin(angles).reshape(len(xy), -1),
                                   np.cos(angles).reshape(len(xy), -1)]).astype(np.float32)
        return numeric, missing, cats, spatial


class SequencePreprocessor:
    def fit(self, values):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean = np.nanmean(values, axis=(0, 1))
            self.sd = np.nanstd(values, axis=(0, 1))
        self.mean = np.nan_to_num(self.mean)
        self.sd = np.where(np.isfinite(self.sd) & (self.sd > 1e-6), self.sd, 1)
        return self

    def transform(self, values, information):
        clean = np.where(np.isfinite(values), values, self.mean)
        value = ((clean-self.mean)/self.sd).astype(np.float32)
        present = np.isfinite(values).astype(np.float32)
        keep = np.zeros(len(SEQUENCE_CHANNELS), dtype=bool)
        if information != "none":
            keep[:len(RENT_CHANNELS)] = True
        if "coe" in information:
            keep[len(RENT_CHANNELS):len(RENT_CHANNELS)+4] = True
        if "stock" in information:
            keep[-1] = True
        value[:, :, ~keep] = 0
        present[:, :, ~keep] = 0
        return np.concatenate([value, present], axis=-1)


def forecast_context(months, origins):
    months, origins = pd.PeriodIndex(months, freq="M"), pd.PeriodIndex(origins, freq="M")
    horizon = np.where(origins.isna(), 1, months.asi8-origins.asi8).astype(float)
    phase = months.month.to_numpy() * 2*np.pi/12
    return np.column_stack([horizon/16, np.log1p(horizon)/np.log(17),
                            np.sin(phase), np.cos(phase)]).astype(np.float32)


class RentalTransformer(nn.Module):
    def __init__(self, n_numeric, cardinalities, spec, d_model=128):
        super().__init__()
        self.spec = spec
        self.numeric_weight = nn.Parameter(torch.randn(n_numeric, d_model)*.02)
        self.numeric_bias = nn.Parameter(torch.randn(n_numeric, d_model)*.02)
        self.missing_weight = nn.Parameter(torch.randn(n_numeric, d_model)*.02)
        self.categories = nn.ModuleList([nn.Embedding(n, d_model) for n in cardinalities])
        self.cls = nn.Parameter(torch.randn(1, 1, d_model)*.02)
        self.spatial = nn.Linear(30, d_model)
        layer = nn.TransformerEncoderLayer(d_model, 4, 4*d_model, .1,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.property_encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.property_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.context = nn.Linear(4, 32)
        if spec.processor == "transformer":
            self.sequence_input = nn.Linear(2*len(SEQUENCE_CHANNELS), 32)
            self.position = nn.Parameter(torch.randn(1, spec.window, 32)*.02)
            temporal = nn.TransformerEncoderLayer(32, 2, 64, .1, batch_first=True,
                                                  norm_first=True, activation="gelu")
            self.sequence_encoder = nn.TransformerEncoder(temporal, 1, enable_nested_tensor=False)
            self.sequence_head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 1))
        elif spec.processor == "lstm":
            self.sequence_encoder = nn.LSTM(2*len(SEQUENCE_CHANNELS), 8, batch_first=True)
            self.sequence_head = nn.Linear(12, 1)
        elif spec.processor == "linear":
            self.sequence_head = nn.Linear(spec.window*2*len(SEQUENCE_CHANNELS)+4, 1)
        if spec.processor != "none":
            self.gate = nn.Parameter(torch.tensor(-2.2))
            nn.init.normal_(self.sequence_head[-1].weight if isinstance(self.sequence_head, nn.Sequential)
                            else self.sequence_head.weight, std=.001)

    def forward(self, numeric, missing, categorical, spatial, sequence, context):
        tokens = numeric[:, :, None]*self.numeric_weight + self.numeric_bias + missing[:, :, None]*self.missing_weight
        cats = [module(categorical[:, i])[:, None] for i, module in enumerate(self.categories)]
        tokens = torch.cat([self.cls.expand(len(numeric), -1, -1), tokens,
                            *cats, self.spatial(spatial)[:, None]], dim=1)
        result = self.property_head(self.property_encoder(tokens)[:, 0]).squeeze(-1)
        if self.spec.processor == "none":
            return result
        if self.spec.processor == "transformer":
            tokens = self.sequence_input(sequence) + self.position
            # The sequence contains only available history; a context token reads it.
            tokens = torch.cat([tokens, self.context(context)[:, None]], dim=1)
            temporal = self.sequence_head(self.sequence_encoder(tokens)[:, -1]).squeeze(-1)
        elif self.spec.processor == "lstm":
            _, (hidden, _) = self.sequence_encoder(sequence)
            temporal = self.sequence_head(torch.cat([hidden[-1], context], dim=1)).squeeze(-1)
        else:
            temporal = self.sequence_head(torch.cat([sequence.flatten(1), context], dim=1)).squeeze(-1)
        return result + self.gate.sigmoid()*temporal
