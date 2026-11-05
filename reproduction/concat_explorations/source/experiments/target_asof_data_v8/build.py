"""Build the target-month as-of v8 dataset.

One row per transaction. For a transaction approved in month `t`, the only external inputs
are the calendar months `[t-12, ..., t-1]` of DBS/COE aggregates. Nothing from month `t` or
later enters. The target is `MONTHLY_RENT / 1000`.

No rent history, no anchor, no random horizon: those are the things this dataset exists to
avoid, so they are absent by construction rather than by omission.

## Time-availability rules

*DBS*: a daily row's `DATE` decides availability. A month's aggregates use only rows whose
date lies inside that month, so a month's series is complete at that month's end.

*COE*: the source has `YEAR/MONTH/ROUND` only, with no publication timestamp. The declared
experimental assumption is that a month's premiums become known at that month's end and are
therefore usable from the following month. This is an **assumption, not a verified fact**,
and is recorded as such in the manifest.

## Masks

Three different things are kept distinct:

* `observed` — the value exists for that (window, step, channel).
* `step_valid` — the step is inside the modelled calendar window. Leading padding before the
  series starts is 0; an interior missing month is still 1, so the time axis is never
  compressed.
* a sample whose window is entirely padding keeps its row and gets the learned missing
  vector, rather than being dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.target_asof_data_v8 import features as F

DBS_SYMBOL = "D05.SI"
MAX_CALENDAR_GAP_DAYS = 5


class AsOfDataError(ValueError):
    pass


# --------------------------------------------------------------------------- market series
class MarketSeries:
    """Per-month market aggregates over a contiguous month index.

    `values[m, c]` is channel `c` for month `m`, NaN where unobserved. `observed` mirrors it
    as a boolean so a genuine 0 is never confused with a missing value.
    """

    def __init__(self, channels, values, observed, n_months):
        self.channels = list(channels)
        self.values = np.asarray(values, dtype=np.float64)
        self.observed = np.asarray(observed, dtype=bool)
        self.n_months = int(n_months)

    def copy_with_spike(self, from_month: int, magnitude: float = 1e6) -> "MarketSeries":
        """Test helper: poison every month >= `from_month` with an extreme value."""
        values = self.values.copy()
        observed = self.observed.copy()
        values[from_month:, :] = magnitude
        observed[from_month:, :] = True
        return MarketSeries(self.channels, values, observed, self.n_months)

    def copy_with_gap(self, month: int) -> "MarketSeries":
        """Test helper: mark one month as entirely unobserved."""
        values = self.values.copy()
        observed = self.observed.copy()
        values[month, :] = np.nan
        observed[month, :] = False
        return MarketSeries(self.channels, values, observed, self.n_months)


def _month_index_from_period(period_index: pd.PeriodIndex, first: pd.Period) -> np.ndarray:
    return np.asarray([(p - first).n for p in period_index], dtype=np.int64)


def build_market_series(stock: pd.DataFrame, coe: pd.DataFrame,
                        last_month: int) -> MarketSeries:
    """Aggregate DBS and COE into one channel matrix over months 0..last_month."""
    n_months = int(last_month) + 1
    channels = F.MARKET_CHANNEL_ORDER
    values = np.full((n_months, len(channels)), np.nan)
    observed = np.zeros((n_months, len(channels)), dtype=bool)
    position = {name: i for i, name in enumerate(channels)}

    # ------------------------------------------------------------------ DBS
    dbs = stock.loc[stock["SYMBOL"] == DBS_SYMBOL].copy()
    if dbs.empty:
        raise AsOfDataError(f"no rows for {DBS_SYMBOL}")
    dbs["DATE"] = pd.to_datetime(dbs["DATE"], errors="raise")
    dbs = dbs.sort_values("DATE").reset_index(drop=True)
    dbs["month"] = pd.PeriodIndex(dbs["DATE"], freq="M")

    close = pd.to_numeric(dbs["ADJUSTED_CLOSE"], errors="coerce").to_numpy(float)
    volume = pd.to_numeric(dbs["VOLUME"], errors="coerce").to_numpy(float)
    valid_close = np.isfinite(close) & (close > 0)

    # daily log return, masked across a multi-row or multi-day gap
    log_return = np.full(len(dbs), np.nan)
    last_valid_index = None
    for i in range(len(dbs)):
        if not valid_close[i]:
            continue
        if last_valid_index is not None:
            step_rows = i - last_valid_index
            gap_days = (dbs["DATE"].iloc[i] - dbs["DATE"].iloc[last_valid_index]).days
            if step_rows == 1 and gap_days <= MAX_CALENDAR_GAP_DAYS:
                log_return[i] = float(np.log(close[i] / close[last_valid_index]))
            # otherwise leave NaN: crossing a gap is not a one-day return
        last_valid_index = i
    dbs["log_return"] = log_return
    # per-day volume is kept raw; the monthly channel is log1p of the monthly MEAN
    dbs["volume_raw"] = np.where(np.isfinite(volume) & (volume >= 0), volume, np.nan)

    first_period = dbs["month"].min()

    # ------------------------------------------------------------------ month-end closes
    # `dbs_log_return` is the plan's **month-end to month-end** log ratio, NOT the sum of
    # daily returns. Summing daily returns across a month that contains masked gaps would
    # silently drop those days and understate the move, and it is a different quantity from
    # the one the schema declares. The earlier implementation summed dailies; corrected here.
    month_end_close = {}
    month_last_date = {}
    for month, group in dbs.groupby("month"):
        closes = group.loc[group["ADJUSTED_CLOSE"].notna()
                           & (group["ADJUSTED_CLOSE"] > 0)]
        if closes.empty:
            continue
        last_row = closes.iloc[-1]
        month_end_close[month] = float(last_row["ADJUSTED_CLOSE"])
        month_last_date[month] = last_row["DATE"]

    for month, group in dbs.groupby("month"):
        index = (month - first_period).n
        if index < 0 or index >= n_months:
            continue
        # month-end log return: missing when the previous month has no usable close
        previous = month - 1
        current_close = month_end_close.get(month)
        previous_close = month_end_close.get(previous)
        if current_close is not None and previous_close is not None:
            values[index, position["dbs_log_return"]] = float(
                np.log(current_close / previous_close))
            observed[index, position["dbs_log_return"]] = True

        # within-month daily return dispersion, still a legitimate monthly summary and
        # independent of the month-end level
        returns = group["log_return"].to_numpy(float)
        finite_returns = returns[np.isfinite(returns)]
        if finite_returns.size >= 2:
            values[index, position["dbs_daily_vol_std"]] = float(finite_returns.std(ddof=1))
            observed[index, position["dbs_daily_vol_std"]] = True

        # Volume: the schema says "log1p monthly MEAN volume", i.e. log1p(mean(volume)),
        # not mean(log1p(volume)). Those differ because log is concave. Implemented as
        # declared so the artefact matches its own name.
        raw_volume = group["VOLUME"].to_numpy(float)
        usable_volume = raw_volume[np.isfinite(raw_volume) & (raw_volume >= 0)]
        if usable_volume.size:
            values[index, position["dbs_log1p_volume"]] = float(
                np.log1p(usable_volume.mean()))
            observed[index, position["dbs_log1p_volume"]] = True

    # ------------------------------------------------------------------ COE
    frame = coe.copy()
    frame["month"] = pd.PeriodIndex(
        frame["YEAR"].astype(int).astype(str) + "-"
        + frame["MONTH"].astype(str).str.lower().str.strip(),
        freq="M")
    # Thousands separators are real values, not missing data. `str.replace("[$,]", ...)`
    # handles "", "$" and ","; plain pd.to_numeric would silently turn "1,023" into NaN
    # (47 such quota rows in this source).
    def _numeric(column):
        return pd.to_numeric(
            frame[column].astype(str).str.replace("[$,]", "", regex=True),
            errors="coerce")

    frame["premium"] = _numeric("QUOTA_PREMIUM")
    frame["quota"] = _numeric("QUOTA")
    frame["bids"] = _numeric("BIDS_RECEIVED")

    for month, group in frame.groupby("month"):
        index = (month - first_period).n
        if index < 0 or index >= n_months:
            continue
        for cat in F.COE_CATEGORIES:
            cat_rows = group.loc[group["CATEGORY"].astype(str).str.lower() == cat]
            if cat_rows.empty:
                continue
            premiums = cat_rows["premium"].to_numpy(float)
            finite_premiums = premiums[np.isfinite(premiums)]
            if finite_premiums.size:
                values[index, position[f"coe_{cat}_premium_mean"]] = \
                    float(finite_premiums.mean())
                observed[index, position[f"coe_{cat}_premium_mean"]] = True
            # Quota and bids must come from the SAME set of valid auctions, or the ratio
            # silently divides one round's bids by another round's quota.
            both_valid = cat_rows.loc[
                np.isfinite(cat_rows["quota"].to_numpy(float))
                & np.isfinite(cat_rows["bids"].to_numpy(float))]
            if not both_valid.empty:
                total_quota = float(both_valid["quota"].to_numpy(float).sum())
                values[index, position[f"coe_{cat}_quota_total"]] = total_quota
                observed[index, position[f"coe_{cat}_quota_total"]] = True
                if total_quota > 0:
                    values[index, position[f"coe_{cat}_bids_over_quota"]] = \
                        float(both_valid["bids"].to_numpy(float).sum() / total_quota)
                    observed[index, position[f"coe_{cat}_bids_over_quota"]] = True

    return MarketSeries(channels, values, observed, n_months)


# --------------------------------------------------------------------------- windows
class Window:
    """The 12 calendar months preceding a target month."""

    __slots__ = ("target_month", "month_index", "values", "observed", "step_valid")

    def __init__(self, target_month, month_index, values, observed, step_valid):
        self.target_month = int(target_month)
        self.month_index = np.asarray(month_index, dtype=np.int32)
        self.values = np.asarray(values, dtype=np.float64)
        self.observed = np.asarray(observed, dtype=bool)
        self.step_valid = np.asarray(step_valid, dtype=bool)


def window_for_target(market: MarketSeries, target_month: int) -> Window:
    """Calendar months t-12 .. t-1, oldest to newest. Month t is never read."""
    months = np.arange(target_month - F.WINDOW_MONTHS, target_month, dtype=np.int32)
    values = np.full((F.WINDOW_MONTHS, len(market.channels)), np.nan)
    observed = np.zeros((F.WINDOW_MONTHS, len(market.channels)), dtype=bool)
    step_valid = np.zeros(F.WINDOW_MONTHS, dtype=bool)
    for step, month in enumerate(months):
        if month < 0:
            # leading padding: not part of the series yet, so the step is invalid and the
            # encoder must skip it (never treat it as an observed zero)
            continue
        step_valid[step] = True
        if month >= market.n_months:
            continue
        values[step] = market.values[month]
        observed[step] = market.observed[month]
    return Window(target_month, months, values, observed, step_valid)


def window_stats(window: Window) -> np.ndarray:
    """Simple statistics derived from the SAME window, one row per channel.

    Columns follow `features.stats_feature_names()`.

    **Windows here are CALENDAR windows, not "last N observations".** `mean3` is the mean of
    the calendar months `t-3..t-1`, and `recent3_minus_prior3` compares `t-3..t-1` against
    `t-6..t-4`. Taking "the last three observed values" would silently stretch those
    windows across a gap and report a different quantity than the name promises — and it
    would disagree with the neural stats arm, which computes its own calendar-window stats.
    A mean over a window with no observation is missing, never an average of older data.
    """
    n_channels = window.values.shape[1]
    n_steps = window.values.shape[0]
    out = np.full((n_channels, len(F.STATS_SUFFIXES)), np.nan)
    i_last = F.STATS_SUFFIXES.index("last_valid")
    i_mean3 = F.STATS_SUFFIXES.index("mean3")
    i_mean12 = F.STATS_SUFFIXES.index("mean12")
    i_delta = F.STATS_SUFFIXES.index("recent3_minus_prior3")
    i_nvalid = F.STATS_SUFFIXES.index("n_valid")
    i_gap = F.STATS_SUFFIXES.index("months_since_last_valid")

    for c in range(n_channels):
        series = window.values[:, c]
        mask = window.observed[:, c] & np.isfinite(series)
        valid_positions = np.flatnonzero(mask)

        # last valid value and months since it are inherently "latest observation" statistics
        if valid_positions.size:
            last = valid_positions[-1]
            out[c, i_last] = series[last]
            out[c, i_gap] = float(n_steps - 1 - last)
        out[c, i_nvalid] = float(mask.sum())
        if not valid_positions.size:
            continue

        def calendar_mean(start: int, stop: int) -> float:
            """Mean over calendar steps [start, stop); NaN if none observed."""
            low = max(0, start)
            high = min(n_steps, stop)
            if low >= high:
                return float("nan")
            block = series[low:high]
            block_mask = mask[low:high]
            if not block_mask.any():
                return float("nan")
            return float(block[block_mask].mean())

        out[c, i_mean3] = calendar_mean(n_steps - 3, n_steps)
        out[c, i_mean12] = calendar_mean(0, n_steps)
        recent = calendar_mean(n_steps - 3, n_steps)
        prior = calendar_mean(n_steps - 6, n_steps - 3)
        if np.isfinite(recent) and np.isfinite(prior):
            out[c, i_delta] = recent - prior
    return out


# --------------------------------------------------------------------------- scalers
class ChannelScaler:
    """Per-channel standardisation fitted on training months only.

    One row per month, not one row per transaction: a month with many transactions must not
    dominate the statistics. Fitted on valid values only; after scaling, missing becomes 0
    and the observation mask still records it.
    """

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def fit(cls, market: MarketSeries, months) -> "ChannelScaler":
        months = sorted({int(m) for m in months if 0 <= int(m) < market.n_months})
        if not months:
            raise AsOfDataError("no training months to fit the market scaler")
        block = market.values[months, :]
        observed = market.observed[months, :]
        mean = np.zeros(block.shape[1])
        std = np.ones(block.shape[1])
        for c in range(block.shape[1]):
            valid = block[:, c][observed[:, c] & np.isfinite(block[:, c])]
            if valid.size:
                mean[c] = float(valid.mean())
                std[c] = float(valid.std(ddof=0)) or 1.0
        scaler = cls(mean, std)
        scaler.fitted_months = list(months)
        scaler.n_months_used = len(months)
        return scaler

    def transform(self, values: np.ndarray, observed: np.ndarray):
        scaled = (values - self.mean) / self.std
        scaled = np.where(observed & np.isfinite(scaled), scaled, 0.0)
        return scaled


class NumericScaler:
    """Per-column standardisation for the house branch, fitted on training rows only."""

    def __init__(self, median, mean, std):
        self.median = np.asarray(median, dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "NumericScaler":
        block = frame[F.HOUSE_NUMERIC_ORDER].to_numpy(dtype=np.float64)
        median = np.zeros(block.shape[1])
        mean = np.zeros(block.shape[1])
        std = np.ones(block.shape[1])
        for c in range(block.shape[1]):
            column = block[:, c]
            valid = column[np.isfinite(column)]
            if valid.size:
                median[c] = float(np.median(valid))
                mean[c] = float(valid.mean())
                std[c] = float(valid.std(ddof=0)) or 1.0
        return cls(median, mean, std)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        block = frame[F.HOUSE_NUMERIC_ORDER].to_numpy(dtype=np.float64)
        # median-impute first so a missing attribute becomes the training midpoint rather
        # than a large negative z-score
        block = np.where(np.isfinite(block), block, self.median)
        return ((block - self.mean) / self.std).astype(np.float32)
