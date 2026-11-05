"""DBS daily financial-state branch (``D05.SI`` only).

Design decisions that matter for correctness
--------------------------------------------
* **Positive adjusted closes only.** Returns use ``ADJUSTED_CLOSE``; rows whose
  value is missing or non-positive are skipped when forming a return.
* **A multi-day gap is never a one-day return.** Each return records how many raw
  rows were skipped. When more than one row was skipped the return is *masked*
  rather than reported, because the log ratio across a gap is not a daily return
  and would silently look like a large one-day move. The two missing closes in
  this source (2026-05-01, 2026-06-01) therefore produce masked entries with the
  dates preserved.
* **Everything trailing looks backwards from the as-of date only.** No centred
  window, no forward fill, no interpolation.
* **Trading-observation windows, not calendar windows.** "1 month" is 21 valid
  trading observations, "3 months" is 63, "6 months" is 126. Calendar months have
  ~21 trading days but not exactly, and using calendar arithmetic would make the
  scalar definitions drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.fusion_contract_v3 import contract

GAP_MASK_STEP = 1          # a return is valid only when exactly one row was skipped
# A missing *row* (a trading day absent from the file) leaves the row step at 1 but
# opens a multi-day calendar gap.  Masking on row steps alone would report that as an
# ordinary one-day return, so both conditions are required.  5 calendar days covers a
# weekend plus a public holiday; the observed maximum in this source is 5.
MAX_CALENDAR_GAP_DAYS = 5


def parse_stock_frame(path=None) -> pd.DataFrame:
    """Load every stock row, preserving raw columns and dates."""
    path = path or contract.STOCK_SOURCE
    frame = pd.read_csv(path)
    required = ["SYMBOL", "DATE", "OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"stock source missing columns: {missing}")
    for column in ("OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="raise")
    if frame["DATE"].duplicated().any():
        # Duplicates are only a problem inside one symbol; check after filtering.
        pass
    return frame


@dataclass
class DBSFrame:
    """Daily rows for one symbol with causal trailing statistics attached."""

    dates: pd.DatetimeIndex
    log_close: np.ndarray          # NaN where the close is invalid
    log_return: np.ndarray         # NaN where undefined or across a multi-row gap
    step_rows: np.ndarray          # rows skipped since the previous valid close
    log_volume: np.ndarray
    gap_days: np.ndarray           # calendar days since the previous valid close
    vol_21: np.ndarray
    ma_dev_21: np.ndarray
    drawdown_126: np.ndarray
    observed: np.ndarray           # 1 where the close is valid

    def __len__(self) -> int:
        return len(self.dates)


def build_dbs_frame(frame: pd.DataFrame, symbol: str = contract.DBS_SYMBOL) -> DBSFrame:
    """Build the daily branch for one symbol, sorted by date."""
    subset = frame.loc[frame["SYMBOL"] == symbol].copy()
    if subset.empty:
        raise ValueError(f"no rows for symbol {symbol!r}")
    if subset["DATE"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate dates in the source")
    subset = subset.sort_values("DATE").reset_index(drop=True)

    adjusted = subset["ADJUSTED_CLOSE"].to_numpy(dtype=float)
    volume = subset["VOLUME"].to_numpy(dtype=float)
    valid = np.isfinite(adjusted) & (adjusted > 0)

    n = len(subset)
    log_close = np.where(valid, np.log(np.where(valid, adjusted, 1.0)), np.nan)
    log_volume = np.where(np.isfinite(volume) & (volume >= 0),
                          np.log1p(np.maximum(volume, 0.0)), np.nan)

    # --- returns, with explicit gap accounting -------------------------
    dates = pd.DatetimeIndex(subset["DATE"])
    log_return = np.full(n, np.nan)
    step_rows = np.zeros(n, dtype=np.int64)
    gap_days = np.zeros(n, dtype=np.int64)
    last_valid = -1
    for index in range(n):
        if not valid[index]:
            continue
        if last_valid >= 0:
            rows_skipped = index - last_valid
            step_rows[index] = rows_skipped
            calendar_days = int((dates[index] - dates[last_valid]).days)
            gap_days[index] = calendar_days
            raw = log_close[index] - log_close[last_valid]
            # Valid only when exactly one row was skipped AND the calendar step is
            # an ordinary one-day move (weekend/holiday allowed, longer is masked).
            if rows_skipped == GAP_MASK_STEP and calendar_days <= MAX_CALENDAR_GAP_DAYS:
                log_return[index] = raw
            else:
                log_return[index] = np.nan
        last_valid = index

    # --- trailing statistics (backward-looking only) --------------------
    vol_21 = _trailing_std(log_return, 21)
    ma_dev_21 = _trailing_ma_deviation(log_close, 21)
    drawdown_126 = _trailing_drawdown(adjusted, valid, 126)

    return DBSFrame(
        dates=pd.DatetimeIndex(subset["DATE"]),
        log_close=log_close,
        log_return=log_return,
        step_rows=step_rows,
        log_volume=log_volume,
        gap_days=gap_days,
        vol_21=vol_21,
        ma_dev_21=ma_dev_21,
        drawdown_126=drawdown_126,
        observed=valid.astype(np.float64),
    )


def _trailing_std(values: np.ndarray, window: int) -> np.ndarray:
    """Std of the last ``window`` *valid* observations, inclusive."""
    out = np.full(len(values), np.nan)
    history: list[float] = []
    for index, value in enumerate(values):
        if np.isfinite(value):
            history.append(float(value))
        if len(history) >= max(2, window // 2) and np.isfinite(value):
            tail = np.asarray(history[-window:], dtype=float)
            out[index] = float(np.std(tail, ddof=1)) if tail.size >= 2 else np.nan
    return out


def _trailing_ma_deviation(log_values: np.ndarray, window: int) -> np.ndarray:
    """log(value) - log(mean of the last ``window`` valid values), inclusive."""
    out = np.full(len(log_values), np.nan)
    history: list[float] = []
    for index, value in enumerate(log_values):
        if np.isfinite(value):
            history.append(float(value))
        if np.isfinite(value) and len(history) >= 2:
            tail = np.asarray(history[-window:], dtype=float)
            mean_log = float(np.mean(tail))
            out[index] = float(value) - mean_log
    return out


def _trailing_drawdown(adjusted: np.ndarray, valid: np.ndarray, window: int) -> np.ndarray:
    """value / running max of the last ``window`` valid values - 1, inclusive."""
    out = np.full(len(adjusted), np.nan)
    history: list[float] = []
    for index in range(len(adjusted)):
        if valid[index]:
            history.append(float(adjusted[index]))
        if valid[index] and history:
            tail = np.asarray(history[-window:], dtype=float)
            peak = float(np.max(tail))
            if peak > 0:
                out[index] = float(adjusted[index]) / peak - 1.0
    return out


def asof_position(dbs: DBSFrame, month_index: int) -> int:
    """Index of the last row whose date falls in or before ``month_index``'s end.

    Month ``m`` covers calendar month ``m``. The observation cutoff is the *end* of
    that month, so a row dated anywhere inside the month is eligible.
    """
    year = 2021 + month_index // 12
    month = month_index % 12 + 1
    month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    eligible = np.flatnonzero(dbs.dates <= month_end)
    return int(eligible[-1]) if eligible.size else -1


def window_arrays(dbs: DBSFrame, position: int, length: int = contract.DBS_LOOKBACK_ROWS):
    """Slice ending at ``position`` (inclusive), padded on the left.

    Returns ``(sequence, mask, date_ints)`` where the sequence is
    ``(length, len(DBS_CHANNELS))`` oldest-first and padding is zero with mask 0.
    """
    channels = len(contract.DBS_CHANNELS)
    sequence = np.zeros((length, channels), dtype=np.float32)
    mask = np.zeros(length, dtype=np.float32)
    dates = np.zeros(length, dtype=np.int32)
    if position < 0:
        return sequence, mask, dates

    start = max(0, position - length + 1)
    indices = np.arange(start, position + 1)
    offset = length - len(indices)
    block = np.column_stack([
        dbs.log_return[indices],
        dbs.log_volume[indices],
        dbs.vol_21[indices],
        dbs.ma_dev_21[indices],
        dbs.drawdown_126[indices],
        dbs.log_close[indices],
        dbs.observed[indices],
    ]).astype(np.float32)
    sequence[offset:] = np.nan_to_num(block, nan=0.0)
    # A position is "observed" for masking purposes only where the close is valid.
    mask[offset:] = dbs.observed[indices].astype(np.float32)
    dates[offset:] = np.array(
        [int(stamp.strftime("%Y%m%d")) for stamp in dbs.dates[indices]], dtype=np.int32)
    return sequence, mask, dates


def scalar_arrays(dbs: DBSFrame, position: int, month_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Window scalars at ``position`` plus a mask of which had enough history."""
    names = contract.DBS_SCALARS
    values = np.full(len(names), np.nan)
    mask = np.zeros(len(names), dtype=np.float32)
    if position < 0:
        return values.astype(np.float32), mask

    index = {name: position for position, name in enumerate(names)}

    def put(name: str, value: float, min_obs: int = 1) -> None:
        if not np.isfinite(value):
            return
        values[index[name]] = value
        mask[index[name]] = 1.0 if min_obs > 0 else 0.0

    # cumulative log returns over valid observations
    for name, need in contract.DBS_RET_WINDOWS.items():
        window = np.arange(max(0, position - need + 1), position + 1)
        closes = dbs.log_close[window]
        fin = closes[np.isfinite(closes)]
        if fin.size >= 2:
            put(name, float(fin[-1] - fin[0]), min_obs=2)

    for name, need in contract.DBS_VOL_WINDOWS.items():
        window = np.arange(max(0, position - need + 1), position + 1)
        returns = dbs.log_return[window]
        fin = returns[np.isfinite(returns)]
        if fin.size >= 2:
            put(name, float(np.std(fin, ddof=1)), min_obs=2)

    put("dbs_drawdown_126", float(dbs.drawdown_126[position]))

    # relative volume: last 21 valid vs the 21 before them
    vols = dbs.log_volume[: position + 1]
    fin = vols[np.isfinite(vols)]
    if fin.size >= 21:
        recent = fin[-21:]
        prior = fin[-42:-21] if fin.size >= 42 else np.array([])
        if prior.size:
            put("dbs_volume_rel_21", float(np.mean(recent) - np.mean(prior)), min_obs=len(prior))

    window63 = np.arange(max(0, position - 62), position + 1)
    put("dbs_n_obs_63", float(np.isfinite(dbs.log_close[window63]).sum()))
    window126 = np.arange(max(0, position - 125), position + 1)
    put("dbs_n_obs_126", float(np.isfinite(dbs.log_close[window126]).sum()))

    if len(dbs.dates) and len(window63):
        gaps = np.diff(dbs.dates[window63]).astype("timedelta64[D]").astype(int)
        put("dbs_gap_max_days_63", float(gaps.max()) if gaps.size else 0.0)

    year = 2021 + month_index // 12
    month = month_index % 12 + 1
    month_end = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    put("dbs_asof_age_days", float((month_end - dbs.dates[position]).days))

    return values.astype(np.float32), mask
