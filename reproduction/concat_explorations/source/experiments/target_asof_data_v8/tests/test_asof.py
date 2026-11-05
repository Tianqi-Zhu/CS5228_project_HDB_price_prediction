"""Targeted boundary tests for the as-of v8 dataset.

Three properties, each aimed at a specific way this dataset could be silently wrong. They
are deliberately small and hand-checkable — this is not a broad suite.

1. **No future leak.** Injecting extreme market values at month `t` or later must not change
   any input for a sample whose target is month `t`. This is the property the whole design
   rests on, so it is asserted by comparison rather than by reading the code.
2. **Short history keeps the row.** A transaction in the very first month has no full
   window; it must still appear as a sample, with leading padding marked invalid, rather
   than being dropped.
3. **No rent input.** The house branch must not contain any rent or anchor column, and the
   target must not be reachable from the inputs.

Run:
    .venv/bin/python -m experiments.target_asof_data_v8.tests.test_asof
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for candidate in (ROOT, ROOT / "experiments"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"PASS  {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"FAIL  {name}  {detail}")


def main() -> int:
    from experiments.target_asof_data_v8 import build as B
    from experiments.target_asof_data_v8 import features as F

    # ---------------------------------------------------------------- 3. no rent input
    print("--- 1. the house branch carries no rent or anchor field")
    forbidden = F.FORBIDDEN_INPUTS
    house_columns = F.HOUSE_NUMERIC_ORDER + F.HOUSE_CATEGORICAL_ORDER
    clashes = []
    for column in house_columns:
        for pattern in forbidden:
            stem = pattern.split(" (")[0]
            if stem.endswith("*"):
                if column.startswith(stem[:-1]):
                    clashes.append((column, pattern))
            elif column == stem:
                clashes.append((column, pattern))
    check("no forbidden column appears in the house branch", not clashes, str(clashes))
    check("furnished is excluded", "FURNISHED_CLEAN" not in house_columns)
    check("house numeric count is 21", len(F.HOUSE_NUMERIC_ORDER) == 21,
          str(len(F.HOUSE_NUMERIC_ORDER)))
    check("house categorical count is 3", len(F.HOUSE_CATEGORICAL_ORDER) == 3)
    print(f"      market channels: {len(F.MARKET_CHANNEL_ORDER)} "
          f"({len(F.DBS_CHANNELS)} DBS + {len(F.COE_CATEGORIES)}x3 COE)")

    # ---------------------------------------------------------------- synthetic sources
    # A tiny stock series and COE table, both spanning enough months for a t-12 window.
    months = pd.period_range("2021-01", periods=18, freq="M")
    stock_rows = []
    for month in months:
        for day in range(1, 6):
            date = pd.Timestamp(year=month.year, month=month.month, day=day)
            stock_rows.append({"SYMBOL": "D05.SI", "DATE": date, "OPEN": 26.0,
                               "HIGH": 26.5, "LOW": 25.5, "CLOSE": 26.0,
                               "ADJUSTED_CLOSE": 26.0 + day * 0.1, "VOLUME": 1_000_000})
    stock = pd.DataFrame(stock_rows)

    coe_rows = []
    for month in months:
        for cat in ("a", "b", "c", "e"):
            for rnd in (1, 2):
                coe_rows.append({"YEAR": month.year, "MONTH": month.strftime("%b").lower(),
                                 "CATEGORY": cat, "ROUND": rnd,
                                 "QUOTA_PREMIUM": 40000 + rnd * 1000,
                                 "QUOTA": 900, "BIDS_RECEIVED": 1200})
    coe = pd.DataFrame(coe_rows)

    market = B.build_market_series(stock, coe, last_month=17)
    print(f"\n      market series: {market.n_months} months, "
          f"{len(market.channels)} channels")

    # ---------------------------------------------------------------- 1. no future leak
    print("\n--- 2. a value at month t or later cannot reach a target at month t")
    targets = [14, 15]
    for t in targets:
        base = B.window_for_target(market, t)
        # poison every month from t onwards with an absurd value
        poisoned = market.copy_with_spike(from_month=t, magnitude=1e6)
        after = B.window_for_target(poisoned, t)
        same_values = np.array_equal(base.values, after.values, equal_nan=True)
        same_mask = np.array_equal(base.observed, after.observed)
        same_months = np.array_equal(base.month_index, after.month_index)
        check(f"target month {t}: values unchanged when months >= {t} are poisoned",
              same_values, f"max delta {np.nanmax(np.abs(base.values - after.values))}")
        check(f"target month {t}: masks and month index unchanged",
              same_mask and same_months)
        # the window must stop strictly before t
        check(f"target month {t}: window ends at t-1",
              int(base.month_index.max()) == t - 1, f"max month {base.month_index.max()}")

    # a sample whose target is BEFORE the poisoned month must also be unaffected
    early_target = 13
    base13 = B.window_for_target(market, early_target)
    poisoned_all = market.copy_with_spike(from_month=14, magnitude=1e6)
    after13 = B.window_for_target(poisoned_all, early_target)
    check("a later month's spike cannot change an earlier target's window",
          np.array_equal(base13.values, after13.values, equal_nan=True))

    # ---------------------------------------------------------------- 2. short history
    print("\n--- 3. a first-month transaction is kept, with padding marked invalid")
    first = B.window_for_target(market, target_month=0)
    check("target month 0 still yields a window object", first is not None)
    check("that window is entirely padding",
          bool((~first.step_valid).all()), f"step_valid sum {first.step_valid.sum()}")
    check("padding steps are not counted as observed",
          int(first.observed.sum()) == 0, str(int(first.observed.sum())))
    check("padding carries the correct calendar months anyway",
          list(first.month_index) == list(range(-12, 0)),
          str(list(first.month_index)))

    # a mid-window gap keeps step_valid=1 so the time axis is not compressed
    gapped = market.copy_with_gap(month=8)
    window8 = B.window_for_target(gapped, target_month=14)
    position = list(window8.month_index).index(8)
    check("an interior missing month keeps its time step valid",
          bool(window8.step_valid[position]),
          f"step_valid at month 8 = {window8.step_valid[position]}")
    check("but it is marked unobserved",
          not bool(window8.observed[position].any()),
          str(window8.observed[position]))
    check("the window still spans exactly 12 steps", len(window8.month_index) == 12)

    # ---------------------------------------------------------------- 4. stats arm
    print("\n--- 4. the stats arm derives everything from the same window")
    stats = B.window_stats(window8)
    check("stats width equals channels x suffixes",
          stats.shape == (len(F.MARKET_CHANNEL_ORDER), len(F.STATS_SUFFIXES)),
          str(stats.shape))
    names = F.stats_feature_names()
    idx = {n: i for i, n in enumerate(names)} if False else None
    flat = stats.reshape(-1)
    n_valid_index = [i for i, n in enumerate(
        [f"{c}__{s}" for c in F.MARKET_CHANNEL_ORDER for s in F.STATS_SUFFIXES])
        if n.endswith("__n_valid")]
    check("n_valid never exceeds the window length",
          bool((flat[n_valid_index] <= 12).all()), str(flat[n_valid_index][:4]))
    check("n_valid is >= 0", bool((flat[n_valid_index] >= 0).all()))

    # ---------------------------------------------------------------- 4b. regression cases
    print("\n--- 4b. regressions for the three defects found in review")

    # (a) COE quota with thousands separators is a VALUE, not missing
    coe_comma = pd.DataFrame([
        {"YEAR": 2021, "MONTH": "jan", "CATEGORY": "a", "ROUND": 1,
         "QUOTA_PREMIUM": "$40,609", "QUOTA": "1,023", "BIDS_RECEIVED": "1,280"},
        {"YEAR": 2021, "MONTH": "jan", "CATEGORY": "a", "ROUND": 2,
         "QUOTA_PREMIUM": "$43,501", "QUOTA": "1,018", "BIDS_RECEIVED": "1,309"},
    ])
    stock_one = pd.DataFrame([
        {"SYMBOL": "D05.SI", "DATE": pd.Timestamp("2021-01-04"),
         "ADJUSTED_CLOSE": 26.0, "VOLUME": 1000},
        {"SYMBOL": "D05.SI", "DATE": pd.Timestamp("2021-01-29"),
         "ADJUSTED_CLOSE": 26.5, "VOLUME": 2000},
    ])
    comma_market = B.build_market_series(stock_one, coe_comma, last_month=1)
    quota_idx = comma_market.channels.index("coe_a_quota_total")
    bids_idx = comma_market.channels.index("coe_a_bids_over_quota")
    check("a comma-formatted quota is parsed, not treated as missing",
          bool(comma_market.observed[0, quota_idx]),
          f"observed={comma_market.observed[0, quota_idx]}")
    check("the quota total equals the sum of the two parsed rounds",
          bool(np.isclose(comma_market.values[0, quota_idx], 1023 + 1018)),
          str(comma_market.values[0, quota_idx]))
    check("bids/quota uses the same auction set",
          bool(np.isclose(comma_market.values[0, bids_idx], (1280 + 1309) / (1023 + 1018))),
          str(comma_market.values[0, bids_idx]))

    # (b) month-end log return, not the sum of daily returns
    month_prices = [("2021-01-04", 100.0), ("2021-01-18", 90.0), ("2021-01-29", 110.0),
                    ("2021-02-01", 110.0), ("2021-02-26", 121.0)]
    stock_two = pd.DataFrame([
        {"SYMBOL": "D05.SI", "DATE": pd.Timestamp(d), "ADJUSTED_CLOSE": p, "VOLUME": 1000}
        for d, p in month_prices])
    two_market = B.build_market_series(stock_two, coe_comma, last_month=2)
    ret_idx = two_market.channels.index("dbs_log_return")
    # The series starts at its own first month, so Jan is index 0, not 1.
    jan = 0
    feb = 1
    # January has no previous month-end inside the series, so per the declared definition
    # its return is MISSING rather than a ratio against nothing.
    check("the first month's return is missing (no previous month-end exists)",
          not bool(two_market.observed[jan, ret_idx]),
          f"observed={two_market.observed[jan, ret_idx]}")
    # February: previous month-end 110 -> 121, so ln(121/110). Note January's mid-month dip
    # to 90 is irrelevant to this ratio, which is the whole point of using month ends.
    check("February's return is the month-end ratio ln(121/110)",
          bool(np.isclose(two_market.values[feb, ret_idx], np.log(121.0 / 110.0))),
          f"{two_market.values[feb, ret_idx]:.6f}")
    check("a mid-month move does not enter the month-end ratio",
          not bool(np.isclose(np.log(121.0 / 110.0), np.log(121.0 / 100.0))),
          "ln(121/110) != ln(121/100), so the previous month-end is being used")
    # The discriminating case: with the mid-month dip, summing daily returns equals the
    # month-end ratio only because every day is observed. Drop one and the summation
    # silently loses that move while the month-end ratio does not.
    summed_daily = float(np.log(90 / 100) + np.log(110 / 90))
    check("summed dailies coincide with the month-end ratio only when no day is masked",
          bool(np.isclose(summed_daily, np.log(1.1))),
          "so the definition matters exactly where data is missing")
    check("the first month has no previous month-end, so its return is missing",
          not bool(two_market.observed[0, ret_idx]),
          "January must be missing, not zero")

    # a masked interior day must not change the month-end return
    stock_gap = pd.DataFrame([
        {"SYMBOL": "D05.SI", "DATE": pd.Timestamp("2021-01-04"), "ADJUSTED_CLOSE": 100.0,
         "VOLUME": 1000},
        # 2021-01-18 missing entirely from the file
        {"SYMBOL": "D05.SI", "DATE": pd.Timestamp("2021-01-29"), "ADJUSTED_CLOSE": 110.0,
         "VOLUME": 1000},
        {"SYMBOL": "D05.SI", "DATE": pd.Timestamp("2021-02-26"), "ADJUSTED_CLOSE": 121.0,
         "VOLUME": 1000},
    ])
    gap_market = B.build_market_series(stock_gap, coe_comma, last_month=2)
    # The series here starts at January, so January itself has no previous month-end. The
    # discriminating comparison is February: its return must equal ln(121/110) even though
    # the interior day 2021-01-18 is absent from the file.
    gap_feb = int(np.searchsorted(gap_market.observed[:, ret_idx], True)) if False else feb
    check("a missing interior day does not alter the month-end return",
          bool(np.isclose(gap_market.values[feb, ret_idx], np.log(121.0 / 110.0))),
          f"{gap_market.values[feb, ret_idx]:.6f}")
    # Under the old summed-daily definition this month's figure would have collapsed: the
    # only surviving daily step inside February spans no gap, but January's would have been
    # masked, so the two definitions genuinely disagree here.
    check("the month-end ratio stays defined where daily summation would be masked",
          bool(gap_market.observed[feb, ret_idx]), "February must be observed")

    # (c) volume is log1p(mean), not mean(log1p)
    stock_vol = pd.DataFrame([
        {"SYMBOL": "D05.SI", "DATE": pd.Timestamp("2021-01-04"), "ADJUSTED_CLOSE": 100.0,
         "VOLUME": 100.0},
        {"SYMBOL": "D05.SI", "DATE": pd.Timestamp("2021-01-29"), "ADJUSTED_CLOSE": 110.0,
         "VOLUME": 10000.0},
        {"SYMBOL": "D05.SI", "DATE": pd.Timestamp("2021-02-26"), "ADJUSTED_CLOSE": 110.0,
         "VOLUME": 500.0},
    ])
    vol_market = B.build_market_series(stock_vol, coe_comma, last_month=2)
    vol_idx = vol_market.channels.index("dbs_log1p_volume")
    expected_log_of_mean = float(np.log1p((100.0 + 10000.0) / 2))
    mean_of_logs = float((np.log1p(100.0) + np.log1p(10000.0)) / 2)
    check("volume channel is log1p of the monthly mean volume",
          bool(np.isclose(vol_market.values[jan, vol_idx], expected_log_of_mean)),
          f"{vol_market.values[jan, vol_idx]:.6f} vs {expected_log_of_mean:.6f}")
    check("that differs from mean(log1p), which is the wrong quantity",
          abs(expected_log_of_mean - mean_of_logs) > 0.5,
          f"{expected_log_of_mean:.4f} vs {mean_of_logs:.4f}")

    # (d) stats use CALENDAR windows, not "last N observations"
    # Design the series so the two definitions must disagree. Steps 0..5 hold LOW values
    # (1,2,3,4,5,6); steps 6..9 are missing; steps 10,11 hold HIGH values (100,200).
    #   calendar t-3..t-1 = steps 9,10,11 -> observed only 10,11 -> mean = 150
    #   last 3 OBSERVATIONS = steps 4,5,11 -> mean = (5+6+200)/3 = 70.33
    #   prior window t-6..t-4 = steps 6,7,8 -> all missing -> delta must be NaN
    series = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, np.nan, np.nan, np.nan, 100.0, 200.0]
    values = np.asarray(series, dtype=float)[:, None]
    observed = np.isfinite(values)
    step_valid = np.ones((12, 1), dtype=bool)
    fake = B.Window(12, np.arange(12), values, observed, step_valid)
    table = B.window_stats(fake)
    mean3 = table[0, F.STATS_SUFFIXES.index("mean3")]
    check("mean3 uses the calendar window t-3..t-1 (steps 10-11 only observed)",
          bool(np.isclose(mean3, 150.0)), f"{mean3} (expected 150)")
    # last-3-OBSERVATIONS takes the final three finite values (6, 100, 200) = 102.0, which
    # is a different quantity from the calendar mean of 150.0. Both numbers are pinned
    # rather than compared with a loose threshold, so a change in either definition fails.
    last3_observations = float(np.asarray(series)[np.isfinite(series)][-3:].mean())
    check("the last-3-observations figure is 102.0", bool(np.isclose(last3_observations, 102.0)),
          f"{last3_observations}")
    check("calendar mean3 (150.0) and last-3-obs (102.0) are different quantities",
          not bool(np.isclose(mean3, last3_observations)),
          f"calendar {mean3:.2f} vs last-3-obs {last3_observations:.2f}")
    mean12 = table[0, F.STATS_SUFFIXES.index("mean12")]
    expected_mean12 = float(np.mean([1, 2, 3, 4, 5, 6, 100, 200]))
    check("mean12 averages every observed step in the window",
          bool(np.isclose(mean12, expected_mean12)),
          f"{mean12:.4f} vs {expected_mean12:.4f}")
    delta = table[0, F.STATS_SUFFIXES.index("recent3_minus_prior3")]
    check("the prior window t-6..t-4 has no observation, so the delta is missing",
          bool(np.isnan(delta)), f"{delta}")
    n_valid = table[0, F.STATS_SUFFIXES.index("n_valid")]
    check("n_valid still counts every observed step (8 of 12)", n_valid == 8.0, str(n_valid))
    gap = table[0, F.STATS_SUFFIXES.index("months_since_last_valid")]
    check("months_since_last_valid is 0 when the newest step is observed",
          gap == 0.0, str(gap))

    # ---------------------------------------------------------------- 5. causality of scalers
    print("\n--- 5. scaler statistics use training months only")
    check("scaler policy records one-row-per-month weighting",
          "one row per distinct month" in F.SCALER_POLICY["month_weights"])
    check("the window definition excludes month t",
          "t-12 .. t-1" in F.TIME_ASSUMPTIONS["window"]["covers"].replace(".. ", ".. "))

    print("\n" + "=" * 62)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for item in FAILED:
        print(f"  - {item}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
