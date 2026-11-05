"""Gate-1 tests for the v3 macro contract.

Two test families, both required by the plan:

* **Tamper tests** rebuild windows after mutating observations *later than* the
  declared cutoff and prove the earlier inputs are unchanged. They are not
  equivalent to editing an exported target column: they exercise the reconstruction
  path, so a builder that ignores its input fails a positive control.
* **Exact-date oracles** pin the cutoff boundary to a specific calendar day, so an
  off-by-one month or a "whole month shifted" mistake cannot pass.

No assertion here is decorative: no ``or True``, no ``>= 0`` on counts, no bare
"is finite" where an exact value exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.fusion_contract_v3 import coe, contract, dbs, windows

# ---------------------------------------------------------------- fixtures


def geometric_stock_frame(rows: int = 140, start: str = "2021-01-04",
                          daily_log_return: float = 0.001,
                          volume: float = 1_000_000.0,
                          missing_rows: tuple[int, ...] = (),
                          gap_from: int | None = None) -> pd.DataFrame:
    """Strictly geometric prices: every valid log return is a known constant."""
    dates = pd.bdate_range(start=start, periods=rows)
    price = np.exp(np.arange(rows) * daily_log_return) * 20.0
    frame = pd.DataFrame({
        "NAME": "Synthetic Bank",
        "SYMBOL": contract.DBS_SYMBOL,
        "DATE": dates.strftime("%Y-%m-%d"),
        "OPEN": price, "HIGH": price, "LOW": price, "CLOSE": price,
        "ADJUSTED_CLOSE": price,
        "VOLUME": np.full(rows, volume),
    })
    for index in missing_rows:
        frame.loc[index, ["OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME"]] = np.nan
    if gap_from is not None:
        # Remove rows so a multi-day gap exists between adjacent valid closes.
        keep = [i for i in range(rows) if not (gap_from <= i < gap_from + 3)]
        frame = frame.iloc[keep].reset_index(drop=True)
        dates_kept = pd.bdate_range(start=start, periods=rows)[keep]
        frame["DATE"] = dates_kept.strftime("%Y-%m-%d")
    return frame


def two_round_coe_frame() -> pd.DataFrame:
    """Two complete months with two rounds each, plus a partial final month."""
    rows = [
        # month 0 (2021-01): two rounds
        (2021, "a", "jan", 1, "$40,000", 1000, "1,200"),
        (2021, "a", "jan", 2, "$42,000", 1000, "1,300"),
        # month 1 (2021-02)
        (2021, "a", "feb", 1, "$44,000", 900, "1,100"),
        (2021, "a", "feb", 2, "$46,000", 900, "1,200"),
        # month 2 (2021-03): single round -> incomplete
        (2021, "a", "mar", 1, "$50,000", 800, "1,000"),
    ]
    columns = ["YEAR", "CATEGORY", "MONTH", "ROUND", "QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"]
    frame = pd.DataFrame(rows, columns=columns)
    for category in ("b", "c", "e"):
        copy = frame.copy()
        copy["CATEGORY"] = category
        frame = pd.concat([frame, copy], ignore_index=True)
    return frame


# ---------------------------------------------------------------- DBS parsing


def test_dbs_row_count_and_missing_closes_are_identified():
    raw = dbs.parse_stock_frame()
    frame = dbs.build_dbs_frame(raw)
    assert len(frame) == 1403
    assert int(frame.observed.sum()) == 1401
    # exactly two invalid closes, and their dates are the documented ones
    invalid = frame.dates[frame.observed == 0]
    assert [stamp.strftime("%Y-%m-%d") for stamp in invalid] == ["2026-05-01", "2026-06-01"]


def test_dbs_synthetic_log_return_is_exactly_the_constant():
    raw = geometric_stock_frame(rows=30, daily_log_return=0.002)
    frame = dbs.build_dbs_frame(raw)
    returns = frame.log_return[np.isfinite(frame.log_return)]
    np.testing.assert_allclose(returns, 0.002, rtol=0.0, atol=1e-12)
    assert returns.size == 29


def test_dbs_multi_row_gap_is_masked_not_reported_as_one_day():
    """Two consecutive missing closes create a 3-row step whose return must be masked."""
    raw = geometric_stock_frame(rows=60, daily_log_return=0.002, missing_rows=(30, 31))
    frame = dbs.build_dbs_frame(raw)
    big_steps = frame.step_rows > 1
    assert big_steps.sum() == 1
    assert frame.step_rows[big_steps][0] == 3
    # a gap exists, its return is masked, and no oversized daily return survives
    assert np.isnan(frame.log_return[big_steps]).all()
    finite = frame.log_return[np.isfinite(frame.log_return)]
    np.testing.assert_allclose(finite, 0.002, rtol=0.0, atol=1e-12)


def test_dbs_missing_row_calendar_gap_is_also_masked():
    """A trading day absent from the file leaves the row step at 1 but spans days.

    Masking on row steps alone would report the whole move as an ordinary daily
    return, which is exactly the failure the plan warns about.
    """
    rows = 60
    base = geometric_stock_frame(rows=rows, daily_log_return=0.002)
    # drop five consecutive business days: row step stays 1, calendar gap becomes 7
    keep = [i for i in range(rows) if not (30 <= i < 35)]
    holed = base.iloc[keep].reset_index(drop=True)
    frame = dbs.build_dbs_frame(holed)
    # every row after the first has a row step of exactly 1 ...
    assert (frame.step_rows[1:] == 1).all()
    joined = int(frame.gap_days.argmax())
    # removing five consecutive business days spans a weekend: 10 calendar days
    assert frame.gap_days[joined] == 10
    assert frame.gap_days[joined] > dbs.MAX_CALENDAR_GAP_DAYS
    assert np.isnan(frame.log_return[joined]), "7-day gap was reported as a daily return"
    finite = frame.log_return[np.isfinite(frame.log_return)]
    np.testing.assert_allclose(finite, 0.002, rtol=0.0, atol=1e-12)


def test_dbs_missing_close_does_not_become_a_zero_return():
    raw = geometric_stock_frame(rows=40, daily_log_return=0.001, missing_rows=(20,))
    frame = dbs.build_dbs_frame(raw)
    assert frame.observed[20] == 0.0
    # the row after the hole has a 2-row step and is therefore masked
    assert frame.step_rows[21] == 2
    assert np.isnan(frame.log_return[21])
    # no zero-return was invented anywhere
    finite = frame.log_return[np.isfinite(frame.log_return)]
    assert not np.any(finite == 0.0)


def test_dbs_trailing_volatility_matches_hand_computation():
    raw = geometric_stock_frame(rows=40, daily_log_return=0.001)
    frame = dbs.build_dbs_frame(raw)
    position = 30
    window = frame.log_return[position - 20: position + 1]
    expected = float(np.std(window[np.isfinite(window)], ddof=1))
    assert frame.vol_21[position] == pytest.approx(expected, abs=1e-12)


def test_dbs_drawdown_is_zero_on_a_monotone_rising_path():
    raw = geometric_stock_frame(rows=60, daily_log_return=0.001)
    frame = dbs.build_dbs_frame(raw)
    values = frame.drawdown_126[np.isfinite(frame.drawdown_126)]
    np.testing.assert_allclose(values, 0.0, atol=1e-12)


# ---------------------------------------------------------------- DBS cutoff


def test_dbs_asof_position_is_the_last_trading_day_of_the_cutoff_month():
    raw = dbs.parse_stock_frame()
    frame = dbs.build_dbs_frame(raw)
    # 2024-03 cutoff month: last trading row in March 2024
    position = dbs.asof_position(frame, contract.month_index("2024-03"))
    stamp = frame.dates[position]
    assert stamp.year == 2024 and stamp.month == 3
    assert stamp.strftime("%Y-%m-%d") == "2024-03-28"
    # the next row belongs to April, proving nothing from April was read
    assert frame.dates[position + 1].month == 4


def test_dbs_window_ends_exactly_at_the_asof_date():
    raw = dbs.parse_stock_frame()
    frame = dbs.build_dbs_frame(raw)
    month = contract.month_index("2024-03")
    position = dbs.asof_position(frame, month)
    sequence, mask, dates = dbs.window_arrays(frame, position, contract.DBS_LOOKBACK_ROWS)
    assert dates.shape == (63,)
    assert int(dates[-1]) == 20240328
    assert mask[-1] == 1.0
    # oldest-first, strictly increasing
    valid_dates = dates[mask == 1]
    assert np.all(np.diff(valid_dates) > 0)
    # exactly 63 real observations are available this late in the series
    assert mask.sum() == 63


def test_dbs_window_pads_without_backfill_early_in_the_series():
    raw = dbs.parse_stock_frame()
    frame = dbs.build_dbs_frame(raw)
    # January 2021 contains 20 trading days (Jan 4..29), so the window holds 20
    # real observations and pads the older 43 positions.
    position = dbs.asof_position(frame, contract.month_index("2021-01"))
    sequence, mask, dates = dbs.window_arrays(frame, position, contract.DBS_LOOKBACK_ROWS)
    assert mask.sum() == 20
    assert int(dates[-1]) == 20210129
    assert int(dates[-20]) == 20210104
    # padding is zero in the sequence and zero in dates, never a copied value
    assert np.all(sequence[:-20] == 0.0)
    assert np.all(dates[:-20] == 0)
    assert np.all(mask[:-20] == 0.0)


def test_dbs_scalars_respect_the_cutoff_month():
    raw = dbs.parse_stock_frame()
    frame = dbs.build_dbs_frame(raw)
    month = contract.month_index("2023-06")
    position = dbs.asof_position(frame, month)
    values, mask = dbs.scalar_arrays(frame, position, month)
    named = dict(zip(contract.DBS_SCALARS, values))
    # as-of age must be non-negative and inside the month
    assert 0 <= named["dbs_asof_age_days"] <= 31
    assert mask[contract.DBS_SCALARS.index("dbs_n_obs_63")] == 1.0
    assert named["dbs_n_obs_63"] == 63
    # cumulative return over 21 valid observations equals the endpoint log ratio
    window = np.arange(max(0, position - 20), position + 1)
    closes = frame.log_close[window]
    finite = closes[np.isfinite(closes)]
    assert named["dbs_ret_1m"] == pytest.approx(float(finite[-1] - finite[0]), abs=1e-10)


# ---------------------------------------------------------------- tamper tests


def test_future_observations_cannot_change_earlier_dbs_windows():
    """Mutate everything after the cutoff and rebuild; earlier windows are identical."""
    cutoff_month = contract.month_index("2023-06")
    raw = dbs.parse_stock_frame()
    baseline = dbs.build_dbs_frame(raw)

    cutoff_end = pd.Timestamp(year=2023, month=6, day=1) + pd.offsets.MonthEnd(0)
    tampered = raw.copy()
    future = pd.to_datetime(tampered["DATE"]) > cutoff_end
    tampered.loc[future, ["ADJUSTED_CLOSE", "VOLUME"]] = tampered.loc[
        future, ["ADJUSTED_CLOSE", "VOLUME"]] * 1000.0
    rebuilt = dbs.build_dbs_frame(tampered)

    position = dbs.asof_position(baseline, cutoff_month)
    a_seq, a_mask, a_dates = dbs.window_arrays(baseline, position, contract.DBS_LOOKBACK_ROWS)
    b_seq, b_mask, b_dates = dbs.window_arrays(rebuilt, dbs.asof_position(rebuilt, cutoff_month),
                                               contract.DBS_LOOKBACK_ROWS)
    np.testing.assert_array_equal(a_seq, b_seq)
    np.testing.assert_array_equal(a_mask, b_mask)
    np.testing.assert_array_equal(a_dates, b_dates)

    a_values, a_scalar_mask = dbs.scalar_arrays(baseline, position, cutoff_month)
    b_values, b_scalar_mask = dbs.scalar_arrays(rebuilt, position, cutoff_month)
    np.testing.assert_allclose(a_values, b_values, rtol=0.0, atol=0.0, equal_nan=True)
    np.testing.assert_array_equal(a_scalar_mask, b_scalar_mask)


def test_positive_control_in_window_labels_do_change_later_windows():
    """Guards against a builder that ignores its input entirely."""
    cutoff_month = contract.month_index("2023-06")
    raw = dbs.parse_stock_frame()
    baseline = dbs.build_dbs_frame(raw)

    within = pd.Timestamp(year=2023, month=5, day=1)
    within_end = pd.Timestamp(year=2023, month=5, day=1) + pd.offsets.MonthEnd(0)
    tampered = raw.copy()
    mask = (pd.to_datetime(tampered["DATE"]) >= within) & (pd.to_datetime(tampered["DATE"]) <= within_end)
    tampered.loc[mask, "ADJUSTED_CLOSE"] = tampered.loc[mask, "ADJUSTED_CLOSE"] * 5.0
    rebuilt = dbs.build_dbs_frame(tampered)

    position = dbs.asof_position(baseline, cutoff_month)
    a_seq, _, _ = dbs.window_arrays(baseline, position, contract.DBS_LOOKBACK_ROWS)
    b_seq, _, _ = dbs.window_arrays(rebuilt, dbs.asof_position(rebuilt, cutoff_month),
                                    contract.DBS_LOOKBACK_ROWS)
    assert not np.allclose(a_seq, b_seq), "window ignored in-window changes"


# ---------------------------------------------------------------- COE parsing


def test_coe_money_strings_are_parsed_not_cast():
    assert coe._parse_money("$40,609") == 40609.0
    assert coe._parse_money("1,280") == 1280.0
    assert coe._parse_money("$0") == 0.0
    assert np.isnan(coe._parse_money(""))
    assert np.isnan(coe._parse_money("nan"))
    with pytest.raises(ValueError):
        coe._parse_money("forty thousand")


def test_coe_categories_and_month_names():
    frame = coe.parse_coe_frame()
    assert len(frame) == 539
    assert sorted(frame.category.unique()) == ["a", "b", "c", "e"]
    assert set(frame.month.unique()) == set(range(1, 9)) | set(range(1, 13))
    # months map through the table, never lexically (feb < jan would be wrong)
    january = frame[frame.month_name == "jan"]
    assert (january.month == 1).all()


def test_coe_category_e_zero_prevailing_premium_is_structural_not_missing():
    """Category E's PREVAILING column is 0 everywhere; QUOTA_PREMIUM carries the real value."""
    raw = coe.parse_coe_frame()
    assert (raw[raw.category == "e"].premium > 0).all()
    monthly = coe.build_coe_monthly(raw)
    e_index = list(monthly.categories).index("e")
    observed = monthly.observed[e_index] > 0
    assert observed.sum() == 67
    # every observed category-E month has a positive premium, none is NaN
    premia = monthly.premium[e_index][observed]
    assert np.all(np.isfinite(premia))
    assert np.all(premia > 0)


def test_coe_genuine_zero_premium_is_preserved():
    """A real 0 must stay 0.0 with observed = 1, not become a missing value."""
    frame = two_round_coe_frame()
    frame.loc[(frame["CATEGORY"] == "e") & (frame["ROUND"] == 1), "QUOTA_PREMIUM"] = "$0"
    monthly = coe.build_coe_monthly(coe.parse_coe_frame() if False else _parsed(frame))
    e_index = list(monthly.categories).index("e")
    column = monthly.month_position(0)
    assert monthly.observed[e_index, column] == 1.0
    # mean of the two rounds: $0 and $42,000
    assert monthly.premium[e_index, column] == 21000.0
    # and it is genuinely 0-free, i.e. not treated as missing
    assert np.isfinite(monthly.premium[e_index, column])


def _parsed(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the parser's own coercion on an already-in-memory frame."""
    out = pd.DataFrame({
        "year": frame["YEAR"].astype(int),
        "category": frame["CATEGORY"].astype(str).str.lower(),
        "month_name": frame["MONTH"].astype(str).str.lower(),
        "round": frame["ROUND"].astype(int),
        "premium": frame["QUOTA_PREMIUM"].map(coe._parse_money),
        "quota": frame["QUOTA"].map(coe._parse_money),
        "bids": frame["BIDS_RECEIVED"].map(coe._parse_money),
    })
    out["month"] = out["month_name"].map(contract.MONTH_ABBREVIATIONS).astype(int)
    out["month_index"] = (out["year"] - 2021) * 12 + out["month"] - 1
    return out


# ---------------------------------------------------------------- COE aggregation


def test_coe_monthly_aggregation_sums_then_divides():
    monthly = coe.build_coe_monthly(_parsed(two_round_coe_frame()))
    column = monthly.month_position(0)
    a_index = list(monthly.categories).index("a")
    # premium = mean of 40000 and 42000
    assert monthly.premium[a_index, column] == 41000.0
    # quota and bids are summed before the ratio
    assert monthly.quota[a_index, column] == 2000.0
    assert monthly.bids[a_index, column] == 2500.0
    assert monthly.ratio[a_index, column] == pytest.approx(2500.0 / 2000.0, abs=1e-12)
    assert monthly.auctions[a_index, column] == 2.0
    assert monthly.observed[a_index, column] == 1.0
    assert monthly.incomplete[a_index, column] == 0.0


def test_coe_zero_quota_is_masked_for_the_ratio_only():
    frame = two_round_coe_frame()
    frame.loc[frame["YEAR"] == 2021, "QUOTA"] = "0"
    monthly = coe.build_coe_monthly(_parsed(frame))
    a_index = list(monthly.categories).index("a")
    column = monthly.month_position(0)
    assert monthly.observed[a_index, column] == 1.0
    assert np.isnan(monthly.ratio[a_index, column])
    assert monthly.premium[a_index, column] == 41000.0
    assert monthly.quota[a_index, column] == 0.0


def test_coe_single_round_month_is_flagged_incomplete():
    monthly = coe.build_coe_monthly(_parsed(two_round_coe_frame()))
    a_index = list(monthly.categories).index("a")
    assert monthly.incomplete[a_index, monthly.month_position(2)] == 1.0
    assert monthly.incomplete[a_index, monthly.month_position(0)] == 0.0


def test_coe_final_source_month_is_always_incomplete():
    monthly = coe.build_coe_monthly(coe.parse_coe_frame())
    last = monthly.month_position(monthly.last_source_month)
    assert contract.month_label(monthly.last_source_month) == "2026-08"
    # every category is flagged for the final partial month
    assert monthly.incomplete[:, last].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_coe_missing_category_month_is_absent_not_zero():
    """2026-08 has no category-E row at all: it must read as unobserved."""
    monthly = coe.build_coe_monthly(coe.parse_coe_frame())
    e_index = list(monthly.categories).index("e")
    column = monthly.month_position(monthly.last_source_month)
    assert monthly.observed[e_index, column] == 0.0
    assert np.isnan(monthly.premium[e_index, column])
    # while A/B/C do have one (partial) round that month
    a_index = list(monthly.categories).index("a")
    assert monthly.observed[a_index, column] == 1.0


# ---------------------------------------------------------------- COE windows


def test_coe_window_is_twelve_months_oldest_first():
    monthly = coe.build_coe_monthly(coe.parse_coe_frame())
    cutoff = contract.month_index("2024-06")
    sequence, mask, months, incomplete = coe.window_arrays(monthly, cutoff)
    assert months.shape == (12,)
    # month ids are YYYYMM, so 0 is unambiguous padding
    assert int(months[-1]) == 202406
    assert int(months[0]) == 202307
    # consecutive calendar months, checked through the index, not raw subtraction
    indices = [(value // 100 - 2021) * 12 + value % 100 - 1 for value in months]
    assert np.all(np.diff(indices) == 1)
    # four categories, six channels
    assert sequence.shape == (4, 12, 6)
    assert mask.shape == (4, 12)


def test_coe_window_pads_before_the_first_source_month():
    monthly = coe.build_coe_monthly(coe.parse_coe_frame())
    cutoff = contract.month_index("2021-03")
    sequence, mask, months, incomplete = coe.window_arrays(monthly, cutoff)
    # A 12-month window ending at 2021-03 starts at 2020-04, which is before the
    # source: those leading positions are padding, and the last three are real.
    assert (months[:9] == 0).all()
    assert [int(value) for value in months[9:]] == [202101, 202102, 202103]
    assert mask[:, :9].sum() == 0.0
    assert np.all(sequence[:, :9, :] == 0.0)
    assert mask[:, 9:].sum() == 12.0  # 4 categories x 3 months


def test_future_months_cannot_change_earlier_coe_windows():
    """Tamper with every auction after the cutoff and rebuild; earlier window is identical."""
    cutoff = contract.month_index("2023-06")
    raw = coe.parse_coe_frame()
    baseline = coe.build_coe_monthly(raw)

    tampered = raw.copy()
    tampered.loc[tampered["month_index"] > cutoff, "premium"] *= 10.0
    tampered.loc[tampered["month_index"] > cutoff, "quota"] *= 10.0
    rebuilt = coe.build_coe_monthly(tampered)

    a = coe.window_arrays(baseline, cutoff)
    b = coe.window_arrays(rebuilt, cutoff)
    for left, right in zip(a, b):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)

    a_scalars = coe.scalar_arrays(baseline, cutoff)
    b_scalars = coe.scalar_arrays(rebuilt, cutoff)
    for left, right in zip(a_scalars, b_scalars):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)


def test_positive_control_within_window_coe_changes_are_detected():
    cutoff = contract.month_index("2023-06")
    raw = coe.parse_coe_frame()
    baseline = coe.build_coe_monthly(raw)

    tampered = raw.copy()
    target_month = contract.month_index("2023-05")
    tampered.loc[tampered["month_index"] == target_month, "premium"] *= 10.0
    rebuilt = coe.build_coe_monthly(tampered)

    a_seq = coe.window_arrays(baseline, cutoff)[0]
    b_seq = coe.window_arrays(rebuilt, cutoff)[0]
    assert not np.allclose(a_seq, b_seq)


def test_coe_scalars_track_the_latest_observed_month():
    monthly = coe.build_coe_monthly(coe.parse_coe_frame())
    cutoff = contract.month_index("2024-06")
    values, mask = coe.scalar_arrays(monthly, cutoff)
    a_index = list(monthly.categories).index("a")
    named = dict(zip(contract.COE_SCALARS, values[a_index]))
    position = monthly.month_position(cutoff)
    assert named["coe_premium_latest"] == pytest.approx(monthly.premium[a_index, position])
    assert named["coe_n_months_12"] == 12.0
    assert named["coe_auctions_12m"] == 24.0
    assert named["coe_incomplete_latest"] == 0.0


# ---------------------------------------------------------------- policy


def test_origin_asof_cutoff_equals_the_row_origin():
    samples = pd.DataFrame({
        "origin_month_index": [10, 10, 36],
        "target_month_index": [12, 15, 38],
    })
    cutoffs = windows.macro_cutoff_months(samples, contract.POLICY_ORIGIN_ASOF)
    assert cutoffs.tolist() == [10, 10, 36]


def test_target_previous_month_cutoff_is_the_month_before_the_target():
    samples = pd.DataFrame({
        "origin_month_index": [10, 10, 36],
        "target_month_index": [12, 15, 38],
    })
    cutoffs = windows.macro_cutoff_months(samples, contract.POLICY_TARGET_PREVIOUS_MONTH)
    assert cutoffs.tolist() == [11, 14, 37]
    # never reaches the target month itself
    assert ((cutoffs - samples["target_month_index"].to_numpy()) == -1).all()


def test_windows_are_deduplicated_by_cutoff_month():
    samples = pd.DataFrame({
        "origin_month_index": [20, 20, 20, 21, 21, 30],
        "target_month_index": [22, 25, 29, 23, 28, 32],
    })
    row_window, unique, counts = windows.window_index(
        windows.macro_cutoff_months(samples, contract.POLICY_ORIGIN_ASOF))
    assert unique.tolist() == [20, 21, 30]
    assert counts.tolist() == [3, 2, 1]
    assert counts.sum() == len(samples)
    assert row_window.tolist() == [0, 0, 0, 1, 1, 2]


# ---------------------------------------------------------------- contract facts


def test_split_definitions_match_the_plan():
    assert contract.split_spec("orig")["cutoff"] == "2024-03"
    assert contract.split_spec("orig")["val_end"] == "2025-03"
    assert contract.split_spec("orig")["n_validation_months"] == 12
    assert contract.split_spec("early")["cutoff"] == "2023-03"
    assert contract.split_spec("early")["n_validation_months"] == 12
    assert contract.split_spec("long")["cutoff"] == "2023-11"
    assert contract.split_spec("long")["n_validation_months"] == 16


def test_dbs_branch_uses_only_the_declared_symbol():
    assert contract.DBS_SYMBOL == "D05.SI"
    raw = dbs.parse_stock_frame()
    assert raw.SYMBOL.nunique() == 77
    frame = dbs.build_dbs_frame(raw)
    assert len(frame) == 1403


def test_arms_add_macro_blocks_without_changing_base_roles():
    for arm in contract.ARM_INPUTS:
        roles = contract.arm_scalar_roles(arm)
        assert contract.ROLE_V21_SCALARS in roles
    assert not contract.arm_uses_dbs_sequence("macro_none")
    assert contract.arm_uses_dbs_sequence("macro_dbs")
    assert not contract.arm_uses_coe_sequence("macro_dbs")
    assert contract.arm_uses_coe_sequence("macro_coe")
    assert contract.arm_uses_dbs_sequence("macro_both")
    assert contract.arm_uses_coe_sequence("macro_both")


def test_scalar_name_lists_match_the_declared_shape():
    assert len(contract.DBS_SCALARS) == 12
    assert len(contract.COE_SCALARS) == 9
    assert len(contract.coe_scalar_names()) == 4 * 9 == 36
    assert len(contract.DBS_CHANNELS) == 7
    assert len(contract.COE_CHANNELS) == 6


def test_package_imports_are_qualified():
    source = (contract.PACKAGE_DIR / "windows.py").read_text()
    assert "from experiments.fusion_contract_v3 import" in source
    assert "\nimport config" not in source
    assert "\nfrom config import" not in source
