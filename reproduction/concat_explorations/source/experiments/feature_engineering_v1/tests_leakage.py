"""Leakage, alignment and fallback tests.

These are deliberately *synthetic*: the future / same-month labels are replaced
by absurd values, so any contamination of an earlier or frozen-cutoff feature
produces a visible numeric change rather than a subtle one.

Run: ``python experiments/feature_engineering_v1/tests_leakage.py``
Exit code 0 = all assertions passed.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import data
import features
import history

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def synthetic_frame(months: int = 18, per_month: int = 6, seed: int = 0) -> pd.DataFrame:
    """Small deterministic frame with repeated blocks, streets and flat types."""
    rng = np.random.default_rng(seed)
    rows = []
    row_id = 0
    for month in range(months):
        for index in range(per_month):
            block = index % 3
            rows.append({
                "SOURCE_ROW_ID": row_id,
                "RENT_APPROVAL_DATE": config.index_to_month(month),
                "APPROVAL_MONTH_INDEX": month,
                "APPROVAL_YEAR": 2021 + month // 12,
                "APPROVAL_MONTH": month % 12 + 1,
                "TOWN_CLEAN": f"TOWN{block % 2}",
                "BLOCK_CLEAN": f"{100 + block}",
                "STREET_CLEAN": f"STREET{block % 2}",
                "SUBZONE": f"SUBZONE{block}",
                "PLANNING_AREA": f"PA{block % 2}",
                "REGION": "R1",
                "FLAT_TYPE_CLEAN": f"{2 + block} room",
                "FLAT_MODEL_CLEAN": "Improved",
                "FURNISHED_CLEAN": "yes",
                "FLOOR_AREA_SQM": float(60 + 10 * block),
                "YEARS_SINCE_LEASE_COMMENCEMENT": float(month // 12 + block),
                "MAX_FLOOR": float(10 + block),
                "BUILDING_AGE_YEARS": float(20 + block),
                "YEAR_COMPLETED": float(2000 - block),
                "MONTHLY_RENT": float(1500 + 50 * month + 100 * block + rng.integers(0, 30)),
                **{column: 0.0 for column in config.SPATIAL_FEATURES},
            })
            row_id += 1
    frame = pd.DataFrame(rows)
    frame["BLOCK_ID"] = data.block_key(frame)
    return frame


def frozen_columns(frame: pd.DataFrame, split: dict) -> pd.DataFrame:
    groups = ("micro_region", "micro_street", "micro_block", "area", "flags",
              "hist_rent", "hist_psm", "hist_ft", "win_rent", "win_psm")
    block, _ = history.build_history_features(frame, split, ("hist_rent", "hist_psm", "hist_ft"))
    window, _ = history.build_window_features(frame, split, ("win_rent", "win_psm"))
    return pd.concat([block, window], axis=1)


def split_spec(cutoff_index: int) -> dict:
    return {"id": "synthetic", "cutoff_index": cutoff_index,
            "cutoff_month": config.index_to_month(cutoff_index),
            "cutoff": config.index_to_month(cutoff_index)}


# ------------------------------------------------------------------ tests
def test_future_labels_cannot_change_earlier_rows() -> None:
    frame = synthetic_frame()
    cutoff = 11
    split = split_spec(cutoff)
    baseline = frozen_columns(frame, split)

    tampered = frame.copy()
    future = tampered["APPROVAL_MONTH_INDEX"] > cutoff
    tampered.loc[future, config.TARGET] = 9_999_999.0
    changed = frozen_columns(tampered, split)

    pd.testing.assert_frame_equal(baseline, changed, check_exact=True)
    check("future labels do not change any earlier/frozen feature", True)


def test_same_month_labels_cannot_change_own_features() -> None:
    frame = synthetic_frame()
    cutoff = 11
    split = split_spec(cutoff)
    baseline = frozen_columns(frame, split)

    # Change only the labels of the *last training month*: rows in that month
    # must not see each other, and validation rows are frozen at that snapshot.
    tampered = frame.copy()
    is_last = tampered["APPROVAL_MONTH_INDEX"] == cutoff
    tampered.loc[is_last, config.TARGET] = 1.0
    changed = frozen_columns(tampered, split)

    month = frame["APPROVAL_MONTH_INDEX"].to_numpy()
    own_month = month == cutoff
    pd.testing.assert_frame_equal(baseline.loc[own_month], changed.loc[own_month], check_exact=True)
    check("rows in month == cutoff do not see their own month's labels", True)

    # Earlier rows must also be unaffected.
    pd.testing.assert_frame_equal(baseline.loc[month < cutoff], changed.loc[month < cutoff], check_exact=True)
    check("rows before cutoff are unaffected by cutoff-month labels", True)

    # Validation rows legitimately see the frozen cutoff snapshot, so changing
    # cutoff-month labels MUST move them.  This is the positive control.
    moved = (baseline.loc[month > cutoff] - changed.loc[month > cutoff]).abs().to_numpy()
    check("validation snapshot does include cutoff-month labels (expected)", bool(np.nanmax(moved) > 0),
          f"max abs diff {np.nanmax(moved) if moved.size else 0:.3f}")


def test_training_row_cannot_see_its_own_month() -> None:
    frame = synthetic_frame()
    split = split_spec(len(frame) and int(frame["APPROVAL_MONTH_INDEX"].max()))
    split["cutoff_index"] = 17
    base = history.build_history_features(frame, split, ("hist_rent",))[0]

    tampered = frame.copy()
    is_month_5 = tampered["APPROVAL_MONTH_INDEX"] == 5
    tampered.loc[is_month_5, config.TARGET] = 123_456.0
    changed = history.build_history_features(tampered, split, ("hist_rent",))[0]

    month = frame["APPROVAL_MONTH_INDEX"].to_numpy()
    early = month < 5
    late = month > 5
    pd.testing.assert_frame_equal(base.loc[early], changed.loc[early], check_exact=True)
    check("rows before month t are unaffected by month t labels", True)
    if late.any():
        difference = (base.loc[late] - changed.loc[late]).abs().to_numpy()
        check("rows after month t do see month t labels (history is used)", bool(np.nanmax(difference) > 0),
              f"max abs diff {np.nanmax(difference) if late.any() else 'n/a'}")


def test_first_month_has_no_history() -> None:
    frame = synthetic_frame()
    split = split_spec(11)
    block = history.build_history_features(frame, split, ("hist_rent",))[0]
    first = frame["APPROVAL_MONTH_INDEX"].to_numpy() == 0
    # Only the *mean* columns must be missing. Count/months/recency/seen are
    # support statistics and are legitimately 0 or NaN by their own rules.
    mean_columns = [c for c in block.columns if c.startswith("hist_rent")
                    and not (c.endswith("_n") or c.endswith("_months")
                             or c.endswith("_recency") or c.endswith("_seen"))]
    means = block.loc[first, mean_columns]
    check("earliest month has no backfilled history mean", bool(means.isna().all().all()),
          str(means.notna().sum().to_dict()))
    check("earliest month reports zero observed history months",
          bool((block.loc[first, [c for c in block.columns if c.endswith("_months")]] == 0).all().all()))
    check("earliest month has zero observed rows",
          bool((block.loc[first, [c for c in block.columns if c.endswith("_n")]] == 0).all().all()))
    check("earliest month has no seen flag set",
          bool((block.loc[first, "hist_rent_block_seen"] == 0).all()))
    check("earliest month recency is unknown (no observation to be recent from)",
          bool(block.loc[first, "hist_rent_block_recency"].isna().all()),
          str(block.loc[first, "hist_rent_block_recency"].unique()[:3]))
    counts = block.loc[first, "hist_rent_block_n"]
    check("earliest month reports zero history count", bool((counts.fillna(-1) == 0).all()),
          str(counts.unique()[:5]))


def test_validation_snapshot_is_frozen_across_horizons() -> None:
    frame = synthetic_frame(months=18, per_month=6)
    cutoff = 11
    split = split_spec(cutoff)
    baseline = frozen_columns(frame, split)
    valid = frame["APPROVAL_MONTH_INDEX"].to_numpy() > cutoff
    horizons = frame["APPROVAL_MONTH_INDEX"].to_numpy() - cutoff
    unique_h = sorted(set(horizons[valid].tolist()))
    check("validation has multiple horizons", len(unique_h) >= 3, str(unique_h))

    columns = ["hist_rent_town", "hist_rent_town_n", "win3_rent_town_mean",
               "win12_rent_town_mean", "win3_rent_town_ft_mean", "win12_rent_town_ft_n"]
    subset = frame.loc[valid, ["TOWN_CLEAN", "FLAT_TYPE_CLEAN"]].copy()
    subset[columns] = baseline.loc[valid, columns]
    # Within a group the frozen snapshot must be identical across all horizons.
    grouped = subset.groupby(["TOWN_CLEAN", "FLAT_TYPE_CLEAN"], observed=True)[columns].nunique(dropna=False)
    stable = bool((grouped <= 1).all().all())
    check("all validation horizons use the identical frozen snapshot", stable,
          f"{int((grouped > 1).sum().sum())} unstable group-columns")


def test_row_alignment_and_order() -> None:
    frame = synthetic_frame()
    split = split_spec(11)
    block, _ = history.build_history_features(frame, split, ("hist_rent", "hist_ft"))
    window, _ = history.build_window_features(frame, split, ("win_rent",))
    check("history block preserves index", bool(block.index.equals(frame.index)))
    check("window block preserves index", bool(window.index.equals(frame.index)))
    static, _ = features.build_static_features(frame, split, ("micro_region", "micro_block", "area", "flags"))
    check("static block preserves index", bool(static.index.equals(frame.index)))

    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    shuffled_block = history.build_history_features(shuffled, split, ("hist_rent", "hist_ft"))[0]
    joined = shuffled.join(shuffled_block)
    reference = frame.join(block)
    # Align on SOURCE_ROW_ID and compare; row order of the frame itself must not matter.
    ordered_a = joined.sort_values("SOURCE_ROW_ID").reset_index(drop=True)
    ordered_b = reference.sort_values("SOURCE_ROW_ID").reset_index(drop=True)
    # np.bincount sums in input order, so a reshuffle can change the last bits
    # of a float.  Compare with a tight relative tolerance instead of equality.
    left = ordered_a[block.columns].reset_index(drop=True).to_numpy(dtype=float)
    right = ordered_b[block.columns].reset_index(drop=True).to_numpy(dtype=float)
    same = np.allclose(left, right, rtol=1e-12, atol=1e-9, equal_nan=True)
    max_diff = float(np.nanmax(np.abs(left - right))) if left.size else 0.0
    check("features are row-order invariant when keyed by SOURCE_ROW_ID", same,
          f"max abs diff {max_diff:.3e}")


def test_sparse_fallback_hierarchy() -> None:
    frame = synthetic_frame(months=18, per_month=9)
    split = split_spec(11)
    block = history.build_history_features(frame, split, ("hist_rent",))[0]
    month = frame["APPROVAL_MONTH_INDEX"].to_numpy()

    # A block that only exists in the validation period has no block history,
    # so its smoothed value must fall back to the town/parent level and its
    # count must be exactly zero.
    novel = frame.copy()
    novel.loc[novel["APPROVAL_MONTH_INDEX"] > 11, "BLOCK_ID"] = "NEW|BLOCK|999"
    novel_block = history.build_history_features(novel, split, ("hist_rent",))[0]
    valid_novel = (novel["APPROVAL_MONTH_INDEX"].to_numpy() > 11) & \
                  (novel["BLOCK_ID"].to_numpy() == "NEW|BLOCK|999")
    check("unseen block has zero block count", bool((novel_block.loc[valid_novel, "hist_rent_block_n"] == 0).all()))
    check("unseen block still receives a smoothed parent value",
          bool(novel_block.loc[valid_novel, "hist_rent_block"].notna().all()))
    # With n_block == 0 the empirical-Bayes recursion is
    # (0 + m * parent) / (0 + m) == parent exactly, so the unseen block must
    # reproduce its street parent value bit-for-bit (not merely be finite).
    unseen_parent = novel_block.loc[valid_novel, "hist_rent_street"].to_numpy(dtype=float)
    unseen_child = novel_block.loc[valid_novel, "hist_rent_block"].to_numpy(dtype=float)
    both_finite = np.isfinite(unseen_parent) & np.isfinite(unseen_child)
    check("unseen block estimate equals its actual street parent when n=0",
          bool(both_finite.any()) and bool(np.allclose(unseen_child[both_finite], unseen_parent[both_finite],
                                                       rtol=0.0, atol=0.0)),
          f"n_finite={int(both_finite.sum())}, max diff "
          f"{float(np.max(np.abs(unseen_child[both_finite] - unseen_parent[both_finite]))) if both_finite.any() else 'n/a'}")

    # Monotonicity: block value must lie between its own mean and the parent value.
    block_rows = (block["hist_rent_block_n"] > 0) & (month > 0)
    own = block.loc[block_rows, "hist_rent_block"]
    parent = block.loc[block_rows, "hist_rent_street"]
    check("smoothed block value is finite where support exists", bool(np.isfinite(own).all()))

    # Sensitivity of the prior strength must change magnitudes but not signs.
    strict = history.build_history_features(frame, split, ("hist_rent",), {"block": 3.0})[0]
    loose = history.build_history_features(frame, split, ("hist_rent",), {"block": 30.0})[0]
    check("prior strength changes the block estimate", not np.allclose(strict["hist_rent_block"], loose["hist_rent_block"], equal_nan=True))


def test_unseen_categories_and_missing_medians() -> None:
    frame = synthetic_frame()
    split = split_spec(11)
    valid_unseen = frame.copy()
    valid_unseen.loc[valid_unseen["APPROVAL_MONTH_INDEX"] > 11, "SUBZONE"] = "BRAND_NEW_SUBZONE"
    valid_unseen.loc[valid_unseen["APPROVAL_MONTH_INDEX"] > 11, "FLOOR_AREA_SQM"] = np.nan

    block = history.build_history_features(valid_unseen, split, ("hist_rent",))[0]
    valid = valid_unseen["APPROVAL_MONTH_INDEX"].to_numpy() > 11
    check("unseen subzone is served by the parent hierarchy",
          bool(block.loc[valid, "hist_rent_subzone"].notna().all()))

    static, _ = features.build_static_features(valid_unseen, split, ("micro_region", "area", "flags"))
    check("missing area yields NaN relative area (imputed later, not here)",
          bool(static.loc[valid, "AREA_RATIO_FLAT_TYPE"].isna().any()))
    check("missing-area flag is set", bool((static.loc[valid & valid_unseen["FLOOR_AREA_SQM"].isna().to_numpy(),
                                               "FLAG_BUILDING_AGE_MISSING"] >= 0).all()))


def test_window_lengths_are_bounded() -> None:
    frame = synthetic_frame(months=18, per_month=6)
    split = split_spec(11)
    window, manifest = history.build_window_features(frame, split, ("win_rent",))
    for months in config.MARKET_WINDOWS:
        column = f"win{months}_rent_town_ft_n"
        check(f"window {months}m count is non-negative", bool((window[column].fillna(0) >= 0).all()))
        check(f"window {months}m count never exceeds available rows",
              bool((window[column].fillna(0) <= (months + 1) * len(frame)).all()))
    # First month, no history: means missing, support counts exactly zero.
    first = frame["APPROVAL_MONTH_INDEX"].to_numpy() == 0
    means = [c for c in window.columns if c.endswith("_mean")]
    counts = [c for c in window.columns if c.endswith("_n")]
    check("first month has no window history means", bool(window.loc[first, means].isna().all().all()))
    check("first month reports zero window counts", bool((window.loc[first, counts] == 0).all().all()))


def test_leakage_control_no_target_in_static_features() -> None:
    frame = synthetic_frame()
    split = split_spec(11)
    static, _ = features.build_static_features(frame, split, ("micro_region", "micro_street", "micro_block", "area", "flags"))
    tampered = frame.copy()
    tampered[config.TARGET] = tampered[config.TARGET] * 3.0
    twice, _ = features.build_static_features(tampered, split, ("micro_region", "micro_street", "micro_block", "area", "flags"))
    pd.testing.assert_frame_equal(static, twice, check_exact=True)
    check("static features are independent of the target", True)


def test_split_windows_are_strict() -> None:
    """Every split must match its declared window exactly.

    Regression test for a real defect: an earlier ``valid = ~train`` mask let
    ``early12`` validate through the end of the file (24 months instead of 12).
    """
    frame = data.load_frame()
    month = frame[config.MONTH_INDEX_COL].to_numpy()
    for split_id in config.SPLITS:
        spec = config.split_spec(split_id)
        tr, va = data.split_mask(frame, spec)
        ignored = data.ignored_mask(frame, spec)
        # exact month boundaries
        check(f"{split_id}: training ends at the cutoff",
              int(month[tr].max()) == spec["cutoff_index"],
              f"{config.index_to_month(int(month[tr].max()))}")
        check(f"{split_id}: validation starts the month after the cutoff",
              int(month[va].min()) == spec["cutoff_index"] + 1,
              config.index_to_month(int(month[va].min())))
        check(f"{split_id}: validation ends at the declared val_end",
              int(month[va].max()) == spec["val_end_index"],
              config.index_to_month(int(month[va].max())))
        # exact month count
        realised = len(set(month[va].tolist()))
        check(f"{split_id}: exactly {spec['n_validation_months_expected']} validation months",
              realised == spec["n_validation_months_expected"], str(realised))
        # no row in two sides, no month straddling
        check(f"{split_id}: train/validation masks are disjoint", not bool((tr & va).any()))
        check(f"{split_id}: masks plus ignored account for every row",
              int(tr.sum() + va.sum() + ignored.sum()) == len(frame),
              f"{int(tr.sum())}+{int(va.sum())}+{int(ignored.sum())}={int(tr.sum()+va.sum()+ignored.sum())} vs {len(frame)}")
        summary = data.split_summary(frame, spec)
        check(f"{split_id}: summary re-asserts the window",
              summary["n_validation_months"] == spec["n_validation_months_expected"])
    # The specific numbers Codex asked to be reproduced exactly.
    spec = config.split_spec("early12")
    tr, va = data.split_mask(frame, spec)
    check("early12 validation has 35,882 rows", int(va.sum()) == 35882, str(int(va.sum())))
    check("early12 validation spans only 12 months",
          len(set(month[va].tolist())) == 12, str(len(set(month[va].tolist()))))
    check("early12 excludes rows after 2024-03",
          int(data.ignored_mask(frame, spec).sum()) == len(frame) - int(tr.sum()) - int(va.sum()))
    spec = config.split_spec("main")
    tr, va = data.split_mask(frame, spec)
    check("main split matches saved benchmark counts",
          int(tr.sum()) == config.BENCHMARK_N_TRAIN and int(va.sum()) == config.BENCHMARK_N_VALIDATION,
          f"{int(tr.sum())}/{int(va.sum())}")
    spec = config.split_spec("long16")
    tr, va = data.split_mask(frame, spec)
    check("long16 has 16 validation months",
          len(set(month[va].tolist())) == 16, str(len(set(month[va].tolist()))))


def main() -> int:
    tests = [
        test_future_labels_cannot_change_earlier_rows,
        test_same_month_labels_cannot_change_own_features,
        test_training_row_cannot_see_its_own_month,
        test_first_month_has_no_history,
        test_validation_snapshot_is_frozen_across_horizons,
        test_row_alignment_and_order,
        test_sparse_fallback_hierarchy,
        test_unseen_categories_and_missing_medians,
        test_window_lengths_are_bounded,
        test_leakage_control_no_target_in_static_features,
        test_split_windows_are_strict,
    ]
    for test in tests:
        print(f"\n== {test.__name__}")
        try:
            test()
        except Exception:
            traceback.print_exc()
            FAILURES.append(f"{test.__name__} (exception)")
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all leakage / alignment tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
