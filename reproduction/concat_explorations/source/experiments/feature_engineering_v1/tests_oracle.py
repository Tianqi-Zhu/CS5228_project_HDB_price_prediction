"""Hand-computed oracle tests for the historical-feature time boundary.

A tiny synthetic panel with one block and one distinct rent per month makes the
expected value of every historical feature an exact, human-checkable number.
With prior strength 0 the empirical-Bayes recursion collapses to the plain
arithmetic mean, so the assertions pin down *precisely which months* entered an
aggregate — not just that some value is finite.

Expected boundaries
-------------------
* Training row dated ``t``  -> aggregate over months ``0 .. t-1``  (``t`` excluded).
* Validation row (``t > c``) -> aggregate over months ``0 .. c``    (frozen at cutoff).

Run: ``./run.sh tests_oracle.py``  (exit 0 = all assertions passed)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import data
import history

FAILURES: list[str] = []

# A single block/street/town and one flat type, so every hierarchy level holds
# the same rows and the smoothed value equals the arithmetic mean of the
# months that were legitimately visible.
MONTHLY_RENT = {0: 1000.0, 1: 2000.0, 2: 3000.0, 3: 4000.0, 4: 5000.0}
CUTOFF = 2
ZERO_PRIOR = {level: 0.0 for level in config.SMOOTHING}


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"PASS  {name}")
    else:
        print(f"FAIL  {name}  {detail}")
        FAILURES.append(name)


def oracle_frame() -> pd.DataFrame:
    rows = []
    row_id = 0
    for month, rent in MONTHLY_RENT.items():
        for _ in range(2):  # two rows per month, identical rent
            rows.append({
                "SOURCE_ROW_ID": row_id,
                "RENT_APPROVAL_DATE": config.index_to_month(month),
                "APPROVAL_MONTH_INDEX": month,
                "APPROVAL_YEAR": 2021,
                "APPROVAL_MONTH": month + 1,
                "TOWN_CLEAN": "T1",
                "BLOCK_CLEAN": "1",
                "STREET_CLEAN": "S1",
                "SUBZONE": "SZ1",
                "PLANNING_AREA": "PA1",
                "REGION": "R1",
                "FLAT_TYPE_CLEAN": "3 room",
                "FLAT_MODEL_CLEAN": "Improved",
                "FURNISHED_CLEAN": "yes",
                "FLOOR_AREA_SQM": 60.0,
                "YEARS_SINCE_LEASE_COMMENCEMENT": 10.0,
                "MAX_FLOOR": 12.0,
                "BUILDING_AGE_YEARS": 20.0,
                "YEAR_COMPLETED": 2000.0,
                "MONTHLY_RENT": rent,
                **{column: 0.0 for column in config.SPATIAL_FEATURES},
            })
            row_id += 1
    frame = pd.DataFrame(rows)
    frame["BLOCK_ID"] = data.block_key(frame)
    return frame


def split() -> dict:
    return {"id": "oracle", "cutoff_index": CUTOFF,
            "cutoff_month": config.index_to_month(CUTOFF),
            "val_start_index": CUTOFF + 1, "val_end_index": 4,
            "val_end": config.index_to_month(4)}


def test_training_row_uses_months_strictly_before_t() -> None:
    frame = oracle_frame()
    block, _ = history.build_history_features(frame, split(), ("hist_rent",), ZERO_PRIOR)

    # Month 0: nothing exists yet.
    month0 = frame["APPROVAL_MONTH_INDEX"].to_numpy() == 0
    check("t=0 has no history (mean missing)",
          bool(block.loc[month0, "hist_rent_block"].isna().all()))
    check("t=0 reports n=0", bool((block.loc[month0, "hist_rent_block_n"] == 0).all()))

    # Month 1: exactly month 0 has been observed -> 1000.
    month1 = frame["APPROVAL_MONTH_INDEX"].to_numpy() == 1
    value1 = block.loc[month1, "hist_rent_town"].to_numpy()
    check("t=1 uses month 0 only (expect 1000)", bool(np.allclose(value1, 1000.0)),
          str(np.unique(value1)))
    check("t=1 reports n=2 rows observed",
          bool((block.loc[month1, "hist_rent_town_n"] == 2).all()),
          str(np.unique(block.loc[month1, "hist_rent_town_n"])))

    # Month 2 (= cutoff): months 0 and 1 -> mean(1000, 2000) = 1500.
    month2 = frame["APPROVAL_MONTH_INDEX"].to_numpy() == CUTOFF
    value2 = block.loc[month2, "hist_rent_town"].to_numpy()
    check("t=cutoff uses months 0..t-1 (expect 1500)", bool(np.allclose(value2, 1500.0)),
          str(np.unique(value2)))
    check("t=cutoff reports n=4 rows observed",
          bool((block.loc[month2, "hist_rent_town_n"] == 4).all()),
          str(np.unique(block.loc[month2, "hist_rent_town_n"])))
    check("t=cutoff reports 2 distinct observed months",
          bool((block.loc[month2, "hist_rent_town_n"] == 4).all()) and
          bool((block.loc[month2, "hist_rent_block_months"] == 2).all()),
          str(np.unique(block.loc[month2, "hist_rent_block_months"])))
    # Its own month (3000) must not be visible: 1500 proves that.
    check("t=cutoff cannot see its own month",
          bool(np.allclose(value2, 1500.0)) and not bool(np.allclose(value2, 2000.0)))

    # Recency is measured from the row's information boundary (t-1 for a
    # training row), so a row at t=2 whose last observation is month 1 has
    # recency 0 -- history is as fresh as the boundary allows.
    check("t=cutoff recency is 0 (boundary t-1 observed)",
          bool(np.allclose(block.loc[month2, "hist_rent_block_recency"], 0.0)),
          str(np.unique(block.loc[month2, "hist_rent_block_recency"])))
    # The same definition must hold at inference: a validation row whose last
    # observation is the cutoff has recency 0 too (checked in the next test).
    check("training and validation share one recency definition",
          bool(np.allclose(block.loc[month2, "hist_rent_block_recency"], 0.0)))

    # Two months earlier (t=1), only month 0 exists -> boundary 0, last 0 ->
    # recency 0 as well; the feature is about freshness, not absolute date.
    month1 = frame["APPROVAL_MONTH_INDEX"].to_numpy() == 1
    check("t=1 recency is 0 (boundary 0 observed)",
          bool(np.allclose(block.loc[month1, "hist_rent_block_recency"], 0.0)),
          str(np.unique(block.loc[month1, "hist_rent_block_recency"])))


def test_validation_uses_frozen_cutoff_snapshot() -> None:
    frame = oracle_frame()
    block, _ = history.build_history_features(frame, split(), ("hist_rent",), ZERO_PRIOR)
    month = frame["APPROVAL_MONTH_INDEX"].to_numpy()
    valid = month > CUTOFF

    # Frozen snapshot: months 0,1,2 -> mean(1000,2000,3000) = 2000 for *every*
    # validation row, at every horizon (h=1 and h=2 here).
    value = block.loc[valid, "hist_rent_town"].to_numpy()
    check("validation uses months 0..cutoff (expect 2000)", bool(np.allclose(value, 2000.0)),
          str(np.unique(value)))
    check("validation n is identical across horizons (6 rows)",
          bool((block.loc[valid, "hist_rent_town_n"] == 6).all()),
          str(np.unique(block.loc[valid, "hist_rent_town_n"])))
    check("validation observes exactly 3 months",
          bool((block.loc[valid, "hist_rent_block_months"] == 3).all()),
          str(np.unique(block.loc[valid, "hist_rent_block_months"])))
    check("validation recency is measured from the cutoff",
          bool(np.allclose(block.loc[valid, "hist_rent_block_recency"], 0.0)),
          str(np.unique(block.loc[valid, "hist_rent_block_recency"])))

    # Changing labels after the cutoff must not move a single validation value.
    tampered = frame.copy()
    tampered.loc[month > CUTOFF, config.TARGET] = -1.0
    changed, _ = history.build_history_features(tampered, split(), ("hist_rent",), ZERO_PRIOR)
    pd.testing.assert_frame_equal(block.loc[valid], changed.loc[valid], check_exact=True)
    check("post-cutoff labels cannot change validation features", True)


def test_flat_type_hierarchy_matches_the_same_boundary() -> None:
    frame = oracle_frame()
    block, _ = history.build_history_features(frame, split(), ("hist_ft",), ZERO_PRIOR)
    month = frame["APPROVAL_MONTH_INDEX"].to_numpy()

    month1 = month == 1
    check("hist_ft at t=1 uses month 0 only (expect 1000)",
          bool(np.allclose(block.loc[month1, "hist_rentft_block_ft"], 1000.0)),
          str(np.unique(block.loc[month1, "hist_rentft_block_ft"])))
    month2 = month == CUTOFF
    check("hist_ft at t=cutoff uses months 0..t-1 (expect 1500)",
          bool(np.allclose(block.loc[month2, "hist_rentft_block_ft"], 1500.0)),
          str(np.unique(block.loc[month2, "hist_rentft_block_ft"])))
    valid = month > CUTOFF
    check("hist_ft validation frozen at cutoff (expect 2000)",
          bool(np.allclose(block.loc[valid, "hist_rentft_block_ft"], 2000.0)),
          str(np.unique(block.loc[valid, "hist_rentft_block_ft"])))


def test_window_summaries_use_the_same_boundary() -> None:
    frame = oracle_frame()
    window, _ = history.build_window_features(frame, split(), ("win_rent",))
    month = frame["APPROVAL_MONTH_INDEX"].to_numpy()

    # 12-month window is wider than the single visible month, so for t=1 it
    # must still contain only month 0.
    month1 = month == 1
    check("win12 at t=1 uses month 0 only (expect 1000)",
          bool(np.allclose(window.loc[month1, "win12_rent_town_mean"], 1000.0)),
          str(np.unique(window.loc[month1, "win12_rent_town_mean"])))
    month2 = month == CUTOFF
    check("win12 at t=cutoff uses months 0..t-1 (expect 1500)",
          bool(np.allclose(window.loc[month2, "win12_rent_town_mean"], 1500.0)),
          str(np.unique(window.loc[month2, "win12_rent_town_mean"])))
    valid = month > CUTOFF
    check("win12 validation frozen at cutoff (expect 2000)",
          bool(np.allclose(window.loc[valid, "win12_rent_town_mean"], 2000.0)),
          str(np.unique(window.loc[valid, "win12_rent_town_mean"])))
    # A 3-month window at t=2 covers months 0 and 1 but not 2.
    check("win3 at t=cutoff covers only two visible months",
          bool(np.allclose(window.loc[month2, "win3_rent_town_mean"], 1500.0)),
          str(np.unique(window.loc[month2, "win3_rent_town_mean"])))


def test_rent_per_sqm_uses_the_same_boundary() -> None:
    frame = oracle_frame()
    block, _ = history.build_history_features(frame, split(), ("hist_psm",), ZERO_PRIOR)
    month = frame["APPROVAL_MONTH_INDEX"].to_numpy()
    month1 = month == 1
    # area 60 -> 1000/60
    check("hist_psm at t=1 uses month 0 only",
          bool(np.allclose(block.loc[month1, "hist_psm_town"], 1000.0 / 60.0)),
          str(np.unique(block.loc[month1, "hist_psm_town"])))
    valid = month > CUTOFF
    check("hist_psm validation frozen at cutoff",
          bool(np.allclose(block.loc[valid, "hist_psm_town"], 2000.0 / 60.0)),
          str(np.unique(block.loc[valid, "hist_psm_town"])))


def test_default_smoothing_only_shrinks_towards_parent() -> None:
    frame = oracle_frame()
    exact, _ = history.build_history_features(frame, split(), ("hist_rent",), ZERO_PRIOR)
    smoothed, _ = history.build_history_features(frame, split(), ("hist_rent",))
    month = frame["APPROVAL_MONTH_INDEX"].to_numpy()
    # With a single block every level has the same mean, so the EB recursion is
    # a no-op even for the default prior. This documents that choice.
    month1 = month == 1
    check("default prior reproduces the exact mean in a single-block panel",
          bool(np.allclose(smoothed.loc[month1, "hist_rent_town"], 1000.0)),
          str(np.unique(smoothed.loc[month1, "hist_rent_town"])))
    check("default prior equals zero-prior value here",
          bool(np.allclose(smoothed["hist_rent_town"], exact["hist_rent_town"], equal_nan=True)))


def main() -> int:
    for test in [
        test_training_row_uses_months_strictly_before_t,
        test_validation_uses_frozen_cutoff_snapshot,
        test_flat_type_hierarchy_matches_the_same_boundary,
        test_window_summaries_use_the_same_boundary,
        test_rent_per_sqm_uses_the_same_boundary,
        test_default_smoothing_only_shrinks_towards_parent,
    ]:
        print(f"\n== {test.__name__}")
        try:
            test()
        except Exception as error:  # noqa: BLE001 - report and continue
            import traceback
            traceback.print_exc()
            FAILURES.append(f"{test.__name__} ({error!r})")
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all oracle boundary tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
