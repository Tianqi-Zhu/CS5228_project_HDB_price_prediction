"""Hand-checkable boundary tests for the v9 history summaries.

Each block targets one way this feature set could silently be wrong. The numbers are small
enough to verify by hand.

1. **No self-label leak.** A row's naive expected value is absent from its own history even
   if the group is otherwise dense, and injecting an extreme rent at the row's own month (or
   later) must not change that row's features.
2. **Recency is measured to the target month**, not to the cutoff, and `history_cutoff_age`
   is the separate policy gap.
3. **Empty history is not a zero.** count=0, means/std NaN, recency=-1.
4. **std needs 2 observations.** count=1 gives NaN std, not 0.
5. **EW mean weights recent months more**, computed by hand for a 12-month half life.
6. **Blocks never join across streets.**
7. **The cutoff never reaches the target month** and validation never passes the train end.

Run:
    .venv/bin/python -m experiments.history_features_v9.tests.test_history
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

from experiments.history_features_v9 import features as F  # noqa: E402

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"PASS  {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"FAIL  {name}  {detail}")


def make_frame(rows):
    """Build a minimal source-shaped frame from (month, rent_sgd, town, street, block) rows."""
    records = []
    for i, row in enumerate(rows):
        month, rent, town, street, block = row[:5]
        records.append({
            "SOURCE_ROW_ID": 1000 + i,
            "RENT_APPROVAL_DATE": (pd.Period("2021-01", freq="M") + month).strftime("%Y-%m"),
            "TOWN_CLEAN": town,
            "STREET_CLEAN": street,
            "BLOCK_CLEAN": block,
            "FLAT_TYPE_CLEAN": row[5] if len(row) > 5 else "4-ROOM",
            "FLAT_MODEL_CLEAN": "model a",
            "SUBZONE": row[6] if len(row) > 6 else "sz1",
            "FLOOR_AREA_SQM": row[7] if len(row) > 7 else 92.0,
            "LEASE_COMMENCE_DATE": 2000,
            "MONTHLY_RENT": rent,
        })
    return pd.DataFrame(records)


def build(frame):
    months = np.asarray(
        [(pd.Period(d, freq="M") - pd.Period("2021-01", freq="M")).n
         for d in frame["RENT_APPROVAL_DATE"]], dtype=np.int64)
    rent_ksgd = frame["MONTHLY_RENT"].to_numpy(float) / 1000.0
    return F.GroupHistory(frame, months, rent_ksgd)


def stats_of(history, level, key, target, cutoff):
    values = history.summarise(level, key, target, cutoff)
    return dict(zip(F.STATS, values))


def main() -> int:
    # ---------------------------------------------------------------- 1. no self leak
    print("--- 1. a row's own label cannot enter its own history")
    frame = make_frame([
        (0, 2000, "T1", "S1", "10"),
        (1, 2100, "T1", "S1", "10"),
        (2, 2200, "T1", "S1", "10"),
        (3, 2300, "T1", "S1", "10"),
    ])
    history = build(frame)
    key = ("t1", "s1", "10", "4-room")
    # A row targeting month 3 has cutoff 0, so it may only see month 0.
    s = stats_of(history, "town_street_block_ft", key, target=3, cutoff=0)
    check("cutoff 0 sees exactly one observation", s["count"] == 1.0, str(s["count"]))
    check("that observation is month 0's rent", bool(np.isclose(s["mean"], 2.0)), str(s["mean"]))
    check("months 1..3 are excluded even though they exist", s["count"] < 4.0)

    # inflate the label at the target month and later: nothing may change
    poisoned = make_frame([
        (0, 2000, "T1", "S1", "10"),
        (1, 2100, "T1", "S1", "10"),
        (2, 2200, "T1", "S1", "10"),
        (3, 999999, "T1", "S1", "10"),
    ])
    poisoned_history = build(poisoned)
    s2 = stats_of(poisoned_history, "town_street_block_ft", key, target=3, cutoff=0)
    check("an extreme rent at the target month leaves the history unchanged",
          np.array_equal(np.array(list(s.values()), dtype=float),
                         np.array(list(s2.values()), dtype=float), equal_nan=True),
          f"{s} vs {s2}")

    # ---------------------------------------------------------------- 2. recency basis
    print("\n--- 2. recency is measured to the target month, not the cutoff")
    frame2 = make_frame([
        (0, 2000, "T1", "S1", "10"),
        (1, 2100, "T1", "S1", "10"),
    ])
    h2 = build(frame2)
    # target 5, cutoff 2 -> newest observation is month 1, so recency = 5-1 = 4
    s = stats_of(h2, "town_street_block_ft", key, target=5, cutoff=2)
    check("recency is t - last_observed_month (5-1=4)", s["recency"] == 4.0, str(s["recency"]))
    check("recency is NOT c - last_observed_month (which would be 1)",
          s["recency"] != 1.0, str(s["recency"]))
    check("history_cutoff_age is the separate policy gap (5-2=3)",
          F.cutoff_age(5, 2) == 3, str(F.cutoff_age(5, 2)))

    # ---------------------------------------------------------------- 3. empty history
    print("\n--- 3. a group with no history is not a zero")
    frame3 = make_frame([(0, 2000, "T1", "S1", "10")])
    h3 = build(frame3)
    absent = ("t9", "s9", "99", "4-room")
    s = stats_of(h3, "town_street_block_ft", absent, target=5, cutoff=2)
    check("count is 0", s["count"] == 0.0, str(s["count"]))
    check("mean is NaN, not 0", bool(np.isnan(s["mean"])), str(s["mean"]))
    check("sample_std is NaN, not 0", bool(np.isnan(s["sample_std"])), str(s["sample_std"]))
    check("recency is the -1 sentinel", s["recency"] == -1.0, str(s["recency"]))

    # ---------------------------------------------------------------- 4. std needs n>=2
    print("\n--- 4. sample_std needs at least two observations")
    s1 = stats_of(h3, "town_street_block_ft", ("t1", "s1", "10", "4-room"), target=5, cutoff=2)
    check("count 1 gives NaN std, not 0", bool(np.isnan(s1["sample_std"])),
          str(s1["sample_std"]))
    frame4 = make_frame([(0, 2000, "T1", "S1", "10"), (1, 3000, "T1", "S1", "10")])
    h4 = build(frame4)
    s2 = stats_of(h4, "town_street_block_ft", key, target=5, cutoff=2)
    expected_std = float(np.std([2.0, 3.0], ddof=1))
    check("count 2 gives the ddof=1 sample std", bool(np.isclose(s2["sample_std"], expected_std)),
          f"{s2['sample_std']} vs {expected_std}")

    # ---------------------------------------------------------------- 5. EW mean by hand
    print("\n--- 5. the EW mean weights the recent month more (12-month half life)")
    frame5 = make_frame([
        (0, 1000, "T1", "S1", "10"),
        (12, 3000, "T1", "S1", "10"),
    ])
    h5 = build(frame5)
    # cutoff 12: month 12 has weight 2^0 = 1, month 0 has weight 2^-1 = 0.5
    s = stats_of(h5, "town_street_block_ft", key, target=15, cutoff=12)
    expected_ew = (3.0 * 1.0 + 1.0 * 0.5) / 1.5
    check("EW mean equals the hand-computed weighted average",
          bool(np.isclose(s["ew_mean"], expected_ew)), f"{s['ew_mean']} vs {expected_ew}")
    check("the EW mean sits above the plain mean, as recency-weighted should",
          s["ew_mean"] > s["mean"], f"{s['ew_mean']} vs {s['mean']}")

    # ---------------------------------------------------------------- 6. block vs street
    print("\n--- 6. an identical block number on another street is a different group")
    frame6 = make_frame([
        (0, 1000, "T1", "S1", "10"),
        (1, 5000, "T1", "S2", "10"),     # same town+block, different street
    ])
    h6 = build(frame6)
    s = stats_of(h6, "town_street_block_ft", ("t1", "s1", "10", "4-room"), target=5, cutoff=4)
    check("block 10 on street S1 does not absorb street S2's rent",
          bool(np.isclose(s["mean"], 1.0)) and s["count"] == 1.0,
          f"count {s['count']} mean {s['mean']}")

    # ---------------------------------------------------------------- 7. cutoff rule
    print("\n--- 7. the cutoff rule never reaches the target month")
    for t in (0, 1, 2, 3, 26, 27, 38):
        c_train = F.cutoff_for(t, train_end_month=26)
        check(f"target {t}: training cutoff {c_train} < {t}", c_train < t, str(c_train))
    check("validation at t=27 stops at 24 (t-3, below the train end)",
          F.cutoff_for(27, train_end_month=26) == 24, str(F.cutoff_for(27, train_end_month=26)))
    check("validation at t=30 is capped at the train end 26",
          F.cutoff_for(30, train_end_month=26) == 26, str(F.cutoff_for(30, train_end_month=26)))
    check("target 2 gives cutoff -1, i.e. no history at all",
          F.cutoff_for(2, train_end_month=26) == -1, str(F.cutoff_for(2, train_end_month=26)))
    s_neg = stats_of(h6, "town_street_block_ft", key, target=2, cutoff=-1)
    check("a negative cutoff yields empty history rather than dropping the row",
          s_neg["count"] == 0.0 and s_neg["recency"] == -1.0, str(s_neg["count"]))

    # ---------------------------------------------------------------- 8. scaler
    print("\n--- 8. scaler: log1p count, impute after masking, mask preserved")
    names = F.feature_names()
    raw = np.full((3, len(names)), np.nan)
    count_idx = names.index("hist_flat_type__count")
    mean_idx = names.index("hist_flat_type__mean")
    raw[0, count_idx] = 0.0
    raw[1, count_idx] = 3.0
    raw[2, count_idx] = 10.0
    raw[1, mean_idx] = 2.0
    raw[2, mean_idx] = 4.0
    scaler = F.HistoryScaler.fit(raw, names)
    scaled, observed = scaler.transform(raw)
    check("count is log1p-transformed before scaling",
          np.allclose(scaler.median[count_idx], np.log1p(3.0)),
          f"{scaler.median[count_idx]} vs {np.log1p(3.0)}")
    check("a count of 0 counts as observed information", bool(observed[0, count_idx]))
    check("a missing mean is marked unobserved", not bool(observed[0, mean_idx]))
    check("the imputed mean equals the fit median, not 0",
          bool(np.isclose(scaled[0, mean_idx], (3.0 - scaler.mean[mean_idx])
                          / scaler.std[mean_idx])),
          f"{scaled[0, mean_idx]}")
    check("scaled output is finite everywhere", bool(np.isfinite(scaled).all()))

    print("\n" + "=" * 62)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for item in FAILED:
        print(f"  - {item}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
