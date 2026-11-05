"""One small hand-checkable fixture for the global-rent bank.

Three properties only, as the plan requires — this is not a broad suite:

1. **Monthly mean is not the cumulative mean**, and multi-month windows average the monthly
   means (equal month weight), not the pooled transactions.
2. **An empty month is missing, never zero** — and a window containing it averages only the
   observed months.
3. **Future months cannot change a feature** at this origin.

Run:
    .venv/bin/python -m experiments.global_rent_mean_v6.tests_series
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402

PASSED, FAILED = [], []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"PASS  {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"FAIL  {name}  {detail}")


def main() -> int:
    import pandas as pd
    from experiments.global_rent_mean_v6 import series as S

    # ------------------------------------------------------------------ 1. mean vs pooled
    # Month 0: two transactions, 1000 and 5000 -> monthly mean 3000 from 2 units.
    # Month 1: four transactions, all 6000   -> monthly mean 6000 from 4 units.
    # Equal-month-weight mean3 over months 0..1 is (3000 + 6000) / 2 = 4500.
    # Pooling all six transactions gives (1000+5000+4*6000) / 6 = 5000.
    # A cumulative mean is the same as pooling here, so both alternatives are distinguishable.
    rows = [
        {"month": 0, "rent": 1000.0},
        {"month": 0, "rent": 5000.0},
        {"month": 1, "rent": 6000.0},
        {"month": 1, "rent": 6000.0},
        {"month": 1, "rent": 6000.0},
        {"month": 1, "rent": 6000.0},
    ]
    frame = pd.DataFrame(rows)
    means, counts = S.build_series(frame, rent_col="rent", month_col="month", fit_end=2)
    print(f"monthly means {means.tolist()}  counts {counts.tolist()}")
    check("month 0 mean is the within-month mean", means[0] == 3000.0, str(means[0]))
    check("month 1 mean is the within-month mean", means[1] == 6000.0, str(means[1]))

    names = S.feature_names()
    vec = S.features_at(means, counts, origin=1)
    mean3f = float(np.exp(vec[names.index("grm_mean3_log")]))
    pooled = (1000.0 + 5000.0 + 4 * 6000.0) / 6
    check("mean3 averages the monthly means, not the pooled transactions",
          bool(np.isclose(mean3f, 4500.0, rtol=1e-12)),
          f"{mean3f} vs equal-month 4500 (pooled would be {pooled:.2f})")
    check("the pooled alternative is materially different, so this test discriminates",
          abs(mean3f - pooled) > 400.0, f"{mean3f} vs {pooled:.2f}")
    check("the bank is not a cumulative average either",
          abs(mean3f - pooled) > 400.0, f"mean3={mean3f} cumulative={pooled:.2f}")

    # ------------------------------------------------------------------ 2. empty != zero
    sparse = [
        {"month": 0, "rent": 2000.0},
        # month 1 has no transactions at all
        {"month": 2, "rent": 4000.0},
    ]
    sp = pd.DataFrame(sparse)
    m2, c2 = S.build_series(sp, rent_col="rent", month_col="month", fit_end=2)
    print(f"sparse means {m2.tolist()}  counts {c2.tolist()}")
    check("an empty month is NaN, not zero", bool(np.isnan(m2[1])), str(m2[1]))
    check("an empty month has count 0", c2[1] == 0.0, str(c2[1]))

    v2 = S.features_at(m2, c2, origin=2)
    mean3_sparse = float(np.exp(v2[names.index("grm_mean3_log")]))
    valid3 = float(v2[names.index("grm_valid3")])
    # only months 0 and 2 exist -> (2000 + 4000) / 2 = 3000, never 6000/3 = 2000
    check("a window with an empty month averages only the observed months",
          bool(np.isclose(mean3_sparse, 3000.0, rtol=1e-12)),
          f"{mean3_sparse} (a zero-filled mean would be 2000)")
    check("the support count reports how many months were used", valid3 == 2.0, str(valid3))

    # latest with age: at origin 1 the latest observed month is 0, so age = 1
    v1 = S.features_at(m2, c2, origin=1)
    age = float(v1[names.index("grm_latest_age")])
    check("the latest-level age counts months since the newest observation", age == 1.0, str(age))

    # The name says "latest available", so an EMPTY origin month must still yield the last
    # observed level rather than NaN. In the present data every origin month trades, which is
    # exactly why this edge case needs a fixture instead of relying on the run to expose it.
    latest_empty = float(np.exp(v1[names.index("grm_latest_log")]))
    check("an empty origin month still reports the last observed level",
          bool(np.isclose(latest_empty, 2000.0, rtol=1e-12)),
          f"{latest_empty} (origin month 1 is empty; month 0 had 2000)")
    dlog_empty = v1[names.index("grm_dlog_1")]
    check("the log change is measured from that last observation, not the empty month",
          bool(np.isnan(dlog_empty)),
          f"{dlog_empty} (no earlier month exists, so missing is correct)")

    # with two observed months separated by a gap, dlog compares the two observed levels
    gap = [{"month": 0, "rent": 2000.0},
           # month 1 empty
           {"month": 2, "rent": 2500.0},
           # month 3 empty
           ]
    gm, gc = S.build_series(pd.DataFrame(gap), rent_col="rent", month_col="month", fit_end=3)
    v3 = S.features_at(gm, gc, origin=3)
    lvl = float(np.exp(v3[names.index("grm_latest_log")]))
    ag = float(v3[names.index("grm_latest_age")])
    dl = float(v3[names.index("grm_dlog_1")])
    check("a run of empty months does not zero out the latest level",
          bool(np.isclose(lvl, 2500.0, rtol=1e-12)), f"{lvl}")
    check("the age reflects the whole empty run", ag == 1.0, f"{ag}")
    check("the log change spans the observed gap",
          bool(np.isclose(dl, np.log(2500.0) - np.log(2000.0), rtol=1e-12)),
          f"{dl:.6f} vs {np.log(1.25):.6f}")

    # ------------------------------------------------------------------ 3. causality
    base_rows = [{"month": m, "rent": 2000.0 + 100 * m} for m in range(4)]
    base_means, base_counts = S.build_series(
        pd.DataFrame(base_rows), rent_col="rent", month_col="month", fit_end=3)
    base_vec = S.features_at(base_means, base_counts, origin=2)

    perturbed_rows = base_rows + [
        {"month": 3, "rent": 99999.0},
        {"month": 4, "rent": 1.0},
    ]
    per_means, per_counts = S.build_series(
        pd.DataFrame(perturbed_rows), rent_col="rent", month_col="month", fit_end=4)
    per_vec = S.features_at(per_means, per_counts, origin=2)
    check("perturbing months after the origin leaves the vector unchanged",
          np.array_equal(base_vec, per_vec, equal_nan=True),
          f"{base_vec} vs {per_vec}")

    refused = False
    try:
        S.build_series(pd.DataFrame(perturbed_rows), rent_col="rent",
                       month_col="month", fit_end=2)
    except Exception as exc:                                   # noqa: BLE001
        refused = "fit_end" in str(exc)
    check("a frame containing months beyond fit_end is refused", refused)

    print("\n" + "=" * 60)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for item in FAILED:
        print(f"  - {item}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
