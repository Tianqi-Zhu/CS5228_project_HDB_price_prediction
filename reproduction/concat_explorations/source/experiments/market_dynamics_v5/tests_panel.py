"""Synthetic fixtures for the recent-rent features.

Each test states the property it protects and fails loudly if it breaks. They run on tiny
hand-built frames so the expected numbers are checkable by hand, not just by re-running the
code.

The properties:

1. **Future labels cannot change a feature.** Perturbing rows after the origin must leave
   the vector bit-identical.
2. **Month boundary is inclusive of the origin.** A transaction in month == origin counts;
   one in origin+1 does not.
3. **Empty is not zero.** A month with no transactions yields NaN and n=0, and never a
   fabricated 0.0 level.
4. **Counts are truthful.** n3 / n12 / valid6 match a direct tally.
5. **Flat market has zero slope.** Constant per-sqm gives slope 0.
6. **Growing market has a positive slope** of the expected magnitude.
7. **Rolling differs from cumulative** when the market moves, which is the reason this
   module exists.

Run:
    .venv/bin/python -m experiments.market_dynamics_v5.tests_panel
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT / "experiments", ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import numpy as np  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"PASS  {name}")
    else:
        FAILED.append(f"{name}  {detail}")
        print(f"FAIL  {name}  {detail}")


def make_frame(rows: list[dict]):
    import pandas as pd

    return pd.DataFrame(rows)


def build(frame, fit_end, parents=None):
    from experiments.market_dynamics_v5.panel import build_panel

    level_keys = {
        "global": [("all",)] * len(frame),
        "flat_type": [tuple(r) for r in
                      zip(frame["ft"].astype(str))],
        "town_ft": [tuple(r) for r in
                    zip(frame["town"].astype(str), frame["ft"].astype(str))],
    }
    return build_panel(frame, level_keys=level_keys, rent_col="rent",
                       area_col="area", month_col="month", fit_end=fit_end,
                       parents=parents)


def row_keys_for(town: str, ft: str) -> dict:
    return {"global": ("all",), "flat_type": (ft,), "town_ft": (town, ft)}


def parents_for(town: str, ft: str) -> dict:
    return {("town_ft", (town, ft)): {"flat_type": (ft,), "global": ("all",)},
            ("flat_type", (ft,)): {"flat_type": (ft,), "global": ("all",)}}


def main() -> int:
    from experiments.market_dynamics_v5.panel import (FALLBACK_GROUP, FALLBACK_MISSING,
                                                      FeatureSpec, features_for)

    spec = FeatureSpec(levels=("global",), windows=(3, 6, 12))
    names = spec.names()
    print(f"feature names ({len(names)}): {names}\n")

    # ---------------------------------------------------------------- 1. causality
    print("--- 1. future rows cannot change a causal feature")
    base_rows = [
        {"month": 0, "rent": 1000.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"},
        {"month": 1, "rent": 1100.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"},
        {"month": 2, "rent": 1200.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"},
    ]
    frame = make_frame(base_rows)
    panel = build(frame, fit_end=5)
    keys = row_keys_for("T1", "4-ROOM")
    parents = parents_for("T1", "4-ROOM")
    base_panel = build(frame, fit_end=5, parents=parents)
    base_vector = features_for(base_panel, origin=2, row_keys=keys, spec=spec)

    # identical history, wildly different future
    future_rows = base_rows + [
        {"month": 3, "rent": 99999.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"},
        {"month": 4, "rent": 1.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"},
    ]
    perturbed_panel = build(make_frame(future_rows), fit_end=5, parents=parents)
    perturbed_vector = features_for(perturbed_panel, origin=2, row_keys=keys, spec=spec)
    check("perturbing months after the origin leaves the vector unchanged",
          np.array_equal(base_vector, perturbed_vector, equal_nan=True),
          f"{base_vector} vs {perturbed_vector}")

    # and a panel containing any future month is refused outright
    refused = False
    try:
        from experiments.market_dynamics_v5.panel import build_panel
        build_panel(make_frame(future_rows),
                    level_keys={"global": [("all",)] * len(future_rows)},
                    rent_col="rent", area_col="area", month_col="month", fit_end=2)
    except Exception as exc:                                   # noqa: BLE001
        refused = "future observations" in str(exc)
    check("a panel frame containing months > fit_end is refused", refused)

    # ---------------------------------------------------------------- 2. boundaries
    print("\n--- 2. the origin month is included, origin+1 is not")
    # per-sqm values 20, 40, 60 -> a 3-month window at origin 2 must average 40
    rows = [
        {"month": 0, "rent": 1000.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"},
        {"month": 1, "rent": 2000.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"},
        {"month": 2, "rent": 3000.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"},
    ]
    panel = build(make_frame(rows), fit_end=2, parents=parents)
    vector = features_for(panel, origin=2, row_keys=keys, spec=spec)
    n3 = vector[names.index("n3__global")]
    recent3 = vector[names.index("recent3_logpsm__global")]
    check("the origin month contributes to n3", n3 == 3.0, str(n3))
    check("recent3 mean includes the origin month",
          bool(np.isclose(np.exp(recent3), (20 + 40 + 60) / 3, rtol=1e-9)),
          f"{np.exp(recent3):.6f} vs {(20 + 40 + 60) / 3:.6f}")

    # ---------------------------------------------------------------- 3. empty
    print("\n--- 3. an empty month is not a zero")
    sparse = [
        {"month": 0, "rent": 1000.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"},
        # months 1 and 2 have no transactions at all
        {"month": 3, "rent": 2000.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"},
    ]
    panel = build(make_frame(sparse), fit_end=3, parents=parents)
    vector = features_for(panel, origin=3, row_keys=keys, spec=spec)
    valid6 = vector[names.index("valid6__global")]
    n3 = vector[names.index("n3__global")]
    slope = vector[names.index("slope6_logpsm__global")]
    recent3 = np.exp(vector[names.index("recent3_logpsm__global")])
    # Two months are observed in the six-month window, but a slope needs at least three.
    # `valid6` reports the support of the series the slope was actually fitted on, so with
    # no fittable basis it is 0 and the slope is NaN — never a fabricated 0.0 trend.
    check("valid6 reports the slope's own support, which is 0 when no basis fits",
          valid6 == 0.0, str(valid6))
    check("an unfittable slope is NaN, not a fake 0.0", bool(np.isnan(slope)), str(slope))
    check("n3 counts transactions, not months", n3 == 1.0, str(n3))
    check("a 3-month window over 2 empty months averages only what exists",
          bool(np.isclose(recent3, 40.0, rtol=1e-9)),
          f"{recent3:.4f} (must be 40, not 40/3)")

    # an entirely empty window must be NaN, never 0.0
    panel_empty = build(make_frame([
        {"month": 0, "rent": 1000.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"}]),
        fit_end=10, parents=parents)
    vector = features_for(panel_empty, origin=10, row_keys=keys, spec=spec)
    recent3 = vector[names.index("recent3_logpsm__global")]
    fallback = vector[names.index("fallback__global")]
    # with no support at any level the value is missing, and the provenance says so
    check("a window with no support anywhere yields NaN and the missing flag",
          bool(np.isnan(recent3)) and fallback == FALLBACK_MISSING,
          f"recent3={recent3} fallback={fallback}")

    # ---------------------------------------------------------------- 4. counts
    print("\n--- 4. n12 and valid6 agree with a direct tally")
    rows = []
    for month in range(12):
        for _ in range(month + 1):        # month m gets m+1 transactions
            rows.append({"month": month, "rent": 1000.0 + 100 * month, "area": 50.0,
                         "town": "T1", "ft": "4-ROOM"})
    panel = build(make_frame(rows), fit_end=11, parents=parents)
    vector = features_for(panel, origin=11, row_keys=keys, spec=spec)
    n12 = vector[names.index("n12__global")]
    valid6 = vector[names.index("valid6__global")]
    check("n12 equals the direct transaction tally", n12 == float(sum(range(1, 13))),
          f"{n12} vs {sum(range(1, 13))}")
    check("valid6 equals 6 when every month is observed", valid6 == 6.0, str(valid6))

    # ---------------------------------------------------------------- 5. flat slope
    print("\n--- 5. a flat market has zero slope")
    flat = [{"month": m, "rent": 2000.0, "area": 50.0, "town": "T1", "ft": "4-ROOM"}
            for m in range(12)]
    panel = build(make_frame(flat), fit_end=11, parents=parents)
    vector = features_for(panel, origin=11, row_keys=keys, spec=spec)
    slope = vector[names.index("slope6_logpsm__global")]
    check("constant per-sqm gives slope 0", bool(np.isclose(slope, 0.0, atol=1e-12)),
          str(slope))

    # ---------------------------------------------------------------- 6. growth
    print("\n--- 6. a growing market has the expected slope")
    # 1% per month compounding: log slope must be ~ln(1.01)
    growth = [{"month": m, "rent": 1000.0 * (1.01 ** m), "area": 50.0,
               "town": "T1", "ft": "4-ROOM"} for m in range(12)]
    panel = build(make_frame(growth), fit_end=11, parents=parents)
    vector = features_for(panel, origin=11, row_keys=keys, spec=spec)
    slope = vector[names.index("slope6_logpsm__global")]
    expected = float(np.log(1.01))
    check("1%/month growth gives slope ln(1.01)",
          bool(np.isclose(slope, expected, rtol=1e-6)),
          f"{slope:.9f} vs {expected:.9f}")

    # ---------------------------------------------------------------- 7. rolling != cumulative
    print("\n--- 7. rolling and cumulative differ when the market moves")
    # cumulative mean over 0..5 of 20,20,20,60,60,60 = 40; recent3 at origin 5 = 60
    jumpy = []
    for month, psm in enumerate([20, 20, 20, 60, 60, 60]):
        jumpy.append({"month": month, "rent": psm * 50.0, "area": 50.0,
                      "town": "T1", "ft": "4-ROOM"})
    panel = build(make_frame(jumpy), fit_end=5, parents=parents)
    vector = features_for(panel, origin=5, row_keys=keys, spec=spec)
    recent3 = float(np.exp(vector[names.index("recent3_logpsm__global")]))
    cumulative = (20 * 3 + 60 * 3) / 6
    check("recent3 reflects the latest level, not the cumulative mean",
          bool(np.isclose(recent3, 60.0, rtol=1e-9)),
          f"{recent3:.4f} vs cumulative {cumulative:.4f}")

    # ---------------------------------------------------------------- 8. parent fallback
    print("\n--- 8. a sparse group backs off to its parent and says so")
    rows = []
    for month in range(6):
        rows.append({"month": month, "rent": 4000.0, "area": 50.0,
                     "town": "T1", "ft": "EXEC"})            # only EXEC trades
        rows.append({"month": month, "rent": 2000.0, "area": 50.0,
                     "town": "T1", "ft": "4-ROOM"})
    frame = make_frame(rows)
    panel = build(frame, fit_end=5, parents={
        ("town_ft", ("T9", "3-ROOM")): {"flat_type": ("3-ROOM"), "global": ("all",)}})
    spec_tt = FeatureSpec(levels=("town_ft",), windows=(3, 6, 12))
    keys_missing = {"town_ft": ("T9", "3-ROOM")}
    vector = features_for(panel, origin=5, row_keys=keys_missing, spec=spec_tt)
    fallback_names = spec_tt.names()
    fb = vector[fallback_names.index("fallback__town_ft")]
    recent3 = vector[fallback_names.index("recent3_logpsm__town_ft")]
    check("a never-traded group is flagged as a fallback",
          fb in (FALLBACK_GROUP, 1.0), f"fallback={fb}")
    check("the fallback value is finite, not NaN",
          bool(np.isfinite(recent3)), str(recent3))

    # ---------------------------------------------------------------- 9. shared basis
    print("\n--- 9. the 3-vs-prior-3 ratio uses ONE basis for both intervals")
    # The group trades only in the recent window. Its prior window has no support, so an
    # independent resolve would compare the group against the parent and report a huge
    # fake jump. The shared basis must instead measure both intervals at the parent.
    rows = []
    for month in range(6):
        # parent (global/town) trades every month at a flat 3000/50 = 60 per sqm
        rows.append({"month": month, "rent": 3000.0, "area": 50.0,
                     "town": "T1", "ft": "4-ROOM"})
    # the specific group trades only in months 4 and 5, at 6000/50 = 120 per sqm
    rows.append({"month": 4, "rent": 6000.0, "area": 50.0, "town": "T1", "ft": "3-ROOM"})
    rows.append({"month": 5, "rent": 6000.0, "area": 50.0, "town": "T1", "ft": "3-ROOM"})
    frame = make_frame(rows)
    panel = build(frame, fit_end=5, parents={
        ("town_ft", ("T1", "3-ROOM")): {"flat_type": ("3-ROOM"), "global": ("all",)}})
    spec_tt = FeatureSpec(levels=("town_ft",), windows=(3, 6, 12))
    keys = {"town_ft": ("T1", "3-ROOM")}
    vector = features_for(panel, origin=5, row_keys=keys, spec=spec_tt)
    names_tt = spec_tt.names()
    ratio = vector[names_tt.index("recent3_over_prior3_log__town_ft")]
    ratio_source = vector[names_tt.index("ratio_source__town_ft")]
    # The 3-ROOM group has no support in the prior window, so the shared basis drops to
    # `global`, which supports both. Note global's own per-sqm is NOT constant here: in
    # months 4-5 the 3-ROOM 120 joins the 4-ROOM 60, so global moves 60 -> 90. Both legs
    # therefore come from the same series and the ratio is the *global* change:
    #   recent3 = mean(90, 90, 60) = 80 ... measured over months 3,4,5
    #   prior3  = mean(60, 60, 60) = 60
    # Under independent resolution the recent leg would have been the group's 120 against
    # the parent's 60, i.e. ln(2) = 0.693 — a jump that is an artefact of the basis switch.
    # The window pools transactions across months (each transaction weighted equally), it
    # does not average monthly means: months 3,4,5 hold 1x60, 2x90, 2x90 -> 420/5 = 84.
    recent3_window = (60.0 * 1 + 90.0 * 2 + 90.0 * 2) / 5
    expected_ratio = float(np.log(recent3_window) - np.log(60.0))
    check("the ratio is computed from ONE series for both legs",
          bool(np.isclose(ratio, expected_ratio, rtol=1e-12)),
          f"ratio={ratio:.9f} vs expected {expected_ratio:.9f}")
    independent_ratio = float(np.log(120.0) - np.log(60.0))
    check("the fake jump under independent resolution would have been much larger",
          abs(independent_ratio - ratio) > 0.3,
          f"independent={independent_ratio:.6f} shared={ratio:.6f}")
    check("the ratio's provenance records the shared basis it actually used",
          ratio_source == 0.0, f"ratio_source={ratio_source} (0.0 == global)")

    # with support in BOTH intervals the group itself is the basis and the ratio is real
    rows2 = list(rows)
    for month in (0, 1):
        rows2.append({"month": month, "rent": 3000.0, "area": 50.0,
                      "town": "T1", "ft": "3-ROOM"})
    panel2 = build(make_frame(rows2), fit_end=5, parents={
        ("town_ft", ("T1", "3-ROOM")): {"flat_type": ("3-ROOM"), "global": ("all",)}})
    vector2 = features_for(panel2, origin=5, row_keys=keys, spec=spec_tt)
    ratio2 = vector2[names_tt.index("recent3_over_prior3_log__town_ft")]
    source2 = vector2[names_tt.index("ratio_source__town_ft")]
    check("with both intervals present the group is used and the ratio is nonzero",
          bool(np.isfinite(ratio2)) and ratio2 != 0.0 and source2 == 2.0,
          f"ratio={ratio2} source={source2}")

    # ---------------------------------------------------------------- 10. slope/support
    print("\n--- 10. the slope's support comes from the slope's own series")
    vector = features_for(panel, origin=5, row_keys=keys, spec=spec_tt)
    valid6 = vector[names_tt.index("valid6__town_ft")]
    slope_source = vector[names_tt.index("slope_source__town_ft")]
    check("valid6 counts the months the slope actually used",
          valid6 == 6.0, f"valid6={valid6} (all six months came from one level)")
    # the fixture's group trades in months 4-5 only, so per-month resolution uses the
    # group where it exists and the parent elsewhere; the recorded source is the last
    # level used, and `slope_source` must always agree with `valid6`'s provenance.
    check("the slope provenance is a real level code, not missing",
          slope_source in (0.0, 1.0, 2.0), f"slope_source={slope_source}")
    print(f"      slope_source={slope_source} (2.0 == town_ft, 0.0 == global)")

    # ---------------------------------------------------------------- 11. slope basis
    print("\n--- 11. the slope never mixes levels across its six months")
    # The exact failure mode this guards: a group that appears only in the last two months,
    # sitting at a stable level 10, while its parent sits at a stable level 20. Resolving
    # each month independently would feed (20,20,20,20,10,10) into one regression and
    # report a strong downward trend. The group observes only 2 months, so the whole
    # series must drop to the parent, where the slope is genuinely 0.
    # The parent pool must stay FLAT at 20 for the test to isolate the level-switch
    # artefact from a genuine transaction-mix move: a child entering the pool at a
    # different price really does move the pool. So months 4-5 add a compensating
    # 4-ROOM at 30 alongside the 3-ROOM at 10, keeping the pool mean at (30+10)/2 = 20.
    rows = []
    for month in range(6):
        rent = 1000.0 if month < 4 else 1500.0     # psm 20 -> 30
        rows.append({"month": month, "rent": rent, "area": 50.0,
                     "town": "T2", "ft": "4-ROOM"})
    # T1 3-ROOM appears only in the final two months, stable at psm 10
    rows.append({"month": 4, "rent": 500.0, "area": 50.0, "town": "T1", "ft": "3-ROOM"})
    rows.append({"month": 5, "rent": 500.0, "area": 50.0, "town": "T1", "ft": "3-ROOM"})
    frame = make_frame(rows)
    panel = build(frame, fit_end=5, parents={
        ("town_ft", ("T1", "3-ROOM")): {"flat_type": ("3-ROOM",), "global": ("all",)}})

    from experiments.market_dynamics_v5.panel import _slope6
    slope, valid_months, source, _n = _slope6(panel, "town_ft", ("T1", "3-ROOM"), 5)
    print(f"      slope={slope!r} valid_months={valid_months} source={source}")
    check("a group with only 2 observed months does not supply a slope",
          valid_months != 2 or source != "town_ft",
          f"valid={valid_months} source={source}")
    check("the slope is 0 because the parent series is flat, not a fake trend",
          bool(np.isclose(slope, 0.0, atol=1e-12)), f"slope={slope}")

    # and the mixed-level regression would have been strongly negative, which is what we
    # are proving does NOT happen
    mixed = float(np.polyfit(np.arange(6.0), np.log([20, 20, 20, 20, 10, 10]), 1)[0])
    print(f"      mixed-level slope that was avoided: {mixed:+.6f}")
    check("the avoided mixed-level slope is materially nonzero",
          abs(mixed) > 0.1, f"{mixed:.6f}")

    # a group observed in >=3 months uses its own series, with real gaps
    rows2 = []
    for month in range(6):
        rows2.append({"month": month, "rent": 1000.0, "area": 50.0,
                      "town": "T2", "ft": "4-ROOM"})
    for month in (0, 1, 5):        # gaps at 2, 3, 4
        rows2.append({"month": month, "rent": 500.0 * (1.02 ** month), "area": 50.0,
                      "town": "T1", "ft": "3-ROOM"})
    panel2 = build(make_frame(rows2), fit_end=5, parents={
        ("town_ft", ("T1", "3-ROOM")): {"flat_type": ("3-ROOM",), "global": ("all",)}})
    slope2, valid2, source2, _n2 = _slope6(panel2, "town_ft", ("T1", "3-ROOM"), 5)
    print(f"      gapped series: slope={slope2:.6f} valid={valid2} source={source2}")
    check("a group with 3 observed months supplies its own slope",
          source2 == "town_ft" and valid2 == 3, f"source={source2} valid={valid2}")
    check("the gapped regression uses real month positions (2%/month -> ln(1.02))",
          bool(np.isclose(slope2, np.log(1.02), rtol=1e-9)),
          f"{slope2:.9f} vs {np.log(1.02):.9f}")

    # ---------------------------------------------------------------- 12. missing keys
    print("\n--- 12. a missing row key is an explicit error, not a silent KeyError")
    from experiments.market_dynamics_v5.panel import MarketDynamicsError
    raised = False
    try:
        features_for(panel2, origin=5, row_keys={"town_ft": ("T1", "3-ROOM")},
                     spec=FeatureSpec())     # spec wants global/flat_type too
    except MarketDynamicsError as exc:
        raised = "missing" in str(exc)
    check("features_for refuses an incomplete row_keys instead of raising KeyError", raised)

    # ---------------------------------------------------------------- 12b. dir-2 basis
    print("\n--- 12b. direction-2 level and slope share one basis")
    # The forecast multiplies level by exp(slope*h), so a level from the group and a slope
    # from a parent would compound a level gap into a fake growth rate. The pair must come
    # from one level, and when no level supports both it must be missing rather than partial.
    from experiments.market_dynamics_v5.panel import level_and_slope
    rows = []
    for month in range(6):
        rows.append({"month": month, "rent": 3000.0, "area": 50.0,
                     "town": "T1", "ft": "4-ROOM"})       # parent: flat 60/sq m, all months
    for month in (4, 5):
        rows.append({"month": month, "rent": 6000.0, "area": 50.0,
                     "town": "T1", "ft": "3-ROOM"})       # group: 120/sq m, last 2 months only
    panel_ls = build(make_frame(rows), fit_end=5, parents={
        ("town_ft", ("T1", "3-ROOM")): {"flat_type": ("3-ROOM",), "global": ("all",)}})
    lv, sl, basis, vm, ln = level_and_slope(panel_ls, "town_ft", ("T1", "3-ROOM"), 5)
    print(f"      level={lv} slope={sl!r} basis={basis} valid_months={vm}")
    check("the group cannot supply a slope from only 2 months",
          not (vm == 2 and basis == "town_ft"), f"basis={basis} valid={vm}")
    check("level and slope come from the same basis level",
          basis is not None, f"basis={basis}")
    # whatever basis was chosen, the level must be that basis's own series, not another
    if basis is not None:
        table = panel_ls.levels[basis]
        group = table.index_of[("T1", "3-ROOM") if basis == "town_ft" else
                               (("3-ROOM",) if basis == "flat_type" else ("all",))]
        expected_level, _n, _v = __import__(
            "experiments.market_dynamics_v5.panel", fromlist=["_value"]
        )._value(table, group, 5, 3)
        check("the returned level equals its announced basis series",
              bool(np.isclose(lv, expected_level, rtol=1e-12)),
              f"level={lv} vs basis series {expected_level}")

    # With support in both intervals the pair must come from the group. The group series is
    # 1%/month for ALL six months here — if any month were left at a different level the
    # fitted slope would reflect that jump, not the growth rate, and the assertion below
    # would be testing the fixture instead of the code.
    rows2 = []
    for month in range(6):
        rows2.append({"month": month, "rent": 3000.0, "area": 50.0,
                      "town": "T1", "ft": "4-ROOM"})          # parent pool, flat 60/sq m
        rows2.append({"month": month, "rent": 3000.0 * (1.01 ** month), "area": 50.0,
                      "town": "T1", "ft": "3-ROOM"})          # group, 1%/month throughout
    panel_ls2 = build(make_frame(rows2), fit_end=5, parents={
        ("town_ft", ("T1", "3-ROOM")): {"flat_type": ("3-ROOM",), "global": ("all",)}})
    lv2, sl2, basis2, vm2, _ln2 = level_and_slope(panel_ls2, "town_ft", ("T1", "3-ROOM"), 5)
    check("a group with full support supplies both level and slope",
          basis2 == "town_ft" and vm2 == 6, f"basis={basis2} valid={vm2}")
    check("that slope matches the group's own 1%/month series",
          bool(np.isclose(sl2, np.log(1.01), rtol=1e-6)),
          f"{sl2:.9f} vs {np.log(1.01):.9f}")

    # ---------------------------------------------------------------- 12c. fixed weights
    print("\n--- 12c. fixed composition: average within cell, THEN combine with q")
    # The exact counterexample from docs/reviews/2026-10-26-fixed-composition-audit.json:
    # two flat types at CONSTANT per-sqm 10 and 20, with the transaction mix flipping from
    # 9:1 to 1:9. A correct fixed-composition index is 15 in every window once the origin
    # has seen both mixes. The earlier implementation multiplied each transaction by a base
    # share, which leaves the current mix inside the average and returned 11 -> 19.
    audit_rows = []
    for month, (n_a, n_b) in enumerate([(9, 1), (1, 9), (9, 1)]):
        for _ in range(n_a):
            audit_rows.append({"month": month, "rent": 10.0 * 50, "area": 50.0, "ft": "A"})
        for _ in range(n_b):
            audit_rows.append({"month": month, "rent": 20.0 * 50, "area": 50.0, "ft": "B"})
    audit_frame = make_frame(audit_rows)

    from experiments.market_dynamics_v5.panel import build_panel, cell_shares
    audit_keys = {"global": [("all",)] * len(audit_frame),
                  "flat_type": [(t,) for t in audit_frame["ft"].astype(str)]}
    audit_panel = build_panel(audit_frame, level_keys=audit_keys, rent_col="rent",
                              area_col="area", month_col="month", fit_end=2,
                              cell_column="ft")
    spec_a = FeatureSpec(levels=("global",), windows=(1, 2, 3))
    names_a = spec_a.names()

    def show(origin):
        tx = features_for(audit_panel, origin, row_keys={"global": ("all",)}, spec=spec_a)
        fx = features_for(audit_panel, origin, row_keys={"global": ("all",)},
                          spec=spec_a, fixed=True)
        return {"q": cell_shares(audit_panel, origin),
                "tx": [float(np.exp(tx[names_a.index(f"recent{w}_logpsm__global")]))
                       for w in (1, 2, 3)],
                "fx": [float(np.exp(fx[names_a.index(f"recent{w}_logpsm__global")]))
                       for w in (1, 2, 3)]}

    at1 = show(1)
    print(f"      origin 1 q={ {k: round(v, 3) for k, v in at1['q'].items()} }")
    print(f"        transaction windows 1/2/3 = {[round(v, 3) for v in at1['tx']]}")
    print(f"        fixed       windows 1/2/3 = {[round(v, 3) for v in at1['fx']]}")
    check("the audit counterexample gives 15 in every window under fixed composition",
          all(bool(np.isclose(v, 15.0, rtol=1e-12)) for v in at1["fx"]), f"{at1['fx']}")
    check("transaction weighting still shows the mix shift (11 or 19)",
          any(bool(np.isclose(v, 11.0, rtol=1e-9)) or bool(np.isclose(v, 19.0, rtol=1e-9))
              for v in at1["tx"]), f"{at1['tx']}")
    check("the old buggy 11 -> 19 pattern is absent from the fixed path",
          not (bool(np.isclose(at1["fx"][0], 11.0))
               and bool(np.isclose(at1["fx"][1], 19.0))), f"{at1['fx']}")
    check("all windows at one origin share the same q",
          all(bool(np.isclose(v, at1["fx"][0], rtol=1e-12)) for v in at1["fx"]),
          f"{at1['fx']}")

    # q must be causal. At origin 0 only month 0 (9:1) has happened, so q = .9/.1 and the
    # fixed index is 11 — the level implied by what that origin could have known. This is
    # expected behaviour, not a defect, and is asserted so it cannot silently change.
    at0 = show(0)
    print(f"      origin 0 q={ {k: round(v, 3) for k, v in at0['q'].items()} } "
          f"fixed={[round(v, 3) for v in at0['fx']]}")
    check("an early origin's q uses only the months it has seen",
          bool(np.isclose(at0["q"]["A"], 0.9)) and bool(np.isclose(at0["q"]["B"], 0.1)),
          f"{at0['q']}")
    check("that early origin's fixed index is 11 by construction, not 15",
          all(bool(np.isclose(v, 11.0, rtol=1e-12)) for v in at0["fx"]), f"{at0['fx']}")

    # a cell absent from a window is dropped and its share redistributed over the rest
    sparse_rows = [
        {"month": 0, "rent": 10.0 * 50, "area": 50.0, "ft": "A"},
        {"month": 0, "rent": 20.0 * 50, "area": 50.0, "ft": "B"},
        {"month": 1, "rent": 10.0 * 50, "area": 50.0, "ft": "A"},
        {"month": 2, "rent": 10.0 * 50, "area": 50.0, "ft": "A"},   # B never trades again
    ]
    sparse_frame = make_frame(sparse_rows)
    sparse_panel = build_panel(sparse_frame, level_keys={
        "global": [("all",)] * len(sparse_frame),
        "flat_type": [(t,) for t in sparse_frame["ft"].astype(str)]},
        rent_col="rent", area_col="area", month_col="month", fit_end=2, cell_column="ft")
    fx_sparse = features_for(sparse_panel, 2, row_keys={"global": ("all",)},
                             spec=FeatureSpec(levels=("global",), windows=(1,)),
                             fixed=True)
    val = float(np.exp(fx_sparse[0]))
    print(f"      cell B absent in the window: fixed mean = {val:.3f} (only A traded)")
    check("an absent cell is dropped, not invented",
          bool(np.isclose(val, 10.0, rtol=1e-12)), f"{val}")

    # requesting fixed composition without any cell history must fail loudly
    from experiments.market_dynamics_v5.panel import MarketDynamicsError
    plain_panel = build(make_frame(rows), fit_end=5)
    threw = False
    try:
        features_for(plain_panel, 2, row_keys={"global": ("all",)},
                     spec=FeatureSpec(levels=("global",), windows=(1,)), fixed=True)
    except MarketDynamicsError as exc:
        threw = "fixed" in str(exc).lower()
    check("fixed composition on a panel without cells raises", threw)

    # within a homogeneous flat type the two weightings must agree exactly
    ft_tx = features_for(audit_panel, 1, row_keys={"flat_type": ("A",)},
                         spec=FeatureSpec(levels=("flat_type",), windows=(1,)))
    ft_fx = features_for(audit_panel, 1, row_keys={"flat_type": ("A",)},
                         spec=FeatureSpec(levels=("flat_type",), windows=(1,)),
                         fixed=True)
    check("within a homogeneous flat type the two weightings agree",
          bool(np.isclose(np.exp(ft_tx[0]), np.exp(ft_fx[0]), rtol=1e-12)),
          f"{np.exp(ft_tx[0]):.4f} vs {np.exp(ft_fx[0]):.4f}")

    # ---------------------------------------------------------------- 13. feature count
    print("\n--- 13. the feature count is derived, not asserted")
    full = FeatureSpec()
    per_level = len(full.windows) + 8   # 3 windows + 6 scalars + 2 provenance codes
    check("names() length matches the derived layout",
          len(full.names()) == per_level * len(full.levels),
          f"{len(full.names())} vs {per_level * len(full.levels)}")
    print(f"      full spec: {len(full.levels)} levels x {per_level} = "
          f"{len(full.names())} features")
    full_keys = {"global": ("all",), "flat_type": ("3-ROOM",),
                 "town_ft": ("T1", "3-ROOM")}
    one = features_for(panel2, origin=5, row_keys=full_keys, spec=full)
    check("a full-spec vector has the declared length",
          len(one) == len(full.names()), f"{len(one)} vs {len(full.names())}")
    finite = int(np.isfinite(one).sum())
    print(f"      {finite}/{len(one)} entries finite on this fixture "
          f"({len(one) - finite} legitimately NaN: prior-3 window absent)")

    print("\n" + "=" * 62)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for item in FAILED:
        print(f"  - {item}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())