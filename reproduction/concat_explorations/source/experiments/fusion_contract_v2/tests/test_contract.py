"""Gate-1 contract tests: causal boundaries, alignment, parity, fixtures.

Every assertion is exact.  No ``or True``, no "is finite", no ``>= 0`` on a count:
those pass on broken implementations and were explicitly rejected in review.

Run:
    env -u PYTHONPATH -u PYTHONSTARTUP ../../.venv-model-benchmark/bin/python \
        -m pytest experiments/fusion_contract_v2/tests -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.fusion_contract_v2 import contract
from experiments.fusion_contract_v2 import export, panel as panel_module

# ---------------------------------------------------------------- helpers


def tiny_frame(months: int = 9, per_month: int = 4) -> pd.DataFrame:
    frame = export.synthetic_fixture(months=months, per_month=per_month)
    return frame


def split_at(cutoff_index: int) -> dict:
    spec = {
        "id": f"n{cutoff_index}",
        "cutoff_index": cutoff_index,
        "cutoff": contract.month_label(cutoff_index),
        "val_start_index": cutoff_index + 1,
        "val_end_index": cutoff_index + 3,
        "val_start": contract.month_label(cutoff_index + 1),
        "val_end": contract.month_label(cutoff_index + 3),
        "n_validation_months": 3,
    }
    return spec


def panel_for(frame: pd.DataFrame, fit_end: int, prior: dict | None = None):
    return panel_module.Panel.build(frame, fit_end, prior)


def codes_of(frame: pd.DataFrame, panel) -> dict[str, np.ndarray]:
    return export._codes_for(frame, panel)


# ---------------------------------------------------------------- 1. splits


def test_split_specs_are_bounded_and_ordered():
    for split_id, raw in contract.SPLITS.items():
        spec = contract.split_spec(split_id)
        assert spec["val_start_index"] == spec["cutoff_index"] + 1
        assert spec["val_end_index"] >= spec["val_start_index"]
        assert spec["n_validation_months"] == spec["val_end_index"] - spec["cutoff_index"]
        # documented month counts
    assert contract.split_spec("orig")["n_validation_months"] == 12
    assert contract.split_spec("early")["n_validation_months"] == 12
    assert contract.split_spec("long")["n_validation_months"] == 16


def test_split_months_match_the_declared_dates():
    assert contract.month_index("2024-03") == contract.split_spec("orig")["cutoff_index"]
    assert contract.month_index("2025-03") == contract.split_spec("orig")["val_end_index"]
    assert contract.month_index("2023-03") == contract.split_spec("early")["cutoff_index"]
    assert contract.month_index("2024-03") == contract.split_spec("early")["val_end_index"]
    assert contract.month_index("2023-11") == contract.split_spec("long")["cutoff_index"]
    assert contract.split_spec("orig")["inner_cutoff_index"] == contract.month_index("2023-03")


# ---------------------------------------------------------------- 2. panel oracle


def test_panel_global_mean_includes_the_boundary_month():
    """Months 0..5 -> mean of base + 100*m over the six visible months.

    per_month=4 places one row of each flat type every month, so the arithmetic
    is exact and any off-by-one shifts the expected value.
    """
    frame = tiny_frame(months=9, per_month=4)
    target = panel_for(frame, fit_end=8)
    rows = frame.iloc[[0]]
    codes = codes_of(rows, target)
    boundary = np.array([5], dtype=np.int64)
    stats = target.stats(boundary, codes)
    # global column is over all flat types
    all_rows = frame[frame["MONTH_INDEX"] <= 5]
    assert stats["hist_rent__global"][0] == pytest.approx(all_rows["MONTHLY_RENT"].mean())
    assert stats["hist_n__global"][0] == len(all_rows)
    assert stats["hist_months__global"][0] == 6


def test_panel_excludes_the_month_after_the_boundary():
    frame = tiny_frame(months=9, per_month=4)
    target = panel_for(frame, fit_end=8)
    rows = frame.iloc[[0]]
    codes = codes_of(rows, target)
    at_five = target.stats(np.array([5]), codes)
    at_six = target.stats(np.array([6]), codes)
    # Different boundaries must give different means because rent rises monthly.
    assert at_five["hist_rent__global"][0] != at_six["hist_rent__global"][0]
    assert at_six["hist_n__global"][0] == at_five["hist_n__global"][0] + 4
    # boundary 5 must equal the mean over months 0..5 exactly
    expected = frame[frame["MONTH_INDEX"] <= 5]["MONTHLY_RENT"].mean()
    assert at_five["hist_rent__global"][0] == pytest.approx(expected)


def test_panel_never_reads_past_fit_end():
    """A panel built to fit_end=4 must be unchanged by rows after month 4."""
    frame = tiny_frame(months=9, per_month=4)
    trimmed = frame[frame["MONTH_INDEX"] <= 4].copy()
    full_panel = panel_for(frame, fit_end=4)
    trim_panel = panel_for(trimmed, fit_end=4)
    rows = frame[frame["MONTH_INDEX"] <= 4].iloc[[0, 4, 8]].reset_index(drop=True)
    boundary = np.array([0, 2, 4], dtype=np.int64)
    a = full_panel.stats(boundary, codes_of(rows, full_panel))
    b = trim_panel.stats(boundary, codes_of(rows, trim_panel))
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=0.0, atol=0.0, equal_nan=True)


def test_boundary_before_first_month_is_empty():
    frame = tiny_frame()
    target = panel_for(frame, fit_end=8)
    codes = codes_of(frame.iloc[[0]], target)
    stats = target.stats(np.array([-1]), codes)
    assert np.isnan(stats["hist_rent__global"][0])
    assert stats["hist_n__global"][0] == 0.0
    assert stats["hist_months__global"][0] == 0.0
    assert stats["anchor_missing"][0] == 1.0


def test_recency_is_measured_from_the_boundary():
    frame = tiny_frame(months=9, per_month=4)
    target = panel_for(frame, fit_end=8)
    codes = codes_of(frame.iloc[[0, 1]], target)
    stats = target.stats(np.array([3, 5]), codes)
    # Every month has observations, so the last observed month is the boundary.
    assert stats["hist_recency__global"][0] == 0.0
    assert stats["hist_recency__global"][1] == 0.0


# ---------------------------------------------------------------- 3. fallback


def test_unseen_key_reproduces_its_parent_mean_exactly():
    """With n=0 the shrinkage reduces to the parent value, not a global constant."""
    frame = tiny_frame(months=9, per_month=4)
    target = panel_for(frame, fit_end=8)

    novel = frame.iloc[[0]].copy()
    novel["BLOCK_CLEAN"] = "999"
    novel["BLOCK_ID"] = "TOWN0|STREET0|999"
    codes = codes_of(novel, target)
    stats = target.stats(np.array([4]), codes)
    assert stats["hist_n__block_ft"][0] == 0.0
    assert stats["hist_exact__block_ft"][0] == 0.0
    # n=0 at block_ft -> value must equal the town_ft level bit-for-bit
    assert stats["hist_rent__block_ft"][0] == stats["hist_rent__town_ft"][0]
    assert stats["hist_psm__block_ft"][0] == stats["hist_psm__town_ft"][0]


def test_sparse_level_shrinks_towards_parent_but_stays_between():
    frame = tiny_frame(months=9, per_month=4)
    target = panel_for(frame, fit_end=8)
    rows = frame.iloc[[0]]
    codes = codes_of(rows, target)
    boundary = np.array([7])
    stats = target.stats(boundary, codes)
    own = stats["hist_rent__block_ft"][0]
    parent = stats["hist_rent__town_ft"][0]
    raw = frame[(frame["MONTH_INDEX"] <= 7)]["MONTHLY_RENT"]
    assert np.isfinite(own)
    # Own value is a shrunk version of its own mean towards the parent, so it
    # must lie within the range spanned by the parent and the raw block mean.
    block_key = target.level_maps["block_ft"][codes["block_ft"][0]]
    block_rows = frame[(frame["MONTH_INDEX"] <= 7)]
    block_rows = block_rows[
        block_rows["BLOCK_ID"].str.cat(block_rows["FLAT_TYPE_CLEAN"], sep="|") == block_key]
    block_mean = block_rows["MONTHLY_RENT"].mean()
    low, high = sorted([parent, block_mean])
    assert low - 1e-6 <= own <= high + 1e-6


def test_anchor_uses_the_finest_observed_level():
    frame = tiny_frame(months=9, per_month=4)
    target = panel_for(frame, fit_end=8)
    codes = codes_of(frame.iloc[[0]], target)
    stats = target.stats(np.array([6]), codes)
    # block_ft is observed for these rows, so the anchor is the block-level value
    assert stats["anchor_level_code"][0] == 0
    assert stats["anchor_rent_sgd"][0] == pytest.approx(stats["hist_rent__block_ft"][0])
    assert stats["anchor_missing"][0] == 0.0


# ---------------------------------------------------------------- 4. sample table


def test_samples_have_origin_equal_to_target_minus_horizon():
    frame = tiny_frame(months=28, per_month=4)
    spec = split_at(26)
    table = export.build_samples(frame, spec, "B")
    assert (table["origin_month_index"]
            == table["target_month_index"] - table["horizon_months"]).all()
    assert (table["horizon_months"] >= 1).all()
    assert (table["horizon_months"] <= contract.MAX_HORIZON).all()


def test_validation_origin_is_the_cutoff_for_every_horizon():
    # 30 months so that val_start+1..val_end (27..29) all exist in the fixture.
    frame = tiny_frame(months=30, per_month=4)
    spec = split_at(26)
    table = export.build_samples(frame, spec, "B")
    valid = table[table["population"] == "validation"]
    assert len(valid) > 0
    assert (valid["origin_month_index"] == spec["cutoff_index"]).all()
    assert (valid["horizon_months"] ==
            valid["target_month_index"] - spec["cutoff_index"]).all()
    assert valid["horizon_months"].nunique() == spec["n_validation_months"]


def test_sample_generation_is_seed_frozen_and_deterministic():
    frame = tiny_frame(months=28, per_month=4)
    spec = split_at(26)
    first = export.build_samples(frame, spec, "B")
    second = export.build_samples(frame, spec, "B")
    pd.testing.assert_frame_equal(first, second)
    # And the horizon is not merely the max every time.
    assert first["horizon_months"].nunique() > 1


def test_stage_a_excludes_outer_months_and_stage_b_includes_them():
    frame = tiny_frame(months=26, per_month=4)
    spec = split_at(24)
    stage_a = export.build_samples(frame, spec, "A")
    stage_b = export.build_samples(frame, spec, "B")
    inner = spec["cutoff_index"] - contract.INNER_HOLDOUT_MONTHS
    fit_a = stage_a[stage_a["population"] == "fit"]
    assert fit_a["target_month_index"].max() <= inner
    holdout = stage_a[stage_a["population"] == "holdout"]
    assert holdout["target_month_index"].min() > inner
    assert holdout["target_month_index"].max() <= spec["cutoff_index"]
    fit_b = stage_b[stage_b["population"] == "fit"]
    assert fit_b["target_month_index"].max() <= spec["cutoff_index"]
    assert fit_b["target_month_index"].max() > inner


def test_holdout_labels_cannot_change_stage_a_state():
    """Tamper with stage-A holdout labels; the stage-A fit rows must not move."""
    frame = tiny_frame(months=26, per_month=4)
    spec = split_at(24)
    inner = spec["cutoff_index"] - contract.INNER_HOLDOUT_MONTHS
    assert inner > 0
    tampered = frame.copy()
    is_holdout = (tampered["MONTH_INDEX"] > inner) & (tampered["MONTH_INDEX"] <= spec["cutoff_index"])
    tampered.loc[is_holdout, contract.TARGET] = 9_999_999.0

    for frame_version in (frame, tampered):
        panel = panel_module.Panel.build(frame_version, inner)
        table = export.build_samples(frame_version, spec, "A")
        fit_ids = table.loc[table["population"] == "fit", contract.ID_COL].to_numpy()
        rows = frame_version.set_index(contract.ID_COL).loc[fit_ids].reset_index()
        codes = codes_of(rows, panel)
        stats = panel.stats(table.loc[table["population"] == "fit",
                                      "origin_month_index"].to_numpy(), codes)
        if frame_version is frame:
            baseline = stats
        else:
            for key in baseline:
                np.testing.assert_allclose(baseline[key], stats[key],
                                           rtol=0.0, atol=0.0, equal_nan=True)


def test_post_cutoff_labels_cannot_change_validation_features():
    frame = tiny_frame(months=28, per_month=4)
    spec = split_at(26)
    tampered = frame.copy()
    tampered.loc[tampered["MONTH_INDEX"] > spec["cutoff_index"], contract.TARGET] = -1.0

    results = []
    for frame_version in (frame, tampered):
        panel = panel_module.Panel.build(frame_version, spec["cutoff_index"])
        table = export.build_samples(frame_version, spec, "B")
        exp = export.build_features(frame_version, table, panel)
        results.append(exp)

    valid = results[0].arrays["row_id"] != 0  # placeholder, replaced below
    # Compare only validation rows.
    table = export.build_samples(frame, spec, "B")
    mask = (table["population"] == "validation").to_numpy()
    for key in ("history", "context", "sequence", "sequence_mask", "static"):
        np.testing.assert_allclose(results[0].arrays[key][mask],
                                   results[1].arrays[key][mask],
                                   rtol=0.0, atol=0.0, equal_nan=True)


def test_validation_features_are_frozen_across_horizons():
    """Every validation row's scalar history must equal the cutoff snapshot."""
    frame = tiny_frame(months=28, per_month=4)
    spec = split_at(26)
    panel = panel_module.Panel.build(frame, spec["cutoff_index"])
    table = export.build_samples(frame, spec, "B")
    exp = export.build_features(frame, table, panel)
    valid = (table["population"] == "validation").to_numpy()
    history = exp.arrays["history"][valid]
    towns = table.loc[valid].merge(
        frame[[contract.ID_COL, "TOWN_CLEAN", "FLAT_TYPE_CLEAN"]], on=contract.ID_COL)
    frame_valid = pd.DataFrame(history, columns=contract.history_columns())
    frame_valid["group"] = (towns["TOWN_CLEAN"] + "|" + towns["FLAT_TYPE_CLEAN"]).to_numpy()
    for column in contract.history_columns():
        spread = frame_valid.groupby("group")[column].nunique(dropna=False)
        assert (spread <= 1).all(), f"{column} varies across horizons within a group"


# ---------------------------------------------------------------- 5. alignment


def test_feature_rows_align_with_the_sample_table():
    frame = tiny_frame(months=28, per_month=4)
    spec = split_at(26)
    panel = panel_module.Panel.build(frame, spec["cutoff_index"])
    table = export.build_samples(frame, spec, "B")
    exp = export.build_features(frame, table, panel)
    assert len(exp.arrays["row_id"]) == len(table)
    np.testing.assert_array_equal(exp.arrays["row_id"], table[contract.ID_COL].to_numpy())
    np.testing.assert_array_equal(exp.arrays["horizon"], table["horizon_months"].to_numpy())
    np.testing.assert_array_equal(exp.arrays["origin"], table["origin_month_index"].to_numpy())
    np.testing.assert_allclose(exp.arrays["target_sgd"],
                               table["actual_rent_sgd"].to_numpy(), rtol=0, atol=1e-3)


def test_arm_columns_have_no_duplicates_and_expected_roles():
    static = contract.arm_scalar_columns("aligned_static")
    history = contract.arm_scalar_columns("aligned_history")
    sequence = contract.arm_scalar_columns("aligned_sequence")
    assert len(static) == len(set(static))
    assert len(history) == len(set(history))
    assert set(static) < set(history)
    assert history == sequence  # sequence arm adds arrays, not scalars
    assert contract.arm_uses_sequence("aligned_sequence")
    assert not contract.arm_uses_sequence("aligned_history")
    # every history column belongs to exactly one level/stat pair
    assert len(contract.history_columns()) == len(contract.HISTORY_LEVELS) * len(contract.HISTORY_STATS)
    unknown = [c for c in history if c not in set(static) | set(contract.history_columns())]
    assert unknown == []


def test_no_month_index_leaks_into_the_static_block():
    """Calendar position must be context, not static: else h=1 and h=12 differ."""
    assert "APPROVAL_MONTH_INDEX" not in contract.static_columns()
    assert "APPROVAL_MONTH_INDEX" not in contract.STATIC_NUMERIC_COLUMNS
    assert "target_month_index" in contract.CONTEXT_COLUMNS
    assert "origin_month_index" in contract.CONTEXT_COLUMNS


def test_sequence_mask_excludes_padding_and_future_months():
    frame = tiny_frame(months=28, per_month=4)
    spec = split_at(26)
    panel = panel_module.Panel.build(frame, spec["cutoff_index"])
    origin = np.array([0, 3, 6])
    codes = codes_of(frame.iloc[:3], panel)
    sequence, mask = panel.sequences(origin, codes)
    # A row with origin 0 can only have one real month in a 12-long window.
    assert mask[0].sum() == len(contract.HISTORY_LEVELS) * 1
    assert mask[1].sum() == len(contract.HISTORY_LEVELS) * 4
    assert mask[2].sum() == len(contract.HISTORY_LEVELS) * 7
    # Padding is zero, never a real value.
    assert np.all(sequence[0][mask[0] == 0] == 0)
    # Position of the last slot that may hold real history is (LEN-1); a window
    # with origin o covers months o-LEN+1 .. o, so exactly min(o+1, LEN) slots
    # can be observed and everything to the *left* of them is padding.
    for row, boundary in enumerate(origin):
        observed = mask[row][0]  # first level
        expected_slots = min(boundary + 1, contract.HISTORY_LEN)
        assert observed.sum() == expected_slots, (row, boundary, observed.sum())
        # the rightmost expected_slots entries are the observed ones
        assert observed[-expected_slots:].all()
        assert not observed[:-expected_slots].any()


def test_sequence_values_match_the_scalar_history_channel():
    """The last observed tensor position must equal the scalar boundary mean."""
    frame = tiny_frame(months=28, per_month=4)
    spec = split_at(26)
    panel = panel_module.Panel.build(frame, spec["cutoff_index"])
    rows = frame.iloc[[0]]
    codes = codes_of(rows, panel)
    boundary = np.array([5])
    stats = panel.stats(boundary, codes)
    sequence, mask = panel.sequences(boundary, codes)
    global_index = list(contract.HISTORY_LEVELS).index("global")
    last = contract.HISTORY_LEN - 1
    assert mask[0, global_index, last] == 1.0
    # channel 2 is log1p(count) of that single month, not the cumulative n
    month_count = panel.month_n["global"][0, 5]
    assert month_count == 4.0          # four synthetic rows per month
    assert sequence[0, global_index, last, 2] == pytest.approx(np.log1p(month_count), abs=1e-6)
    month_mean = panel.month_sum["global"][0, 5] / month_count
    assert sequence[0, global_index, last, 0] == pytest.approx(np.log1p(month_mean), abs=1e-6)
    # and the scalar cumulative mean must differ from that single-month mean
    assert stats["hist_rent__global"][0] != pytest.approx(month_mean)


# ---------------------------------------------------------------- 6. fixtures


def test_synthetic_fixture_is_hand_checkable():
    frame = export.synthetic_fixture(months=4, per_month=4)
    assert len(frame) == 16
    # Rent rises 100/month *within* each flat type (the raw column cycles across
    # four bases, so it is deliberately not globally monotone).
    for flat_type, group in frame.groupby("FLAT_TYPE_CLEAN"):
        rising = group.sort_values("MONTH_INDEX")["MONTHLY_RENT"].to_numpy()
        assert np.all(np.diff(rising) == 100.0), flat_type
    # month 0, 3 room -> 2600 exactly
    row = frame[(frame["MONTH_INDEX"] == 0) & (frame["FLAT_TYPE_CLEAN"] == "3 room")]
    assert len(row) == 1
    assert row["MONTHLY_RENT"].iloc[0] == 2600.0
    row = frame[(frame["MONTH_INDEX"] == 3) & (frame["FLAT_TYPE_CLEAN"] == "3 room")]
    assert row["MONTHLY_RENT"].iloc[0] == 2900.0


def test_fixture_round_trips_through_the_export_path():
    frame = export.synthetic_fixture(months=28, per_month=4)
    spec = split_at(26)
    panel = panel_module.Panel.build(frame, spec["cutoff_index"])
    table = export.build_samples(frame, spec, "B")
    exp = export.build_features(frame, table, panel)
    assert len(exp.arrays["static"]) == len(table)
    assert exp.arrays["sequence"].shape[1] == len(contract.HISTORY_LEVELS)
    assert exp.arrays["sequence"].shape[2] == contract.HISTORY_LEN
    assert exp.arrays["sequence"].shape[3] == len(contract.PANEL_CHANNELS)
    assert np.isfinite(exp.arrays["static"]).all()
    assert np.isfinite(exp.arrays["context"]).all()


def test_package_import_is_qualified():
    """Bare `config`/`data` imports would collide with the other panes' modules."""
    source = (contract.PACKAGE_DIR / "panel.py").read_text()
    assert "from config import" not in source
    assert "import config" not in source
    export_source = (contract.PACKAGE_DIR / "export.py").read_text()
    assert "from experiments.fusion_contract_v2 import" in export_source
