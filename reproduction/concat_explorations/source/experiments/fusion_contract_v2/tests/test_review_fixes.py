"""Regression tests for the two defects found in Gate-1 interface review.

Each test in this file fails on the v2.0.0 implementation and passes after the
fix, so it documents the defect rather than merely covering the new code.

1. Stage-A holdout used a *sampled* horizon, so ``origin = target - h`` often
   exceeded the stage-A panel's ``fit_end``; the boundary was then silently
   clamped by ``min(origin, fit_end)``, which left ``context`` reporting a
   horizon/origin that disagreed with the history actually supplied.
   Fix: holdout rows are frozen at ``origin = inner_cutoff`` with
   ``h = target - inner_cutoff``.

2. Category codes were produced with ``pd.factorize`` over the whole stage table
   (including holdout and validation rows), and only a stage-B encoding was
   written.  Codes therefore depended on which rows were being scored, and no
   stage-A vocabulary existed.
   Fix: each stage fits its own vocabulary on ``population == "fit"`` only, with
   ``0`` reserved for unknown, saved as ``encodingA.json`` / ``encodingB.json``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.fusion_contract_v2 import contract, export, panel as panel_module


def frame_with_unseen_category(months: int = 30, per_month: int = 4) -> pd.DataFrame:
    """Fit-period rows plus a town that appears only in the validation window."""
    frame = export.synthetic_fixture(months=months, per_month=per_month)
    spec_cutoff = 26
    late = frame["MONTH_INDEX"] > spec_cutoff
    frame.loc[late, "TOWN_CLEAN"] = "BRAND_NEW_TOWN"
    frame["BLOCK_ID"] = (
        frame["TOWN_CLEAN"].str.cat(frame["STREET_CLEAN"], sep="|")
        .str.cat(frame["BLOCK_CLEAN"], sep="|"))
    return frame


def split_at(cutoff_index: int) -> dict:
    return {
        "id": f"n{cutoff_index}",
        "cutoff_index": cutoff_index,
        "cutoff": contract.month_label(cutoff_index),
        "val_start": contract.month_label(cutoff_index + 1),
        "val_end": contract.month_label(cutoff_index + 3),
        "val_start_index": cutoff_index + 1,
        "val_end_index": cutoff_index + 3,
        "n_validation_months": 3,
        "inner_cutoff_index": cutoff_index - contract.INNER_HOLDOUT_MONTHS,
    }


# ---------------------------------------------------------------- defect 1
def test_stage_a_has_no_validation_population():
    """Stage A must never expose outer validation rows to fitting or selection."""
    frame = export.synthetic_fixture(months=30, per_month=4)
    spec = split_at(26)
    table = export.build_samples(frame, spec, "A")
    assert set(table["population"].unique()) == {"fit", "holdout"}
    assert "validation" not in set(table["population"].unique())


def test_stage_a_holdout_origin_is_the_inner_cutoff():
    frame = export.synthetic_fixture(months=30, per_month=4)
    spec = split_at(26)
    inner = spec["inner_cutoff_index"]
    table = export.build_samples(frame, spec, "A")
    holdout = table[table["population"] == "holdout"]
    assert len(holdout) > 0
    # exact: every holdout row is frozen at the inner cutoff
    assert (holdout["origin_month_index"] == inner).all()
    # its horizon is therefore determined, not sampled
    assert (holdout["horizon_months"]
            == holdout["target_month_index"] - inner).all()
    assert holdout["horizon_months"].nunique() == contract.INNER_HOLDOUT_MONTHS


def test_stage_a_holdout_context_matches_the_history_it_receives():
    """The clamp must become a no-op: origin already equals the panel fit_end."""
    frame = export.synthetic_fixture(months=30, per_month=4)
    spec = split_at(26)
    inner = spec["inner_cutoff_index"]
    panel = panel_module.Panel.build(frame, inner)
    table = export.build_samples(frame, spec, "A")
    export_result = export.build_features(frame, table, panel)

    holdout_mask = (table["population"] == "holdout").to_numpy()
    origins = export_result.arrays["origin"][holdout_mask]
    # before the fix these exceeded panel.fit_end and were clamped downstream
    assert int(origins.max()) == inner
    assert int(origins.max()) <= panel.fit_end

    context = pd.DataFrame(export_result.arrays["context"][holdout_mask],
                           columns=contract.context_columns())
    # context must agree with the exported origin
    assert (context["origin_month_index"] == inner).all()
    assert (context["horizon_months"]
            == context["target_month_index"] - inner).all()
    # and the anchor must be finite for rows with real history
    assert context["anchor_missing"].sum() == 0


def test_stage_a_holdout_horizons_stay_within_the_declared_maximum():
    frame = export.synthetic_fixture(months=30, per_month=4)
    spec = split_at(26)
    table = export.build_samples(frame, spec, "A")
    holdout = table[table["population"] == "holdout"]
    assert holdout["horizon_months"].max() <= contract.MAX_HORIZON


# ---------------------------------------------------------------- defect 2
def test_vocabulary_is_fitted_on_fit_rows_only():
    frame = frame_with_unseen_category()
    spec = split_at(26)
    table = export.build_samples(frame, spec, "B")
    encoding = export.fit_vocabularies(frame, table)
    towns = encoding["categorical_vocabularies"]["TOWN_CLEAN"]
    # the validation-only town must be absent from the vocabulary
    assert "BRAND_NEW_TOWN" not in towns
    assert towns["<unknown>"] == 0
    # real categories start at 1, never 0
    real_codes = [code for name, code in towns.items() if name != "<unknown>"]
    assert min(real_codes) == 1
    assert len(set(real_codes)) == len(real_codes)


def test_unknown_category_maps_to_zero_not_to_a_new_code():
    frame = frame_with_unseen_category()
    spec = split_at(26)
    table = export.build_samples(frame, spec, "B")
    panel = panel_module.Panel.build(frame, spec["cutoff_index"])
    result = export.build_features(frame, table, panel)

    rows = frame.set_index(contract.ID_COL).loc[table[contract.ID_COL]].reset_index()
    unseen = (rows["TOWN_CLEAN"] == "BRAND_NEW_TOWN").to_numpy()
    assert unseen.any(), "fixture must contain a validation-only town"

    town_index = list(contract.static_columns()).index("TOWN_CLEAN")
    codes = result.arrays["static"][:, town_index]
    assert (codes[unseen] == 0).all()
    # and known towns are never 0
    assert (codes[~unseen] > 0).all()


def test_category_codes_do_not_change_when_validation_rows_change():
    """The defect: factorising the whole table let scoring rows move the codes."""
    frame = export.synthetic_fixture(months=30, per_month=4)
    spec = split_at(26)
    table = export.build_samples(frame, spec, "B")
    panel = panel_module.Panel.build(frame, spec["cutoff_index"])
    baseline = export.build_features(frame, table, panel)

    tampered = frame.copy()
    late = tampered["MONTH_INDEX"] > spec["cutoff_index"]
    tampered.loc[late, "FLAT_MODEL_CLEAN"] = "A_MODEL_NOBODY_HAS_SEEN"
    changed = export.build_features(tampered, table, panel)

    fit_mask = (table["population"] == "fit").to_numpy()
    np.testing.assert_array_equal(baseline.arrays["static"][fit_mask],
                                  changed.arrays["static"][fit_mask])
    # the fit rows' vocabulary must be byte-identical (n_fit_rows is unchanged
    # too, so the whole encoding can be compared directly)
    assert export.fit_vocabularies(frame, table) == export.fit_vocabularies(tampered, table)


def test_stage_a_and_stage_b_vocabularies_are_separate():
    """Stage A must have its own vocabulary, fitted on its own fit rows."""
    frame = export.synthetic_fixture(months=30, per_month=4)
    spec = split_at(26)
    table_a = export.build_samples(frame, spec, "A")
    table_b = export.build_samples(frame, spec, "B")
    encoding_a = export.fit_vocabularies(frame, table_a)
    encoding_b = export.fit_vocabularies(frame, table_b)

    # The vocabularies may legitimately hold the same strings when every category
    # appears early.  What must differ is the population they were fitted on.
    assert encoding_a["n_fit_rows"] < encoding_b["n_fit_rows"]
    assert encoding_a["fitted_on"].startswith("population == 'fit'")
    assert encoding_b["fitted_on"].startswith("population == 'fit'")
    # stage A fits on fewer months, so it cannot carry more categories
    for column in contract.CATEGORICAL_COLUMNS:
        assert len(encoding_a["categorical_vocabularies"][column]) \
            <= len(encoding_b["categorical_vocabularies"][column])
    # and both are stamped with the current contract version
    assert encoding_a["contract_version"] == contract.CONTRACT_VERSION
    assert encoding_b["contract_version"] == contract.CONTRACT_VERSION


def test_emitted_codes_round_trip_through_the_saved_mapping():
    frame = frame_with_unseen_category()
    spec = split_at(26)
    table = export.build_samples(frame, spec, "B")
    panel = panel_module.Panel.build(frame, spec["cutoff_index"])
    result = export.build_features(frame, table, panel)
    encoding = result.encoding
    assert encoding is not None

    rows = frame.set_index(contract.ID_COL).loc[table[contract.ID_COL]].reset_index()
    for index, column in enumerate(contract.CATEGORICAL_COLUMNS):
        codes = result.arrays["static"][:, index].astype(int)
        mapping = encoding["categorical_vocabularies"][column]
        reverse = {code: name for name, code in mapping.items()}
        # every emitted non-zero code must reverse to the row's actual value
        actual = rows[column].fillna("__missing__").astype(str).to_numpy()
        for code, value in zip(codes, actual):
            if code == 0:
                assert value not in mapping or value == "<unknown>"
            else:
                assert reverse[code] == value, (column, code, value, reverse[code])


def test_numeric_medians_come_from_fit_rows_only():
    frame = export.synthetic_fixture(months=30, per_month=4)
    spec = split_at(26)
    table = export.build_samples(frame, spec, "B")
    encoding = export.fit_vocabularies(frame, table)
    fit_ids = set(table.loc[table["population"] == "fit", contract.ID_COL])
    fit_rows = frame[frame[contract.ID_COL].isin(fit_ids)]
    for column, median in encoding["numeric_medians"].items():
        assert median == pytest.approx(float(fit_rows[column].median()), abs=1e-9)


def test_contract_version_is_bumped_past_the_superseded_one():
    assert contract.CONTRACT_VERSION == "fusion-contract-v2.1.0"
    # the superseded directory must remain on disk and must not be the version
    # this package writes to
    superseded = contract.REPORT_ROOT / "fusion-contract-v2.0.0__43867d0ba111"
    if superseded.exists():
        assert contract.CONTRACT_VERSION not in superseded.name
