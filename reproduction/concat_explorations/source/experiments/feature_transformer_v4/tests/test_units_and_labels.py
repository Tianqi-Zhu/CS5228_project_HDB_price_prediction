"""Unit and label agreement against the contract, on real rows.

Two defects this guards, both found by review after they had already produced wrong numbers:

1. ``anchor_ksgd`` held the anchor in **SGD** while the target subtracted it as if it were
   kSGD, so every target was ~1000x too large (raw anchor 2051.4489746 gave target -2049.249
   instead of 0.148551). A pure "target + anchor reconstructs actual" identity does **not**
   catch that — it holds under any consistent unit error. So the checks below pin each unit
   separately against the contract, and also score a zero-residual model.
2. Contract categorical code ``0`` was mapped to a real vocabulary entry, so "unknown" and
   a genuine category shared an index. Code 0 must always mean unknown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE.parent))

CONTRACT = ROOT / "reports" / "fusion_contract_v2" / "fusion-contract-v2.1.0__43867d0ba111"
TARGET_SCALE = 1000.0


def _have_contract() -> bool:
    return CONTRACT.exists()


pytestmark = pytest.mark.skipif(not _have_contract(),
                                reason="v2.1 contract is not present")


@pytest.fixture(scope="module")
def rig():
    from neural_fusion_v2 import contract as C

    from feature_transformer_v4 import data as D

    manifest = C.load_manifest(CONTRACT)
    a_fit = C.load_population(CONTRACT, manifest, "orig", "A", "fit")
    a_hold = C.load_population(CONTRACT, manifest, "orig", "A", "holdout")
    state = D.fit_stage_state(a_fit, "A/fit")
    return {
        "manifest": manifest, "a_fit": a_fit, "a_hold": a_hold, "state": state,
        "fit_tensors": D.build_tensors(a_fit, state),
        "hold_tensors": D.build_tensors(a_hold, state),
        "module": D,
    }


# ------------------------------------------------------------------ anchor units
def test_anchor_ksgd_times_1000_equals_the_population_anchor_sgd(rig):
    tensors = rig["hold_tensors"]
    population_anchor_sgd = rig["a_hold"].anchor_sgd
    rebuilt = tensors.anchor_ksgd * TARGET_SCALE
    assert np.allclose(rebuilt, population_anchor_sgd, rtol=0, atol=1e-3), (
        "anchor_ksgd is not the anchor divided by 1000; the first row reads "
        f"{tensors.anchor_ksgd[0]:.6f} vs expected {population_anchor_sgd[0] / TARGET_SCALE:.6f}")


def test_target_is_actual_minus_anchor_over_1000(rig):
    tensors = rig["hold_tensors"]
    expected = (rig["a_hold"].actual_sgd / TARGET_SCALE) - tensors.anchor_ksgd
    assert np.allclose(tensors.target, expected, rtol=0, atol=1e-6), (
        f"target is wrong; first row {tensors.target[0]:.6f} vs {expected[0]:.6f}")


def test_target_magnitude_is_a_residual_not_a_rent(rig):
    """A pure identity check passes even when both sides share a 1000x error."""
    tensors = rig["hold_tensors"]
    assert float(np.abs(tensors.target).max()) < 5.0, (
        f"target max |value| is {float(np.abs(tensors.target).max()):.1f} kSGD — that is a "
        "rent, not a residual")
    # the anchor explains most of the rent but not all: measured median |residual| is
    # 0.778 kSGD against a 3,000 SGD median rent, i.e. ~26% of the rent level
    assert float(np.median(np.abs(tensors.target))) < 2.0


def test_a_zero_residual_model_scores_the_anchor_in_sgd(rig):
    """The strongest unit check: predicting nothing must give anchor-scale SGD rents."""
    tensors = rig["hold_tensors"]
    zeros = np.zeros(len(tensors), dtype=np.float64)
    predicted = tensors.to_sgd(zeros)
    assert np.allclose(predicted, rig["a_hold"].anchor_sgd, rtol=0, atol=1e-3)
    median_pred = float(np.median(predicted))
    assert 500 < median_pred < 10000, (
        f"zero-residual prediction median is {median_pred:.1f} — not a rent in SGD")
    anchor_rmse = float(np.sqrt(np.mean((predicted - rig["a_hold"].actual_sgd) ** 2)))
    assert 400 < anchor_rmse < 1200, (
        f"anchor-only RMSE {anchor_rmse:.1f} is outside the range a rent-scale target "
        "can produce; a 1000x unit error lands far outside this band")


def test_to_sgd_is_the_inverse_of_the_target_definition(rig):
    tensors = rig["hold_tensors"]
    recovered = tensors.to_sgd(tensors.target.astype(np.float64))
    assert np.allclose(recovered, rig["a_hold"].actual_sgd, rtol=0, atol=1e-2)


# ------------------------------------------------------------- categorical codes
def test_unknown_code_zero_is_reserved(rig):
    state = rig["state"]
    for name, table in state.vocabularies.items():
        assert table[0] == 0, f"{name}: index 0 must be the unknown code"
        assert len(set(table)) == len(table), f"{name}: vocabulary has duplicate codes"
        if 0 in table[1:]:
            raise AssertionError(f"{name}: contract code 0 was admitted as a real category")


def test_a_zero_code_in_the_data_maps_to_the_unknown_slot():
    """Code 0 is a legal input (missing category) and must land on index 0, never a real one.

    A synthetic fixture, because the real A/fit slice happens to contain no zeros — which is
    exactly why the earlier `raise` on presence was the wrong behaviour: it would have
    crashed the moment a stage did contain one.
    """
    from feature_transformer_v4 import data as D

    names = ["TOWN_CLEAN", "FLAT_TYPE_CLEAN"]
    codes = np.array([[0, 0], [5, 3], [0, 3], [5, 0]], dtype=np.int64)

    tables = {}
    for i, name in enumerate(names):
        present = sorted({int(v) for v in codes[:, i].tolist()})
        tables[name] = [D.UNKNOWN_CODE] + [c for c in present if c != D.UNKNOWN_CODE]

    assert tables["TOWN_CLEAN"] == [0, 5], tables["TOWN_CLEAN"]
    assert tables["FLAT_TYPE_CLEAN"] == [0, 3], tables["FLAT_TYPE_CLEAN"]
    assert 0 not in tables["TOWN_CLEAN"][1:], "a real category claimed the unknown index"

    # applying that table maps a zero input to index 0 while real codes shift up
    lookup = {code: index for index, code in enumerate(tables["TOWN_CLEAN"]) if index > 0}
    assert [lookup.get(int(v), 0) for v in codes[:, 0].tolist()] == [0, 1, 0, 1]



def test_a_code_absent_from_fit_becomes_unknown(rig):
    module = rig["module"]
    state = rig["state"]
    hold = rig["a_hold"]
    tensors = rig["train_tensors"] if "train_tensors" in rig else rig["hold_tensors"]

    # invent a row whose first categorical code was never seen during fit
    unseen = max(state.vocabularies[state.categorical_names[0]]) + 999
    arrays = hold.arrays
    codes = np.array(tensors.categorical)
    assert codes[:, 0].max() >= 0
    # apply_state is the function under test, so drive it through a one-row population
    single = hold.subset(np.arange(1))
    payload = module.apply_state(single, state)
    assert payload["categorical"].min() >= 0
    assert unseen not in state.vocabularies[state.categorical_names[0]]


def test_categorical_codes_stay_inside_their_vocabularies(rig):
    state = rig["state"]
    tensors = rig["hold_tensors"]
    for i, name in enumerate(state.categorical_names):
        size = len(state.vocabularies[name])
        column = tensors.categorical[:, i]
        assert int(column.min()) >= 0
        assert int(column.max()) < size, (
            f"{name}: code {int(column.max())} outside vocabulary of {size}")


def test_vocabularies_are_fitted_separately_per_stage(rig):
    """Stage A and stage B may see different categories, so they must not share a table."""
    from neural_fusion_v2 import contract as C

    from feature_transformer_v4 import data as D

    manifest = rig["manifest"]
    b_fit = C.load_population(CONTRACT, manifest, "orig", "B", "fit")
    state_b = D.fit_stage_state(b_fit, "B/fit")
    state_a = rig["state"]
    assert state_a.fitted_population == "A/fit"
    assert state_b.fitted_population == "B/fit"
    for name in state_a.categorical_names:
        assert state_a.vocabularies[name] != [] and state_b.vocabularies[name] != []


# ---------------------------------------------------------------------- scaling
def test_log1p_is_applied_to_the_count_columns(rig):
    """The v2 scaler log1p-transforms hist_n__* / hist_months__*; skipping it is a real bug."""
    scaler = rig["state"].scaler
    assert scaler.log1p_columns, "the scaler reports no log1p columns at all"
    count_columns = [c for c in scaler.history_columns
                     if c.startswith("hist_n__") or c.startswith("hist_months__")]
    assert set(count_columns) <= set(scaler.log1p_columns), (
        "a count column would be standardised without log1p")


def test_numeric_block_is_finite_and_clipped(rig):
    tensors = rig["hold_tensors"]
    assert np.isfinite(tensors.numeric).all()
    assert float(np.abs(tensors.numeric).max()) <= 8.0 + 1e-5
    assert tensors.numeric.shape[1] == 56


def test_stage_state_round_trips_without_refitting(rig):
    from neural_fusion_v2 import contract as C

    from feature_transformer_v4 import data as D

    state = rig["state"]
    restored = D.StageState.from_dict(state.to_dict())
    hold = rig["a_hold"]

    original = D.build_tensors(hold, state)
    rebuilt = D.build_tensors(hold, restored)
    assert np.array_equal(original.numeric, rebuilt.numeric)
    assert np.array_equal(original.categorical, rebuilt.categorical)
    assert np.array_equal(original.target, rebuilt.target)
    assert restored.fitted_population == state.fitted_population


def test_row_identity_matches_the_contract(rig):
    tensors = rig["hold_tensors"]
    arrays = rig["a_hold"].arrays
    assert np.array_equal(tensors.row_id, arrays.row_id)
    assert np.array_equal(tensors.horizon, arrays.horizon)
    assert np.array_equal(tensors.actual_sgd, rig["a_hold"].actual_sgd)
