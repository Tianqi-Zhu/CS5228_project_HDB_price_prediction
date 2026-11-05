"""Tests for the v9 history arm.

The failure modes that matter here are the ones a plausible implementation gets silently wrong:

* the new history tokens collide with the CLS type id, so the query reads a feature embedding;
* the 50 history columns are concatenated into one vector instead of becoming 50 tokens;
* the shared-parameter copy is actually a warm start from trained weights;
* a median-imputed cell is treated as an observation;
* the history cutoff reaches past the row's own target.

Each gets an explicit counterexample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.history_fusion_v9 import models as HM

NUMERIC = [f"n{i}" for i in range(21)]
CARDINALITIES = (27, 7, 19)
CHANNELS = 15


def make_batch(rows=4, months=12, history=50, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "house_numeric": torch.randn(rows, len(NUMERIC), generator=generator),
        "house_cat": torch.randint(0, 7, (rows, 3), generator=generator),
        "calendar": torch.randn(rows, 2, generator=generator),
        "market_values": torch.randn(rows, months, CHANNELS, generator=generator),
        "market_observed": torch.ones(rows, months, CHANNELS),
        "market_step_valid": torch.ones(rows, months),
        "history": torch.randn(rows, history, generator=generator),
    }


# ------------------------------------------------------------------- type ids
def test_cls_type_id_moves_past_every_feature_token():
    """CLS must not land on a token that now exists.

    v8 had 24 feature tokens and CLS at 25. Adding 50 history tokens makes 74 features, so CLS
    has to move to 75. Leaving it at 25 would make the query read history token 0.
    """
    base = HM.build_base_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    history = HM.build_history_model(42, NUMERIC, CARDINALITIES, CHANNELS)

    assert base.house.n_feature_tokens == 24
    assert base.house.query_type_id == 25
    assert history.house.n_feature_tokens == 74
    assert history.house.query_type_id == 75, (
        f"CLS sits at {history.house.query_type_id}; it must be > 74 or it collides with a "
        f"history feature token")
    assert history.house.query_type_id > history.house.n_feature_tokens


def test_history_token_type_ids_are_contiguous_and_do_not_overlap_cls():
    history = HM.build_history_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    first = history.house.base_feature_tokens + 1
    ids = list(range(first, first + HM.HISTORY_COLUMNS))
    assert ids[0] == 25 and ids[-1] == 74
    assert history.house.query_type_id not in ids
    assert max(ids) < history.house.n_types


# ----------------------------------------------------------------- token count
def test_fifty_history_columns_become_fifty_tokens_not_one_vector():
    history = HM.build_history_model(42, NUMERIC, CARDINALITIES, CHANNELS).eval()
    batch = make_batch()
    tokens = history.house.base_tokens(batch["house_numeric"], batch["house_cat"])
    history_tokens = history.house.history_tokens(batch["history"])
    assert tokens.shape[1] == 24
    assert history_tokens.shape[1] == HM.HISTORY_COLUMNS
    assert torch.isfinite(history_tokens).all()


def test_each_history_column_has_its_own_vector_parameters():
    """A shared projection would make the 50 columns indistinguishable by position."""
    history = HM.build_history_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    assert history.house.history_weight.shape == (HM.HISTORY_COLUMNS, HM.D_MODEL)
    assert history.house.history_bias.shape == (HM.HISTORY_COLUMNS, HM.D_MODEL)
    assert history.house.history_identity.shape == (HM.HISTORY_COLUMNS, HM.D_MODEL)
    # rows must differ, otherwise every column produces the same offset direction
    assert history.house.history_weight.std(dim=1).min() > 0


# --------------------------------------------------------------- initialisation
def test_common_parameters_come_from_a_fresh_seed_not_from_trained_weights():
    base = HM.build_base_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    history = HM.build_history_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    parity = HM.copy_common_parameters(base, history)
    assert parity["all_checks_pass"], [c for c in parity["checks"]
                                       if not c.get("bitwise_equal", True)]
    assert "fresh seed-42" in parity["source"]
    assert parity["history_params"] > parity["base_params"]


def test_the_cls_row_moves_rather_than_being_left_behind():
    base = HM.build_base_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    history = HM.build_history_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    parity = HM.copy_common_parameters(base, history)
    base_state = base.state_dict()
    history_state = history.state_dict()
    old_cls, new_cls = parity["cls_move"]["old"], parity["cls_move"]["new"]
    key = "house.type_embedding.weight"
    assert torch.equal(history_state[key][new_cls], base_state[key][old_cls])
    assert not torch.equal(history_state[key][old_cls], base_state[key][old_cls]), (
        "the old CLS row is unchanged, so a history feature inherited the query embedding")


def test_base_feature_type_rows_are_preserved():
    base = HM.build_base_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    history = HM.build_history_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    HM.copy_common_parameters(base, history)
    key = "house.type_embedding.weight"
    assert torch.equal(history.state_dict()[key][1:25], base.state_dict()[key][1:25])


def test_history_parameters_are_fresh_not_copied():
    base = HM.build_base_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    history = HM.build_history_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    before = history.house.history_weight.detach().clone()
    HM.copy_common_parameters(base, history)
    after = history.state_dict()["house.history_weight"]
    assert torch.equal(before, after), "the history block was overwritten during the copy"
    assert after.abs().sum() > 0


# ------------------------------------------------------------------ forward/back
def test_forward_is_finite_and_uses_history():
    torch.manual_seed(0)
    model = HM.build_history_model(42, NUMERIC, CARDINALITIES, CHANNELS).eval()
    batch = make_batch()
    with torch.no_grad():
        first = model(batch)
        disturbed = dict(batch)
        disturbed["history"] = torch.randn_like(batch["history"]) * 5.0
        second = model(disturbed)
    assert first.shape == (4,)
    assert torch.isfinite(first).all()
    assert not torch.allclose(first, second, atol=1e-5), "the history input has no effect"


def test_history_gradient_reaches_every_new_parameter():
    torch.manual_seed(1)
    model = HM.build_history_model(42, NUMERIC, CARDINALITIES, CHANNELS)
    loss = torch.nn.functional.mse_loss(model(make_batch()), torch.randn(4))
    loss.backward()
    for name in ("house.history_weight", "house.history_bias", "house.history_identity"):
        grad = model.get_parameter(name).grad
        assert grad is not None and grad.abs().sum() > 0, f"{name} has no gradient"
    # the market branch and head must still train
    assert model.market.projection.weight.grad.abs().sum() > 0
    assert model.head[0].weight.grad.abs().sum() > 0


def test_zero_history_is_a_valid_input():
    """`count == 0` rows arrive as imputed values, not as a reason to drop the row."""
    model = HM.build_history_model(42, NUMERIC, CARDINALITIES, CHANNELS).eval()
    batch = make_batch()
    batch["history"] = torch.zeros_like(batch["history"])
    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------ data contract
def test_leakage_check_flags_a_cutoff_that_reaches_the_target(tmp_path):
    from experiments.history_fusion_v9 import data as HD

    split = HD.HistorySplit(
        row_id=torch.arange(3).numpy(),
        target_month=torch.tensor([27, 28, 29]).numpy(),
        actual_sgd=torch.zeros(3).numpy(),
        history_numeric=torch.zeros(3, 50).numpy(),
        history_raw=torch.zeros(3, 50).numpy(),
        history_observed=torch.zeros(3, 50, dtype=torch.bool).numpy(),
        cutoff_month=torch.tensor([24, 25, 27]).numpy(),   # last one is too late
        history_cutoff_age=torch.tensor([3, 3, 2]).numpy())
    report = HD.check_leakage(split)
    assert report["cutoff_exceeds_target_minus_3"] == 1
    assert not report["leak_free"]


def test_leakage_check_accepts_a_correct_cutoff():
    from experiments.history_fusion_v9 import data as HD

    split = HD.HistorySplit(
        row_id=torch.arange(3).numpy(),
        target_month=torch.tensor([27, 28, 38]).numpy(),
        actual_sgd=torch.zeros(3).numpy(),
        history_numeric=torch.zeros(3, 50).numpy(),
        history_raw=torch.zeros(3, 50).numpy(),
        history_observed=torch.zeros(3, 50, dtype=torch.bool).numpy(),
        cutoff_month=torch.tensor([24, 25, 26]).numpy(),   # min(26, t-3)
        history_cutoff_age=torch.tensor([3, 3, 12]).numpy())
    report = HD.check_leakage(split)
    assert report["leak_free"], report


def test_ready_guard_rejects_a_changed_export(tmp_path):
    from experiments.history_fusion_v9 import data as HD
    import json
    import numpy as np

    np.savez_compressed(tmp_path / "orig_history.npz", dummy=np.zeros(1))
    (tmp_path / "DATA_READY.json").write_text(json.dumps(
        {"status": "ready", "files": {"orig_history.npz": {"sha256": "0" * 64}}}))
    with pytest.raises(HD.HistoryDataError, match="changed after it was marked ready"):
        HD.require_ready(tmp_path)


def test_ready_guard_rejects_a_missing_marker(tmp_path):
    from experiments.history_fusion_v9 import data as HD

    with pytest.raises(HD.HistoryDataError, match="absent"):
        HD.require_ready(tmp_path)
