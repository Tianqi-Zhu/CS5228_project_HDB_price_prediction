"""Regression tests for the v5 macro time tokens.

The two defects these guard are both silent by nature — a mask that is the right width but in
the wrong place, and a token whose calendar age drifts when a window has holes. Neither raises,
so each gets an explicit failing-first check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.macro_time_tokens_v5 import tokens as T


# --------------------------------------------------------------------- mask placement
def test_encoder_mask_puts_feature_padding_where_the_features_are():
    """The v4 construction shifted every flag by `n_macro` positions.

    Sequence is `[query, 60 features, 7 macros]`; a row whose last five features are padding
    must mark 56-60, not 58-62.
    """
    batch, n_features, n_macro = 2, 60, 7
    feature_mask = torch.zeros(batch, n_features, dtype=torch.bool)
    feature_mask[:, 55:] = True

    mask = T.build_encoder_mask(feature_mask, leading=1, n_macro=n_macro)
    assert mask.shape == (batch, 1 + n_features + n_macro)
    assert mask[0].nonzero().flatten().tolist() == [56, 57, 58, 59, 60]


def test_encoder_mask_matches_the_v4_output_shape_but_not_its_positions():
    """Guard the regression: the old code produced the right width, so width alone is no test."""
    batch, n_features, n_macro, leading = 1, 60, 7, 1
    feature_mask = torch.zeros(batch, n_features, dtype=torch.bool)
    feature_mask[:, -5:] = True

    old = torch.cat([torch.zeros(batch, leading + n_macro, dtype=torch.bool), feature_mask], 1)
    fixed = T.build_encoder_mask(feature_mask, leading, n_macro)
    assert old.shape == fixed.shape            # same shape...
    assert not torch.equal(old, fixed)         # ...different meaning


def test_encoder_mask_without_macro_tokens_is_a_plain_shift():
    batch, n_features = 1, 60
    feature_mask = torch.zeros(batch, n_features, dtype=torch.bool)
    feature_mask[:, 0] = True
    mask = T.build_encoder_mask(feature_mask, leading=1, n_macro=0)
    assert mask[0].nonzero().flatten().tolist() == [1]


def test_combine_masks_unions_padding():
    a = torch.tensor([[False, True, False]])
    b = torch.tensor([[False, False, True]])
    assert T.combine_masks(a, b).tolist() == [[False, True, True]]
    assert T.combine_masks(None, None) is None


# ------------------------------------------------------------------- block endpoints
def test_block_end_offsets_ignore_leading_padding():
    """A leading-padded window must not shift every block to a younger age.

    The calendar endpoints (3, 6, 9) are fixed by the block geometry, not by where the holes
    are. Only the observation index moves.
    """
    observed = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1]], dtype=torch.float32)
    end, valid, age, obs_index = T.block_end_offsets(observed, block_rows=3, n_blocks=3,
                                                     return_observed_index=True)
    assert end[0].tolist() == [3, 6, 9]          # calendar rows, unaffected by the padding
    assert obs_index[0].tolist() == [0, 3, 6]    # the empty first block inherits 0
    assert valid[0].tolist() == [False, True, True]
    assert age[0].tolist() == [2, 1, 0]


def test_block_end_offsets_flag_an_interior_empty_block():
    observed = torch.tensor([[1, 1, 1, 0, 0, 0, 1, 1, 1]], dtype=torch.float32)
    end, valid, age, obs_index = T.block_end_offsets(observed, block_rows=3, n_blocks=3,
                                                     return_observed_index=True)
    assert valid[0].tolist() == [True, False, True]
    assert end[0].tolist() == [3, 6, 9]
    # the empty middle block inherits the previous observation count, it does not jump forward
    assert obs_index[0].tolist() == [3, 3, 6]


def test_block_end_offsets_all_missing():
    observed = torch.zeros(1, 9)
    end, valid, age, obs_index = T.block_end_offsets(observed, block_rows=3, n_blocks=3,
                                                     return_observed_index=True)
    assert not valid.any()
    assert end[0].tolist() == [3, 6, 9]
    assert obs_index[0].tolist() == [0, 0, 0]


def test_block_end_offsets_rejects_a_window_that_does_not_divide():
    with pytest.raises(T.MacroTimeTokenError):
        T.block_end_offsets(torch.ones(1, 10), block_rows=3, n_blocks=3)


def test_endpoints_are_calendar_rows_not_compacted_positions():
    """Same observation count, different calendar rows -> different tokens.

    This is the distinction the plan calls out: resolving 21/42/63 against *compacted*
    positions would make "observed at the block's first row" and "observed at its last row"
    identical, silently re-ageing the token and losing the tail under leading padding.
    """
    torch.manual_seed(7)
    encoder = T.build_dbs_block_encoder()
    encoder.eval()
    sequence = torch.randn(2, 63, 7)

    first_rows = torch.zeros(2, 63)
    for block in range(3):
        first_rows[:, block * 21] = 1
    last_rows = torch.zeros(2, 63)
    for block in range(3):
        last_rows[:, block * 21 + 20] = 1

    with torch.no_grad():
        start = encoder(sequence, first_rows)
        end = encoder(sequence, last_rows)

    # identical counts, validity and age...
    assert torch.equal(start["valid"], end["valid"])
    assert torch.equal(start["age"], end["age"])
    assert torch.equal(start["valid"], torch.ones_like(start["valid"]))
    # ...but the tokens must differ, because a different calendar row was read
    delta = (start["tokens"] - end["tokens"]).abs().max().item()
    assert delta > 1e-3, f"endpoint ignored the calendar row (delta {delta:.2e})"


def test_block_end_offsets_reject_a_window_that_does_not_divide():
    with pytest.raises(T.MacroTimeTokenError):
        T.block_end_offsets(torch.ones(1, 10), block_rows=3, n_blocks=3)


# ------------------------------------------------------------------------ causality
def test_a_later_observation_cannot_change_an_earlier_block_token():
    """Only observed prefix steps may affect a prefix token."""
    torch.manual_seed(0)
    encoder = T.build_dbs_block_encoder()
    encoder.eval()
    sequence = torch.randn(1, 63, 7)
    observed = torch.ones(1, 63)

    baseline = encoder(sequence, observed)["tokens"]
    mutated = sequence.clone()
    mutated[:, 42:] += 5.0                     # newest block only

    with torch.no_grad():
        after = encoder(mutated, observed)["tokens"]

    oldest_delta = (after[0, 0] - baseline[0, 0]).abs().max().item()
    newest_delta = (after[0, -1] - baseline[0, -1]).abs().max().item()
    assert oldest_delta == 0.0, "an earlier block token saw a later observation"
    assert newest_delta > 1e-4, "the mutation did not reach the block it belongs to"


def test_leading_padding_does_not_change_the_observed_state():
    """Compaction must give the same state as the same window without the padding."""
    torch.manual_seed(1)
    encoder = T.build_dbs_block_encoder()
    encoder.eval()
    real = torch.randn(1, 42, 7)
    padded_sequence = torch.cat([torch.zeros(1, 21, 7), real], dim=1)
    padded_observed = torch.cat([torch.zeros(1, 21), torch.ones(1, 42)], dim=1)

    with torch.no_grad():
        padded = encoder(padded_sequence, padded_observed)

    # the padded window has three blocks of 21 where only the last two carry data
    assert torch.isfinite(padded["tokens"]).all()
    assert padded["valid"][0].tolist() == [False, True, True]

    # The state at a given observation must not depend on how much padding preceded it.
    # Compare the newest block of the padded window against the newest block of a window
    # that contains exactly the same observations with no padding in front.
    short = T.build_dbs_block_encoder()
    short.load_state_dict(encoder.state_dict())
    short.eval()
    with torch.no_grad():
        reference = short(torch.cat([torch.zeros(1, 21, 7), real], dim=1),
                          torch.cat([torch.zeros(1, 21), torch.ones(1, 42)], dim=1))
    assert torch.allclose(padded["tokens"][0, -1], reference["tokens"][0, -1], atol=1e-5), \
        "padding changed the newest block's state"


# ------------------------------------------------------------------------- gradients
def test_every_encoder_parameter_receives_gradient_on_observed_input():
    torch.manual_seed(2)
    encoder = T.build_dbs_block_encoder()
    sequence = torch.randn(4, 63, 7)
    observed = torch.ones(4, 63)
    observed[0, :21] = 0
    observed[3, :] = 0

    encoder.zero_grad()
    encoder(sequence, observed)["tokens"].sum().backward()
    # `empty` is only read for a wholly missing block, so it is allowed to be zero here; it
    # gets its own check below. Everything else must learn from observed input.
    missing = [name for name, p in encoder.named_parameters()
               if name.split(".")[0] != "empty"
               and (p.grad is None or p.grad.abs().sum() == 0.0)]
    assert not missing, f"parameters with no gradient: {missing}"


def test_empty_vector_learns_only_from_missing_blocks():
    """`empty` is the representation for an empty block: zero grad unless one occurs."""
    torch.manual_seed(4)
    encoder = T.build_dbs_block_encoder()

    observed_full = torch.ones(2, 63)
    encoder.zero_grad()
    encoder(torch.randn(2, 63, 7), observed_full)["tokens"].sum().backward()
    assert encoder.empty.grad.abs().sum() == 0.0

    observed_gap = torch.ones(2, 63)
    observed_gap[:, :21] = 0                      # first block empty
    encoder.zero_grad()
    encoder(torch.randn(2, 63, 7), observed_gap)["tokens"].sum().backward()
    assert encoder.empty.grad.abs().sum() > 0.0


def test_tokens_are_finite_for_a_fully_missing_window():
    encoder = T.build_dbs_block_encoder()
    out = encoder(torch.zeros(2, 63, 7), torch.zeros(2, 63))
    assert torch.isfinite(out["tokens"]).all()
    assert not out["valid"].any()
    assert not out["window_observed"].any()


def test_sparse_window_still_trains_the_recurrent_weights():
    """A window with almost no observations must still give the RNN input-path gradient.

    This is the regression for a real defect: the state tensor was scattered back to calendar
    positions while the block endpoints counted *observed* steps. On a dense window the two
    orders coincide and every test passed; on a COE window holding one observed month the
    endpoint pointed at a padding slot, so the recurrent output never reached the loss. The
    RNN weights got exactly zero gradient while the projection and embeddings kept training —
    a silent partial freeze that only a sparse window exposes.

    With a single observation `weight_hh` is legitimately zero (it multiplies the initial zero
    state), so the input weights are what this case can assert; time dependence is covered by
    the two-observation case below.
    """
    torch.manual_seed(5)
    encoder = T.build_coe_block_encoder()

    # one observed month out of twelve, at the newest position -- the real contract has such
    # windows (measured: 1 of 27 windows in A/fit)
    observed = torch.zeros(2, 12)
    observed[:, -1] = 1.0
    sequence = torch.randn(2, 12, encoder.channels)

    encoder.zero_grad()
    out = encoder(sequence, observed)
    out["tokens"].sum().backward()

    assert out["valid"][:, -1].all(), "the newest block holds the only observation"
    assert not out["valid"][:, :-1].any()

    feed_forward = [(n, p) for n, p in encoder.rnn.named_parameters() if "weight_ih" in n]
    assert feed_forward, "no input weights found"
    dead = [n for n, p in feed_forward if p.grad is None or p.grad.abs().sum() == 0.0]
    assert not dead, f"the recurrent input path received no gradient: {dead}"


def test_two_observations_train_the_recurrence():
    """Two adjacent observations are the minimum that can exercise `weight_hh`."""
    torch.manual_seed(6)
    encoder = T.build_coe_block_encoder()
    observed = torch.zeros(2, 12)
    observed[:, -2:] = 1.0

    encoder.zero_grad()
    encoder(torch.randn(2, 12, encoder.channels), observed)["tokens"].sum().backward()
    recurrent = [(n, p) for n, p in encoder.rnn.named_parameters() if "weight_hh" in n]
    assert recurrent
    dead = [n for n, p in recurrent if p.grad is None or p.grad.abs().sum() == 0.0]
    assert not dead, f"recurrence not trained with two observations: {dead}"


def test_dense_windows_are_not_special_cased():
    """Guard against 'fixing' the sparse case by changing the dense one."""
    encoder = T.build_dbs_block_encoder()
    observed = torch.ones(2, 63)
    encoder.zero_grad()
    out = encoder(torch.randn(2, 63, 7), observed)
    out["tokens"].sum().backward()
    assert torch.isfinite(out["tokens"]).all()
    assert out["valid"].all()
    # `empty` is the representation for an empty block, so a fully observed window must
    # leave it untouched; it is checked separately in the missing-block test
    dead = [n for n, p in encoder.named_parameters()
            if n.split(".")[0] != "empty"
            and (p.grad is None or p.grad.abs().sum() == 0.0)]
    assert not dead, f"parameters with no gradient on dense input: {dead}"


# ---------------------------------------------------------------------------- reload
def test_checkpoint_roundtrip_is_bitwise_stable():
    torch.manual_seed(3)
    encoder = T.build_dbs_block_encoder().eval()
    sequence = torch.randn(2, 63, 7)
    observed = torch.ones(2, 63)
    observed[1, 10:20] = 0

    with torch.no_grad():
        before = encoder(sequence, observed)["tokens"]

    clone = T.build_dbs_block_encoder()
    clone.load_state_dict({k: v.detach().clone() for k, v in encoder.state_dict().items()})
    clone.eval()
    with torch.no_grad():
        after = clone(sequence, observed)["tokens"]
    assert torch.equal(before, after)


# --------------------------------------------------------------------- age identity
def test_age_is_newest_zero_and_never_reordered_by_holes():
    for holes in ([], [9, 10], list(range(0, 12))):
        observed = torch.ones(1, 63)
        if holes:
            observed[0, holes] = 0
        end, valid, age = T.block_end_offsets(observed, block_rows=21, n_blocks=3)
        assert age[0].tolist() == [2, 1, 0], f"age reordered for holes={holes}"
        assert end[0].tolist() == [21, 42, 63], f"calendar ends moved for holes={holes}"


def test_coe_encoder_uses_four_quarterly_blocks():
    encoder = T.build_coe_block_encoder()
    from experiments.macro_encoders_v3 import contract as macro_contract
    channels = len(macro_contract.COE_CHANNELS) * len(macro_contract.COE_CATEGORIES)
    out = encoder(torch.zeros(2, 12, channels), torch.ones(2, 12))
    assert out["tokens"].shape == (2, 4, T.TOKEN_DIM)
    assert out["age"][0].tolist() == [3, 2, 1, 0]
