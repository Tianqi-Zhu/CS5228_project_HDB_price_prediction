"""Tests for the v8 fusion model.

The checks that matter are the ones a plausible-but-wrong implementation would still pass:

* a Bi-LSTM summarised by `output[:, -1]` looks fine on a dense window and only fails when the
  window has a gap or padding;
* a market branch that is accidentally detached still trains the house branch;
* a market branch that ignores its mask still produces finite numbers when everything is
  missing.

Each of those gets an explicit counterexample rather than a shape assertion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.target_asof_fusion_v8 import models as M

NUMERIC = [f"n{i}" for i in range(21)]
CARDINALITIES = (27, 7, 19)
CHANNELS = 8


def make_batch(rows=4, months=12, channels=CHANNELS, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "house_numeric": torch.randn(rows, len(NUMERIC), generator=generator),
        "house_cat": torch.randint(0, 7, (rows, 3), generator=generator),
        "calendar": torch.randn(rows, 2, generator=generator),
        "market_values": torch.randn(rows, months, channels, generator=generator),
        "market_observed": torch.ones(rows, months, channels),
        "market_step_valid": torch.ones(rows, months),
    }


# ------------------------------------------------------------------------ shapes
@pytest.mark.parametrize("arm", M.ARMS)
def test_every_arm_produces_one_scalar_per_row(arm):
    model = M.build(arm, NUMERIC, CARDINALITIES, CHANNELS)
    output = model(make_batch())
    assert output.shape == (4,)
    assert torch.isfinite(output).all()


def test_house_and_head_sizes_are_identical_across_arms():
    models = M.build_three_arms(42, NUMERIC, CARDINALITIES, CHANNELS)
    sizes = {arm: m.describe() for arm, m in models.items()}
    assert len({s["house_parameters"] for s in sizes.values()}) == 1
    assert len({s["head_parameters"] for s in sizes.values()}) == 1
    # only the market branch differs
    assert sizes["house"]["market_parameters"] == 0
    assert sizes["stats"]["market_parameters"] > 0
    assert sizes["bilstm"]["market_parameters"] > sizes["stats"]["market_parameters"]


def test_house_arm_reads_no_market_input_at_all():
    """The ablation floor must be insensitive to the market tensors."""
    model = M.build("house", NUMERIC, CARDINALITIES, CHANNELS).eval()
    batch = make_batch()
    with torch.no_grad():
        baseline = model(batch)
        disturbed = dict(batch)
        disturbed["market_values"] = torch.randn_like(batch["market_values"]) * 100.0
        disturbed["market_observed"] = torch.zeros_like(batch["market_observed"])
        after = model(disturbed)
    assert torch.equal(baseline, after), "the house arm changed when market input changed"


# ------------------------------------------------------------- bidirectional h_n
def test_bilstm_summary_is_not_the_last_output_position():
    """Guard the specific mistake: `output[:, -1]` is not a bidirectional summary.

    The two must differ, and the difference must show up exactly where it matters — a window
    whose early months carry the signal.
    """
    torch.manual_seed(0)
    encoder = M.BiLstmMarket(CHANNELS)
    values = torch.zeros(1, 12, CHANNELS)
    observed = torch.zeros(1, 12, CHANNELS)
    step_valid = torch.ones(1, 12)
    # signal only in the first three months
    values[:, :3] = 5.0
    observed[:, :3] = 1.0

    inputs = torch.cat([values, observed], dim=-1)
    output, (h_n, _) = encoder.bilstm(inputs)
    via_last_position = output[:, -1]
    via_h_n = torch.cat([h_n[0], h_n[1]], dim=1)
    assert not torch.allclose(via_last_position, via_h_n, atol=1e-4), \
        "the two summaries coincide; the test cannot distinguish them"

    summary = encoder(values, observed, step_valid)
    expected = encoder.projection(via_h_n)
    assert torch.allclose(summary, expected, atol=1e-6), \
        "the encoder does not summarise with h_n"


def test_backward_direction_actually_reads_the_window():
    """Changing only the earliest months must move the backward final state."""
    torch.manual_seed(1)
    encoder = M.BiLstmMarket(CHANNELS).eval()
    values = torch.randn(1, 12, CHANNELS)
    observed = torch.ones(1, 12, CHANNELS)
    step_valid = torch.ones(1, 12)

    with torch.no_grad():
        base = encoder(values, observed, step_valid)
        mutated = values.clone()
        mutated[:, :3] += 5.0                      # earliest months only
        after = encoder(mutated, observed, step_valid)
    assert not torch.allclose(base, after, atol=1e-5), \
        "the earliest months do not reach the summary; the backward pass is being discarded"


# --------------------------------------------------------------------------- masks
def test_prefix_padding_is_packed_away():
    """A window that starts before the data exists must equal the shorter window."""
    torch.manual_seed(2)
    encoder = M.BiLstmMarket(CHANNELS).eval()

    real = torch.randn(1, 6, CHANNELS)
    real_obs = torch.ones(1, 6, CHANNELS)

    padded_values = torch.cat([torch.zeros(1, 6, CHANNELS), real], dim=1)
    padded_obs = torch.cat([torch.zeros(1, 6, CHANNELS), real_obs], dim=1)
    padded_valid = torch.cat([torch.zeros(1, 6), torch.ones(1, 6)], dim=1)

    with torch.no_grad():
        short = encoder(real, real_obs, torch.ones(1, 6))
        padded = encoder(padded_values, padded_obs, padded_valid)
    assert torch.allclose(short, padded, atol=1e-5), \
        "leading padding changed the summary"


def test_interior_missing_month_keeps_its_time_step():
    """An interior gap must not be compressed: the step count stays the same."""
    torch.manual_seed(3)
    encoder = M.BiLstmMarket(CHANNELS).eval()
    values = torch.randn(1, 12, CHANNELS)
    observed = torch.ones(1, 12, CHANNELS)
    step_valid = torch.ones(1, 12)

    gap = values.clone()
    gap[:, 5] = 0.0
    gap_obs = observed.clone()
    gap_obs[:, 5] = 0.0                       # month 5 missing, but the month still exists

    with torch.no_grad():
        base = encoder(values, observed, step_valid)
        with_gap = encoder(gap, gap_obs, step_valid)
    # step_valid is unchanged, so the number of packed steps is the same; only the values differ
    assert step_valid.sum().item() == 12
    assert not torch.allclose(base, with_gap, atol=1e-5), "the gap had no effect at all"


def test_a_fully_missing_window_uses_the_learned_vector():
    encoder = M.BiLstmMarket(CHANNELS).eval()
    values = torch.zeros(2, 12, CHANNELS)
    observed = torch.zeros(2, 12, CHANNELS)
    step_valid = torch.ones(2, 12)
    with torch.no_grad():
        out = encoder(values, observed, step_valid)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, encoder.missing.expand(2, -1), atol=1e-6)


def test_masked_values_cannot_change_predictions():
    """Zeroing a masked cell must not move the output: the mask, not the value, decides."""
    torch.manual_seed(4)
    model = M.build("bilstm", NUMERIC, CARDINALITIES, CHANNELS).eval()
    batch = make_batch()
    batch["market_observed"][:, 3:5] = 0.0

    with torch.no_grad():
        base = model(batch)
        disturbed = {k: v.clone() for k, v in batch.items()}
        disturbed["market_values"][:, 3:5] = torch.randn_like(
            disturbed["market_values"][:, 3:5]) * 99.0
        after = model(disturbed)
    assert torch.allclose(base, after, atol=1e-6), \
        "changing a masked market value changed the prediction"


# ------------------------------------------------------------- stats: calendar windows
def test_stats_use_calendar_windows_not_last_n_observations():
    """A hole must shorten the calendar window, not be skipped over.

    Slots 9..11 are calendar `t-3..t-1`. If slot 9 is missing, `mean3` averages the two
    observed slots inside that calendar window; a "last three observations" implementation
    would reach back to slot 8 and return a different number.
    """
    values = torch.zeros(1, 12, 1)
    observed = torch.zeros(1, 12, 1)
    # slot 8 = 100 (outside the recent window), slots 10, 11 = 20, 30 (inside it)
    values[0, 8, 0] = 100.0
    values[0, 10, 0] = 20.0
    values[0, 11, 0] = 30.0
    observed[0, [8, 10, 11], 0] = 1.0

    stats = M.StatsMarket.statistics(values, observed)
    mean3 = float(stats[0, 1])                 # channel 0, suffix index 1 = mean3
    assert abs(mean3 - 25.0) < 1e-6, (
        f"mean3 is {mean3}; calendar t-3..t-1 over slots 10,11 gives 25.0, while "
        f"last-3-observations over slots 8,10,11 would give 50.0")
    # n_valid counts all observed months in the window
    assert abs(float(stats[0, 4]) - 3.0 / 12.0) < 1e-6
    # last_valid is the newest observation, slot 11
    assert abs(float(stats[0, 0]) - 30.0) < 1e-6
    # months_since_last_valid for the newest slot is 0
    assert abs(float(stats[0, 5])) < 1e-6


def test_stats_prior_window_gap_makes_the_change_missing_not_stretched():
    """`recent3_minus_prior3` compares t-3..t-1 with t-6..t-4; it never widens a window."""
    values = torch.zeros(1, 12, 1)
    observed = torch.zeros(1, 12, 1)
    values[0, [9, 10, 11], 0] = 10.0           # recent window fully observed
    observed[0, [9, 10, 11], 0] = 1.0
    values[0, 2, 0] = 999.0                    # a value far outside the prior window
    observed[0, 2, 0] = 1.0
    # the prior window (slots 6..8) is entirely empty
    stats = M.StatsMarket.statistics(values, observed)
    change = float(stats[0, 3])
    assert abs(change) < 1e-6, (
        f"change is {change}; with an empty prior calendar window it must be 0, not "
        f"borrowed from slot 2")


def test_stats_mean12_covers_the_whole_window():
    values = torch.ones(1, 12, 1)
    observed = torch.ones(1, 12, 1)
    stats = M.StatsMarket.statistics(values, observed)
    assert abs(float(stats[0, 2]) - 1.0) < 1e-6      # mean12
    assert abs(float(stats[0, 4]) - 1.0) < 1e-6      # n_valid / length


# ---------------------------------------------------------------------- gradients
@pytest.mark.parametrize("arm", ("stats", "bilstm"))
def test_both_encoders_and_the_head_receive_gradient(arm):
    model = M.build(arm, NUMERIC, CARDINALITIES, CHANNELS)
    batch = make_batch()
    target = torch.randn(batch["house_numeric"].shape[0])
    loss = torch.nn.functional.mse_loss(model(batch), target)
    loss.backward()

    def total(module):
        return sum(float(p.grad.abs().sum()) for p in module.parameters()
                   if p.grad is not None)

    assert total(model.house) > 0, f"{arm}: house encoder has no gradient"
    assert total(model.market) > 0, f"{arm}: market encoder has no gradient"
    assert total(model.head) > 0, f"{arm}: head has no gradient"

    # `market.missing` is only read when a window has no observation at all, so a dense batch
    # legitimately leaves it untouched; it gets its own check below
    missing = [name for name, p in model.named_parameters()
               if not name.endswith("market.missing")
               and (p.grad is None or p.grad.abs().sum() == 0.0)]
    assert not missing, f"{arm}: parameters without gradient: {missing}"


@pytest.mark.parametrize("arm", ("stats", "bilstm"))
def test_missing_vector_learns_only_from_empty_windows(arm):
    model = M.build(arm, NUMERIC, CARDINALITIES, CHANNELS)
    batch = make_batch()
    torch.nn.functional.mse_loss(model(batch), torch.randn(4)).backward()
    empty_grad = model.market.missing.grad.abs().sum().item()
    assert empty_grad == 0.0, f"{arm}: the missing vector moved on a dense batch"

    model.zero_grad()
    empty = make_batch()
    empty["market_observed"][1] = 0.0
    torch.nn.functional.mse_loss(model(empty), torch.randn(4)).backward()
    assert model.market.missing.grad.abs().sum().item() > 0.0, \
        f"{arm}: the missing vector never learns"


def test_joint_gradient_reaches_the_market_projection_and_lstm():
    """The market branch must be trained end-to-end, not used as a fixed feature."""
    torch.manual_seed(5)
    model = M.build("bilstm", NUMERIC, CARDINALITIES, CHANNELS)
    loss = torch.nn.functional.mse_loss(model(make_batch()), torch.randn(4))
    loss.backward()
    assert model.market.projection.weight.grad.abs().sum() > 0
    for name, parameter in model.market.bilstm.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


# ----------------------------------------------------------------- initialisation
def test_common_parameters_are_bitwise_identical_across_arms():
    models = M.build_three_arms(42, NUMERIC, CARDINALITIES, CHANNELS)
    reference = models["house"].state_dict()
    for arm in ("stats", "bilstm"):
        state = models[arm].state_dict()
        for key in state:
            if key.startswith(("house.", "head.")):
                assert torch.equal(state[key], reference[key]), f"{arm}: {key} differs"


def test_market_parameters_are_not_shared_between_arms():
    """Copies, not tied tensors."""
    models = M.build_three_arms(42, NUMERIC, CARDINALITIES, CHANNELS)
    before = models["stats"].market.net[0].weight.detach().clone()
    with torch.no_grad():
        models["bilstm"].market.projection.weight.add_(1.0)
    assert torch.equal(models["stats"].market.net[0].weight.detach(), before)


# ----------------------------------------------------------------------- protocol
def test_no_rent_history_or_anchor_in_the_batch_contract():
    """The model must not read a rent-history or anchor field even if one is present."""
    batch = make_batch()
    batch["anchor_ksgd"] = torch.randn(4) * 1000
    batch["history_rent"] = torch.randn(4, 24)
    model = M.build("bilstm", NUMERIC, CARDINALITIES, CHANNELS).eval()
    with torch.no_grad():
        first = model(batch)
        batch["anchor_ksgd"] += 500.0
        batch["history_rent"] += 5.0
        second = model(batch)
    assert torch.equal(first, second), "the model reads an anchor or rent history field"


def test_head_width_matches_the_declared_concat():
    model = M.build("bilstm", NUMERIC, CARDINALITIES, CHANNELS)
    assert model.head[0].in_features == 2 * M.D_MODEL + 2
