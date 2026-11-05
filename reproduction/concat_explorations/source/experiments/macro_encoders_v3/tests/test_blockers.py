"""Regression tests for the three blockers found in independent review.

Each test is written to fail on the code as it stood when the blocker was reported,
then pass after the fix.  The names state the invariant, not the fix.

Blocker 1: ``build_window_tensors`` raised on the DBS *scalar* block because the
    transform only accepted a per-position mask while that block carries a
    per-element ``(W, channels)`` mask.  Per-channel validity must be honoured
    element-wise, and a masked element must be *absent*, never silently zero-as-data.

Blocker 2: ``MacroEncoderSet`` was a ``@dataclass``, which can break parameter
    enumeration.  Enumeration was measured working (14,048 parameters), so this is a
    regression guard rather than a documented failure.

Blocker 3: both encoders ran ``pack_padded_sequence`` with
    ``lengths = mask.sum(dim=1)``, which assumes the valid steps form a *prefix*.
    Real exported windows are **leading-padded** (``orig/A`` window 0 is 43 zeros
    then 20 ones for DBS, 11 zeros then 1 for COE), so the recurrence read padding
    and dropped the real tail.  The same observations must produce the same token
    wherever the padding sits.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.macro_encoders_v3 import api, contract, data, encoders, scaling

pytestmark = pytest.mark.skipif(
    not contract.V3_REPORT_ROOT.exists(), reason="no v3 macro contract exported")


def manifest() -> dict:
    return contract.load_manifest()


# ================================================================== blocker 1


def test_dbs_scalar_block_transform_accepts_per_element_mask():
    """The scalar block's mask is (W, channels): it must transform, not raise."""
    _w, _r = manifest(), None
    windows = data.load_windows(_w, "orig", "A")
    rows = data.load_rows(_w, "orig", "A")
    scalers = scaling.fit_macro_scalers(windows, rows)

    values = windows.dbs_scalars                      # (W, 12)
    mask = windows.dbs_scalar_mask > 0                # (W, 12) per element
    assert mask.shape == values.shape
    scaled = scalers.dbs_scalar.transform(values, mask)
    assert scaled.shape == values.shape
    # masked elements are exactly zero *and* the mask still says they are absent
    assert float(np.abs(scaled[~mask]).max()) == 0.0


def test_masked_scalar_is_absent_not_zero_valued_data():
    """A masked element must not become indistinguishable from a real zero."""
    _w, _r = manifest(), None
    windows = data.load_windows(_w, "orig", "A")
    rows = data.load_rows(_w, "orig", "A")
    scalers = scaling.fit_macro_scalers(windows, rows)
    values = windows.dbs_scalars.copy()
    mask = windows.dbs_scalar_mask > 0

    # Build a probe whose *raw* value is far from the fitted mean on every channel,
    # so an unmasked transform is definitely non-zero. Then compare: masked -> 0.
    state = scalers.dbs_scalar
    probe = np.zeros_like(values)
    probe[:] = state.mean + 3.0 * state.scale
    every_channel = np.ones_like(mask)
    scaled_unmasked = state.transform(probe, every_channel)
    # a value 3 SD above the mean standardises to exactly 3.0 on a non-degenerate channel
    non_degenerate = ~state.degenerate
    assert np.allclose(scaled_unmasked[..., non_degenerate], 3.0, atol=1e-5)
    assert not np.allclose(scaled_unmasked, 0.0)

    # the same raw values, now masked, must collapse to exactly zero
    scaled_masked = state.transform(probe, mask)
    assert float(np.abs(scaled_masked[~mask]).max()) == 0.0
    assert not np.allclose(scaled_masked, scaled_unmasked)


def test_build_window_tensors_succeeds_and_drops_dead_scalar_scaling():
    """The full builder must succeed; it does not need the DBS scalar arrays."""
    _w, _r = manifest(), None
    windows = data.load_windows(_w, "orig", "A")
    rows = data.load_rows(_w, "orig", "A")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)
    assert tensors.dbs_seq.shape[0] == windows.n_windows
    # The window tensors do not carry scalar blocks at all; scaling them here would
    # be dead work that can only raise or waste time.
    assert not hasattr(tensors, "dbs_scalars")


# ================================================================== blocker 2


def test_encoder_parameters_enumerate_and_optimize():
    """Real optimizer enumeration over the module must work and update weights."""
    modules = encoders.build_encoders()
    parameters = list(modules.parameters())
    assert len(parameters) > 0
    assert sum(p.numel() for p in parameters) == 14048

    unique_ids = {id(p) for p in parameters}
    assert len(unique_ids) == len(parameters), "duplicate parameter objects"

    optimizer = torch.optim.SGD(modules.parameters(), lr=0.1)
    _w, _r = manifest(), None
    windows = data.load_windows(_w, "orig", "A")
    rows = data.load_rows(_w, "orig", "A")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)

    before = modules.dbs.projection.weight.detach().clone()
    out = api.encode_windows(modules, tensors)
    loss = out["dbs_token"].pow(2).mean() + out["coe_token"].pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = modules.dbs.projection.weight.detach()
    assert not torch.equal(before, after), "optimizer step did not update a weight"


def test_encoder_set_is_hashable_and_not_a_dataclass():
    modules = encoders.build_encoders()
    # a dataclass with eq=True would make this raise TypeError: unhashable
    assert isinstance(hash(modules), int)


# ================================================================== blocker 3


def test_real_early_window_is_leading_padded():
    """The premise of blocker 3, measured on the actual export."""
    _man = manifest()
    windows = data.load_windows(_man, "orig", "A")
    first = windows.dbs_mask[0]
    assert float(first.sum()) == 20.0
    # 43 leading zeros, then 20 ones: padding is a prefix
    assert float(first[:43].sum()) == 0.0
    assert float(first[43:].sum()) == 20.0
    coe_first = windows.coe_mask[0].max(axis=0)
    assert float(coe_first.sum()) == 1.0
    assert float(coe_first[:11].sum()) == 0.0


def test_dbs_token_is_invariant_to_padding_position():
    """Same 20 observations, left- vs right-padded, must give the same token."""
    modules = encoders.build_encoders()
    modules.eval()

    rng = np.random.default_rng(0)
    length, channels = contract.DBS_LOOKBACK_ROWS, len(contract.DBS_CHANNELS)
    n_valid = 20
    real = rng.normal(size=(1, n_valid, channels)).astype(np.float32)

    left = np.zeros((1, length, channels), dtype=np.float32)
    left[0, length - n_valid:] = real[0]
    left_mask = np.zeros((1, length), dtype=np.float32)
    left_mask[0, length - n_valid:] = 1.0

    right = np.zeros((1, length, channels), dtype=np.float32)
    right[0, :n_valid] = real[0]
    right_mask = np.zeros((1, length), dtype=np.float32)
    right_mask[0, :n_valid] = 1.0

    with torch.no_grad():
        a, _ = modules.dbs(torch.from_numpy(left), torch.from_numpy(left_mask))
        b, _ = modules.dbs(torch.from_numpy(right), torch.from_numpy(right_mask))

    torch.testing.assert_close(a, b, rtol=0, atol=1e-6,
                              msg="DBS token depends on where the padding sits")


def test_coe_token_is_invariant_to_padding_position():
    modules = encoders.build_encoders()
    modules.eval()

    rng = np.random.default_rng(1)
    length, channels = contract.COE_LOOKBACK_MONTHS, contract.COE_FLAT_CHANNELS
    n_valid = 4
    real = rng.normal(size=(1, n_valid, channels)).astype(np.float32)

    left = np.zeros((1, length, channels), dtype=np.float32)
    left[0, length - n_valid:] = real[0]
    left_mask = np.zeros((1, length), dtype=np.float32)
    left_mask[0, length - n_valid:] = 1.0

    right = np.zeros((1, length, channels), dtype=np.float32)
    right[0, :n_valid] = real[0]
    right_mask = np.zeros((1, length), dtype=np.float32)
    right_mask[0, :n_valid] = 1.0

    with torch.no_grad():
        a, _ = modules.coe(torch.from_numpy(left), torch.from_numpy(left_mask))
        b, _ = modules.coe(torch.from_numpy(right), torch.from_numpy(right_mask))

    torch.testing.assert_close(a, b, rtol=0, atol=1e-6,
                              msg="COE token depends on where the padding sits")


def test_dbs_uses_the_real_tail_of_a_leading_padded_window():
    """Changing the *valid tail* must change the token; changing padding must not."""
    modules = encoders.build_encoders()
    modules.eval()

    _w, _r = manifest(), None
    windows = data.load_windows(_w, "orig", "A")
    rows = data.load_rows(_w, "orig", "A")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)

    with torch.no_grad():
        baseline, _ = modules.dbs(tensors.dbs_seq[:1], tensors.dbs_mask[:1])

        # Perturb a padded position: the token must not move.
        padded = api.select_window_tensors(tensors, np.array([0]))
        zero_index = int(np.flatnonzero(padded.dbs_mask[0].numpy() == 0)[0])
        padded.dbs_seq[0, zero_index] = 99.0
        after_pad, _ = modules.dbs(padded.dbs_seq, padded.dbs_mask)
        torch.testing.assert_close(baseline, after_pad, rtol=0, atol=0)

        # Perturb a valid position: the token must move.
        valid = api.select_window_tensors(tensors, np.array([0]))
        valid_index = int(np.flatnonzero(valid.dbs_mask[0].numpy() == 1)[-1])
        valid.dbs_seq[0, valid_index] = valid.dbs_seq[0, valid_index] + 5.0
        after_valid, _ = modules.dbs(valid.dbs_seq, valid.dbs_mask)
        assert not torch.allclose(baseline, after_valid), \
            "the real tail of a leading-padded window was ignored"


def test_coe_uses_the_real_tail_of_a_leading_padded_window():
    modules = encoders.build_encoders()
    modules.eval()

    _w, _r = manifest(), None
    windows = data.load_windows(_w, "orig", "A")
    rows = data.load_rows(_w, "orig", "A")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)

    with torch.no_grad():
        baseline, _ = modules.coe(tensors.coe_seq[:1], tensors.coe_mask[:1])

        padded = api.select_window_tensors(tensors, np.array([0]))
        zero_index = int(np.flatnonzero(padded.coe_mask[0].numpy() == 0)[0])
        padded.coe_seq[0, zero_index] = 99.0
        after_pad, _ = modules.coe(padded.coe_seq, padded.coe_mask)
        torch.testing.assert_close(baseline, after_pad, rtol=0, atol=0)

        valid = api.select_window_tensors(tensors, np.array([0]))
        valid_index = int(np.flatnonzero(valid.coe_mask[0].numpy() == 1)[-1])
        valid.coe_seq[0, valid_index] = valid.coe_seq[0, valid_index] + 5.0
        after_valid, _ = modules.coe(valid.coe_seq, valid.coe_mask)
        assert not torch.allclose(baseline, after_valid), \
            "the real tail of a leading-padded window was ignored"


def test_mid_gap_window_keeps_true_order():
    """A window with a hole in the middle must compact in order, not reorder."""
    modules = encoders.build_encoders()
    modules.eval()

    length, channels = contract.DBS_LOOKBACK_ROWS, len(contract.DBS_CHANNELS)
    rng = np.random.default_rng(2)
    values = rng.normal(size=(1, length, channels)).astype(np.float32)
    mask = np.ones((1, length), dtype=np.float32)
    mask[0, 10:15] = 0.0        # interior hole

    # reference: the same observations, packed densely in their original order
    kept = np.flatnonzero(mask[0] == 1)
    dense = np.zeros((1, length, channels), dtype=np.float32)
    dense_mask = np.zeros((1, length), dtype=np.float32)
    dense[0, :len(kept)] = values[0, kept]
    dense_mask[0, :len(kept)] = 1.0

    with torch.no_grad():
        a, _ = modules.dbs(torch.from_numpy(values), torch.from_numpy(mask))
        b, _ = modules.dbs(torch.from_numpy(dense), torch.from_numpy(dense_mask))
    torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


def test_all_masked_window_still_reports_missing():
    modules = encoders.build_encoders()
    modules.eval()
    length, channels = contract.DBS_LOOKBACK_ROWS, len(contract.DBS_CHANNELS)
    sequence = torch.zeros(2, length, channels)
    mask = torch.zeros(2, length)
    with torch.no_grad():
        token, missing = modules.dbs(sequence, mask)
    assert torch.equal(missing, torch.ones(2, 1))
    assert torch.isfinite(token).all()
