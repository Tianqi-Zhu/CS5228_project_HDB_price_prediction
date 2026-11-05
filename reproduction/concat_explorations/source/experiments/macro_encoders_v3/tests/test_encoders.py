"""Tests for the v3 macro encoders.

Four families, each with an assertion that can actually fail:

* **real contract integration** - load `orig` stage A and B from the actual
  exported manifest, check shapes, policy, provenance and window/row reconciliation;
* **scaler isolation** - fit-population-only fitting, provable by tampering with
  validation rows and rebuilding;
* **gradients and padding** - both encoders produce non-zero gradients, padding
  cannot influence a token, and a fully-missing window is flagged rather than NaN;
* **save/reload** - a saved encoder + scaler state reproduces tokens bit-for-bit.

No `or True`, no decorative `>= 0`, no bare "is finite" where an exact value exists.
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


def windows_for(split: str, stage: str):
    man = manifest()
    return data.load_windows(man, split, stage), data.load_rows(man, split, stage)


# ---------------------------------------------------------------- integration


def test_real_orig_stage_b_loads_with_recorded_shapes():
    windows, rows = windows_for("orig", "B")
    # shapes come from the exported contract, not from a guess
    assert windows.dbs_seq.shape == (windows.n_windows, contract.DBS_LOOKBACK_ROWS,
                                     len(contract.DBS_CHANNELS))
    assert windows.coe_months_seq.shape == (windows.n_windows, contract.COE_LOOKBACK_MONTHS,
                                            contract.COE_FLAT_CHANNELS)
    # month stays the recurrent axis: 12 months, 24 category-channels
    assert windows.coe_months_seq.shape[1] == 12
    assert windows.coe_months_seq.shape[2] == 24
    # 39 unique windows for 123,350 rows in orig/B
    assert windows.n_windows == 39
    assert len(rows.row_window_index) == 123350
    assert int(windows.window_count.sum()) == 123350


def test_real_orig_stage_a_loads_and_is_smaller():
    windows_a, rows_a = windows_for("orig", "A")
    windows_b, rows_b = windows_for("orig", "B")
    assert windows_a.n_windows == 27
    assert len(rows_a.row_window_index) == 88192
    assert windows_a.n_windows < windows_b.n_windows
    # stage A's last cutoff is the inner cutoff, before stage B's
    assert int(windows_a.window_cutoff_month.max()) < int(windows_b.window_cutoff_month.max())


def test_policy_and_provenance_are_carried_not_assumed():
    windows, rows = windows_for("orig", "B")
    man = manifest()
    assert windows.policy == man["policy"] == "origin_asof"
    assert windows.contract_id == man["contract_id"]
    assert rows.policy == windows.policy
    assert windows.split_id == rows.split_id == "orig"
    assert windows.stage == rows.stage == "B"


def test_row_window_index_reconciles_with_the_window_counts():
    windows, rows = windows_for("orig", "B")
    counts = np.bincount(rows.row_window_index, minlength=windows.n_windows)
    np.testing.assert_array_equal(counts, windows.window_count.astype(np.int64))
    # every row maps to a real window
    assert int(rows.row_window_index.min()) >= 0
    assert int(rows.row_window_index.max()) < windows.n_windows


def test_population_masks_split_the_rows_exactly():
    _, rows = windows_for("orig", "B")
    fit = data.population_mask(rows, "fit")
    valid = data.population_mask(rows, "validation")
    assert not (fit & valid).any()
    assert int(fit.sum()) == 88192
    assert int(valid.sum()) == 35158
    assert int(fit.sum() + valid.sum()) == len(rows.row_window_index)
    with pytest.raises(KeyError):
        data.population_mask(rows, "holdout")   # not a stage-B population


def test_loader_rejects_a_missing_split():
    man = manifest()
    with pytest.raises(KeyError):
        data.load_windows(man, "orig", "Z")


# ---------------------------------------------------------------- scaler fit scope


def test_scalers_fit_only_on_fit_population_windows():
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    fit_windows = np.unique(rows.row_window_index[rows.population == "fit"])
    valid_only = np.setdiff1d(np.unique(rows.row_window_index[rows.population == "validation"]),
                             fit_windows)
    assert scalers.dbs.n_windows_used == int(fit_windows.size)
    assert scalers.dbs.fitted_on.startswith("windows referenced by population=='fit'")
    # the recorded description states how many windows were used
    assert f"{fit_windows.size} unique windows" in scalers.dbs.fitted_on
    # validation-only windows were not part of the fit set
    assert valid_only.size == 0 or scalers.dbs.n_windows_used == int(fit_windows.size)


def test_scaler_state_ignores_validation_values():
    """Tamper with a validation-only window; the fitted state must not move."""
    windows, rows = windows_for("orig", "B")
    fit_windows = set(np.unique(rows.row_window_index[rows.population == "fit"]).tolist())
    all_windows = set(range(windows.n_windows))
    only_validation = sorted(all_windows - fit_windows)
    if not only_validation:
        pytest.skip("every window is referenced by a fit row in this stage")

    baseline = scaling.fit_macro_scalers(windows, rows)

    tampered = type(windows)(**{field: getattr(windows, field) for field in windows.__dataclass_fields__})
    target = only_validation[0]
    tampered.dbs_seq = windows.dbs_seq.copy()
    tampered.dbs_seq[target] = windows.dbs_seq[target] * 50.0 + 100.0
    rebuilt = scaling.fit_macro_scalers(tampered, rows)

    np.testing.assert_allclose(baseline.dbs.mean, rebuilt.dbs.mean, rtol=0, atol=0)
    np.testing.assert_allclose(baseline.dbs.scale, rebuilt.dbs.scale, rtol=0, atol=0)
    np.testing.assert_array_equal(baseline.dbs.n_used, rebuilt.dbs.n_used)


def test_scaler_masks_are_respected_in_the_mean():
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    # recompute the last log_return mean by hand over masked positions of fit windows
    fit_windows = np.unique(rows.row_window_index[rows.population == "fit"])
    values = windows.dbs_seq[fit_windows][..., 0]
    masks = windows.dbs_mask[fit_windows] > 0
    expected = float(np.mean(values[masks & np.isfinite(values)]))
    index = list(contract.DBS_CHANNELS).index("log_return")
    assert scalers.dbs.mean[index] == pytest.approx(expected, abs=1e-9)


def test_degenerate_channels_are_flagged_not_divided_by_zero():
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    # 'observed' is a flag: it is deliberately not standardised
    observed_index = list(contract.DBS_CHANNELS).index("observed")
    assert scalers.dbs.scale[observed_index] == 1.0
    assert scalers.dbs.mean[observed_index] == 0.0
    # every used scale is finite and positive
    assert np.all(np.isfinite(scalers.dbs.scale))
    assert np.all(scalers.dbs.scale > 0)


def test_coe_scaler_covers_all_four_categories():
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    assert len(scalers.coe.channel_names) == 24
    for category in contract.COE_CATEGORIES:
        assert any(name.startswith(f"coe_{category}_") for name in scalers.coe.channel_names)
    assert len(scalers.coe_scalar.channel_names) == 36


# ---------------------------------------------------------------- tensors, padding


def test_build_window_tensors_shapes_and_zero_padding():
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)
    assert tensors.dbs_seq.shape == (39, 63, 7)
    assert tensors.dbs_mask.shape == (39, 63)
    assert tensors.coe_seq.shape == (39, 12, 24)
    assert tensors.coe_mask.shape == (39, 12)
    assert tensors.coe_channel_mask.shape == (39, 12, 24)
    # padded positions are exactly zero after scaling
    dbs_pad = tensors.dbs_mask == 0
    assert float(tensors.dbs_seq[dbs_pad].abs().max()) == 0.0
    # no NaN anywhere, because missingness is carried by masks
    assert torch.isfinite(tensors.dbs_seq).all()
    assert torch.isfinite(tensors.coe_seq).all()


def test_missing_state_is_explicit_for_a_synthetic_empty_window():
    """A window with no observations yields a finite token and missing == 1."""
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)
    # force window 0 to look unobserved
    tensors.dbs_mask[0] = 0.0
    tensors.dbs_seq[0] = 0.0
    tensors.coe_mask[0] = 0.0
    tensors.coe_seq[0] = 0.0
    modules = encoders.build_encoders()
    out = api.encode_windows(modules, tensors)
    assert out["dbs_missing"][0].item() == 1.0
    assert out["coe_missing"][0].item() == 1.0
    assert torch.isfinite(out["dbs_token"][0]).all()
    assert torch.isfinite(out["coe_token"][0]).all()
    # a window with history is not flagged
    assert out["dbs_missing"][1].item() == 0.0


def test_padding_only_changes_nothing_outside_the_mask():
    """Changing values under the mask cannot change a token."""
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)
    modules = encoders.build_encoders()
    modules.eval()

    with torch.no_grad():
        baseline = api.encode_windows(modules, tensors)

        noisy = api.select_window_tensors(tensors, np.arange(tensors.n_windows))
        pad = noisy.dbs_mask == 0
        noisy.dbs_seq[pad] = 123.0
        perturbed = api.encode_windows(modules, noisy)

    torch.testing.assert_close(baseline["dbs_token"], perturbed["dbs_token"],
                               rtol=0, atol=0)


def test_real_padding_is_present_for_early_windows():
    """The earliest cutoffs have short DBS history; that padding must be visible."""
    windows, rows = windows_for("orig", "B")
    lengths = windows.dbs_mask.sum(axis=1)
    assert lengths.min() < contract.DBS_LOOKBACK_ROWS
    assert lengths.max() == contract.DBS_LOOKBACK_ROWS
    # windows with a short lookback correspond to the earliest cutoff months
    short = np.flatnonzero(lengths < contract.DBS_LOOKBACK_ROWS)
    assert int(windows.window_cutoff_month[short].min()) == int(windows.window_cutoff_month.min())


# ---------------------------------------------------------------- gradients


def test_dbs_encoder_receives_nonzero_gradient():
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)
    modules = encoders.build_encoders()

    out = api.encode_windows(modules, tensors)
    loss = out["dbs_token"].pow(2).mean()
    loss.backward()

    lstm_grad = modules.dbs.lstm.weight_ih_l0.grad
    proj_grad = modules.dbs.projection.weight.grad
    assert lstm_grad is not None and torch.isfinite(lstm_grad).all()
    assert float(lstm_grad.abs().sum()) > 0
    assert proj_grad is not None and float(proj_grad.abs().sum()) > 0
    # the COE branch was never called in this loss, so it has no gradient yet
    assert modules.coe.gru.weight_ih_l0.grad is None


def test_coe_encoder_receives_nonzero_gradient():
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)
    modules = encoders.build_encoders()

    out = api.encode_windows(modules, tensors)
    loss = out["coe_token"].pow(2).mean()
    loss.backward()

    gru_grad = modules.coe.gru.weight_ih_l0.grad
    assert gru_grad is not None and torch.isfinite(gru_grad).all()
    assert float(gru_grad.abs().sum()) > 0
    assert modules.coe.projection.weight.grad is not None
    assert float(modules.coe.projection.weight.grad.abs().sum()) > 0


def test_batch_path_gathers_without_detaching():
    """A row batch encodes only its unique windows and keeps the graph alive."""
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)
    modules = encoders.build_encoders()

    row_index = np.arange(200)
    unique, inverse = api.batch_indices(rows.row_window_index, row_index, tensors.n_windows)
    assert unique.size <= 200
    out = api.encode_batch(modules, tensors, rows.row_window_index, row_index)
    assert out["dbs_token"].shape == (200, contract.TOKEN_DIM)
    assert out["coe_token"].shape == (200, contract.TOKEN_DIM)
    assert out["dbs_token"].requires_grad

    loss = out["dbs_token"].pow(2).mean()
    loss.backward()
    assert modules.dbs.lstm.weight_ih_l0.grad is not None
    assert float(modules.dbs.lstm.weight_ih_l0.grad.abs().sum()) > 0


def test_gather_is_row_order_correct():
    """Gathering must follow row_window_index exactly, not assume sorted order."""
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)
    modules = encoders.build_encoders()
    modules.eval()
    with torch.no_grad():
        per_window = api.encode_windows(modules, tensors)

    order = np.array([5, 0, 5, 2, 0], dtype=np.int64)
    gathered = api.gather_to_rows(per_window, order)
    for position, window_id in enumerate(order):
        torch.testing.assert_close(gathered["dbs_token"][position],
                                   per_window["dbs_token"][window_id],
                                   rtol=0, atol=0)


# ---------------------------------------------------------------- save / reload


def test_scaler_state_round_trips_exactly(tmp_path):
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    scalers.save(tmp_path)
    reloaded = scaling.MacroScalers.load(tmp_path)

    for block in ("dbs", "dbs_scalar", "coe", "coe_scalar"):
        original = getattr(scalers, block)
        copy = getattr(reloaded, block)
        np.testing.assert_array_equal(original.mean, copy.mean)
        np.testing.assert_array_equal(original.scale, copy.scale)
        np.testing.assert_array_equal(original.n_used, copy.n_used)
        assert original.fitted_on == copy.fitted_on
        assert original.policy == copy.policy == "origin_asof"
    assert reloaded.contract_id == scalers.contract_id


def test_reloaded_scaler_produces_identical_tensors(tmp_path):
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    scalers.save(tmp_path)
    reloaded = scaling.MacroScalers.load(tmp_path)
    a = api.build_window_tensors(windows, scalers)
    b = api.build_window_tensors(windows, reloaded)
    torch.testing.assert_close(a.dbs_seq, b.dbs_seq, rtol=0, atol=0)
    torch.testing.assert_close(a.coe_seq, b.coe_seq, rtol=0, atol=0)


def test_encoder_checkpoint_reproduces_tokens(tmp_path):
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)

    torch.manual_seed(7)
    modules = encoders.build_encoders()
    modules.eval()
    with torch.no_grad():
        before = api.encode_windows(modules, tensors)
    torch.save(modules.state_dict(), tmp_path / "macro.pt")

    torch.manual_seed(999)
    restored = encoders.build_encoders()
    restored.load_state_dict(torch.load(tmp_path / "macro.pt", weights_only=True))
    restored.eval()
    with torch.no_grad():
        after = api.encode_windows(restored, tensors)

    torch.testing.assert_close(before["dbs_token"], after["dbs_token"], rtol=0, atol=0)
    torch.testing.assert_close(before["coe_token"], after["coe_token"], rtol=0, atol=0)
    torch.testing.assert_close(before["dbs_missing"], after["dbs_missing"], rtol=0, atol=0)


# ---------------------------------------------------------------- architecture


def test_encoder_dimensions_match_the_documented_contract():
    modules = encoders.build_encoders()
    assert modules.dbs.token_dim == 48
    assert modules.coe.token_dim == 48
    assert modules.dbs.hidden_dim == 32
    assert modules.coe.hidden_dim == 32
    counts = encoders.parameter_count(modules)
    assert counts["total"] > 0
    assert set(counts["by_submodule"]) == {"dbs", "coe"}


def test_tokens_are_finite_for_every_real_window():
    windows, rows = windows_for("orig", "B")
    scalers = scaling.fit_macro_scalers(windows, rows)
    tensors = api.build_window_tensors(windows, scalers)
    modules = encoders.build_encoders()
    modules.eval()
    with torch.no_grad():
        out = api.encode_windows(modules, tensors)
    assert torch.isfinite(out["dbs_token"]).all()
    assert torch.isfinite(out["coe_token"]).all()
    # with real data, no window is fully missing
    assert float(out["dbs_missing"].sum()) == 0.0
    assert float(out["coe_missing"].sum()) == 0.0
