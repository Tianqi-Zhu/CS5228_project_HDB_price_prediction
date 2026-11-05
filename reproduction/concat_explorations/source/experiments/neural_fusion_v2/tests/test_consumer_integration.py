"""Consumer integration tests on a producer-shaped fixture.

These exercise the *real* layout and the real population roles:

* loading A/fit, A/holdout, A/validation, B/fit, B/validation from the producer's files
* boundary checks: fitting populations never contain post-cutoff labels; validation is one
  frozen origin; A/holdout does not overlap A/fit's target months
* the deviation channel: a holdout that is not evaluated from the single inner cutoff is
  recorded as a deviation, not silently accepted
* per-stage scaler fitting: A state from A/fit, B state from B/fit, transform never fits
* anchor bookkeeping: kept in SGD, converted to kSGD exactly once
* unit handling on real-scale inputs (area ~215, anchor ~4000, counts ~1e5)
* parity of every non-sequence input across the four controls
* sequence flattening is month-major and padding stays inert
* determinism, checkpoint reload

What they do **not** prove: producer-side causality. The fixture writes the arrays
directly, so a test that edits a label here cannot show that the producer's history
windows stop at their origins — only that the consumer never reads labels as inputs.
Proving producer causality requires the producer's own tamper test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE.parent))
sys.path.insert(0, str(PACKAGE / "tests"))

from neural_fusion_v2 import contract as C  # noqa: E402
from neural_fusion_v2.adapter import AdapterError, build_tensors, flatten_sequence  # noqa: E402
from neural_fusion_v2.models import Shapes, build_control  # noqa: E402
from neural_fusion_v2.preprocessing import ScalerState  # noqa: E402
from neural_fusion_v2.settings import CAPACITY  # noqa: E402
from neural_fusion_v2.training import (FitConfig, build, fit_early_stopping,  # noqa: E402
                                       fit_fixed_epochs, metrics, predict, set_seed)
from producer_fixture import write_producer_fixture  # noqa: E402


@pytest.fixture(scope="module")
def contract_dir(tmp_path_factory):
    return write_producer_fixture(tmp_path_factory.mktemp("contract"))


@pytest.fixture(scope="module")
def loaded(contract_dir):
    manifest = C.load_manifest(contract_dir)
    encoding_a = C.load_encoding(contract_dir, manifest, "orig", "A")
    encoding_b = C.load_encoding(contract_dir, manifest, "orig", "B")
    pops = {
        "a_fit": C.load_population(contract_dir, manifest, "orig", "A", "fit"),
        "a_hold": C.load_population(contract_dir, manifest, "orig", "A", "holdout"),
        "b_fit": C.load_population(contract_dir, manifest, "orig", "B", "fit"),
        "b_val": C.load_population(contract_dir, manifest, "orig", "B", "validation"),
    }
    report = C.validate_boundaries(manifest, "orig", pops["a_fit"], pops["a_hold"],
                                   pops["b_fit"], pops["b_val"])
    return manifest, {"A": encoding_a, "B": encoding_b}, pops, report


def shapes_for(pop, use_sequence, encoding) -> Shapes:
    a = pop.arrays
    vocab = encoding.get("categorical_vocabularies", {})
    cards = []
    for i, column in enumerate(a.categorical_columns):
        observed = int(a.static[:, i].max()) + 1
        cards.append(max(observed, len(vocab.get(column, [])) or observed))
    return Shapes(
        n_categorical=len(a.categorical_columns), static_dim=len(a.numeric_columns),
        history_dim=len(a.history_order), context_dim=len(a.context_order) - 1,
        cardinalities=tuple(cards),
        sequence_channels=a.n_channels if use_sequence else 0,
        sequence_length=a.n_months * a.n_levels if use_sequence else 0)


# ------------------------------------------------------------- loading & boundaries
def test_populations_load_with_expected_roles(loaded):
    _, _, pops, report = loaded
    assert len(pops["a_fit"]) > 0 and len(pops["a_hold"]) > 0
    assert len(pops["b_fit"]) > 0 and len(pops["b_val"]) > 0
    assert report["stage_a_has_validation"] is False, "v2.1 stage A has no validation population"
    assert report["blocking_checks"] == "passed"
    assert report["deviations"] == [], "the fixture should satisfy the holdout requirement"
    assert pops["a_fit"].stage == "A" and pops["b_fit"].stage == "B"


def test_a_fit_has_no_post_cutoff_labels(loaded):
    manifest, _, pops, _ = loaded
    inner = C.split_spec(manifest, "orig")["inner_cutoff_index"]
    assert pops["a_fit"].samples["target_month_index"].max() <= inner
    assert pops["a_fit"].arrays.origin.max() <= inner


def test_a_holdout_does_not_overlap_fit_labels(loaded):
    _, _, pops, _ = loaded
    fit_max = pops["a_fit"].samples["target_month_index"].max()
    assert pops["a_hold"].samples["target_month_index"].min() > fit_max


def test_validation_uses_one_frozen_origin(loaded):
    manifest, _, pops, _ = loaded
    cutoff = C.split_spec(manifest, "orig")["cutoff_index"]
    assert pops["b_val"].arrays.origin.min() == pops["b_val"].arrays.origin.max() == cutoff


def test_holdout_not_at_inner_cutoff_is_a_recorded_deviation(tmp_path):
    """The real contract's current defect: the consumer records it, it does not crash or hide it."""
    directory = write_producer_fixture(tmp_path, contract_id="bad-holdout",
                                       holdout_single_origin=False)
    manifest = C.load_manifest(directory)
    pops = [C.load_population(directory, manifest, "orig", stage, pop)
            for stage, pop in (("A", "fit"), ("A", "holdout"),
                               ("B", "fit"), ("B", "validation"))]
    report = C.validate_boundaries(manifest, "orig", *pops)
    assert report["blocking_checks"] == "passed"
    assert report["deviations"], "a multi-origin holdout must be reported"
    requirement = report["deviations"][0]["requirement"]
    assert "single inner cutoff" in requirement
    assert len(report["a_holdout_origin"]) > 1


def test_row_alignment_between_csv_and_npz_is_enforced(tmp_path):
    directory = write_producer_fixture(tmp_path, contract_id="misaligned")
    path = directory / "samples" / "orig" / "stageA.csv"
    samples = pd.read_csv(path)
    samples.loc[[5, 6]] = samples.loc[[6, 5]].to_numpy()
    samples.to_csv(path, index=False)
    manifest = C.load_manifest(directory)
    with pytest.raises(C.ContractInvalid, match="not row-aligned"):
        C.load_population(directory, manifest, "orig", "A", "fit")


def test_post_cutoff_label_in_fitting_population_is_rejected(tmp_path):
    directory = write_producer_fixture(tmp_path, contract_id="leaky")
    path = directory / "samples" / "orig" / "stageA.csv"
    samples = pd.read_csv(path)
    idx = samples.index[samples.population == "fit"][0]
    new_horizon = int(samples.loc[idx, "horizon_months"])
    new_target, new_origin = 99, 99 - new_horizon
    samples.loc[idx, "target_month_index"] = new_target
    samples.loc[idx, "origin_month_index"] = new_origin
    samples.to_csv(path, index=False)
    # keep the npz consistent with the csv so the alignment check does not fire first
    npz_path = directory / "features" / "orig" / "stageA.npz"
    stored = dict(np.load(npz_path))
    stored["origin"] = stored["origin"].copy()
    stored["origin"][idx] = new_origin
    np.savez(npz_path, **stored)
    manifest = C.load_manifest(directory)
    pops = [C.load_population(directory, manifest, "orig", stage, pop)
            for stage, pop in (("A", "fit"), ("A", "holdout"),
                               ("B", "fit"), ("B", "validation"))]
    with pytest.raises(C.ContractInvalid, match="beyond inner cutoff"):
        C.validate_boundaries(manifest, "orig", *pops)


def test_missing_manifest_raises_not_ready(tmp_path):
    with pytest.raises(C.ContractNotReady):
        C.load_manifest(tmp_path / "absent")


# ------------------------------------------------------------ scaler states per stage
def test_stage_a_state_fits_only_a_fit_rows(loaded):
    _, _, pops, _ = loaded
    state = ScalerState.fit(pops["a_fit"].arrays, population="A/fit")
    assert state.fitted_population == "A/fit"
    assert state.fitted_rows == len(pops["a_fit"])
    expected = float(np.nanmedian(pops["a_fit"].arrays.static[:, 4]))
    assert state.static_median["FLOOR_AREA_SQM"] == pytest.approx(expected)


def test_transform_of_holdout_does_not_change_the_state(loaded):
    _, _, pops, _ = loaded
    state = ScalerState.fit(pops["a_fit"].arrays, population="A/fit")
    before = state.to_dict()
    state.transform(pops["a_hold"].arrays)
    state.transform(pops["b_val"].arrays)
    after = state.to_dict()
    assert before == after, "transform must never mutate fitted state"


def test_stage_a_and_stage_b_states_differ(loaded):
    _, _, pops, _ = loaded
    state_a = ScalerState.fit(pops["a_fit"].arrays, population="A/fit")
    state_b = ScalerState.fit(pops["b_fit"].arrays, population="B/fit")
    assert state_a.fitted_population != state_b.fitted_population
    assert state_a.fitted_rows != state_b.fitted_rows
    assert state_a.static_median != state_b.static_median


def test_state_roundtrip_preserves_transform(loaded):
    _, _, pops, _ = loaded
    state = ScalerState.fit(pops["b_fit"].arrays, population="B/fit")
    restored = ScalerState.from_dict(state.to_dict())
    a = state.transform(pops["b_val"].arrays)
    b = restored.transform(pops["b_val"].arrays)
    for key in ("static", "history", "context", "anchor_ksgd"):
        np.testing.assert_array_equal(a[key], b[key])


# ------------------------------------------------------------------ units and anchor
def test_anchor_is_kept_in_sgd_and_converted_once(loaded):
    _, _, pops, _ = loaded
    state = ScalerState.fit(pops["b_fit"].arrays, population="B/fit")
    tensors = build_tensors(pops["b_val"], state, use_sequence=False)
    raw = pops["b_val"].anchor_sgd
    np.testing.assert_allclose(tensors.anchor_sgd, raw, rtol=0, atol=1e-3)
    np.testing.assert_allclose(tensors.anchor_ksgd, raw / 1000.0, rtol=1e-6, atol=1e-6)
    rebuilt = tensors.to_sgd(tensors.target)
    np.testing.assert_allclose(rebuilt, tensors.actual_sgd, rtol=0, atol=1e-3)


def test_real_scale_inputs_become_trainable(loaded):
    _, _, pops, _ = loaded
    state = ScalerState.fit(pops["b_fit"].arrays, population="B/fit")
    tensors = build_tensors(pops["b_fit"], state, use_sequence=False)
    for name in ("static", "history", "context"):
        array = getattr(tensors, name)
        assert np.isfinite(array).all(), f"{name} still has NaN after imputation"
        assert np.abs(array).max() <= state.clip_std + 1e-3, f"{name} not clipped"
        assert np.abs(array).max() > 1.0, f"{name} suspiciously flat; scaling may be a no-op"
    # a raw count column reaching ~1e5 must not survive as 1e5
    assert np.abs(tensors.history).max() < 10.0


def test_missing_history_values_are_imputed_not_left_nan(loaded):
    _, _, pops, _ = loaded
    state = ScalerState.fit(pops["b_fit"].arrays, population="B/fit")
    raw_history = pops["b_fit"].arrays.history
    assert np.isnan(raw_history).any(), "fixture should contain NaNs to exercise this"
    tensors = build_tensors(pops["b_fit"], state, use_sequence=False)
    assert np.isfinite(tensors.history).all()


def test_anchor_column_must_exist(tmp_path):
    """A contract whose context has no anchor column must be rejected, not guessed at."""
    directory = write_producer_fixture(tmp_path, contract_id="no-anchor")
    manifest = C.load_manifest(directory)
    keep = [i for i, c in enumerate(manifest["splits"]["orig"]["feature_order"]["context"])
            if c != C.ANCHOR_SGD]
    manifest["splits"]["orig"]["feature_order"]["context"] = [
        manifest["splits"]["orig"]["feature_order"]["context"][i] for i in keep]
    npz_path = directory / "features" / "orig" / "stageA.npz"
    stored = dict(np.load(npz_path))
    stored["context"] = stored["context"][:, keep]
    np.savez(npz_path, **stored)
    with pytest.raises(C.ContractInvalid, match="anchor_rent_sgd"):
        C.load_stage_arrays(directory, manifest, "orig", "A")


# -------------------------------------------------------------------------- parity
@pytest.mark.parametrize("control", ["mlp", "static_transformer", "lstm_concat", "lstm_fusion"])
def test_every_control_forward_is_finite(loaded, control):
    _, encoding, pops, _ = loaded
    use_sequence = control in ("lstm_concat", "lstm_fusion")
    state = ScalerState.fit(pops["b_fit"].arrays, population="B/fit")
    tensors = build_tensors(pops["b_fit"].subset(np.arange(96)), state, use_sequence=use_sequence)
    shapes = shapes_for(pops["b_fit"], use_sequence, encoding["B"])
    model = build_control(control, shapes)
    features = next(iter(tensors.batches(48, shuffle=False)))
    out = model(features)
    assert out.shape == (len(features["static"]),)
    assert torch.isfinite(out).all()
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_all_controls_share_requested_inputs(loaded):
    _, _, pops, _ = loaded
    state = ScalerState.fit(pops["b_fit"].arrays, population="B/fit")
    static = build_tensors(pops["b_fit"], state, use_sequence=False)
    sequence = build_tensors(pops["b_fit"], state, use_sequence=True)
    assert static.parity_signature() == sequence.parity_signature()
    for name in ("static", "history", "context", "categorical", "target", "anchor_ksgd",
                 "anchor_sgd", "actual_sgd"):
        np.testing.assert_array_equal(getattr(static, name), getattr(sequence, name))
    assert static.sequence is None and sequence.sequence is not None


def test_static_controls_never_receive_sequence(loaded):
    _, _, pops, _ = loaded
    state = ScalerState.fit(pops["b_fit"].arrays, population="B/fit")
    tensors = build_tensors(pops["b_fit"].subset(np.arange(24)), state, use_sequence=False)
    features = next(iter(tensors.batches(24, shuffle=False)))
    assert "sequence" not in features and "sequence_mask" not in features


def test_sequence_flattening_is_month_major_and_masked(loaded):
    _, _, pops, _ = loaded
    arrays = pops["b_fit"].arrays
    flat, mask = flatten_sequence(arrays.sequence, arrays.sequence_mask)
    n, levels, months, channels = arrays.sequence.shape
    assert flat.shape == (n, months * levels, channels)
    assert mask.shape == (n, months * levels)
    # step t = month*levels + level
    step = 3 * levels + 2
    np.testing.assert_array_equal(flat[:, step, :], arrays.sequence[:, 2, 3, :].reshape(n, channels))
    assert np.abs(flat[mask == 0]).max(initial=0.0) == 0.0


def test_sequence_models_ignore_padding_content(loaded):
    _, encoding, pops, _ = loaded
    state = ScalerState.fit(pops["b_fit"].arrays, population="B/fit")
    full_mask = pops["b_fit"].arrays.sequence_mask
    padded_rows = np.flatnonzero((full_mask == 0).any(axis=(1, 2)))[:96]
    if padded_rows.size == 0:
        pytest.skip("fixture has no padded rows")
    tensors = build_tensors(pops["b_fit"].subset(padded_rows), state, use_sequence=True)
    clean = next(iter(tensors.batches(len(tensors), shuffle=False)))
    noisy = dict(clean)
    noisy["sequence"] = clean["sequence"] + (1 - clean["sequence_mask"]).unsqueeze(-1) * 5.0
    for control in ("lstm_concat", "lstm_fusion"):
        set_seed(2)
        model = build_control(control, shapes_for(pops["b_fit"], True, encoding["B"]))
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(clean), model(noisy), rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------- training plumbing
def test_seed_reproduces_training(loaded):
    _, encoding, pops, _ = loaded
    state = ScalerState.fit(pops["b_fit"].arrays, population="B/fit")
    tensors = build_tensors(pops["b_fit"].subset(np.arange(128)), state, use_sequence=False)
    config = FitConfig(control="mlp", epochs=2, batch_size=64, seed=4)
    runs = []
    for _ in range(2):
        set_seed(4)
        model, _ = build("mlp", shapes_for(pops["b_fit"], False, encoding["B"]))
        fit_fixed_epochs(model, tensors, config, 2)
        runs.append(predict(model, tensors))
    np.testing.assert_allclose(runs[0], runs[1], rtol=0, atol=1e-6)


def test_checkpoint_reload_reproduces_predictions(loaded, tmp_path):
    _, encoding, pops, _ = loaded
    state = ScalerState.fit(pops["b_fit"].arrays, population="B/fit")
    tensors = build_tensors(pops["b_val"].subset(np.arange(64)), state, use_sequence=True)
    shapes = shapes_for(pops["b_fit"], True, encoding["B"])
    set_seed(9)
    model, _ = build("lstm_fusion", shapes)
    fit_fixed_epochs(model, tensors, FitConfig(control="lstm_fusion", epochs=1,
                                               batch_size=32, seed=9), 1)
    reference = predict(model, tensors)
    path = tmp_path / "ckpt.pt"
    torch.save({"control": "lstm_fusion", "state_dict": model.state_dict(),
                "shapes": shapes.__dict__, "stage_b_state": state.to_dict()}, path)
    stored = torch.load(path, map_location="cpu", weights_only=False)
    rebuilt = build_control(stored["control"], Shapes(**stored["shapes"]))
    rebuilt.load_state_dict(stored["state_dict"])
    restored = ScalerState.from_dict(stored["stage_b_state"])
    tensors2 = build_tensors(pops["b_val"].subset(np.arange(64)), restored, use_sequence=True)
    np.testing.assert_allclose(predict(rebuilt, tensors2), reference, rtol=0, atol=1e-6)


def test_early_stopping_uses_holdout_only(loaded):
    _, encoding, pops, _ = loaded
    state = ScalerState.fit(pops["a_fit"].arrays, population="A/fit")
    train = build_tensors(pops["a_fit"].subset(np.arange(320)), state, use_sequence=False)
    holdout = build_tensors(pops["a_hold"].subset(np.arange(160)), state, use_sequence=False)
    config = FitConfig(control="mlp", epochs=5, patience=2, batch_size=64, seed=1)
    set_seed(1)
    model, _ = build("mlp", shapes_for(pops["a_fit"], False, encoding["A"]))
    result = fit_early_stopping(model, train, holdout, config)
    assert 1 <= result.best_epoch <= result.epochs_run
    assert all("holdout_rmse_sgd" in row for row in result.history)
    assert np.isfinite(result.best_inner_rmse)


def test_actual_manifest_reports_code_capacity(loaded):
    _, encoding, pops, _ = loaded
    config = FitConfig(control="lstm_fusion", seed=42, epochs=7, batch_size=128)
    manifest = config.actual_manifest(shapes_for(pops["b_fit"], True, encoding["B"]))
    assert manifest["capacity"]["layers"] == CAPACITY["layers"] == 2
    assert manifest["capacity"]["d_model"] == 48
    assert manifest["capacity"]["heads"] == 4
    assert manifest["capacity"]["lstm_hidden"] == 32
    assert manifest["batch_size"] == 128 and manifest["max_epochs"] == 7
    assert manifest["threads"] == 2
    assert manifest["early_stopping"].startswith("A/fit")
    assert manifest["refit"].startswith("B/fit")


def test_metrics_are_signed_sgd():
    out = metrics(np.array([2000.0, 3000.0]), np.array([2100.0, 2900.0]))
    assert out["bias_sgd"] == pytest.approx(0.0)
    assert out["rmse_sgd"] == pytest.approx(100.0)
    assert out["n"] == 2
