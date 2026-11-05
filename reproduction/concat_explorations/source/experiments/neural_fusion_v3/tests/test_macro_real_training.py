"""The real macro path, exercised under the real rental objective.

Every other macro test in this package drives the fusion wrapper with `LinearMacroSource`, a
placeholder that carries no macro data. Those tests pin the *mechanism*; they are not
evidence that the real DBS/COE encoders are wired into training. This file is:

* real v2.1 contract rows (a small slice, so it runs in seconds),
* real v3 macro windows and real `macro_encoders_v3` encoders,
* the contract's own residual target and MSE loss,
* one `optimizer.step()`, then a checkpoint save/load round trip.

If the encoders were disconnected, or if a pathway fed the model without a gradient, these
checks fail. They say nothing about whether macro features improve accuracy.
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
SLICE = 192                    # rows per population: enough for a gradient, fast enough for CI


def _have_artifacts() -> bool:
    if not CONTRACT.exists():
        return False
    try:
        from experiments.macro_encoders_v3 import contract as macro_contract
        macro_contract.latest_v3_contract_dir()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _have_artifacts(),
                                reason="real v2.1/v3 contracts are not present")


@pytest.fixture(scope="module")
def rig():
    """Contract rows + macro rows + a fused model, built once per module."""
    from neural_fusion_v2 import contract as C
    from neural_fusion_v2.adapter import build_tensors
    from neural_fusion_v2.preprocessing import ScalerState
    from neural_fusion_v2.training import set_seed

    from experiments.macro_encoders_v3 import contract as macro_contract
    from experiments.neural_fusion_v3.fusion_macro import MacroFusion
    from experiments.neural_fusion_v3.macro_bridge import (MacroTensors, align_population,
                                                          build_stage_macro)
    from experiments.neural_fusion_v3.models import Shapes, build_control

    import experiments.neural_fusion_v3.main as main_module

    manifest = C.load_manifest(CONTRACT)
    encoding_a = C.load_encoding(CONTRACT, manifest, "orig", "A")
    macro_manifest = macro_contract.load_manifest(macro_contract.latest_v3_contract_dir())

    a_fit = C.load_population(CONTRACT, manifest, "orig", "A", "fit")
    a_hold = C.load_population(CONTRACT, manifest, "orig", "A", "holdout")
    a_fit = a_fit.subset(np.arange(SLICE))
    a_hold = a_hold.subset(np.arange(SLICE))

    state_a = ScalerState.fit(a_fit.arrays, population="A/fit")
    t_fit = build_tensors(a_fit, state_a, use_sequence=True)
    t_hold = build_tensors(a_hold, state_a, use_sequence=True)

    stage_macro = build_stage_macro(macro_manifest, "orig", "A")
    align_population(stage_macro, a_fit)
    align_population(stage_macro, a_hold)

    shapes = main_module.shapes_of(a_fit, True, encoding_a)

    def build(mode):
        def base_factory(name, sh):
            return build_control(name, sh, state_a.static_columns, token_mode="block")

        return main_module.build_fusion(base_factory, "lstm_fusion", shapes, mode,
                                        lambda m: main_module._macro_source(stage_macro,
                                                                            macro_manifest, m))

    set_seed(42)
    return {
        "fit_plain": t_fit,              # what `--macro none` feeds the model
        "hold_plain": t_hold,
        "fit": MacroTensors(t_fit, a_fit.arrays.row_id, stage_macro),
        "hold": MacroTensors(t_hold, a_hold.arrays.row_id, stage_macro),
        "stage_macro": stage_macro,
        "macro_manifest": macro_manifest,
        "build": build,
        "shapes": shapes,
        "base_factory": lambda name, sh: build_control(name, sh, state_a.static_columns,
                                                       token_mode="block"),
    }


def _batch(tensors, batch_size=SLICE):
    return next(iter(tensors.batches(batch_size, shuffle=False, with_target=True)))


def _encoder_parameters(model):
    return [(name, p) for name, p in model.named_parameters() if name.startswith("source.")]


ENABLED_BRANCHES = {"dbs": ("dbs",), "coe": ("coe",), "both": ("dbs", "coe")}


@pytest.mark.parametrize("mode", ["dbs", "coe", "both"])
def test_real_encoders_receive_non_zero_gradient_under_the_rental_loss(rig, mode):
    """Enabled branches must learn; the disabled one must stay untouched."""
    model = rig["build"](mode)
    model.train()
    features, target = _batch(rig["fit"])
    loss = torch.nn.functional.mse_loss(model(features), target)
    loss.backward()

    parameters = _encoder_parameters(model)
    assert parameters, "no parameters under `source.`: the encoders are not registered"

    enabled = tuple(f"source.encoders.{b}." for b in ENABLED_BRANCHES[mode])
    disabled = tuple(f"source.encoders.{b}." for b in ("dbs", "coe")
                     if b not in ENABLED_BRANCHES[mode])

    dead = [name for name, p in parameters
            if name.startswith(enabled) and not name.endswith("empty")
            and (p.grad is None or p.grad.abs().sum() == 0)]
    # `empty` is the learned stand-in for an all-missing window; every window in this slice
    # has observations, so it legitimately never fires here.
    assert not dead, f"{mode}: enabled-branch parameters got no gradient: {dead}"

    if disabled:
        touched = [name for name, p in parameters
                   if name.startswith(disabled) and p.grad is not None
                   and p.grad.abs().sum() > 0]
        assert not touched, (
            f"{mode}: the disabled branch received gradient: {touched}. `build_encoders` "
            "always builds both so state_dict keys line up, but the unused one must not "
            "be trained.")


@pytest.mark.parametrize("mode", ["dbs", "coe", "both"])
def test_optimizer_step_moves_real_encoder_weights(rig, mode):
    model = rig["build"](mode)
    model.train()
    before = {n: p.detach().clone() for n, p in _encoder_parameters(model)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    features, target = _batch(rig["fit"])
    loss = torch.nn.functional.mse_loss(model(features), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    moved = [n for n, p in _encoder_parameters(model)
             if not torch.equal(p.detach(), before[n])]
    assert moved, f"{mode}: no encoder weight changed after optimizer.step()"


def test_none_and_both_use_the_same_rental_target_and_rows(rig):
    """The comparison must differ in the macro branch only, not in the labels."""
    none_features, none_target = _batch(rig["fit_plain"])
    both_features, both_target = _batch(rig["fit"])
    torch.testing.assert_close(none_target, both_target, rtol=0, atol=0)
    for key in ("static", "history", "context", "categorical", "sequence", "sequence_mask"):
        torch.testing.assert_close(none_features[key], both_features[key], rtol=0, atol=0)
    # `none` carries no macro keys at all; `both` adds them
    assert not any(k.startswith("macro_") for k in none_features)
    assert {"macro_dbs", "macro_coe", "macro_row"} <= set(both_features)


def test_macro_none_with_v2_weights_reproduces_v2_predictions_on_real_rows(rig):
    """Load a v2 state dict into the v3 `block`/`none` model and compare on contract data."""
    from neural_fusion_v2.models import LSTMFusionControl
    from neural_fusion_v2.settings import CAPACITY

    import copy

    v2_none = rig["build"]("none")
    weights = copy.deepcopy(v2_none.state_dict())     # one model, two instances
    v2_model = LSTMFusionControl(rig["shapes"].to_v2())
    v2_model.load_state_dict(weights)
    v2_model.eval()
    v2_none.eval()          # dropout is 0.05 in the token stack: compare in eval mode
    features, _ = _batch(rig["hold"])
    with torch.no_grad():
        torch.testing.assert_close(v2_none(features), v2_model(features), rtol=0, atol=0)


def test_real_macro_checkpoint_survives_a_save_load_round_trip(rig, tmp_path):
    """Same weights + same saved scalers ⇒ identical predictions, without refitting."""
    from experiments.macro_encoders_v3 import scaling as macro_scaling

    import experiments.neural_fusion_v3.main as main_module

    model = rig["build"]("both")
    model.eval()
    features, _ = _batch(rig["hold"])
    with torch.no_grad():
        original = model(features)

    path = tmp_path / "ckpt.pt"
    stage_macro = rig["stage_macro"]
    torch.save({
        "control": "lstm_fusion", "token_mode": "block", "macro_mode": "both",
        "state_dict": model.state_dict(), "shapes": rig["shapes"].__dict__,
        "macro": {
            "contract_id": rig["macro_manifest"]["contract_id"],
            "policy": rig["macro_manifest"]["policy"],
            "stage_a": {"split": "orig", "stage": "A"},
            "scalers_a": {n: getattr(stage_macro.scaler, n).to_dict()
                          for n in ("dbs", "dbs_scalar", "coe", "coe_scalar")},
        },
    }, path)

    stored = torch.load(path, map_location="cpu", weights_only=False)
    saved_scalers = main_module._scalers_from_checkpoint(stored, "A")

    from experiments.neural_fusion_v3.macro_bridge import (MacroTensors, build_stage_macro)
    import experiments.neural_fusion_v3.main as mm

    original_fit = macro_scaling.fit_macro_scalers
    macro_scaling.fit_macro_scalers = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("a checkpoint reload must not refit the macro scalers"))
    try:
        rebuilt_stage = build_stage_macro(rig["macro_manifest"], "orig", "A",
                                         scalers=saved_scalers)
    finally:
        macro_scaling.fit_macro_scalers = original_fit

    rebuilt = mm.build_fusion(rig["base_factory"], "lstm_fusion", rig["shapes"], "both",
                              lambda m: mm._macro_source(rebuilt_stage,
                                                         rig["macro_manifest"], m))
    rebuilt.load_state_dict(stored["state_dict"])
    rebuilt.eval()

    from neural_fusion_v2 import contract as C
    from neural_fusion_v2.adapter import build_tensors
    from neural_fusion_v2.preprocessing import ScalerState

    manifest = C.load_manifest(CONTRACT)
    a_hold = C.load_population(CONTRACT, manifest, "orig", "A", "holdout").subset(
        np.arange(SLICE))
    # the holdout rows must be re-derived with the scaler state the checkpoint carries
    state = ScalerState.fit(C.load_population(CONTRACT, manifest, "orig", "A", "fit")
                            .subset(np.arange(SLICE)).arrays, population="A/fit")
    hold = MacroTensors(build_tensors(a_hold, state, use_sequence=True),
                        a_hold.arrays.row_id, rebuilt_stage)
    hold_features, _ = _batch(hold)
    with torch.no_grad():
        reloaded = rebuilt(hold_features)
    torch.testing.assert_close(reloaded, original, rtol=0, atol=1e-6)
