"""Macro fusion must not disturb `none`, and must genuinely use enabled tokens.

Three properties, each of which a plausible-looking implementation can violate:

1. ``macro=none`` **is** v2: same object, same state-dict keys, identical predictions.
2. Enabled encoders receive non-zero gradients and their tokens really reach the pooled
   representation through attention — not merely concatenated and ignored.
3. ``none`` changes neither the token count nor the pooled positions.

These use `LinearMacroSource`, a placeholder that carries no macro data. They pin the
*mechanism*. They say nothing about whether real DBS/COE features help, which needs the
producer's real artifacts and a real run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PACKAGE.parent))

from neural_fusion_v2 import models as v2_models  # noqa: E402
from neural_fusion_v2.settings import CAPACITY  # noqa: E402

from neural_fusion_v3.fusion_macro import (ATTENTION_CONTROLS, LinearMacroSource,  # noqa: E402
                                           MacroFusion, MacroFusionUnsupported,
                                           build_fusion, describe)
from neural_fusion_v3.models import Shapes, build_control  # noqa: E402

FEATURES = ["FLOOR_AREA_SQM", "LATITUDE", "LONGITUDE", "NEAREST_OPEN_MRT_KM",
            "MAX_FLOOR", "DISTANCE_TO_CBD_KM"]


def shapes_for(control: str) -> Shapes:
    return Shapes(
        n_categorical=4, numeric_dim=len(FEATURES), history_dim=24, context_dim=11,
        cardinalities=(27, 7, 21, 2),
        sequence_channels=6 if control in ("lstm_fusion", "lstm_concat") else 0,
        sequence_length=48 if control in ("lstm_fusion", "lstm_concat") else 0)


def block_factory(name, shapes, token_mode="block"):
    return build_control(name, shapes, FEATURES, token_mode=token_mode)


def make_batch(control: str, n: int = 8, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    batch = {
        "static": torch.randn(n, len(FEATURES)),
        "history": torch.randn(n, 24),
        "context": torch.randn(n, 11),
        "categorical": torch.stack([torch.randint(0, 2, (n,)) for _ in range(4)], dim=1),
    }
    if control in ("lstm_fusion", "lstm_concat"):
        mask = torch.ones(n, 48)
        mask[:, 36:] = 0.0
        batch["sequence"] = torch.randn(n, 48, 6) * mask.unsqueeze(-1)
        batch["sequence_mask"] = mask
    return batch


# ------------------------------------------------------------------ none is v2
@pytest.mark.parametrize("control", ["static_transformer", "lstm_fusion", "mlp", "lstm_concat"])
def test_macro_none_returns_the_v2_object_itself(control):
    model = build_fusion(block_factory, control, shapes_for(control), "none")
    reference = v2_models.CONTROLS[control](shapes_for(control).to_v2())
    assert type(model) is type(reference), "macro=none must be v2's class, not a wrapper"
    assert list(model.state_dict().keys()) == list(reference.state_dict().keys())


@pytest.mark.parametrize("control", ATTENTION_CONTROLS)
def test_macro_none_predictions_equal_v2_under_the_same_weights(control):
    model = build_fusion(block_factory, control, shapes_for(control), "none")
    reference = v2_models.CONTROLS[control](shapes_for(control).to_v2())
    reference.load_state_dict(model.state_dict())
    model.eval()
    reference.eval()
    batch = make_batch(control)
    with torch.no_grad():
        torch.testing.assert_close(model(batch), reference(batch), rtol=0, atol=0)


@pytest.mark.parametrize("control", ATTENTION_CONTROLS)
def test_macro_none_keeps_token_count_and_pooled_positions(control):
    none_model = build_fusion(block_factory, control, shapes_for(control), "none")
    enabled = build_fusion(block_factory, control, shapes_for(control), "both",
                           lambda mode: LinearMacroSource(2))
    d_none, d_enabled = describe(none_model), describe(enabled)
    assert d_none["macro_tokens"] == 0
    assert d_none["pooled_positions"] == d_enabled["pooled_positions"], (
        "enabling macro must not change how many positions are pooled")
    assert d_enabled["sequence_length"] == d_none["sequence_length"] + 2
    assert d_none["type_embedding_rows"] + 2 == d_enabled["type_embedding_rows"]


# ------------------------------------------------------- enabled tokens are used
@pytest.mark.parametrize("control", ATTENTION_CONTROLS)
def test_enabled_macro_encoders_receive_non_zero_gradients(control):
    source = LinearMacroSource(2)
    model = MacroFusion(block_factory(control, shapes_for(control)), source, "both")
    model.train()
    out = model(make_batch(control))
    out.sum().backward()
    for name, parameter in source.named_parameters():
        assert parameter.grad is not None, f"{name} got no gradient at all"
        assert parameter.grad.abs().sum() > 0, f"{name} gradient is identically zero"


@pytest.mark.parametrize("control", ATTENTION_CONTROLS)
def test_an_optimizer_really_steps_the_macro_parameters(control):
    source = LinearMacroSource(2)
    model = MacroFusion(block_factory(control, shapes_for(control)), source, "both")
    model.train()
    before = {name: p.detach().clone() for name, p in source.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    out = model(make_batch(control))
    out.pow(2).mean().backward()
    optimizer.step()
    changed = [name for name, p in source.named_parameters()
               if not torch.equal(p, before[name])]
    assert changed, "no macro parameter moved after optimizer.step()"


@pytest.mark.parametrize("control", ATTENTION_CONTROLS)
def test_macro_tokens_reach_the_pooled_output_through_attention(control):
    """Perturb only the macro tokens; the pooled vector must move."""
    source = LinearMacroSource(2)
    model = MacroFusion(block_factory(control, shapes_for(control)), source, "both")
    model.eval()
    batch = make_batch(control)
    block = model.base.block
    with torch.no_grad():
        base_pooled = model(batch)
        source.bias_tokens.add_(5.0)
        perturbed = model(batch)
    assert not torch.allclose(base_pooled, perturbed), (
        "changing macro tokens left predictions untouched: they are concatenated but not read")


def test_macro_tokens_are_excluded_from_the_pooled_mean():
    """Attending to macro is required; averaging them into the pooled vector is not."""
    source = LinearMacroSource(2)
    model = MacroFusion(block_factory("static_transformer", shapes_for("static_transformer")),
                        source, "both")
    model.eval()
    block = model.base.block
    batch = make_batch("static_transformer")
    captured = {}

    real_pooled = block.pooled

    def spy(sequence, mask=None):
        captured["sequence"] = sequence.detach()
        captured["mask"] = None if mask is None else mask.detach()
        return real_pooled(sequence, mask)

    block.pooled = spy
    try:
        with torch.no_grad():
            model(batch)
    finally:
        block.pooled = real_pooled

    mask = captured["mask"]
    assert mask is not None, "a mask must be passed so macro positions are not averaged"
    n_macro = model.macro_tokens
    assert mask[:, -n_macro:].sum() == 0, "macro positions must be masked out of the mean"
    assert mask[:, :-n_macro].min() == 1, "every tabular position must stay pooled"


@pytest.mark.parametrize("control", ATTENTION_CONTROLS)
def test_type_embedding_is_extended_not_replaced(control):
    none_model = block_factory(control, shapes_for(control))
    block_before = none_model.block if hasattr(none_model, "block") else none_model
    original = block_before.type_embedding.weight.detach().clone()
    model = MacroFusion(none_model, LinearMacroSource(2), "both")
    grown = model.base.block.type_embedding if hasattr(model.base, "block") else model.base.type_embedding
    assert grown.num_embeddings == original.shape[0] + 2
    torch.testing.assert_close(grown.weight[:original.shape[0]], original, rtol=0, atol=0)


# ------------------------------------------------------------------- guardrails
@pytest.mark.parametrize("control", ["mlp", "lstm_concat"])
def test_macro_tokens_are_rejected_on_controls_without_attention(control):
    with pytest.raises(MacroFusionUnsupported, match="no attention memory"):
        build_fusion(block_factory, control, shapes_for(control), "dbs",
                     lambda mode: LinearMacroSource(1))


def test_enabled_mode_without_a_source_is_an_error_not_a_silent_no_op():
    with pytest.raises(MacroFusionUnsupported, match="needs a token source"):
        build_fusion(block_factory, "static_transformer",
                     shapes_for("static_transformer"), "dbs", None)
    with pytest.raises(MacroFusionUnsupported, match="returned None"):
        build_fusion(block_factory, "static_transformer",
                     shapes_for("static_transformer"), "dbs", lambda mode: None)


def test_unknown_macro_mode_is_rejected():
    with pytest.raises(MacroFusionUnsupported, match="unknown macro mode"):
        build_fusion(block_factory, "static_transformer",
                     shapes_for("static_transformer"), "dbs+coe", None)


def test_source_with_wrong_token_width_is_rejected():
    class WrongWidth(torch.nn.Module):
        n_tokens = 1

        def forward(self, batch):
            return torch.zeros(batch["context"].shape[0], 1, 7)

    model = MacroFusion(block_factory("static_transformer", shapes_for("static_transformer")),
                        WrongWidth(), "dbs")
    with pytest.raises(ValueError, match="must return"):
        model(make_batch("static_transformer"))


def test_describe_records_what_was_actually_built():
    model = build_fusion(block_factory, "lstm_fusion", shapes_for("lstm_fusion"), "both",
                         lambda mode: LinearMacroSource(2))
    payload = model.describe()
    assert payload["macro_mode"] == "both"
    assert payload["macro_tokens"] == 2
    assert payload["control"] == "lstm_fusion"
    assert payload["context_projection_kept"] is True
    assert "not a validated measure of economic health" in payload["semantics"]
    assert payload["sequence_length"] == payload["pooled_positions"] + 2


def test_describe_of_a_none_model_reports_no_macro_branch():
    payload = describe(build_fusion(block_factory, "static_transformer",
                                    shapes_for("static_transformer"), "none"))
    assert payload["macro_mode"] == "none"
    assert payload["macro_tokens"] == 0
    assert payload["source"] is None
