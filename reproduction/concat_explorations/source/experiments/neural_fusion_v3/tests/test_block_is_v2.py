"""Block mode must be v2, verifiably.

The previous draft of v3 silently changed the context pathway as well as the numeric block,
which made `block` vs `per_feature` a comparison of several modifications at once. These
tests pin the property that was missing: a v3 `block` model loaded with a v2 state dict,
given the same batch, must produce the same numbers as the v2 model itself.
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

from neural_fusion_v3.models import Shapes, build_control, model_description  # noqa: E402
from neural_fusion_v3.tokens import TokenisationError  # noqa: E402

FEATURES = ["FLOOR_AREA_SQM", "LATITUDE", "LONGITUDE", "NEAREST_OPEN_MRT_KM",
            "MAX_FLOOR", "DISTANCE_TO_CBD_KM"]
CONTROLS = ["mlp", "static_transformer", "lstm_concat", "lstm_fusion"]


def shapes_for(control: str) -> Shapes:
    return Shapes(
        n_categorical=4, numeric_dim=len(FEATURES), history_dim=24, context_dim=11,
        cardinalities=(27, 7, 21, 2),
        sequence_channels=6 if control in ("lstm_concat", "lstm_fusion") else 0,
        sequence_length=48 if control in ("lstm_concat", "lstm_fusion") else 0)


def make_batch(control: str, seed: int = 0, n: int = 12) -> dict:
    torch.manual_seed(seed)
    batch = {
        "static": torch.randn(n, len(FEATURES)),
        "history": torch.randn(n, 24),
        "context": torch.randn(n, 11),
        "categorical": torch.stack([torch.randint(0, 2, (n,)) for _ in range(4)], dim=1),
    }
    if control in ("lstm_concat", "lstm_fusion"):
        mask = torch.ones(n, 48)
        mask[:, 36:] = 0.0
        batch["sequence"] = torch.randn(n, 48, 6) * mask.unsqueeze(-1)
        batch["sequence_mask"] = mask
    return batch


@pytest.mark.parametrize("control", CONTROLS)
def test_block_mode_is_v2_with_the_same_state_dict(control):
    v2_shapes = shapes_for(control).to_v2()
    torch.manual_seed(7)
    reference = v2_models.CONTROLS[control](v2_shapes)
    reference.eval()

    block = build_control(control, shapes_for(control), FEATURES, token_mode="block")
    block.load_state_dict(reference.state_dict())          # must match key-for-key
    block.eval()

    batch = make_batch(control, seed=3)
    with torch.no_grad():
        expected = reference(batch)
        actual = block(batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("control", CONTROLS)
def test_block_mode_has_identical_parameter_names_and_count(control):
    v2_shapes = shapes_for(control).to_v2()
    reference = v2_models.CONTROLS[control](v2_shapes)
    block = build_control(control, shapes_for(control), FEATURES, token_mode="block")
    v2_keys = list(reference.state_dict().keys())
    block_keys = list(block.state_dict().keys())
    assert block_keys == v2_keys, "block mode must expose v2's exact parameter names"
    v2_params = sum(p.numel() for p in reference.parameters())
    assert sum(p.numel() for p in block.parameters()) == v2_params


@pytest.mark.parametrize("control", CONTROLS)
def test_per_feature_keeps_every_v2_pathway_except_the_numeric_block(control):
    """Every v2 parameter key must survive except `static_proj`."""
    v2_shapes = shapes_for(control).to_v2()
    reference = v2_models.CONTROLS[control](v2_shapes)
    per = build_control(control, shapes_for(control), FEATURES, token_mode="per_feature")

    v2_keys = {k for k in reference.state_dict() if "static_proj" not in k}
    per_keys = {k for k in per.state_dict() if "feature_tokens" not in k}
    assert per_keys == v2_keys, (
        f"{control}: per_feature changed parameters beyond the numeric block: "
        f"missing={sorted(v2_keys - per_keys)[:5]} extra={sorted(per_keys - v2_keys)[:5]}")
    # the context projection must still exist and be used
    encoder = per.encoder if hasattr(per, "encoder") else per.block.encoder
    assert encoder.context_proj is not None
    assert encoder.static_proj is None


@pytest.mark.parametrize("control", CONTROLS)
def test_per_feature_tokens_are_width_48_not_8(control):
    per = build_control(control, shapes_for(control), FEATURES, token_mode="per_feature")
    description = model_description(per)
    numeric = description["encoder"]["numeric"]
    assert numeric["n_tokens"] == len(FEATURES)
    assert numeric["token_width"] == CAPACITY["d_model"] == 48
    assert numeric["bottleneck"] is None
    assert description["encoder"]["uses_context_projection"] is True


def test_block_description_reports_the_8_wide_bottleneck():
    block = build_control("mlp", shapes_for("mlp"), FEATURES, token_mode="block")
    numeric = model_description(block)["encoder"]["numeric"]
    assert numeric["n_tokens"] == 1
    assert numeric["token_width"] == 8
    assert numeric["bottleneck"] == f"Linear({len(FEATURES)}, 8)"


@pytest.mark.parametrize("control", CONTROLS)
def test_per_feature_forward_is_finite_and_backpropagates(control):
    per = build_control(control, shapes_for(control), FEATURES, token_mode="per_feature")
    batch = make_batch(control)
    out = per(batch)
    assert out.shape == (12,)
    assert torch.isfinite(out).all()
    out.sum().backward()
    grads = [p.grad for p in per.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_per_feature_rejects_wrong_numeric_width():
    per = build_control("mlp", shapes_for("mlp"), FEATURES, token_mode="per_feature")
    batch = make_batch("mlp")
    batch["static"] = torch.randn(12, len(FEATURES) - 1)
    with pytest.raises(TokenisationError, match="numeric width"):
        per(batch)


def test_block_and_per_feature_differ_only_where_intended():
    block = model_description(build_control("lstm_fusion", shapes_for("lstm_fusion"),
                                            FEATURES, token_mode="block"))
    per = model_description(build_control("lstm_fusion", shapes_for("lstm_fusion"),
                                          FEATURES, token_mode="per_feature"))
    assert block["encoder"]["categorical_tokens"] == per["encoder"]["categorical_tokens"]
    assert block["encoder"]["history_tokens"] == per["encoder"]["history_tokens"] == 1
    assert block["encoder"]["context_tokens"] == per["encoder"]["context_tokens"] == 1
    assert block["encoder"]["uses_context_projection"] == per["encoder"]["uses_context_projection"]
    assert block["capacity"] == per["capacity"] == dict(CAPACITY)
    assert block["encoder"]["numeric"]["n_tokens"] == 1
    assert per["encoder"]["numeric"]["n_tokens"] == len(FEATURES)
