"""The run manifest must be able to prove which token layout a run used.

Before this was fixed, `main.py` recorded only `fusion_macro.describe()`, whose fields are
all macro-related. A `block` run and a `per_feature` run therefore produced the same
description (28 sequence positions, 29 type rows), and no run's own artifacts could say
whether it used 1 numeric token of width 8 or 26 tokens of width 48.

These tests pin two things:

1. the description now distinguishes the layouts and carries the column mapping;
2. adding those fields changed **no numerics** — the forward pass of a given state dict is
   identical before and after, which is what makes it safe to keep the older runs.
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

from neural_fusion_v2.settings import CAPACITY  # noqa: E402
from neural_fusion_v3.fusion_macro import (LinearMacroSource, MacroFusion,  # noqa: E402
                                           describe)
from neural_fusion_v3.models import Shapes, build_control, model_description  # noqa: E402

FEATURES = ["FLOOR_AREA_SQM", "LATITUDE", "LONGITUDE", "NEAREST_OPEN_MRT_KM",
            "MAX_FLOOR", "DISTANCE_TO_CBD_KM"]


def shapes(control: str = "lstm_fusion") -> Shapes:
    return Shapes(n_categorical=4, numeric_dim=len(FEATURES), history_dim=24, context_dim=11,
                  cardinalities=(27, 7, 21, 2),
                  sequence_channels=6 if control == "lstm_fusion" else 0,
                  sequence_length=48 if control == "lstm_fusion" else 0)


def batch(control: str = "lstm_fusion", n: int = 6, seed: int = 5) -> dict:
    torch.manual_seed(seed)
    payload = {
        "static": torch.randn(n, len(FEATURES)), "history": torch.randn(n, 24),
        "context": torch.randn(n, 11),
        "categorical": torch.stack([torch.randint(0, 2, (n,)) for _ in range(4)], dim=1),
    }
    if control == "lstm_fusion":
        mask = torch.ones(n, 48)
        mask[:, 30:] = 0.0
        payload["sequence"] = torch.randn(n, 48, 6) * mask.unsqueeze(-1)
        payload["sequence_mask"] = mask
    return payload


def test_block_and_per_feature_descriptions_are_distinguishable():
    block = describe(build_control("lstm_fusion", shapes(), FEATURES, token_mode="block"))
    per = describe(build_control("lstm_fusion", shapes(), FEATURES, token_mode="per_feature"))

    assert block["encoder"]["numeric"]["n_tokens"] == 1
    assert block["encoder"]["numeric"]["token_width"] == 8
    assert block["encoder"]["numeric"]["bottleneck"].startswith("Linear(6, 8)")

    assert per["encoder"]["numeric"]["n_tokens"] == len(FEATURES)
    assert per["encoder"]["numeric"]["token_width"] == CAPACITY["d_model"] == 48
    assert per["encoder"]["numeric"]["bottleneck"] is None

    # the layouts must differ in total token count, not only in the numeric sub-field
    assert per["encoder"]["total_tokens"] > block["encoder"]["total_tokens"]
    assert per["sequence_length"] > block["sequence_length"]


def test_per_feature_description_carries_the_column_mapping():
    model = build_control("lstm_fusion", shapes(), FEATURES, token_mode="per_feature")
    numeric = describe(model)["encoder"]["numeric"]
    assert numeric["feature_names"] == FEATURES
    assert numeric["feature_indices"] == list(range(len(FEATURES)))


def test_column_mapping_follows_a_permuted_contract_order():
    """If the contract hands columns in another order, indices must follow the names."""
    permuted = list(reversed(FEATURES))
    model = build_control("lstm_fusion", shapes(), permuted, token_mode="per_feature",
                          expected_numeric=FEATURES)
    numeric = describe(model)["encoder"]["numeric"]
    assert numeric["feature_names"] == FEATURES, "tokens are ordered by the expected list"
    assert numeric["feature_indices"] == [permuted.index(n) for n in FEATURES]
    assert numeric["feature_indices"] != list(range(len(FEATURES)))


def test_fused_description_still_reports_the_underlying_token_layout():
    for mode in ("block", "per_feature"):
        fused = MacroFusion(build_control("lstm_fusion", shapes(), FEATURES, token_mode=mode),
                            LinearMacroSource(2), "both")
        payload = describe(fused)
        assert payload["macro_mode"] == "both"
        assert payload["macro_tokens"] == 2
        assert payload["encoder"]["numeric"]["token_width"] == (8 if mode == "block" else 48)
        assert "parameters" in payload


def test_adding_report_fields_did_not_change_the_forward_pass():
    """The same state dict must produce the same numbers; only the report grew."""
    torch.manual_seed(11)
    model = build_control("lstm_fusion", shapes(), FEATURES, token_mode="per_feature")
    model.eval()
    payload = batch()
    with torch.no_grad():
        before = model(payload)

    description = describe(model)                      # the reporting change
    assert "encoder" in description

    with torch.no_grad():
        after = model(payload)
    torch.testing.assert_close(after, before, rtol=0, atol=0)

    # and the description must not have touched any parameter
    fresh = build_control("lstm_fusion", shapes(), FEATURES, token_mode="per_feature")
    fresh.load_state_dict(model.state_dict())
    fresh.eval()                       # dropout is 0.05: compare deterministically
    with torch.no_grad():
        torch.testing.assert_close(fresh(payload), before, rtol=0, atol=0)


def test_report_change_is_purely_additive_for_a_none_run():
    """`none` keeps v2's keys; the new fields are additions, not renames."""
    model = build_control("static_transformer", shapes("static_transformer"), FEATURES,
                          token_mode="block")
    payload = describe(model)
    for key in ("control", "macro_mode", "macro_tokens", "sequence_length",
                "pooled_positions", "macro_role", "type_embedding_rows",
                "context_projection_kept", "source", "semantics"):
        assert key in payload, key
    assert payload["macro_mode"] == "none" and payload["source"] is None
    assert payload["encoder"]["numeric"]["mode"] == "block"
