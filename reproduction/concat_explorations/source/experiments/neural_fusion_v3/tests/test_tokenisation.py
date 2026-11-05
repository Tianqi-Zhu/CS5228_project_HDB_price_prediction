"""Per-feature token semantics: named indices, sentinels, permutation, width.

Column-identity failures keep every tensor shape valid, so these tests check *which*
columns reach which token using unique sentinel values, and check that permuting the input
moves tokens with their columns rather than their positions.

Nothing here re-tests the block reference: `test_block_is_v2.py` proves block mode is v2
itself by loading a v2 state dict and comparing predictions.
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

from neural_fusion_v2.settings import CAPACITY  # noqa: E402

from neural_fusion_v3.tokens import (PerFeatureTokens, TokenisationError,  # noqa: E402
                                     feature_token_fingerprint, resolve_feature_index)

FEATURES = ["FLOOR_AREA_SQM", "LATITUDE", "LONGITUDE", "NEAREST_OPEN_MRT_KM",
            "MAX_FLOOR", "DISTANCE_TO_CBD_KM"]


# ------------------------------------------------------------------ name resolution
def test_index_map_resolves_by_name_not_position():
    shuffled = ["LATITUDE", "NEAREST_OPEN_MRT_KM", "FLOOR_AREA_SQM", "DISTANCE_TO_CBD_KM",
                "LONGITUDE", "MAX_FLOOR"]
    index = resolve_feature_index(shuffled, FEATURES)
    assert [shuffled[i] for i in index] == FEATURES
    assert index != list(range(len(FEATURES))), "positional order must differ, else this is blind"


def test_missing_duplicate_and_unnamed_columns_are_rejected():
    with pytest.raises(TokenisationError, match="absent from the contract"):
        resolve_feature_index(FEATURES[:-1], FEATURES)
    with pytest.raises(TokenisationError, match="not covered by the feature list"):
        resolve_feature_index(FEATURES + ["EXTRA_FEATURE"], FEATURES)
    with pytest.raises(TokenisationError, match="duplicate"):
        resolve_feature_index(["A", "A", "B"], ["A", "B"])


# --------------------------------------------------------------- token semantics
def test_token_width_is_48_not_8():
    tokens = PerFeatureTokens(FEATURES, out_dim=CAPACITY["d_model"])
    out = tokens(torch.randn(5, len(FEATURES)))
    assert out.shape == (5, len(FEATURES), 48)
    assert tokens.describe()["bottleneck"] is None
    assert tokens.describe()["token_width"] == 48


def test_each_token_reads_its_own_column_by_sentinel():
    tokens = PerFeatureTokens(FEATURES, out_dim=8)
    with torch.no_grad():
        tokens.weight.fill_(1.0)
        tokens.bias.zero_()
        tokens.scale.fill_(1.0)
        tokens.identity.zero_()
    sentinel = torch.tensor([[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]])
    out = tokens(sentinel)[0]
    recovered = torch.tanh(torch.tensor([11.0, 22.0, 33.0, 44.0, 55.0, 66.0])).tolist()
    np.testing.assert_allclose(out[:, 0].tolist(), recovered, rtol=1e-6)


def test_identity_makes_equal_values_distinguishable():
    tokens = PerFeatureTokens(FEATURES, out_dim=8)
    with torch.no_grad():
        tokens.weight.fill_(1.0)
        tokens.bias.zero_()
        tokens.scale.fill_(1.0)
    identical = torch.ones(1, len(FEATURES))
    out = tokens(identical)[0]
    assert not torch.allclose(out[0], out[1])
    assert torch.unique(out, dim=0).shape[0] == len(FEATURES)


def test_tokens_carry_no_shared_linear_bottleneck():
    tokens = PerFeatureTokens(FEATURES, out_dim=48)
    assert not any(isinstance(m, torch.nn.Linear) for m in tokens.modules())
    assert tokens.weight.shape == (len(FEATURES),)
    assert tokens.bias.shape == (len(FEATURES),)
    assert tokens.scale.shape == (len(FEATURES), 48)


# ------------------------------------------------------------- permutation behaviour
def test_permuting_columns_with_their_name_map_preserves_each_token():
    base = torch.arange(1.0, len(FEATURES) + 1.0).unsqueeze(0)
    order = [3, 0, 5, 1, 4, 2]
    permuted_names = [FEATURES[i] for i in order]

    # each module's weight follows *its own* declared name order, so a token keeps the
    # same value only if the name-to-column mapping travelled with the permutation
    a = PerFeatureTokens(FEATURES, out_dim=8)
    b = PerFeatureTokens(permuted_names, out_dim=8)
    weight_by_name = {name: float(i + 1) for i, name in enumerate(FEATURES)}
    with torch.no_grad():
        a.weight.copy_(torch.tensor([weight_by_name[n] for n in FEATURES]))
        b.weight.copy_(torch.tensor([weight_by_name[n] for n in permuted_names]))
        for module in (a, b):
            module.bias.zero_()
            module.scale.fill_(1.0)
            module.identity.zero_()

    out_a = a(base)[0][:, 0]
    out_b = b(base[:, order])[0][:, 0]
    by_name_a = dict(zip(FEATURES, out_a.tolist()))
    by_name_b = dict(zip(permuted_names, out_b.tolist()))
    for name in FEATURES:
        assert by_name_a[name] == pytest.approx(by_name_b[name]), name


def test_permuting_without_remapping_changes_tokens():
    tokens = PerFeatureTokens(FEATURES, out_dim=8)
    with torch.no_grad():
        tokens.weight.copy_(torch.arange(1.0, len(FEATURES) + 1.0))
        tokens.bias.zero_()
        tokens.scale.fill_(1.0)
        tokens.identity.zero_()
    base = torch.arange(1.0, len(FEATURES) + 1.0).unsqueeze(0)
    order = [3, 0, 5, 1, 4, 2]
    assert not torch.allclose(tokens(base), tokens(base[:, order]))


def test_fingerprint_records_names_and_indices():
    columns = ["LATITUDE", "FLOOR_AREA_SQM", "MAX_FLOOR", "NEAREST_OPEN_MRT_KM",
               "LONGITUDE", "DISTANCE_TO_CBD_KM"]
    index = resolve_feature_index(columns, FEATURES)
    module = PerFeatureTokens([columns[i] for i in index], out_dim=48)
    module.feature_indices = index
    payload = feature_token_fingerprint(module)
    assert payload["feature_names"] == FEATURES
    assert payload["feature_indices"] == [columns.index(n) for n in FEATURES]
    assert payload["mode"] == "per_feature"


def test_wrong_numeric_width_is_rejected():
    tokens = PerFeatureTokens(FEATURES, out_dim=48)
    with pytest.raises(TokenisationError, match="numeric width"):
        tokens(torch.randn(3, len(FEATURES) - 1))
