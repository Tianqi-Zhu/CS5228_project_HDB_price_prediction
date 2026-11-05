"""Regression tests for numeric-group column assignment.

The bug these guard against: ``GroupedStaticEncoder`` used ``torch.split`` on the
numeric matrix, so each group received a contiguous *positional* slice of
``StaticPreprocessor.numeric_matrix`` order instead of its declared columns. Every
tensor shape stayed valid, so all earlier tests passed while all five groups were fed
the wrong features. The tests below assert *which columns* reach each group, using
distinct sentinel values rather than shapes or sums.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from models import GroupedStaticEncoder, build_model, group_index_map  # noqa: E402
from preprocessing import StaticPreprocessor  # noqa: E402


def expanded_columns() -> list[str]:
    columns: list[str] = []
    for name in config.NUMERIC_FEATURES:
        columns.append(name)
        if name in config.INDICATOR_FEATURES:
            columns.append(f"{name}__missing")
    return columns


def test_index_map_resolves_declared_columns_by_name():
    columns = expanded_columns()
    indices, names = group_index_map(columns)
    assert list(names) == list(config.NUMERIC_GROUPS)
    for group, members in config.NUMERIC_GROUPS.items():
        assert names[group] == members
        assert [columns[i] for i in indices[list(names).index(group)]] == members


def test_index_map_is_not_identity_for_this_layout():
    """The input order differs from the declaration order; the map must show that."""
    columns = expanded_columns()
    indices, names = group_index_map(columns)
    flat = [i for index in indices for i in index]
    assert flat != list(range(len(columns))), (
        "if this ever becomes identity the positional split bug would be invisible again")
    # concretely: property must NOT be columns[0:2]
    property_index = indices[list(names).index("property")]
    assert property_index != [0, 1]
    assert columns[property_index[0]] == "FLOOR_AREA_SQM"
    assert columns[property_index[1]] == "YEARS_SINCE_LEASE_COMMENCEMENT"


def test_encoder_reads_declared_columns_via_sentinels():
    """Give every numeric column a unique sentinel; each group must see only its own."""
    columns = expanded_columns()
    indices, names = group_index_map(columns)
    sentinel = {name: float(i + 1) for i, name in enumerate(columns)}
    numeric = torch.tensor([[sentinel[c] for c in columns]], dtype=torch.float32)
    categorical = torch.zeros((1, len(config.CAT_FEATURES)), dtype=torch.long)

    encoder = GroupedStaticEncoder([2] * len(config.CAT_FEATURES), indices, names, embed_dim=4)
    # linear layers are randomly initialised; what matters is which columns they receive
    seen: dict[str, list[float]] = {}
    for group, linear in enumerate(encoder.group_linear):
        selected = numeric.index_select(1, encoder._index(group, numeric.device))
        seen[list(names)[group]] = selected.squeeze(0).tolist()
    for group, members in names.items():
        assert seen[group] == [sentinel[m] for m in members], group


def test_old_positional_split_would_fail_these_assertions():
    """Demonstrates the bug: positional chunking disagrees with the declared groups."""
    columns = expanded_columns()
    _, names = group_index_map(columns)
    sizes = [len(members) for members in names.values()]
    positional = torch.split(torch.arange(len(columns)).unsqueeze(0), sizes, dim=1)
    mismatches = 0
    for chunk, (group, members) in zip(positional, names.items()):
        got = [columns[int(i)] for i in chunk.squeeze(0)]
        if got != members:
            mismatches += 1
    assert mismatches == len(names), "all five groups were mis-assigned by the old split"


def test_encoder_rejects_wrong_input_width():
    columns = expanded_columns()
    indices, names = group_index_map(columns)
    encoder = GroupedStaticEncoder([2] * len(config.CAT_FEATURES), indices, names, embed_dim=4)
    with pytest.raises(ValueError, match="numeric width"):
        encoder.flat(torch.zeros((3, len(columns) - 1)), torch.zeros((3, len(config.CAT_FEATURES)), dtype=torch.long))


def test_group_linear_shapes_match_declared_members():
    columns = expanded_columns()
    indices, names = group_index_map(columns)
    encoder = GroupedStaticEncoder([2] * len(config.CAT_FEATURES), indices, names, embed_dim=4)
    for linear, (group, members) in zip(encoder.group_linear, names.items()):
        assert linear.in_features == len(members), group


def test_unassigned_or_duplicated_column_is_rejected():
    columns = expanded_columns()
    with pytest.raises(KeyError, match="not assigned"):
        group_index_map(columns + ["SOMETHING_ELSE"])
    with pytest.raises(KeyError, match="Duplicate numeric columns"):
        group_index_map(columns + ["LATITUDE"])


def test_real_preprocessor_columns_produce_a_valid_map():
    """End-to-end on the real CSV column order, not a hand-built list."""
    frame = pd.read_csv(config.TRAIN_SPATIAL, nrows=200)
    prep = StaticPreprocessor.fit(frame)
    assert prep.numeric_columns == expanded_columns()
    indices, names = group_index_map(prep.numeric_columns)
    flattened = [i for index in indices for i in index]
    assert sorted(flattened) == list(range(len(prep.numeric_columns)))
    for group, members in config.NUMERIC_GROUPS.items():
        assert names[group] == members
    assert all(len(index) for index in indices)


@pytest.mark.parametrize("name", ["mlp", "static_transformer", "lstm_concat", "lstm_transformer"])
def test_every_model_uses_the_index_map(name):
    columns = expanded_columns()
    indices, names = group_index_map(columns)
    model = build_model(name, [2] * len(config.CAT_FEATURES), indices, config.CONTEXT_DIM,
                        group_names=names)
    encoder = model.encoder if hasattr(model, "encoder") else model.block.encoder
    assert encoder.group_indices == indices
    assert encoder.group_names == names
    sentinel_numeric = torch.arange(len(columns), dtype=torch.float32).unsqueeze(0)
    categorical = torch.zeros((1, len(config.CAT_FEATURES)), dtype=torch.long)
    batch = {"numeric": sentinel_numeric, "categorical": categorical,
             "context": torch.zeros((1, config.CONTEXT_DIM)), "target": torch.zeros(1)}
    if name in ("lstm_concat", "lstm_transformer"):
        batch["panel"] = torch.zeros((1, config.HISTORY_LEN, len(config.PANEL_CHANNELS)))
        batch["panel_mask"] = torch.ones((1, config.HISTORY_LEN))
    out = model(batch)
    assert out.shape == (1,) and torch.isfinite(out).all()
