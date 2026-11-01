"""Label-blind context partitioning and complete flat-type prediction assembly."""
from __future__ import annotations

import numpy as np
import pandas as pd


def flat_type_partitions(train, query):
    if not train.index.is_unique or not query.index.is_unique:
        raise ValueError("Context and query row IDs must be unique")
    if len(train.index.intersection(query.index)):
        raise ValueError("Context and query IDs must not overlap")
    if not train.columns.equals(query.columns):
        raise ValueError("Context and query feature schemas differ")
    a, b = train.FLAT_TYPE_CLEAN.astype("string"), query.FLAT_TYPE_CLEAN.astype("string")
    if a.isna().any() or b.isna().any():
        raise ValueError("Missing flat type; cannot route inference")
    partitions = {}
    for kind in sorted(b.unique()):
        context = np.flatnonzero(a.eq(kind).to_numpy())
        queries = np.flatnonzero(b.eq(kind).to_numpy())
        if not len(context):
            raise ValueError(f"No matching context for flat type {kind}")
        partitions[str(kind)] = (context, queries)
    np.testing.assert_array_equal(np.sort(np.concatenate([q for _, q in partitions.values()])), np.arange(len(query)))
    return partitions


def assemble_predictions(query, frames):
    result = pd.concat(frames, ignore_index=True).set_index("row_id")
    if not result.index.is_unique or len(result) != len(query) or set(result.index) != set(query.index):
        raise ValueError("Category predictions must cover every query exactly once")
    result = result.loc[query.index]
    if not np.isfinite(result.predicted).all():
        raise ValueError("Non-finite category predictions")
    if not np.array_equal(result.flat_type.astype(str), query.FLAT_TYPE_CLEAN.astype(str)):
        raise ValueError("Prediction flat type disagrees with query")
    return result.reset_index(names="row_id")
