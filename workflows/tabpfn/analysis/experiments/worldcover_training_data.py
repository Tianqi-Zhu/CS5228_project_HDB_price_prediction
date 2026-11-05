"""Validate and append bounded tree fractions to the existing TabPFN inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

TREE_FRACTIONS = ["TREE_CLASS_FRACTION_100M", "TREE_CLASS_FRACTION_300M"]
TREE_COLUMNS = TREE_FRACTIONS + ["TREE_COVER_UNAVAILABLE"]
PREPROCESSING = {
    "fraction_units": "tree-class area / valid mapped area; fraction in [0, 1]",
    "external_scaling": "identity; no additional StandardScaler, MinMaxScaler, log or quantile transformation",
    "reason": "Inputs already have bounded dimensionless units; retain TabPFN-3.5 Fast checkpoint preprocessing",
    "missing_values": "Preserve NaN and explicit 0/1 unavailable flag; TabPFN handles missing values internally",
    "fit_statistics": "No external preprocessing statistics are fitted to either context or queries",
    "baseline_columns": "Preserve all 77 existing values, column order and categorical dtypes",
    "documentation_url": "https://github.com/PriorLabs/TabPFN#usage-tips",
}


def append_tree_features(original, additions):
    """Keep bounded floats and native missingness, preserving every baseline cell."""
    if original.shape[1] != 77 or not original.columns.is_unique:
        raise ValueError("Expected the original unique 77-column baseline")
    if not original.index.is_unique or not additions.index.is_unique:
        raise ValueError("Duplicate row IDs are not allowed")
    if not original.index.equals(additions.index):
        raise ValueError("Tree features must have identical row IDs in identical order")
    if list(additions.columns) != TREE_COLUMNS or set(TREE_COLUMNS) & set(original.columns):
        raise ValueError("Unexpected or overlapping tree feature schema")
    fields = additions.astype("float64").copy()
    values = fields[TREE_FRACTIONS].to_numpy()
    observed = ~np.isnan(values)
    if not np.isfinite(values[observed]).all():
        raise ValueError("Tree fractions contain infinity")
    if ((values[observed] < 0) | (values[observed] > 1)).any():
        raise ValueError("Tree fractions must already be normalized to [0, 1]")
    flag = fields.TREE_COVER_UNAVAILABLE
    if not flag.isin([0., 1.]).all():
        raise ValueError("Unavailable flag must be binary without missing values")
    expected = fields[TREE_FRACTIONS].isna().any(axis=1).astype(float)
    if not flag.equals(expected.rename(flag.name)):
        raise ValueError("Unavailable flag disagrees with missing tree fractions")
    result = original.join(fields, validate="one_to_one")
    pd.testing.assert_frame_equal(result[original.columns], original)
    pd.testing.assert_index_equal(result.index, original.index)
    assert result.shape == (len(original), 80)
    return result
