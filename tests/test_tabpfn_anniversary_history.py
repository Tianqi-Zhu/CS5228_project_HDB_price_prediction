"""Causal lookup checks for TabPFN anniversary rent features."""

import pandas as pd

from analysis.experiments.tabpfn_anniversary_history import AnniversaryHistory


def test_anniversary_history_uses_only_prior_matching_block_and_type():
    rows = pd.DataFrame({
        "RENT_APPROVAL_DATE": ["2023-01", "2024-01", "2024-02", "2024-01"],
        "TOWN": ["a"] * 4,
        "STREET": ["road"] * 4,
        "BLOCK": ["1", "1", "1", "2"],
        "FLAT_TYPE_CLEAN": ["4 room"] * 4,
        "MONTHLY_RENT": [1200.0, 2000.0, 2200.0, 9000.0],
    })
    feature = "ANN_BLOCK_12_R1_"
    encoded = AnniversaryHistory().fit(rows).transform(rows)
    assert encoded.loc[0, feature + "N"] == 0
    assert encoded.loc[1, feature + "N"] == 1
    assert encoded.loc[1, feature + "RENT"] == 1200
    assert encoded.loc[2, feature + "N"] == 1
    assert encoded.loc[3, feature + "N"] == 0

    changed = rows.copy()
    changed.loc[1, "MONTHLY_RENT"] = 100000.0
    changed_encoded = AnniversaryHistory().fit(changed).transform(changed)
    pd.testing.assert_series_equal(encoded.loc[1], changed_encoded.loc[1])
