"""Create a conservative, reversible low-rent anomaly filter for HDB rentals.

These flags are *not* verified bedroom-only leases. The source HDB rental series
is titled "Renting Out of Flats" and does not identify bedrooms rented.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/train.csv"
OUT = ROOT / "data/processed/possible_partial_rentals"
RATIO = 0.50
MIN_TOWN_QUARTER = 30
MIN_BLOCK_YEAR = 5
ELIGIBLE_TYPES = {"3-room", "4-room", "5-room", "executive"}


def build_flags(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data.insert(0, "SOURCE_CSV_LINE", range(2, len(data) + 2))
    for col in ("TOWN", "STREET", "BLOCK", "FLAT_MODEL"):
        data[f"_{col}"] = data[col].astype(str).str.strip().str.lower()
    data["_FLAT_TYPE"] = data.FLAT_TYPE.astype(str).str.strip().str.lower().str.replace(" ", "-", regex=False)
    data["YEAR"] = data.RENT_APPROVAL_DATE.astype(str).str[:4]
    month = data.RENT_APPROVAL_DATE.astype(str).str[5:7].astype(int)
    data["QUARTER"] = data.YEAR + "Q" + (((month - 1) // 3) + 1).astype(str)

    town_group = data.groupby(["_TOWN", "_FLAT_TYPE", "QUARTER"]).MONTHLY_RENT
    data["TOWN_TYPE_QUARTER_N"] = town_group.transform("size")
    data["TOWN_TYPE_QUARTER_MEDIAN"] = town_group.transform("median")
    block_group = data.groupby([
        "_TOWN", "_STREET", "_BLOCK", "_FLAT_TYPE", "_FLAT_MODEL",
        "FLOOR_AREA_SQM", "YEAR",
    ]).MONTHLY_RENT
    data["BLOCK_LAYOUT_YEAR_N"] = block_group.transform("size")
    data["BLOCK_LAYOUT_YEAR_MEDIAN"] = block_group.transform("median")
    data["TOWN_RENT_RATIO"] = data.MONTHLY_RENT / data.TOWN_TYPE_QUARTER_MEDIAN
    data["BLOCK_RENT_RATIO"] = data.MONTHLY_RENT / data.BLOCK_LAYOUT_YEAR_MEDIAN

    data["BROAD_LOW_RENT_FLAG"] = (
        data._FLAT_TYPE.isin(ELIGIBLE_TYPES)
        & data.TOWN_TYPE_QUARTER_N.ge(MIN_TOWN_QUARTER)
        & data.TOWN_RENT_RATIO.lt(RATIO)
    )
    data["CONSERVATIVE_LOW_RENT_FLAG"] = (
        data.BROAD_LOW_RENT_FLAG
        & data.BLOCK_LAYOUT_YEAR_N.ge(MIN_BLOCK_YEAR)
        & data.BLOCK_RENT_RATIO.lt(RATIO)
    )
    return data.drop(columns=[f"_{c}" for c in ("TOWN", "STREET", "BLOCK", "FLAT_MODEL", "FLAT_TYPE")])


def main() -> None:
    original = pd.read_csv(SOURCE)
    if original.MONTHLY_RENT.isna().any() or original.MONTHLY_RENT.le(0).any():
        raise ValueError("Missing or nonpositive rent")
    flagged = build_flags(original)
    candidate = flagged[flagged.CONSERVATIVE_LOW_RENT_FLAG].copy()
    broad = flagged[flagged.BROAD_LOW_RENT_FLAG].copy()
    retained = original.loc[~flagged.CONSERVATIVE_LOW_RENT_FLAG].copy()
    if len(candidate) + len(retained) != len(original):
        raise AssertionError("Filtered rows do not reconcile")

    OUT.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(OUT / "low_rent_candidates.csv", index=False)
    broad.to_csv(OUT / "broad_low_rent_review.csv", index=False)
    retained.to_csv(OUT / "train_without_low_rent_candidates.csv", index=False)
    flagged[["SOURCE_CSV_LINE", "BROAD_LOW_RENT_FLAG", "CONSERVATIVE_LOW_RENT_FLAG"]].to_csv(
        OUT / "all_row_flags.csv", index=False)
    summary = {
        "source": "data/train.csv",
        "total_rows": int(len(original)),
        "eligibility": sorted(ELIGIBLE_TYPES),
        "ratio_threshold_strictly_below": RATIO,
        "minimum_town_type_quarter_rows": MIN_TOWN_QUARTER,
        "minimum_block_layout_year_rows": MIN_BLOCK_YEAR,
        "broad_candidates": int(len(broad)),
        "conservative_candidates_removed": int(len(candidate)),
        "retained_rows": int(len(retained)),
        "removed_by_year": candidate.YEAR.value_counts().sort_index().to_dict(),
        "removed_by_flat_type": candidate.FLAT_TYPE.value_counts().sort_index().to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
