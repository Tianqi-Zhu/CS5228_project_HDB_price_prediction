"""Split labeled HDB rentals by a minimum monthly rent per labeled room.

Run from the repository root:
    .venv/Scripts/python.exe analysis/clean_rent_per_room.py
    .venv/Scripts/python.exe analysis/clean_rent_per_room.py --executive-room-count 5

The numeric HDB flat type is used as the room count (for example, 4-room -> 4).
Executive flats have no numeric room count in the source, so the default is to
write them to a separate unclassified file rather than assign one implicitly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/train.csv"
DESTINATION = ROOT / "data/processed/rent_per_room_500"
THRESHOLD = 500.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executive-room-count", type=int, choices=(5, 6),
                        help="Optional assumption for executive flats; default is unclassified.")
    args = parser.parse_args()

    source = pd.read_csv(SOURCE, dtype={"FLAT_TYPE": "string"})
    if source.MONTHLY_RENT.isna().any() or not pd.api.types.is_numeric_dtype(source.MONTHLY_RENT):
        raise ValueError("MONTHLY_RENT must be complete and numeric")
    source.insert(0, "SOURCE_CSV_LINE", range(2, len(source) + 2))

    labels = source.FLAT_TYPE.str.strip().str.lower()
    executive_source = source[labels.eq("executive")].copy()
    counts = labels.str.extract(r"^([1-5])\s*[- ]\s*room$", expand=False)
    unexpected = labels[counts.isna() & labels.ne("executive")]
    if len(unexpected):
        raise ValueError(f"Unexpected FLAT_TYPE values: {unexpected.drop_duplicates().tolist()}")
    source["ROOM_COUNT"] = pd.to_numeric(counts, errors="coerce").astype("Int64")
    if args.executive_room_count is not None:
        source.loc[labels.eq("executive"), "ROOM_COUNT"] = args.executive_room_count
    source["RENT_PER_ROOM"] = source.MONTHLY_RENT / source.ROOM_COUNT

    unclassified = source[source.ROOM_COUNT.isna()].copy()
    classified = source[source.ROOM_COUNT.notna()].copy()
    removed = classified[classified.RENT_PER_ROOM < THRESHOLD].copy()
    cleaned = classified[classified.RENT_PER_ROOM >= THRESHOLD].copy()
    if len(cleaned) + len(removed) + len(unclassified) != len(source):
        raise AssertionError("Rows were lost during the split")
    if not set(source.SOURCE_CSV_LINE) == set(pd.concat([cleaned, removed, unclassified]).SOURCE_CSV_LINE):
        raise AssertionError("Source line reconciliation failed")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(DESTINATION / "cleaned.csv", index=False)
    removed.to_csv(DESTINATION / "removed_below_500.csv", index=False)
    unclassified.to_csv(DESTINATION / "executive_unclassified.csv", index=False)
    address_audit = executive_source.groupby(
        ["TOWN", "STREET", "BLOCK", "FLAT_MODEL", "FLOOR_AREA_SQM"], dropna=False
    ).agg(rental_rows=("SOURCE_CSV_LINE", "size"),
          first_source_line=("SOURCE_CSV_LINE", "min")).reset_index()
    address_audit["VERIFIED_ROOM_COUNT"] = pd.NA
    address_audit["STATUS"] = "unit_floor_plan_needed"
    address_audit["HDB_BLOCK_DATA_URL"] = "https://data.gov.sg/datasets/d_17f5382f26140b1fdae0ba2ef6239d2f/view"
    address_audit.to_csv(DESTINATION / "executive_address_audit.csv", index=False)

    summary = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "total_rows": int(len(source)),
        "room_count_rule": "Use the numeric flat-type label; e.g. 4-room is divided by 4",
        "executive_room_count": args.executive_room_count,
        "minimum_rent_per_room_sgd_inclusive": THRESHOLD,
        "cleaned_rows": int(len(cleaned)),
        "removed_below_500_rows": int(len(removed)),
        "executive_unclassified_rows": int(len(unclassified)),
        "executive_address_model_area_groups": int(len(address_audit)),
        "equal_to_500_rows_retained": int(cleaned.RENT_PER_ROOM.eq(THRESHOLD).sum()),
    }
    (DESTINATION / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
