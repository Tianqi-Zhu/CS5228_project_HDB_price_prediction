"""Save a first-pass <=S$1,500 rent filter for 3-room-or-larger HDB flats.

This is a price rule for an experimental training copy. A low declared rent does
not establish that the lease covered bedrooms rather than the whole flat.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/train.csv"
OUT = ROOT / "data/processed/whole_flat_rent_le_1500"
TYPES = {"3-room", "4-room", "5-room", "executive"}
CUTOFF = 1500


def main() -> None:
    raw = pd.read_csv(SOURCE)
    normalized_type = raw.FLAT_TYPE.astype(str).str.strip().str.lower().str.replace(" ", "-", regex=False)
    remove = normalized_type.isin(TYPES) & raw.MONTHLY_RENT.le(CUTOFF)
    kept = raw.loc[~remove].copy()
    removed = raw.loc[remove].copy()
    removed.insert(0, "SOURCE_CSV_LINE", removed.index + 2)
    removed["NORMALIZED_FLAT_TYPE"] = normalized_type[remove]
    removed["FILTER_REASON"] = "3-room-or-larger monthly rent <= 1500"
    if len(kept) + len(removed) != len(raw):
        raise AssertionError("Row counts do not reconcile")
    if not kept.columns.equals(raw.columns):
        raise AssertionError("Retained training schema changed")
    if not (removed.MONTHLY_RENT <= CUTOFF).all():
        raise AssertionError("A removed row exceeds the cutoff")

    OUT.mkdir(parents=True, exist_ok=True)
    kept.to_csv(OUT / "train_filtered.csv", index=False)
    removed.to_csv(OUT / "removed_monthly_rent_le_1500.csv", index=False)

    summary = {
        "source": "data/train.csv",
        "rule": "normalized FLAT_TYPE in 3-room, 4-room, 5-room, executive and MONTHLY_RENT <= 1500",
        "total_rows": int(len(raw)),
        "removed_rows": int(len(removed)),
        "retained_rows": int(len(kept)),
        "removed_equal_to_1500": int(removed.MONTHLY_RENT.eq(CUTOFF).sum()),
        "removed_below_1500": int(removed.MONTHLY_RENT.lt(CUTOFF).sum()),
        "removed_by_year": removed.RENT_APPROVAL_DATE.str[:4].value_counts().sort_index().to_dict(),
        "removed_by_flat_type": removed.NORMALIZED_FLAT_TYPE.value_counts().sort_index().to_dict(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
