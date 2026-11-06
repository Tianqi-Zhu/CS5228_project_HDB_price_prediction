"""Audit and split rents strictly below half their year/town/flat-type median.

Run: .venv/Scripts/python.exe analysis/filter_rent_below_half_median.py
Medians use every original training row once, including the row being assessed.
This is a retrospective diagnostic, not a verified error label or a causal
historical feature. The source training file is never overwritten.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/train.csv"
OUT = ROOT / "data/processed/year_town_type_half_median"
KEYS = ["YEAR", "NORMALIZED_TOWN", "NORMALIZED_FLAT_TYPE"]
FLAG = "BELOW_HALF_MEDIAN"


def build_audit(raw: pd.DataFrame) -> pd.DataFrame:
    """Preserve input order and values; append normalized grouping keys and flags."""
    required = ["RENT_APPROVAL_DATE", "TOWN", "FLAT_TYPE", "MONTHLY_RENT"]
    if raw[required].isna().any().any():
        raise ValueError("Missing year/town/flat-type/rent values")
    rent = pd.to_numeric(raw.MONTHLY_RENT, errors="raise")
    if not np.isfinite(rent.to_numpy(dtype=float)).all() or rent.le(0).any():
        raise ValueError("Rents must be finite and positive")
    dates = pd.to_datetime(raw.RENT_APPROVAL_DATE, format="%Y-%m", errors="raise")
    audit = raw.copy()
    audit.insert(0, "SOURCE_CSV_LINE", range(2, len(raw) + 2))
    audit["YEAR"] = dates.dt.year
    audit["NORMALIZED_TOWN"] = raw.TOWN.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
    audit["NORMALIZED_FLAT_TYPE"] = raw.FLAT_TYPE.astype("string").str.strip().str.lower().str.replace(r"\s*-\s*|\s+", "-", regex=True)
    if audit.NORMALIZED_TOWN.eq("").any():
        raise ValueError("Empty town label")
    if not audit.NORMALIZED_FLAT_TYPE.isin(["1-room", "2-room", "3-room", "4-room", "5-room", "executive"]).all():
        raise ValueError("Unexpected flat type")
    audit["MONTHLY_RENT"] = rent
    group = audit.groupby(KEYS, dropna=False).MONTHLY_RENT
    audit["GROUP_N"] = group.transform("size")
    audit["GROUP_MEDIAN_RENT"] = group.transform("median")
    audit["HALF_MEDIAN_CUTOFF"] = audit.GROUP_MEDIAN_RENT * 0.5
    audit["RENT_TO_GROUP_MEDIAN"] = rent / audit.GROUP_MEDIAN_RENT
    audit[FLAG] = rent.lt(audit.HALF_MEDIAN_CUTOFF)
    audit["EQUAL_TO_HALF_MEDIAN"] = rent.eq(audit.HALF_MEDIAN_CUTOFF)
    audit["GROUP_N_LT_30"] = audit.GROUP_N.lt(30)
    return audit


def aggregate(audit: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    result = audit.groupby(keys, dropna=False).agg(
        total_rows=(FLAG, "size"), removed_rows=(FLAG, "sum"),
        equal_to_half_median_retained=("EQUAL_TO_HALF_MEDIAN", "sum"),
    ).reset_index()
    result["retained_rows"] = result.total_rows - result.removed_rows
    result["removed_pct"] = 100 * result.removed_rows / result.total_rows
    return result


def table(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        return f"{value:.2f}" if isinstance(value, float) else str(value)
    return "\n".join([
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
        *["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def main() -> None:
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    raw = pd.read_csv(SOURCE)
    audit = build_audit(raw)
    remove = audit[FLAG]
    removed = audit.loc[remove].sort_values(["RENT_TO_GROUP_MEDIAN", "SOURCE_CSV_LINE"])
    kept = raw.loc[~remove].copy()
    groups = aggregate(audit, KEYS).merge(
        audit.groupby(KEYS).agg(
            group_median_rent=("GROUP_MEDIAN_RENT", "first"),
            half_median_cutoff=("HALF_MEDIAN_CUTOFF", "first"),
            minimum_rent=("MONTHLY_RENT", "min"), maximum_rent=("MONTHLY_RENT", "max"),
        ).reset_index(), on=KEYS, validate="one_to_one",
    )
    by_year = aggregate(audit, ["YEAR"])
    by_type = aggregate(audit, ["NORMALIZED_FLAT_TYPE"])
    by_town = aggregate(audit, ["NORMALIZED_TOWN"]).sort_values("removed_rows", ascending=False)

    # Compare independently computed rules on the same original rows.
    fixed_rule = audit.NORMALIZED_FLAT_TYPE.isin(["3-room", "4-room", "5-room", "executive"]) & audit.MONTHLY_RENT.le(1500)
    comparisons = []
    previous = {"3-room-or-larger rent <= 1500": fixed_rule}
    old_flags_path = ROOT / "data/processed/possible_partial_rentals/all_row_flags.csv"
    if old_flags_path.exists():
        old = pd.read_csv(old_flags_path)
        if old.SOURCE_CSV_LINE.tolist() != audit.SOURCE_CSV_LINE.tolist():
            raise ValueError("Previous flags do not align with source rows")
        for col in ["BROAD_LOW_RENT_FLAG", "CONSERVATIVE_LOW_RENT_FLAG"]:
            if not pd.api.types.is_bool_dtype(old[col]):
                raise ValueError(f"Expected boolean flags in {col}")
            previous[col] = old[col]
    for name, other in previous.items():
        comparisons.append(dict(previous_rule=name, previous_removed=int(other.sum()),
                                both_removed=int((remove & other).sum()),
                                annual_rule_only=int((remove & ~other).sum()),
                                previous_rule_only=int((~remove & other).sum())))
    overlap = pd.DataFrame(comparisons)

    # Add supporting comparisons for manual review without changing the rule.
    removed["ALSO_FIXED_1500_FLAG"] = fixed_rule.loc[removed.index]
    block_keys = [*KEYS, "STREET", "BLOCK", "FLAT_MODEL", "FLOOR_AREA_SQM"]
    supporting = audit.groupby(block_keys, dropna=False).MONTHLY_RENT
    removed["SAME_BLOCK_LAYOUT_YEAR_N"] = supporting.transform("size").loc[removed.index]
    removed["SAME_BLOCK_LAYOUT_YEAR_MEDIAN"] = supporting.transform("median").loc[removed.index]
    removed["RENT_TO_BLOCK_LAYOUT_MEDIAN"] = removed.MONTHLY_RENT / removed.SAME_BLOCK_LAYOUT_YEAR_MEDIAN
    supported = removed.SAME_BLOCK_LAYOUT_YEAR_N.ge(5)
    block_low = supported & removed.RENT_TO_BLOCK_LAYOUT_MEDIAN.lt(0.5)

    # Reconcile the split, preserve the model-input schema, and retain equality.
    assert len(kept) + len(removed) == len(raw)
    assert kept.columns.equals(raw.columns)
    assert removed.MONTHLY_RENT.lt(removed.HALF_MEDIAN_CUTOFF).all()
    assert not audit.loc[~remove, "MONTHLY_RENT"].lt(audit.loc[~remove, "HALF_MEDIAN_CUTOFF"]).any()
    assert not (remove & audit.EQUAL_TO_HALF_MEDIAN).any()
    assert set(audit.loc[~remove, "SOURCE_CSV_LINE"]) | set(removed.SOURCE_CSV_LINE) == set(audit.SOURCE_CSV_LINE)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash

    OUT.mkdir(parents=True, exist_ok=True)
    kept.to_csv(OUT / "train_filtered.csv", index=False)
    removed.to_csv(OUT / "removed_below_half_median.csv", index=False)
    audit.drop(columns=raw.columns).to_csv(OUT / "all_row_flags.csv", index=False)
    for name, frame in [("group_medians", groups), ("summary_by_year", by_year),
                        ("summary_by_flat_type", by_type), ("summary_by_town", by_town),
                        ("comparison_previous_filters", overlap)]:
        frame.to_csv(OUT / f"{name}.csv", index=False)

    summary = dict(
        source="data/train.csv", source_sha256=source_hash,
        rule="MONTHLY_RENT < 0.5 * median(MONTHLY_RENT within approval year x normalized town x normalized flat type)",
        median_reference="All original training rows, including the assessed row; one pass; no deduplication or iterative recalculation",
        types="All 1- through 5-room types and executive; executive is its own type",
        minimum_group_size=None, total_rows=len(raw), removed_rows=len(removed), retained_rows=len(kept),
        removed_pct=100 * len(removed) / len(raw),
        equal_to_half_median_rows_retained=int(audit.EQUAL_TO_HALF_MEDIAN.sum()),
        total_groups=len(groups), groups_with_removed_rows=int(groups.removed_rows.gt(0).sum()),
        groups_with_fewer_than_30_rows=int(groups.total_rows.lt(30).sum()),
        removed_from_groups_with_fewer_than_30_rows=int(removed.GROUP_N_LT_30.sum()),
        minimum_flagged_group_size=int(removed.GROUP_N.min()) if len(removed) else None,
        removed_with_at_least_5_same_block_layout_year_rows=int(supported.sum()),
        removed_also_below_half_same_block_layout_year_median_with_n_ge5=int(block_low.sum()),
        first_approval_month=raw.RENT_APPROVAL_DATE.min(), last_approval_month=raw.RENT_APPROVAL_DATE.max(),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    examples = removed[["SOURCE_CSV_LINE", "RENT_APPROVAL_DATE", "TOWN", "FLAT_TYPE", "MONTHLY_RENT",
                        "GROUP_MEDIAN_RENT", "HALF_MEDIAN_CUTOFF", "RENT_TO_GROUP_MEDIAN", "GROUP_N"]].head(15)
    report = [
        "# Annual town and flat-type low-rent audit", "",
        f"Source: `data/train.csv`; SHA-256 `{source_hash}`.", "",
        f"The requested strict rule flags **{len(removed):,} / {len(raw):,} rows ({summary['removed_pct']:.3f}%)** and retains **{len(kept):,} rows**.", "",
        "## Method", "",
        "Group by approval calendar year, normalized town and normalized flat type. Merge aliases such as `4 room` / `4-room`; keep executive separate. Calculate one median from all original labeled rows in each group, including the row being assessed. Remove only rent < 0.5 x that median. There is no extra minimum group size, flat-type restriction, fallback, or repeated filtering.", "",
        f"Exactly half the median is retained: {summary['equal_to_half_median_rows_retained']:,} rows. Of {len(groups):,} observed groups, {summary['groups_with_fewer_than_30_rows']:,} have fewer than 30 rows; {summary['removed_from_groups_with_fewer_than_30_rows']:,} flagged rows come from these small groups. The data cover {summary['first_approval_month']} through {summary['last_approval_month']}; 2025 contains January-March only.", "",
        "## By year", "", table(by_year), "",
        "## By flat type", "", table(by_type), "",
        "## Towns with most flagged rows", "", table(by_town.head(10)), "",
        "## Comparison with previous rules", "", table(overlap), "",
        "The fixed S$1,500 rule is recomputed directly on the current source. Earlier broad/conservative flags, if present, are read from their existing aligned audit file; they are not part of this rule.", "",
        "## Lowest rent relative to its group median", "", table(examples), "",
        "## What the flags establish", "",
        f"These rents are unusually low for the requested comparison group. As a supplementary check, {int(supported.sum()):,} flagged rows have at least five observations with the same year, town, flat type, street, block, flat model and floor area; {int(block_low.sum()):,} of these also fall below half that more local median. This supports reviewing those records, but does not establish an incorrect label or a bedroom-only lease: the data do not identify individual units or contractual terms. The supplementary check uses raw address/model labels and does not alter selection.", "",
        "Small groups can have unstable medians or miss anomalies when cheap rents dominate. Annual medians can also flag genuine discounted leases and do not account for within-year market changes. No external source verification or model retraining was performed.", "",
        "This is a retrospective audit using labels from the whole observed year. For model evaluation, recompute the filter using only each training fold and leave validation targets unchanged. Do not calculate training cutoffs using held-out labels. A row audit alone does not establish that removal improves prediction.", "",
        "## Outputs and reproduction", "",
        "- `train_filtered.csv`: original columns and original order for retained rows.",
        "- `removed_below_half_median.csv`: original row values, source CSV line (header is line 1), group statistics, and supporting block comparison; sorted by rent/median ascending.",
        "- `all_row_flags.csv`: one row per original record, in original order, with grouping keys, medians, cutoffs and flags.",
        "- `group_medians.csv`: every observed year/town/type median, cutoff, count and removed count.",
        "- `summary_by_year.csv`, `summary_by_flat_type.csv`, `summary_by_town.csv`, `comparison_previous_filters.csv`, `summary.json`: aggregate audit results.", "",
        "Run from repository root: `.venv/Scripts/python.exe analysis/filter_rent_below_half_median.py`.", "",
        "The original training file and existing filters/submissions are preserved.", "",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
