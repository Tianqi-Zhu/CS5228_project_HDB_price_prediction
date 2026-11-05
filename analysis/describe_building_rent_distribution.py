"""Describe observed HDB rental types and rents at block and monthly resolution."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.config import ROOT
from analysis.data import normalized_address, prepare_main, sha256


OUT = ROOT / "reports" / "building_rent_distribution"
TYPES = ("1-room", "2-room", "3-room", "4-room", "5-room", "executive")


def qspread(values):
    q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {"q25": float(q25), "median": float(q50), "q75": float(q75), "iqr": float(q75 - q25)}


def main():
    raw_path = ROOT / "data" / "train.csv"
    blocks_path = ROOT / "data" / "auxiliary" / "sg-hdb-block.csv"
    raw = prepare_main(pd.read_csv(raw_path))
    block = normalized_address(pd.read_csv(blocks_path))
    if block.duplicated(["TOWN", "STREET", "BLOCK"]).any():
        raise ValueError("Block metadata has duplicate address keys")
    data = normalized_address(raw).merge(
        block[["TOWN", "STREET", "BLOCK", "YEAR_COMPLETED"]],
        on=["TOWN", "STREET", "BLOCK"], how="left", validate="many_to_one",
    )
    if len(data) != len(raw) or data.YEAR_COMPLETED.isna().any():
        raise ValueError("Rental-to-block join is incomplete")
    data["building"] = data.TOWN + "|" + data.STREET + "|" + data.BLOCK
    data["street_completion_cohort"] = (
        data.TOWN + "|" + data.STREET + "|" + data.YEAR_COMPLETED.astype(str)
    )
    data["month"] = data.RENT_APPROVAL_DATE
    data["flat_type"] = data.FLAT_TYPE_CLEAN
    data["rent"] = pd.to_numeric(data.MONTHLY_RENT)
    if data.rent.isna().any() or (data.rent <= 0).any():
        raise ValueError("Invalid rent")

    block_types = data.groupby("building").flat_type.nunique()
    building_month = data.groupby(["building", "month"], sort=False).agg(
        n=("rent", "size"), type_count=("flat_type", "nunique"))
    building_type_month = data.groupby(["building", "flat_type", "month"], sort=False).rent.agg(
        n="size", q25=lambda x: x.quantile(.25), median="median", q75=lambda x: x.quantile(.75))
    building_type_month["iqr"] = building_type_month.q75 - building_type_month.q25

    # A completion cohort on one street is an illustrative grouping, not an
    # official HDB project identifier.
    cohort_buildings = data.groupby("street_completion_cohort").building.nunique()
    cohort_month = data.groupby(["street_completion_cohort", "month"], sort=False).agg(
        n=("rent", "size"), buildings=("building", "nunique"), types=("flat_type", "nunique"))
    multi_cohorts = cohort_buildings[cohort_buildings.ge(2)].index
    type_sets = data.groupby("building").flat_type.agg(lambda x: frozenset(x.unique()))
    building_cohort = data[["building", "street_completion_cohort"]].drop_duplicates().set_index("building")
    sets = building_cohort.join(type_sets.rename("types")).groupby("street_completion_cohort").types.agg(
        lambda x: len(set(x)) > 1)

    # Compare flat-type medians only where both types occur in one block-month.
    by_type = building_type_month.reset_index().pivot_table(
        index=["building", "month"], columns="flat_type", values="median")
    pair_gaps = {}
    for smaller, larger in (("3-room", "4-room"), ("4-room", "5-room")):
        delta = (by_type[larger] - by_type[smaller]).dropna()
        pair_gaps[f"{larger}_minus_{smaller}"] = {"block_months": len(delta), **qspread(delta)}

    # A recent quarter gives more support than a single month while keeping
    # temporal drift limited; still descriptive, not a unit-level rent panel.
    recent = data[data.month.ge("2025-01")]
    cohort_recent = recent.groupby("street_completion_cohort").agg(
        rentals=("rent", "size"), buildings=("building", "nunique"), types=("flat_type", "nunique"))
    candidates = cohort_recent.query("buildings >= 3 and buildings <= 6 and types >= 3").sort_values(
        ["rentals", "buildings"], ascending=[False, True])
    example_cohort = candidates.index[0]
    example = recent[recent.street_completion_cohort.eq(example_cohort)]
    example_type = example.groupby("flat_type").rent.agg(
        n="size", q25=lambda x: x.quantile(.25), median="median", q75=lambda x: x.quantile(.75))
    example_type = example_type.reindex([t for t in TYPES if t in example_type.index])
    recent_buildings = recent.groupby("building").agg(rentals=("rent", "size"), types=("flat_type", "nunique"))
    example_building = recent_buildings.query("rentals >= 5 and types >= 2").sort_values(
        ["rentals", "types"], ascending=False).index[0]
    building_example = recent[recent.building.eq(example_building)]
    building_example_type = building_example.groupby("flat_type").rent.agg(
        n="size", q25=lambda x: x.quantile(.25), median="median", q75=lambda x: x.quantile(.75))
    building_example_type = building_example_type.reindex([t for t in TYPES if t in building_example_type.index])

    stats = {
        "source_sha256": {"train": sha256(raw_path), "blocks": sha256(blocks_path)},
        "rental_rows": len(data), "months": int(data.month.nunique()),
        "start_month": data.month.min(), "end_month": data.month.max(),
        "buildings_observed": int(block_types.size),
        "building_type_count_distribution": {str(k): int(v) for k, v in block_types.value_counts().sort_index().items()},
        "building_months_observed": len(building_month),
        "building_month_one_rental_share": float(building_month.n.eq(1).mean()),
        "building_month_type_count_distribution": {str(k): int(v) for k, v in building_month.type_count.value_counts().sort_index().items()},
        "building_month_ge3": int(building_month.n.ge(3).sum()),
        "building_type_months_observed": len(building_type_month),
        "building_type_month_ge2": int(building_type_month.n.ge(2).sum()),
        "building_type_month_ge3": int(building_type_month.n.ge(3).sum()),
        "building_type_month_ge5": int(building_type_month.n.ge(5).sum()),
        "building_type_month_ge3_median_iqr_sgd": float(building_type_month.loc[building_type_month.n.ge(3), "iqr"].median()),
        "building_type_month_ge3_p90_iqr_sgd": float(building_type_month.loc[building_type_month.n.ge(3), "iqr"].quantile(.9)),
        "pair_gaps": pair_gaps,
        "street_completion_cohorts": int(cohort_buildings.size),
        "multi_building_cohorts": int(len(multi_cohorts)),
        "multi_building_cohorts_different_observed_type_sets": int(sets.loc[multi_cohorts].sum()),
        "cohort_months_observed": len(cohort_month),
        "cohort_month_ge5": int(cohort_month.n.ge(5).sum()),
        "cohort_month_ge5_and_ge2_buildings": int((cohort_month.n.ge(5) & cohort_month.buildings.ge(2)).sum()),
        "example_cohort": example_cohort,
        "example_recent_rentals": len(example),
        "example_recent_buildings": int(example.building.nunique()),
        "example_recent_types": int(example.flat_type.nunique()),
        "example_building": example_building,
        "example_building_recent_rentals": len(building_example),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    block_types.value_counts().sort_index().plot.bar(ax=axes[0], color="#3478a8")
    axes[0].set(title="Flat types observed per block, 2021–2025", xlabel="Number of flat types", ylabel="Blocks")
    axes[0].tick_params(axis="x", rotation=0)
    building_month.n.clip(upper=6).value_counts().sort_index().plot.bar(ax=axes[1], color="#4b9b74")
    axes[1].set(title="Rental observations per block-month", xlabel="Transactions (6 means 6+)", ylabel="Block-months")
    axes[1].tick_params(axis="x", rotation=0)
    fig.savefig(OUT / "observed_support.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    series = [example.loc[example.flat_type.eq(t), "rent"].to_numpy() for t in example_type.index]
    ax.boxplot(series, tick_labels=[f"{t}\n(n={len(v)})" for t, v in zip(example_type.index, series)], showfliers=False)
    ax.set(title=f"Street/completion cohort: {example_cohort.replace('|', ' · ')}\nJan–Mar 2025 approvals",
           ylabel="Monthly rent (S$)")
    fig.savefig(OUT / "cohort_example_rents.png", dpi=160)
    plt.close(fig)

    lines = ["# Rental types and rents within HDB blocks", "",
             "Source: 150,000 labeled rental records from January 2021 to March 2025, matched to the local HDB block table by town, street, and block number. Month is rental approval month. A record is a transaction, not a distinct vacant unit or listing; no unit number or official project ID is supplied.", "",
             "## What the observations show", "",
             f"- {len(block_types):,} blocks have at least one observed rental. Observed types per block: " + ", ".join(f"{k} type: {v:,}" for k, v in block_types.value_counts().sort_index().items()) + ".",
             f"- Across {len(building_month):,} observed block-months, {building_month.n.eq(1).mean():.1%} have one transaction. " + ", ".join(f"{k} observed type(s): {v:,}" for k, v in building_month.type_count.value_counts().sort_index().items()) + ".",
             f"- {len(building_type_month):,} block × type × month cells contain a rental; {building_type_month.n.ge(3).sum():,} have at least three and {building_type_month.n.ge(5).sum():,} have at least five. The median within-cell IQR among cells with at least three is S${stats['building_type_month_ge3_median_iqr_sgd']:.0f}; the 90th percentile is S${stats['building_type_month_ge3_p90_iqr_sgd']:.0f}. These supported cells are a small, selected subset.", "",
             "![Observed type and transaction support](observed_support.png)", "",
             "Monthly observed type sets vary mainly because few flats rent in any one block-month. This does not establish changes in the block's built flat mix or its vacant inventory.", "",
             "## Same-block, same-month flat-type gaps", "",
             "| Comparison | Block-months with both types | Median difference (S$) | 25th–75th percentile (S$) |", "| --- | ---: | ---: | ---: |"]
    for pair, values in pair_gaps.items():
        larger, smaller = pair.split("_minus_")
        lines.append(f"| {larger} minus {smaller} | {values['block_months']:,} | {values['median']:.0f} | {values['q25']:.0f}–{values['q75']:.0f} |")
    lines += ["", "Differences are between the type-specific transaction medians of the same block and approval month. A cell often has only one transaction per type, so these gaps describe a noisy observed sample.", "",
              "## One building over a recent quarter", "",
              f"Illustrative high-volume block: `{example_building}`; {len(building_example)} approvals in January–March 2025. The individual flat/unit numbers are unavailable, so repeated approvals cannot be assigned to the same or different units.", "",
              "| Flat type | Transactions | 25th percentile (S$) | Median (S$) | 75th percentile (S$) |", "| --- | ---: | ---: | ---: | ---: |"]
    for kind, row in building_example_type.iterrows():
        lines.append(f"| {kind} | {int(row.n)} | {row.q25:.0f} | {row['median']:.0f} | {row.q75:.0f} |")
    lines += ["",
              "## Street and completion-year cohort example", "",
              "The source has no HDB project name. Town + street + completion year groups nearby blocks for an example, but can merge distinct projects or split one project. These groups are never labeled as official projects.", "",
              f"There are {len(cohort_buildings):,} such cohorts, including {len(multi_cohorts):,} with at least two observed blocks. "
              f"Among those multi-block cohorts, {int(sets.loc[multi_cohorts].sum()):,} have different **observed** flat-type sets across their blocks. Sampling can cause part of that difference. "
              f"Only {(cohort_month.n.ge(5) & cohort_month.buildings.ge(2)).sum():,} of {len(cohort_month):,} observed cohort-months have at least five transactions across at least two blocks.", "",
              f"Illustrative compact cohort: `{example_cohort}`; {len(example):,} January–March 2025 transactions from {example.building.nunique()} blocks.", "",
              "| Flat type | Transactions | 25th percentile (S$) | Median (S$) | 75th percentile (S$) |", "| --- | ---: | ---: | ---: | ---: |"]
    for kind, row in example_type.iterrows():
        lines.append(f"| {kind} | {int(row.n)} | {row.q25:.0f} | {row['median']:.0f} | {row.q75:.0f} |")
    lines += ["", "![Recent rents by type in the illustrative cohort](cohort_example_rents.png)", "",
              "This example is chosen for transaction support and illustrates the mixture of flat types. It cannot establish the full price distribution of a true named HDB project or isolate floor, interior condition, lease terms, and tenant selection.", "",
              "## Modeling implication", "",
              "At month *t*, describe a block's rent as a flat-type mixture: p(rent | block, t) = Σ_type w(type | block, t) × p(rent | block, type, t). The observed transaction shares estimate the mixture of **approved rentals** in this sample; they are not stock shares or listing probabilities. Fit the type-conditioned distribution with partial pooling across nearby blocks/town and recent months, since most block-month/type cells cannot support a free-standing distribution.", "",
              "HDB's Property Information dataset publishes per-block counts by flat type but lacks a project ID; a dated stock snapshot would help define feasible types and stock weights. Historical snapshots would be needed to make stock features point-in-time safe for forecasting. Source: https://data.gov.sg/datasets/d_17f5382f26140b1fdae0ba2ef6239d2f/view", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/describe_building_rent_distribution.py`."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
