"""Join a sourced SLA named-estate sample to HDB rentals and summarize rents."""
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


CONFIG = ROOT / "analysis/configs/hdb_sla_named_estates_2023.json"
OUT = ROOT / "reports/sla_hdb_estate_rents"
START, END = "2024-06", "2025-03"


def quantiles(values):
    return pd.Series({f"p{int(q*100):02d}": values.quantile(q) for q in (.1, .25, .5, .75, .9)})


def main():
    source = json.loads(CONFIG.read_text(encoding="utf-8"))
    blocks = normalized_address(pd.read_csv(ROOT / "data/auxiliary/sg-hdb-block.csv"))
    if blocks.duplicated(["TOWN", "STREET", "BLOCK"]).any():
        raise ValueError("Duplicate local HDB addresses")
    blocks = blocks.rename(columns={"TOWN": "town", "STREET": "street", "BLOCK": "block"})
    raw = prepare_main(pd.read_csv(ROOT / "data/train.csv"))
    rentals = normalized_address(raw).rename(columns={"TOWN": "town", "STREET": "street", "BLOCK": "block"})
    rentals["month"] = rentals.RENT_APPROVAL_DATE
    rentals["flat_type"] = rentals.FLAT_TYPE_CLEAN
    rentals["rent_sgd"] = rentals.MONTHLY_RENT.astype(float)
    if rentals["rent_sgd"].isna().any() or (rentals["rent_sgd"] <= 0).any():
        raise ValueError("Invalid rental target")
    listed = []
    for entry in source["entries"]:
        for street, block_list in entry["addresses"].items():
            for block in block_list.split():
                listed.append({"sla_serial": entry["sn"], "estate": entry["name"],
                               "town": entry["town"], "street": street, "block": block,
                               "source_url": source["source_url"], "source_as_of": "2023-08-07"})
    listed = pd.DataFrame(listed)
    keys = ["town", "street", "block"]
    if listed.duplicated(keys).any() or listed.sla_serial.nunique() != len(source["entries"]):
        raise ValueError("Ambiguous or duplicate sourced addresses")
    lookup = listed.merge(blocks[keys + ["POSTAL_CODE", "YEAR_COMPLETED", "LATITUDE", "LONGITUDE"]],
                          on=keys, how="left", validate="one_to_one", indicator="local_block_join")
    counts = rentals.groupby(keys).size().rename("rental_rows_2021_2025").reset_index()
    lookup = lookup.merge(counts, on=keys, how="left", validate="one_to_one")
    lookup["rental_rows_2021_2025"] = lookup["rental_rows_2021_2025"].fillna(0).astype(int)
    lookup["match_status"] = np.where(lookup.local_block_join.eq("both"), "matched_local_block", "not_in_local_block_table")
    lookup = lookup.drop(columns="local_block_join")
    OUT.mkdir(parents=True, exist_ok=True)
    lookup.to_csv(OUT / "sla_estate_block_mapping.csv", index=False)

    assigned = rentals.merge(listed[keys + ["sla_serial", "estate"]], on=keys, how="inner", validate="many_to_one")
    assigned["block_id"] = assigned.town + "|" + assigned.street + "|" + assigned.block
    assigned["window"] = np.where(assigned.month.between(START, END), "2024-06_to_2025-03", "all_2021-01_to_2025-03")
    if not set(assigned[keys].itertuples(index=False, name=None)).issubset(set(lookup[keys].itertuples(index=False, name=None))):
        raise ValueError("Unexpected rental assignment")
    assigned[["sla_serial", "estate", "town", "street", "block", "month", "flat_type", "rent_sgd"]].to_csv(
        OUT / "estate_rental_transactions.csv", index=False)
    selected = assigned[assigned.month.between(START, END)].copy()

    support = lookup.groupby(["sla_serial", "estate", "town"]).agg(
        listed_blocks=("block", "size"), matched_blocks=("match_status", lambda x: int(x.eq("matched_local_block").sum())),
        rented_blocks=("rental_rows_2021_2025", lambda x: int(x.gt(0).sum())),
        all_rentals=("rental_rows_2021_2025", "sum")).reset_index()
    window_support = selected.groupby("sla_serial").agg(
        validation_rentals=("rent_sgd", "size"), validation_blocks=("block_id", "nunique"),
        validation_flat_types=("flat_type", "nunique")).reset_index()
    support = support.merge(window_support, on="sla_serial", how="left", validate="one_to_one")
    for c in ("validation_rentals", "validation_blocks", "validation_flat_types"):
        support[c] = support[c].fillna(0).astype(int)
    support.to_csv(OUT / "estate_coverage.csv", index=False)

    def distribution(frame, columns):
        counts = frame.groupby(columns).agg(n=("rent_sgd", "size"), blocks=("block_id", "nunique")).reset_index()
        qs = frame.groupby(columns).rent_sgd.apply(quantiles).unstack(-1).reset_index()
        return counts.merge(qs, on=columns, validate="one_to_one")

    full_distribution = distribution(assigned, ["sla_serial", "estate", "town", "flat_type"])
    full_distribution.insert(0, "window", "2021-01_to_2025-03")
    late_distribution = distribution(selected, ["sla_serial", "estate", "town", "flat_type"])
    late_distribution.insert(0, "window", "2024-06_to_2025-03")
    pd.concat([full_distribution, late_distribution], ignore_index=True).to_csv(
        OUT / "estate_type_rent_distribution.csv", index=False)
    block_distribution = distribution(selected, ["sla_serial", "estate", "town", "street", "block", "flat_type"])
    block_distribution.to_csv(OUT / "estate_block_type_rent_distribution_recent.csv", index=False)
    monthly = distribution(assigned, ["sla_serial", "estate", "town", "month", "flat_type"])
    monthly.to_csv(OUT / "estate_month_type_rent_distribution.csv", index=False)

    valid_cells = monthly[monthly.month.between(START, END)]
    supported = valid_cells[(valid_cells.n >= 3) & (valid_cells.blocks >= 2)].copy()
    supported["iqr"] = supported.p75 - supported.p25
    summary = {
        "source_name": source["source"], "source_url": source["source_url"],
        "source_config_sha256": sha256(CONFIG),
        "rental_sha256": sha256(ROOT / "data/train.csv"),
        "block_sha256": sha256(ROOT / "data/auxiliary/sg-hdb-block.csv"),
        "named_estates_sampled": int(len(support)),
        "listed_addresses": int(len(lookup)),
        "matched_local_addresses": int(lookup.match_status.eq("matched_local_block").sum()),
        "addresses_with_rentals": int(lookup.rental_rows_2021_2025.gt(0).sum()),
        "rental_rows_all": int(len(assigned)),
        "rental_rows_validation_window": int(len(selected)),
        "validation_month_type_cells_observed": int(len(valid_cells)),
        "validation_cells_ge3_and_ge2_blocks": int(len(supported)),
        "validation_supported_cell_median_iqr_sgd": float(supported.iqr.median()),
        "validation_supported_cell_p90_iqr_sgd": float(supported.iqr.quantile(.9)),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Compare the 4-room distribution across estates with sufficient support.
    four = late_distribution.query("flat_type == '4-room' and n >= 25 and blocks >= 3").sort_values("n", ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    values = [selected.loc[(selected.sla_serial == row.sla_serial) & selected.flat_type.eq("4-room"), "rent_sgd"].to_numpy()
              for row in four.itertuples()]
    ax.boxplot(values, tick_labels=[f"{row.estate}\n(n={row.n})" for row in four.itertuples()], showfliers=False)
    ax.set(ylabel="Monthly rent (S$)", title="4-room rents in sampled named HDB estates, Jun 2024–Mar 2025")
    ax.tick_params(axis="x", labelrotation=35)
    fig.savefig(OUT / "four_room_estate_distributions.png", dpi=160)
    plt.close(fig)

    target = late_distribution[late_distribution.n >= 20].sort_values(["n", "estate"], ascending=[False, True]).iloc[0]
    sample = selected[selected.sla_serial.eq(target.sla_serial)]
    type_order = [t for t in ("2-room", "3-room", "4-room", "5-room", "executive") if sample.flat_type.eq(t).sum() >= 3]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    vals = [sample.loc[sample.flat_type.eq(t), "rent_sgd"].to_numpy() for t in type_order]
    ax.boxplot(vals, tick_labels=[f"{t}\n(n={len(v)})" for t, v in zip(type_order, vals)], showfliers=False)
    ax.set(ylabel="Monthly rent (S$)", title=f"{target.estate}: rent by flat type, Jun 2024–Mar 2025")
    fig.savefig(OUT / "one_estate_type_distributions.png", dpi=160)
    plt.close(fig)

    lines = ["# Rent distributions in sourced HDB estate groups", "",
             "SLA's *Computerised HDB Estates* list (updated 7 August 2023) explicitly groups multiple blocks under named HDB estates. The saved crosswalk transcribes 27 named, multi-block entries and joins each listed town/street/block address to the local HDB block table and the 150,000 labeled rent approvals. This is a selected sample of named estates, not a complete precinct boundary map. A named estate is used as an observable precinct-scale grouping; legal/upgrade precinct boundaries can differ.", "",
             f"[SLA source PDF]({source['source_url']}) · [HDB's NRP explanation](https://www.hdb.gov.sg/managing-my-home/upgrading-and-redevelopment/neighbourhood-renewal-programme-nrp)", "",
             "## Coverage", "",
             f"The source lists {summary['listed_addresses']:,} addresses across {summary['named_estates_sampled']} selected estates; {summary['matched_local_addresses']:,} match the local block table and {summary['addresses_with_rentals']:,} have at least one labeled rental. There are {summary['rental_rows_all']:,} assigned rentals from January 2021–March 2025 and {summary['rental_rows_validation_window']:,} from June 2024–March 2025. Missing matches can be nonresidential blocks or differences between the 2023 SLA list and the local block table.", "",
             "| SLA estate | Town | Listed blocks | Matched | Rented blocks | Rentals Jun 2024–Mar 2025 |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in support.sort_values("validation_rentals", ascending=False).itertuples():
        lines.append(f"| {row.estate} | {row.town} | {row.listed_blocks} | {row.matched_blocks} | {row.rented_blocks} | {row.validation_rentals} |")
    lines += ["", "## Rent distributions", "",
              "The distribution files report transaction count, contributing blocks, and 10th/25th/50th/75th/90th percentiles by estate and flat type, for both the full labeled period and the ten-month recent window. The monthly file gives the same summaries at a single time step. These are approval distributions; they do not reveal the rents of occupied units that did not transact.", "",
              "![4-room rents in selected estates](four_room_estate_distributions.png)", "",
              "The table compares 4-room approval rents in the recent ten-month window. Each range is the 25th to 75th percentile; these pooled ranges can include changes over time as well as differences between blocks and units.", "",
              "| Named estate | Rentals | Blocks | 25th percentile | Median | 75th percentile |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in late_distribution.query("flat_type == '4-room' and n >= 10 and blocks >= 3").sort_values("n", ascending=False).itertuples():
        lines.append(f"| {row.estate} | {row.n} | {row.blocks} | S${row.p25:,.0f} | S${row.p50:,.0f} | S${row.p75:,.0f} |")
    lines += ["",
              f"In June 2024–March 2025 there are {summary['validation_month_type_cells_observed']:,} observed estate × month × flat-type cells. Only {summary['validation_cells_ge3_and_ge2_blocks']:,} have at least three rentals from at least two blocks. Among those cells, median within-cell interquartile range is S${summary['validation_supported_cell_median_iqr_sgd']:.0f} (90th percentile S${summary['validation_supported_cell_p90_iqr_sgd']:.0f}). Most single-month distributions remain thin even after grouping several blocks.", "",
              f"Illustration: {target.estate} has {int(len(sample))} rentals in the ten-month window. Its type-specific distributions are shown below; read medians alongside sample counts.", "",
              "![Rents by type within one estate](one_estate_type_distributions.png)", "",
              "Within this estate, the recent 4-room median also varies across blocks:", "",
              "| Block | Rentals | Median monthly rent |", "| --- | ---: | ---: |"]
    for row in block_distribution[(block_distribution.sla_serial == target.sla_serial) & (block_distribution.flat_type == "4-room")].sort_values("block").itertuples():
        lines.append(f"| {row.block.upper()} | {row.n} | S${row.p50:,.0f} |")
    lines += ["",
              "## Use in forecasting", "",
              "Estimate rent conditional on estate, flat type, and month, with shrinkage toward town × flat-type and national × flat-type levels. Preserve building identity because a named estate can contain blocks with different property mixes. Use only estate maps that existed before the forecast origin; this SLA list is dated 2023, whereas its public PDF includes older entry effective dates. Estate boundaries and named developments are not identical to every HDB upgrading precinct.", "",
              "Saved data: [block mapping](sla_estate_block_mapping.csv), [estate coverage](estate_coverage.csv), [estate × type distributions](estate_type_rent_distribution.csv), [recent block × type distributions](estate_block_type_rent_distribution_recent.csv), [monthly estate × type distributions](estate_month_type_rent_distribution.csv), [assigned rental rows](estate_rental_transactions.csv), [summary](summary.json).", "",
              "Reproduce: `.venv/Scripts/python.exe analysis/analyze_sla_hdb_estate_rents.py`."]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
