"""Write consumer fixtures with hand-computable expected values.

A consumer must be able to load these and assert exact numbers before touching real
data.  Two fixtures:

``synthetic_dbs.csv``
    Geometric prices where every valid log return is exactly ``0.002``, one missing
    close, and a five-business-day hole that must be masked on the calendar rule.

``synthetic_coe.csv``
    Two complete months with two rounds each, one single-round final month, and one
    category carrying a genuine zero premium.

``expected_values.json`` records the values a consumer should reproduce.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.fusion_contract_v3 import coe, contract, dbs


def write_fixtures(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)

    # ---------------- DBS ----------------
    rows = 90
    daily = 0.002
    dates = pd.bdate_range(start="2021-01-04", periods=rows)
    price = np.exp(np.arange(rows) * daily) * 20.0
    stock = pd.DataFrame({
        "NAME": "Synthetic Bank", "SYMBOL": contract.DBS_SYMBOL,
        "DATE": dates.strftime("%Y-%m-%d"),
        "OPEN": price, "HIGH": price, "LOW": price, "CLOSE": price,
        "ADJUSTED_CLOSE": price,
        "VOLUME": np.full(rows, 1_000_000.0),
    })
    # one missing close (row 40) and five consecutive missing rows (60..64)
    stock.loc[40, ["OPEN", "HIGH", "LOW", "CLOSE", "ADJUSTED_CLOSE", "VOLUME"]] = np.nan
    stock = pd.concat([
        stock.iloc[:60],
        stock.iloc[65:],
    ], ignore_index=True)
    stock_path = output / "synthetic_dbs.csv"
    stock.to_csv(stock_path, index=False)

    frame = dbs.build_dbs_frame(stock)
    finite = frame.log_return[np.isfinite(frame.log_return)]
    expected = {
        "dbs": {
            "symbol": contract.DBS_SYMBOL,
            "rows": int(len(stock)),
            "observed": int(frame.observed.sum()),
            "daily_log_return": daily,
            "n_finite_returns": int(finite.size),
            "all_finite_returns_equal": float(np.ptp(finite)),
            "max_calendar_gap_days": int(frame.gap_days.max()),
            "n_masked_returns": int(np.isnan(frame.log_return).sum() - 1),
            "note": "the observation after a hole is masked; no zero or oversized "
                    "one-day return is invented",
            "consumers_should_call": "dbs.build_dbs_frame(pd.read_csv(<this file>))",
        },
        "paths": {"dbs": str(stock_path.name)},
    }

    # ---------------- COE ----------------
    records = []
    for category in contract.COE_CATEGORIES:
        records += [
            (2021, category, "jan", 1, "$40,000", 1000, "1,200"),
            (2021, category, "jan", 2, "$42,000", 1000, "1,300"),
            (2021, category, "feb", 1, "$44,000", 900, "1,100"),
            (2021, category, "feb", 2, "$46,000", 900, "1,200"),
            (2021, category, "mar", 1, "$50,000", 800, "1,000"),
        ]
    # category e: a genuine zero premium in the first round of January
    records = [row for row in records
               if not (row[1] == "e" and row[2] == "jan" and row[3] == 1)]
    records.append((2021, "e", "jan", 1, "$0", 1000, "1,200"))
    columns = ["YEAR", "CATEGORY", "MONTH", "ROUND", "QUOTA_PREMIUM", "QUOTA", "BIDS_RECEIVED"]
    coe_frame = pd.DataFrame(records, columns=columns)
    coe_path = output / "synthetic_coe.csv"
    coe_frame.to_csv(coe_path, index=False)

    monthly = coe.build_coe_monthly(coe.parse_coe_frame(coe_path))
    a_index = list(monthly.categories).index("a")
    e_index = list(monthly.categories).index("e")
    jan = monthly.month_position(0)
    mar = monthly.month_position(2)
    expected["coe"] = {
        "categories": list(monthly.categories),
        "n_months": int(len(monthly.months)),
        "a_january": {
            "premium": float(monthly.premium[a_index, jan]),
            "quota": float(monthly.quota[a_index, jan]),
            "bids": float(monthly.bids[a_index, jan]),
            "ratio": float(monthly.ratio[a_index, jan]),
            "auctions": float(monthly.auctions[a_index, jan]),
            "incomplete": float(monthly.incomplete[a_index, jan]),
        },
        "e_january_zero_premium": {
            "premium": float(monthly.premium[e_index, jan]),
            "observed": float(monthly.observed[e_index, jan]),
            "note": "a genuine $0 round is kept and averaged, not treated as missing",
        },
        "march_incomplete": {
            "a_incomplete": float(monthly.incomplete[a_index, mar]),
            "note": "single round -> flagged incomplete",
        },
        "last_source_month": contract.month_label(int(monthly.last_source_month)),
        "consumers should call": "coe.build_coe_monthly(coe.parse_coe_frame(<this file>))",
    }
    expected["paths"]["coe"] = coe_path.name

    (output / "expected_values.json").write_text(json.dumps(expected, indent=2) + "\n")
    return expected


if __name__ == "__main__":
    target = contract.REPORT_ROOT / "fixtures"
    payload = write_fixtures(target)
    print(json.dumps(payload, indent=2))
