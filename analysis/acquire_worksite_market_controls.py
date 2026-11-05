"""Cache historical US equity/semiconductor/SGD controls from FRED."""

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from analysis.config import ROOT
from analysis.data import sha256

SERIES = ["SP500", "NASDAQCOM", "NASDAQSOX", "DEXSIUS"]


def main():
    cache = ROOT / ".cache/company_worksite_followup"
    out = ROOT / "reports/company_worksite_followup"
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    path = cache / "fred_controls_download.bin"
    manifest_path = out / "market_source_manifest.json"
    if path.exists() and manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if sha256(path) != prior["download_sha256"]:
            raise ValueError("Cached FRED response changed")
        url = prior["download_url"]
        retrieved = prior["retrieved_at"]
    else:
        session = requests.Session()
        session.trust_env = False
        response = session.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                               params={"id": ",".join(SERIES), "cosd": "2020-12-01", "coed": "2025-03-31"},
                               headers={"Accept-Encoding": "gzip, deflate"}, timeout=40)
        response.raise_for_status()
        path.write_bytes(response.content)
        url, retrieved = response.url, datetime.now(timezone.utc).isoformat()
    content = path.read_bytes()
    frames = []
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".csv"):
                    # Read bytes in memory; never extract paths from a ZIP.
                    frames.append(pd.read_csv(io.BytesIO(archive.read(name))))
    else:
        frames.append(pd.read_csv(io.BytesIO(content)))
    pieces = []
    for f in frames:
        date = next(c for c in f if "date" in c.lower())
        keep = [c for c in SERIES if c in f]
        if keep:
            piece = f[[date, *keep]].rename(columns={date: "date"}).set_index("date")
            pieces.append(piece)
    daily = pd.concat(pieces, axis=1)
    daily.index = pd.to_datetime(daily.index)
    daily = daily.apply(pd.to_numeric, errors="coerce").sort_index().loc["2020-12-01":"2025-03-31"]
    if set(SERIES) - set(daily) or daily.index.duplicated().any():
        raise ValueError("FRED controls missing/duplicated")
    monthly = daily.groupby(daily.index.to_period("M")).last().reindex(pd.period_range("2020-12", "2025-03", freq="M"))
    if monthly[SERIES].isna().any().any():
        raise ValueError("Incomplete historical control months; do not backfill")
    daily.to_csv(cache / "fred_daily_controls.csv", index_label="date")
    monthly.to_csv(cache / "fred_month_end_levels.csv", index_label="month")
    returns = 100 * (monthly / monthly.shift(1)).apply(__import__("numpy").log)
    returns.to_csv(out / "market_monthly_log_returns.csv", index_label="month")
    manifest = {"download_url": url, "retrieved_at": retrieved, "download_sha256": sha256(path),
                "series": {s: f"https://fred.stlouisfed.org/series/{s}" for s in SERIES},
                "monthly_levels_path": str((cache / "fred_month_end_levels.csv").relative_to(ROOT)),
                "monthly_levels_sha256": sha256(cache / "fred_month_end_levels.csv"),
                "monthly_returns_sha256": sha256(out / "market_monthly_log_returns.csv"),
                "start": str(monthly.index.min()), "end": str(monthly.index.max()),
                "interpretation": "US stock indexes are price indices; DEXSIUS is SGD per USD. Later-vintage historical data, not point-in-time release archives. Only dates no later than March 2025 are retained."}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Acquired {len(monthly)} complete month-end control observations: {SERIES}")


if __name__ == "__main__":
    main()
