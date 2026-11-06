"""Cache public OneMap geocodes for independently documented company worksites."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import requests

from analysis.config import ROOT
from analysis.data import sha256

CONFIG = ROOT / "analysis/configs/company_worksites.json"
OUT = ROOT / "reports/company_worksite_stocks"
CACHE = ROOT / ".cache/company_worksite_stocks/geocodes"


def acquire():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    rows, manifest = [], []
    for site in config["sites"]:
        path = CACHE / f"{site['postal']}.json"
        url = "https://www.onemap.gov.sg/api/common/elastic/search"
        params = {"searchVal": site["postal"], "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1}
        if not path.exists():
            for attempt in range(5):
                response = session.get(url, params=params, timeout=25)
                if response.status_code != 429:
                    break
                delay = min(30, max(float(response.headers.get("Retry-After", 3)), 2 ** attempt))
                print(f"OneMap rate limit; retrying {site['site_id']} in {delay:g}s", flush=True)
                time.sleep(delay)
            response.raise_for_status()
            body = response.json()
            path.write_text(json.dumps({"retrieved_at": datetime.now(timezone.utc).isoformat(),
                                        "url": response.url, "response": body}, indent=2), encoding="utf-8")
            time.sleep(1.2)
        saved = json.loads(path.read_text(encoding="utf-8"))
        matches = [r for r in saved["response"].get("results", []) if r["POSTAL"] == site["postal"]]
        # An exact postal match must also match the documented street number.
        matches = [r for r in matches if r["BLK_NO"].upper() == site["address"].split()[0].upper()]
        # A large industrial address may also return a named tenant (for
        # example the vehicle inspection centre at 249 Jalan Boon Lay).
        # Prefer the generic address pin, never an unrelated tenant's pin.
        generic = [r for r in matches if r["BUILDING"] == "NIL"]
        if len(generic) == 1:
            matches = generic
        coordinates = {(float(r["LATITUDE"]), float(r["LONGITUDE"])) for r in matches}
        if len(coordinates) != 1:
            raise ValueError(f"Ambiguous/unavailable documented address: {site['site_id']}: {matches}")
        match = matches[0]
        rows.append({**site, "latitude": float(match["LATITUDE"]), "longitude": float(match["LONGITUDE"]),
                     "onemap_address": match["ADDRESS"], "geocode_source": saved["url"],
                     "geocode_retrieved_at": saved["retrieved_at"]})
        manifest.append({"site_id": site["site_id"], "cache": str(path.relative_to(ROOT)), "sha256": sha256(path)})
        print(f"Verified {site['site_id']}: {match['ADDRESS']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "worksite_locations.csv", index=False)
    (OUT / "geocode_manifest.json").write_text(json.dumps({"config_sha256": sha256(CONFIG), "geocodes": manifest}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    acquire()
