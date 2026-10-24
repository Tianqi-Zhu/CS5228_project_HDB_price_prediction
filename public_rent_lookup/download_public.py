"""Optionally download a fresh public snapshot; the pinned snapshot is included."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from .run import ROOT, SOURCE_PAGE, sha256


DATASET_ID = "d_c9f57187485a850908655db0e8cfe651"
API = "https://data.gov.sg/api/action/datastore_search"
REQUIRED = {"_id", "rent_approval_date", "town", "block", "street_name", "flat_type", "monthly_rent"}


def download(destination: Path, *, page_size: int = 10000, force: bool = False) -> dict:
    destination = Path(destination).resolve()
    data_root = (ROOT / "data").resolve()
    if destination == data_root or data_root in destination.parents:
        raise ValueError("Public downloads cannot overwrite competition data/")
    metadata_path = destination.with_suffix(".json")
    if not force and (destination.exists() or metadata_path.exists()):
        raise FileExistsError("Destination exists; choose a new path or pass --force")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    records, total = [], None
    while total is None or len(records) < total:
        query = urllib.parse.urlencode({"resource_id": DATASET_ID,
                                        "limit": page_size, "offset": len(records)})
        request = urllib.request.Request(API + "?" + query, headers={
            "User-Agent": "CS5228-public-rent-lookup/1.0", "Accept": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
        if not payload.get("success"):
            raise ValueError("Official API reported an unsuccessful request")
        page = payload["result"]
        page_total = int(page["total"])
        if total is not None and total != page_total:
            raise ValueError("Dataset size changed during pagination; retry later")
        total = page_total
        if not page["records"]:
            raise ValueError("Empty page before the declared dataset total")
        records.extend(page["records"])
        print(f"Downloaded {len(records):,}/{total:,}", flush=True)
        if len(records) < total:
            time.sleep(.35)
    public = pd.DataFrame(records)
    if not REQUIRED.issubset(public.columns):
        raise ValueError("Downloaded snapshot has missing required fields")
    if len(public) != total or public["_id"].duplicated().any():
        raise ValueError("Downloaded snapshot has incorrect count or duplicate IDs")
    rents = pd.to_numeric(public["monthly_rent"], errors="raise").to_numpy(float)
    if not np.isfinite(rents).all() or (rents <= 0).any():
        raise ValueError("Downloaded snapshot has invalid rents")
    destination.parent.mkdir(parents=True, exist_ok=True)
    public.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n")
    metadata = {"source_page": SOURCE_PAGE, "api": API, "dataset_id": DATASET_ID,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "rows": len(public), "sha256": sha256(destination)}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "source/hdb_public_rents_latest.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(json.dumps(download(args.output, force=args.force), indent=2))


if __name__ == "__main__":
    main()
