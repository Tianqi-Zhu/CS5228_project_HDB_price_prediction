"""Audit whether competition records occur in public HDB rental disclosures.

Public-label lookup is reported as source reconstruction, never as a temporal
forecast score. This module does not modify modeling inputs or submissions.
"""

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

from analysis.pipeline.config import DATA, ROOT, TARGET, STUDY_REPORTS
from analysis.pipeline.data import check_output_path, load_main, protect_sources, sha256
from analysis.pipeline.evaluation import metrics


DATASET_ID = "d_c9f57187485a850908655db0e8cfe651"
SOURCE_PAGE = f"https://data.gov.sg/datasets/{DATASET_ID}/view"
API = "https://data.gov.sg/api/action/datastore_search"
CACHE = ROOT / ".cache/public_rents/hdb_public_rents.csv"
OUT = STUDY_REPORTS / "source_overlap"
KEYS = ["month", "town", "block", "street", "flat_type"]
STREET_WORDS = {
    "AVENUE": "AVE", "ROAD": "RD", "STREET": "ST", "DRIVE": "DR", "CRESCENT": "CRES",
    "CLOSE": "CL", "PLACE": "PL", "BOULEVARD": "BLVD", "CENTRAL": "CTRL", "JALAN": "JLN",
    "BUKIT": "BT", "UPPER": "UPP", "TANJONG": "TG", "KAMPONG": "KG", "LORONG": "LOR",
    "SAINT": "ST", "TERRACE": "TER", "MOUNT": "MT",
    "NORTH": "NTH", "SOUTH": "STH", "GARDENS": "GDNS", "HEIGHTS": "HTS",
    "PARK": "PK", "COMMONWEALTH": "CWEALTH", "MARKET": "MKT",
}


def download(destination=CACHE):
    destination = check_output_path(Path(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    records, total = [], None
    while total is None or len(records) < total:
        query = urllib.parse.urlencode({"resource_id": DATASET_ID, "limit": 10000, "offset": len(records)})
        request = urllib.request.Request(API + "?" + query,
            headers={"User-Agent": "CS5228-source-audit/1.0", "Accept": "application/json"})
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
            raise ValueError(payload)
        page = payload["result"]
        total = int(page["total"])
        if not page["records"]:
            raise ValueError("Public API returned an empty page before the declared total")
        records.extend(page["records"])
        print(f"Downloaded {len(records):,}/{total:,} public records", flush=True)
        time.sleep(.35)
    frame = pd.DataFrame(records)
    if len(frame) != total or frame["_id"].duplicated().any():
        raise ValueError("Public download is incomplete or contains duplicated record IDs")
    frame.to_csv(destination, index=False)
    metadata = {"source_page": SOURCE_PAGE, "api": API, "dataset_id": DATASET_ID,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "rows": len(frame),
        "sha256": sha256(destination)}
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame


def text(values):
    return values.astype("string").fillna("").str.upper().str.strip().str.replace(r"\s+", " ", regex=True)


def canonical_keys(frame, public=False):
    def column(local, official):
        return frame[official if public else local]
    result = pd.DataFrame(index=frame.index)
    result["month"] = pd.to_datetime(column("RENT_APPROVAL_DATE", "rent_approval_date"), errors="raise").dt.strftime("%Y-%m")
    result["town"] = text(column("TOWN", "town")).str.replace(r"\s*/\s*", "/", regex=True).replace({"CENTRAL AREA": "CENTRAL"})
    result["block"] = text(column("BLOCK", "block")).str.replace(r"\s+", "", regex=True).str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    street = text(column("STREET", "street_name")).str.replace("'", "", regex=False).str.replace(r"[^A-Z0-9 ]", " ", regex=True)
    result["street"] = street.map(lambda value: " ".join(STREET_WORDS.get(word, word) for word in value.split()))
    result["flat_type"] = text(column("FLAT_TYPE", "flat_type")).str.replace(r"[^A-Z0-9]", "", regex=True)
    return result


def audit(cache=CACHE, output_dir=OUT):
    out = check_output_path(Path(output_dir))
    out.mkdir(parents=True, exist_ok=True)
    public = pd.read_csv(cache, dtype={"block": "string"})
    values = canonical_keys(public, public=True)
    values["rent"] = pd.to_numeric(public["monthly_rent"], errors="raise")
    groups = values.groupby(KEYS, dropna=False)["rent"]
    stats = groups.agg(public_mean="mean", public_sum="sum", public_count="size", public_unique_rents="nunique").reset_index()
    label_sets = groups.agg(lambda rents: frozenset(rents)).to_dict()
    sources = [DATA / "train.csv", DATA / "test.csv"]
    with protect_sources(sources) as source_hashes:
        train, test = load_main(sources[0], require_target=True), load_main(sources[1])
        results = {}
        for name, frame in (("train", train), ("test", test)):
            keys = canonical_keys(frame)
            keys["source_row"] = np.arange(len(frame))
            joined = keys.merge(stats, on=KEYS, how="left", validate="many_to_one", sort=False)
            if not np.array_equal(joined.source_row, np.arange(len(frame))):
                raise ValueError("Source audit changed competition row order")
            matched = joined.public_count.notna()
            result = {"rows": len(frame), "matched_rows": int(matched.sum()),
                "key_match_fraction": float(matched.mean()),
                "single_rent_value_rows": int((matched & joined.public_unique_rents.eq(1)).sum()),
                "ambiguous_rent_rows": int((matched & joined.public_unique_rents.gt(1)).sum())}
            joined["matched"] = matched
            joined.groupby("month").agg(rows=("matched", "size"), matched=("matched", "sum")).to_csv(out / f"{name}_coverage_by_month.csv")
            if name == "train":
                labels = frame[TARGET].to_numpy(float)
                found = np.array([label in label_sets.get(tuple(key), ())
                    for key, label in zip(keys[KEYS].itertuples(index=False, name=None), labels)])
                result["label_present_in_public_group_rows"] = int(found.sum())
                result["label_present_fraction"] = float(found.mean())
                result["public_group_mean_reconstruction_matched_rmse_sgd"] = metrics(
                    labels[matched], joined.loc[matched, "public_mean"].to_numpy(float))["rmse_sgd"]
                latest = frame["RENT_APPROVAL_DATE"].ge("2024-04").to_numpy()
                evaluated = latest & matched.to_numpy()
                result["latest_known_label_matched_rows"] = int(evaluated.sum())
                result["latest_public_mean_reconstruction_rmse_sgd"] = metrics(
                    labels[evaluated], joined.loc[evaluated, "public_mean"].to_numpy(float))["rmse_sgd"]
                known_model = ROOT / "output/experiment_runs/comparison_model_search/2024-04/lightgbm/predictions.csv"
                if known_model.exists():
                    baseline = pd.read_csv(known_model).set_index("source_row")
                    positions = np.flatnonzero(latest)
                    reconstructed = joined.loc[positions, "public_mean"].to_numpy(float)
                    reconstructed = np.where(np.isfinite(reconstructed), reconstructed,
                        baseline.loc[positions, "prediction"].to_numpy(float))
                    result["latest_public_lookup_with_model_fallback_rmse_sgd"] = metrics(labels[latest], reconstructed)["rmse_sgd"]
                    count = joined.loc[positions, "public_count"].to_numpy(float) - found[positions].astype(float)
                    total = joined.loc[positions, "public_sum"].to_numpy(float) - labels[positions] * found[positions]
                    excluded = np.divide(total, count, out=np.full(len(positions), np.nan), where=count > 0)
                    excluded = np.where(np.isfinite(excluded), excluded,
                        baseline.loc[positions, "prediction"].to_numpy(float))
                    result["latest_remove_own_public_rent_then_model_fallback_rmse_sgd"] = metrics(labels[latest], excluded)["rmse_sgd"]
            results[name] = result
        report = {"source_page": SOURCE_PAGE, "public_rows": len(public),
            "public_months": [values.month.min(), values.month.max()], "public_sha256": sha256(Path(cache)),
            "competition_source_sha256": source_hashes, "results": results,
            "interpretation": "Public-label reconstruction audit, not a valid past-only forecasting score. Matching uses only month, town, block, canonical street and flat type. No test labels are supplied by the competition files, and no Kaggle submission is generated.",
            "rules_status": "Source reconstruction is an audit, not an assertion that target recovery is permitted. See competition_context.json for verified site wording."}
        context = out / "competition_context.json"
        if context.exists():
            report["competition_context"] = json.loads(context.read_text(encoding="utf-8"))
        (out / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.download:
        download(args.cache)
    return audit(args.cache, args.output_dir)


if __name__ == "__main__":
    main()
