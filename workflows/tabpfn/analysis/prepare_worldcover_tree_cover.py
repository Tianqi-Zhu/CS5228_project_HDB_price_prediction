"""Acquire public Singapore WorldCover tiles and extract 100/300 m tree fields."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from analysis.config import ROOT
from analysis.data import sha256
from analysis.compare_local_economy_landscape import write_json
from analysis.experiments.worldcover_tree_cover import extract_product

CONFIG = ROOT / "analysis/configs/worldcover_tree_cover.json"


def locations():
    cfg = json.loads(CONFIG.read_text())
    out = ROOT / "reports" / cfg["experiment"]
    cache = ROOT / ".cache" / cfg["experiment"]
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    return cfg, out, cache


def acquire():
    cfg, out, cache = locations()
    target = out / "source_manifest.json"
    manifest = json.loads(target.read_text()) if target.exists() else {}
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    for spec in cfg["products"]:
        year, version = spec["year"], spec["version"]
        name = f"ESA_WorldCover_10m_{year}_{version}_{cfg['tile']}_Map.tif"
        path = cache / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        previous = manifest.get(str(year), {})
        if path.exists() and previous.get("sha256") == sha256(path):
            print(f"Cached {year}: {path.stat().st_size:,} bytes", flush=True)
            continue
        if path.exists():
            raise ValueError(f"Unverified existing raster: {path}")
        url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/{version}/{year}/map/{name}"
        temporary = path.with_suffix(".tif.part")
        total = 0
        with session.get(url, timeout=(30, 120), stream=True) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("Expected a full raster download, not a partial response")
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    total += len(chunk)
                    if total > cfg["maximum_download_mib"] * 1024**2:
                        raise ValueError("Tile exceeds configured per-file download limit")
                    stream.write(chunk)
            expected = response.headers.get("Content-Length")
            if expected and total != int(expected):
                raise ValueError("Truncated raster download")
        import rasterio
        with rasterio.open(temporary) as raster:
            if raster.count != 1 or raster.width < 10000 or raster.height < 10000:
                raise ValueError("Unexpected WorldCover tile dimensions")
            metadata = {"crs": str(raster.crs), "width": raster.width, "height": raster.height, "nodata": raster.nodata}
        temporary.replace(path)
        manifest[str(year)] = {**spec, "url": url, "documentation_url": cfg["source_url"],
                              "release_source_url": cfg["release_source_url"], "license": "CC-BY-4.0",
                              "attribution": f"ESA WorldCover project {year}; contains modified Copernicus Sentinel data ({year}) processed by ESA WorldCover consortium",
                              "path": str(path.relative_to(ROOT)), "bytes": total, "sha256": sha256(path),
                              "retrieved_at": datetime.now(timezone.utc).isoformat(), **metadata}
        write_json(target, manifest)
        print(f"Acquired {year}: {total:,} bytes", flush=True)


def extract():
    cfg, out, cache = locations()
    manifest = json.loads((out / "source_manifest.json").read_text())
    building_path = ROOT / "reports" / cfg["parent_experiment"] / "canonical_buildings.csv"
    buildings = pd.read_csv(building_path)
    signatures = {"config_sha256": sha256(CONFIG), "buildings_sha256": sha256(building_path),
                  "code_sha256": sha256(ROOT / "analysis/experiments/worldcover_tree_cover.py"),
                  "source_sha256": {year: entry["sha256"] for year, entry in manifest.items()}}
    target = out / "extraction_manifest.json"
    if target.exists():
        saved = json.loads(target.read_text())
        if saved["inputs"] != signatures:
            raise ValueError("Extraction inputs changed; preserve existing experiment artifacts")
        if all(sha256(ROOT / entry["path"]) == entry["sha256"] for entry in saved["outputs"].values()):
            print("Reused verified tree-cover extraction", flush=True)
            return
        raise ValueError("Extracted feature artifact changed")
    outputs, summaries = {}, []
    for spec in cfg["products"]:
        entry = manifest[str(spec["year"])]
        raster = ROOT / entry["path"]
        if sha256(raster) != entry["sha256"]:
            raise ValueError("Source raster checksum changed")
        frame = extract_product(raster, buildings, cfg)
        path = cache / f"building_features_{spec['year']}.csv"
        frame.to_csv(path, index=False, float_format="%.10f")
        outputs[str(spec["year"])] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "buildings": len(frame)}
        for column in frame.columns.drop("building_id"):
            summaries.append({"year": spec["year"], "feature": column, "buildings": len(frame), "missing": int(frame[column].isna().sum()),
                              "minimum": float(frame[column].min()), "mean": float(frame[column].mean()), "maximum": float(frame[column].max())})
        print(f"Extracted {spec['year']}: {len(frame):,} canonical buildings", flush=True)
    pd.DataFrame(summaries).to_csv(out / "building_feature_summary.csv", index=False)
    write_json(target, {"inputs": signatures, "outputs": outputs, "method": "EPSG:3414 10 m nearest-neighbour grid; pixel centres inside metric circles; valid-pixel denominator",
                        "tree_classes": cfg["tree_classes"], "mangroves_included": False, "minimum_valid_fraction": cfg["minimum_valid_fraction"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["acquire", "extract", "all"], default="all")
    args = parser.parse_args()
    if args.stage in ("acquire", "all"):
        acquire()
    if args.stage in ("extract", "all"):
        extract()
