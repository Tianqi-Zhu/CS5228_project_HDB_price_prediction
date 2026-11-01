"""Small, independently cached public government downloads; no credentials required."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import requests
from analysis.config import ROOT
from analysis.data import sha256

CONFIG = ROOT / "analysis/configs/local_economy_landscape.json"


def acquire(config=CONFIG):
    cfg = json.loads(Path(config).read_text())
    cache = ROOT / ".cache" / cfg["experiment"] / "sources"
    out = ROOT / "reports" / cfg["experiment"]
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    session = requests.Session()
    session.trust_env = False
    for name, spec in cfg["datasets"].items():
        path = cache / f"{name}.{spec['format']}"
        if path.exists() and name in manifest and manifest[name].get("sha256") == sha256(path):
            print(f"Cached {name}: {path.stat().st_size:,} bytes", flush=True)
            continue
        entry = {**spec, "source_url": f"https://data.gov.sg/datasets/{spec['id']}/view",
                 "license": "Singapore Open Data Licence", "license_url": "https://data.gov.sg/open-data-licence",
                 "retrieved_at": datetime.now(timezone.utc).isoformat()}
        try:
            url = f"https://api-open.data.gov.sg/v1/public/api/datasets/{spec['id']}/poll-download"
            for attempt in range(6):
                response = session.get(url, timeout=40)
                response.raise_for_status()
                body = response.json()
                download = body.get("data", {}).get("url")
                if download:
                    break
                if attempt == 0:
                    session.get(url.replace("poll-download", "initiate-download"), timeout=40)
                time.sleep(2)
            else:
                raise RuntimeError("Public API did not return a ready download")
            response = session.get(download, timeout=60, stream=True)
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(1024 * 1024):
                content.extend(chunk)
                if len(content) > 25 * 1024**2:
                    raise RuntimeError("Download exceeded the predeclared 25 MiB source cap")
            if spec["format"] == "geojson":
                document = json.loads(content)
                if document.get("type") != "FeatureCollection":
                    raise ValueError("Expected GeoJSON FeatureCollection")
                entry["source_records"] = len(document["features"])
            path.write_bytes(content)
            entry.update(status="acquired", path=str(path.relative_to(ROOT)), sha256=sha256(path), bytes=len(content))
            print(f"Acquired {name}: {len(content):,} bytes", flush=True)
        except Exception as exc:
            entry.update(status="blocked", reason=f"{type(exc).__name__}: {exc}")
            print(f"Blocked {name}: {entry['reason']}", flush=True)
        manifest[name] = entry
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    acquire(args.config)
